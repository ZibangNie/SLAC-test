from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]|[^\w\s]",
    re.UNICODE,
)


@dataclass
class ProjectorConfig:
    max_chunk_atoms: int = 64
    min_chunk_atoms: int = 2
    max_chunk_chars: int = 1600
    min_chunk_chars: int = 20
    max_chunk_tokens: int = 384
    min_chunk_tokens: int = 48


DEFAULT_PROJECTOR_CONFIG = ProjectorConfig()


def token_len_proxy(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def boundary_vector_to_spans(num_atoms: int, b: Sequence[int]) -> List[Tuple[int, int]]:
    if num_atoms <= 0:
        return []

    gaps = [i for i, x in enumerate(b) if int(x) == 1]
    spans: List[Tuple[int, int]] = []
    start = 0
    for g in gaps:
        end = g + 1
        spans.append((start, end))
        start = end
    spans.append((start, num_atoms))
    return spans


def spans_to_boundary_vector(num_atoms: int, spans: Sequence[Tuple[int, int]]) -> List[int]:
    num_gaps = max(0, num_atoms - 1)
    b = [0] * num_gaps
    for i in range(len(spans) - 1):
        _, e = spans[i]
        g = e - 1
        if 0 <= g < num_gaps:
            b[g] = 1
    return b


def spans_to_units(atoms_text: Sequence[str], spans: Sequence[Tuple[int, int]]) -> List[Dict]:
    units = []
    for uid, (s, e) in enumerate(spans):
        text = "\n".join(atoms_text[s:e]).strip()
        units.append(
            {
                "unit_id": uid,
                "text": text,
                "start_atom": s,
                "end_atom": e,
            }
        )
    return units


def _chunk_text(atoms_text: Sequence[str], span: Tuple[int, int]) -> str:
    s, e = span
    return "\n".join(atoms_text[s:e]).strip()


def _chunk_stats(atoms_text: Sequence[str], span: Tuple[int, int]) -> Dict[str, int]:
    text = _chunk_text(atoms_text, span)
    s, e = span
    return {
        "atoms": e - s,
        "chars": len(text),
        "tokens": token_len_proxy(text),
    }


def _is_overlong(stats: Dict[str, int], cfg: ProjectorConfig) -> bool:
    return (
        stats["atoms"] > cfg.max_chunk_atoms
        or stats["chars"] > cfg.max_chunk_chars
        or stats["tokens"] > cfg.max_chunk_tokens
    )


def _is_too_short(stats: Dict[str, int], cfg: ProjectorConfig) -> bool:
    return (
        stats["atoms"] < cfg.min_chunk_atoms
        or stats["chars"] < cfg.min_chunk_chars
        or stats["tokens"] < cfg.min_chunk_tokens
    )


def _pick_split_gap(
    span: Tuple[int, int],
    gap_scores: Optional[Sequence[float]] = None,
) -> Optional[int]:
    """
    Pick an internal split gap g for span [s,e), where valid internal gaps are [s, e-2].
    Prefer the highest gap score if provided; otherwise use midpoint.
    """
    s, e = span
    if e - s <= 1:
        return None

    internal_gaps = list(range(s, e - 1))
    if not internal_gaps:
        return None

    if gap_scores is not None:
        best_g = max(internal_gaps, key=lambda g: float(gap_scores[g]))
        return best_g

    mid = (s + e) // 2
    # gap index corresponds to boundary between mid-1 and mid
    g = mid - 1
    g = max(s, min(e - 2, g))
    return g


def _hard_max_split_once(
    spans: List[Tuple[int, int]],
    atoms_text: Sequence[str],
    cfg: ProjectorConfig,
    gap_scores: Optional[Sequence[float]] = None,
) -> Tuple[List[Tuple[int, int]], bool]:
    out: List[Tuple[int, int]] = []
    changed = False

    for span in spans:
        stats = _chunk_stats(atoms_text, span)
        if _is_overlong(stats, cfg):
            g = _pick_split_gap(span, gap_scores=gap_scores)
            if g is None:
                out.append(span)
                continue
            s, e = span
            out.append((s, g + 1))
            out.append((g + 1, e))
            changed = True
        else:
            out.append(span)

    return out, changed


def _hard_max_split_until_ok(
    spans: List[Tuple[int, int]],
    atoms_text: Sequence[str],
    cfg: ProjectorConfig,
    gap_scores: Optional[Sequence[float]] = None,
    max_rounds: int = 32,
) -> List[Tuple[int, int]]:
    cur = spans
    for _ in range(max_rounds):
        cur, changed = _hard_max_split_once(cur, atoms_text, cfg, gap_scores=gap_scores)
        if not changed:
            break
    return cur


def _merge_two(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return (a[0], b[1])


def _boundary_confidence(
    left_span: Tuple[int, int],
    gap_scores: Optional[Sequence[float]],
) -> float:
    """
    Confidence of the boundary after left_span, i.e. gap = end(left_span)-1
    """
    if gap_scores is None:
        return 0.0
    g = left_span[1] - 1
    if 0 <= g < len(gap_scores):
        return float(gap_scores[g])
    return 0.0


def _choose_merge_side(
    spans: List[Tuple[int, int]],
    idx: int,
    atoms_text: Sequence[str],
    cfg: ProjectorConfig,
    gap_scores: Optional[Sequence[float]] = None,
) -> str:
    """
    Choose merge side for a too-short chunk at spans[idx].
    Rule:
    1) if only one side exists, use that side
    2) otherwise prefer deleting the lower-confidence boundary
    3) tie-break by merged chunk size closeness to target minimum
    """
    has_left = idx > 0
    has_right = idx < len(spans) - 1

    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    if not has_left and not has_right:
        return "none"

    left_conf = _boundary_confidence(spans[idx - 1], gap_scores)
    right_conf = _boundary_confidence(spans[idx], gap_scores)

    if left_conf < right_conf:
        return "left"
    if right_conf < left_conf:
        return "right"

    cur = spans[idx]
    merged_left = _merge_two(spans[idx - 1], cur)
    merged_right = _merge_two(cur, spans[idx + 1])

    left_stats = _chunk_stats(atoms_text, merged_left)
    right_stats = _chunk_stats(atoms_text, merged_right)

    left_deficit = max(0, cfg.min_chunk_atoms - left_stats["atoms"])
    right_deficit = max(0, cfg.min_chunk_atoms - right_stats["atoms"])

    if left_deficit < right_deficit:
        return "left"
    if right_deficit < left_deficit:
        return "right"

    left_size = left_stats["atoms"]
    right_size = right_stats["atoms"]
    return "left" if left_size <= right_size else "right"


def _soft_min_merge_once(
    spans: List[Tuple[int, int]],
    atoms_text: Sequence[str],
    cfg: ProjectorConfig,
    gap_scores: Optional[Sequence[float]] = None,
) -> Tuple[List[Tuple[int, int]], bool]:
    if not spans:
        return spans, False

    for idx, span in enumerate(spans):
        stats = _chunk_stats(atoms_text, span)
        if _is_too_short(stats, cfg):
            side = _choose_merge_side(spans, idx, atoms_text, cfg, gap_scores=gap_scores)

            if side == "left":
                merged = _merge_two(spans[idx - 1], spans[idx])
                new_spans = spans[: idx - 1] + [merged] + spans[idx + 1 :]
                return new_spans, True

            if side == "right":
                merged = _merge_two(spans[idx], spans[idx + 1])
                new_spans = spans[:idx] + [merged] + spans[idx + 2 :]
                return new_spans, True

            return spans, False

    return spans, False


def _soft_min_merge_until_ok(
    spans: List[Tuple[int, int]],
    atoms_text: Sequence[str],
    cfg: ProjectorConfig,
    gap_scores: Optional[Sequence[float]] = None,
    max_rounds: int = 32,
) -> List[Tuple[int, int]]:
    cur = spans
    for _ in range(max_rounds):
        cur, changed = _soft_min_merge_once(cur, atoms_text, cfg, gap_scores=gap_scores)
        if not changed:
            break
    return cur


def project_boundary_vector(
    atoms_text: Sequence[str],
    b: Sequence[int],
    cfg: ProjectorConfig | None = None,
    gap_scores: Optional[Sequence[float]] = None,
) -> Dict:
    """
    Final projector:
      1) convert b -> spans
      2) hard max split
      3) soft min merge
      4) return projected b*

    gap_scores:
      optional confidence score per gap, larger means boundary more likely to keep/split.
      It is used:
        - during hard split: choose strongest internal split gap
        - during soft merge: delete the weaker neighboring boundary
    """
    cfg = cfg or DEFAULT_PROJECTOR_CONFIG
    num_atoms = len(atoms_text)

    spans = boundary_vector_to_spans(num_atoms, b)
    spans_after_split = _hard_max_split_until_ok(
        spans,
        atoms_text,
        cfg,
        gap_scores=gap_scores,
    )
    spans_after_merge = _soft_min_merge_until_ok(
        spans_after_split,
        atoms_text,
        cfg,
        gap_scores=gap_scores,
    )

    projected_b = spans_to_boundary_vector(num_atoms, spans_after_merge)
    projected_units = spans_to_units(atoms_text, spans_after_merge)

    return {
        "spans_before": spans,
        "spans_after_split": spans_after_split,
        "spans_after_merge": spans_after_merge,
        "projected_b": projected_b,
        "projected_units": projected_units,
    }


def rebuild_chunks_from_boundary_vector(
    atoms_text: Sequence[str],
    b: Sequence[int],
    cfg: ProjectorConfig | None = None,
    gap_scores: Optional[Sequence[float]] = None,
) -> Dict:
    return project_boundary_vector(
        atoms_text=atoms_text,
        b=b,
        cfg=cfg,
        gap_scores=gap_scores,
    )