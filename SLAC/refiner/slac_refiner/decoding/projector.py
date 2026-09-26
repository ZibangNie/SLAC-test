from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Dict, List, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]|[^\w\s]",
    re.UNICODE,
)
Span = Tuple[int, int]
TokenCounter = Callable[[str], int]


@dataclass
class ProjectorConfig:
    max_chunk_atoms: int = 64
    min_chunk_atoms: int = 2
    max_chunk_chars: int = 1600
    min_chunk_chars: int = 20
    max_chunk_tokens: int = 384
    min_chunk_tokens: int = 48


DEFAULT_PROJECTOR_CONFIG = ProjectorConfig()


class ProjectorBudgetError(ValueError):
    """An indivisible atom exceeds at least one configured hard maximum."""

    def __init__(self, overlong_spans: List[Dict]):
        self.overlong_spans = overlong_spans
        indices = [item["start_atom"] for item in overlong_spans]
        super().__init__(f"Hard chunk budget is infeasible at indivisible atoms: {indices}")


def token_len_proxy(text: str) -> int:
    """Legacy regex proxy; this is not a model tokenizer."""
    return len(_TOKEN_RE.findall(text or ""))


def boundary_vector_to_spans(num_atoms: int, b: Sequence[int]) -> List[Span]:
    if num_atoms < 0:
        raise ValueError("num_atoms must be nonnegative")
    if len(b) != max(0, num_atoms - 1):
        raise ValueError("Boundary vector length must equal max(0, num_atoms - 1)")
    if any(value not in (0, 1) for value in b):
        raise ValueError("Boundary vector must contain only 0 or 1")
    if num_atoms == 0:
        return []
    spans: List[Span] = []
    start = 0
    for g, value in enumerate(b):
        if value == 1:
            spans.append((start, g + 1))
            start = g + 1
    spans.append((start, num_atoms))
    return spans


def spans_to_boundary_vector(num_atoms: int, spans: Sequence[Span]) -> List[int]:
    num_gaps = max(0, num_atoms - 1)
    b = [0] * num_gaps
    for _, end in spans[:-1]:
        gap = end - 1
        if 0 <= gap < num_gaps:
            b[gap] = 1
    return b


class _SpanContext:
    """Render and measure the very same span; never sum atom token counts."""

    def __init__(
        self,
        atoms_text: Sequence[str],
        token_counter: Optional[TokenCounter] = None,
        source_text: Optional[str] = None,
        atom_char_spans: Optional[Sequence[Span]] = None,
    ):
        self.atoms_text = atoms_text
        self.token_counter = token_counter if token_counter is not None else token_len_proxy
        self.source_text = source_text
        self.atom_char_spans = atom_char_spans
        self._stats: Dict[Span, Dict[str, int]] = {}
        if (source_text is None) != (atom_char_spans is None):
            raise ValueError("source_text and atom_char_spans must be supplied together")
        if source_text is not None:
            if len(atom_char_spans) != len(atoms_text):
                raise ValueError("One source character span is required per atom")
            if not atoms_text and source_text:
                raise ValueError("Nonempty source_text cannot be represented by zero atoms")
            previous_end = 0
            for atom, (start, end) in zip(atoms_text, atom_char_spans):
                if not isinstance(start, Integral) or not isinstance(end, Integral):
                    raise ValueError("Source character offsets must be integers")
                if not 0 <= previous_end <= start <= end <= len(source_text):
                    raise ValueError("Source character spans must be ordered and nonoverlapping")
                if source_text[start:end] != atom:
                    raise ValueError("Each atom must exactly match its source character span")
                previous_end = end

    def char_span(self, span: Span) -> Span:
        # Inter-atom separators belong to the preceding atom. Leading/trailing
        # source text is retained, so concatenating projected units is lossless.
        start, end = span
        char_start = 0 if start == 0 else self.atom_char_spans[start][0]
        char_end = len(self.source_text) if end == len(self.atoms_text) else self.atom_char_spans[end][0]
        return char_start, char_end

    def text(self, span: Span) -> str:
        if self.source_text is not None:
            start, end = self.char_span(span)
            return self.source_text[start:end]
        start, end = span
        return "\n".join(self.atoms_text[start:end]).strip()

    def stats(self, span: Span) -> Dict[str, int]:
        if span not in self._stats:
            text = self.text(span)
            count = self.token_counter(text)
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                raise ValueError("token_counter must return a nonnegative integer")
            self._stats[span] = {"atoms": span[1] - span[0], "chars": len(text), "tokens": int(count)}
        return self._stats[span]

    def units(self, spans: Sequence[Span]) -> List[Dict]:
        units = []
        for uid, span in enumerate(spans):
            unit = {"unit_id": uid, "text": self.text(span), "start_atom": span[0], "end_atom": span[1]}
            if self.source_text is not None:
                unit["start_char"], unit["end_char"] = self.char_span(span)
            units.append(unit)
        return units


def spans_to_units(
    atoms_text: Sequence[str], spans: Sequence[Span], *,
    source_text: Optional[str] = None, atom_char_spans: Optional[Sequence[Span]] = None,
) -> List[Dict]:
    return _SpanContext(atoms_text, source_text=source_text, atom_char_spans=atom_char_spans).units(spans)


def _is_overlong(stats: Dict[str, int], cfg: ProjectorConfig) -> bool:
    return any(stats[key] > getattr(cfg, f"max_chunk_{key}") for key in ("atoms", "chars", "tokens"))


def _is_too_short(stats: Dict[str, int], cfg: ProjectorConfig) -> bool:
    return any(stats[key] < getattr(cfg, f"min_chunk_{key}") for key in ("atoms", "chars", "tokens"))


def _pick_split_gap(span: Span, gap_scores: Optional[Sequence[float]] = None) -> Optional[int]:
    start, end = span
    if end - start <= 1:
        return None
    if gap_scores is not None:
        return max(range(start, end - 1), key=lambda gap: float(gap_scores[gap]))
    return (start + end) // 2 - 1


def _hard_max_split(spans: Sequence[Span], context: _SpanContext, cfg: ProjectorConfig,
                    gap_scores: Optional[Sequence[float]]) -> List[Span]:
    # A stack avoids both recursion depth and the former arbitrary 32-round cap.
    pending = list(reversed(spans))
    result = []
    while pending:
        span = pending.pop()
        gap = _pick_split_gap(span, gap_scores) if _is_overlong(context.stats(span), cfg) else None
        if gap is None:
            result.append(span)
        else:
            pending.append((gap + 1, span[1]))
            pending.append((span[0], gap + 1))
    return result


def _soft_min_merge(spans: Sequence[Span], context: _SpanContext, cfg: ProjectorConfig,
                    gap_scores: Optional[Sequence[float]]) -> List[Span]:
    result = list(spans)
    while True:
        changed = False
        for index, span in enumerate(result):
            if not _is_too_short(context.stats(span), cfg):
                continue
            candidates = []
            # Hard maxima dominate soft minima and confidence preference.
            for left in (index - 1, index):
                if left < 0 or left + 1 >= len(result):
                    continue
                merged = (result[left][0], result[left + 1][1])
                stats = context.stats(merged)
                if _is_overlong(stats, cfg):
                    continue
                gap = result[left][1] - 1
                confidence = float(gap_scores[gap]) if gap_scores is not None else 0.0
                deficit = max(0, cfg.min_chunk_atoms - stats["atoms"])
                candidates.append((confidence, deficit, stats["atoms"], left, merged))
            if candidates:
                _, _, _, left, merged = min(candidates)
                result[left:left + 2] = [merged]
                changed = True
                break
            # An unmergeable short span must not block later feasible merges.
        if not changed:
            return result


def _validate_config(cfg: ProjectorConfig) -> None:
    for key in ("atoms", "chars", "tokens"):
        maximum = getattr(cfg, f"max_chunk_{key}")
        minimum = getattr(cfg, f"min_chunk_{key}")
        if (isinstance(maximum, bool) or not isinstance(maximum, Integral)
                or isinstance(minimum, bool) or not isinstance(minimum, Integral)
                or not 0 <= minimum <= maximum or maximum <= 0):
            raise ValueError(f"Expected 0 <= min_chunk_{key} <= max_chunk_{key}, with a positive maximum")


def project_boundary_vector(
    atoms_text: Sequence[str], b: Sequence[int], cfg: ProjectorConfig | None = None,
    gap_scores: Optional[Sequence[float]] = None, *,
    token_counter: Optional[TokenCounter] = None,
    source_text: Optional[str] = None,
    atom_char_spans: Optional[Sequence[Span]] = None,
    strict: bool = False,
) -> Dict:
    """Split to hard maxima, then merge toward soft minima only when feasible.

    Existing callers retain regex token counts and newline-joined, stripped text.
    Supply ``token_counter`` to count each complete rendered span with the actual
    downstream tokenizer (including its special-token policy and no truncation).
    Atom token counts cannot be added: tokenization is generally non-additive.

    For lossless source reconstruction supply both ``source_text`` and exact
    ``atom_char_spans``. Offsets must be ordered and each atom must match its slice.
    Inter-atom separators belong to the preceding atom, and all leading/trailing
    text is retained. Budgets measure exactly the returned unit text in both modes.

    ``gap_scores`` selects stronger split boundaries and weaker *feasible* merge
    boundaries. All maxima apply together. An overlarge indivisible atom is kept
    and reported with ``hard_max_satisfied=False``; ``strict=True`` raises
    ``ProjectorBudgetError`` instead. Soft minima may remain unmet.
    """
    cfg = cfg or DEFAULT_PROJECTOR_CONFIG
    _validate_config(cfg)
    num_atoms = len(atoms_text)
    spans = boundary_vector_to_spans(num_atoms, b)
    if gap_scores is not None:
        if len(gap_scores) != max(0, num_atoms - 1):
            raise ValueError("gap_scores length must equal max(0, num_atoms - 1)")
        if any(not math.isfinite(float(score)) for score in gap_scores):
            raise ValueError("gap_scores must be finite")
    context = _SpanContext(atoms_text, token_counter, source_text, atom_char_spans)
    spans_after_split = _hard_max_split(spans, context, cfg, gap_scores)
    spans_after_merge = _soft_min_merge(spans_after_split, context, cfg, gap_scores)

    overlong_spans = []
    short_spans = []
    for start, end in spans_after_merge:
        stats = context.stats((start, end))
        record = {"start_atom": start, "end_atom": end, "stats": dict(stats)}
        over = [key for key in stats if stats[key] > getattr(cfg, f"max_chunk_{key}")]
        under = [key for key in stats if stats[key] < getattr(cfg, f"min_chunk_{key}")]
        if over:
            overlong_spans.append({**record, "violated_dimensions": over, "reason": "indivisible_atom"})
        if under:
            short_spans.append({**record, "unmet_dimensions": under})
    if strict and overlong_spans:
        raise ProjectorBudgetError(overlong_spans)

    return {
        "spans_before": spans,
        "spans_after_split": spans_after_split,
        "spans_after_merge": spans_after_merge,
        "projected_b": spans_to_boundary_vector(num_atoms, spans_after_merge),
        "projected_units": context.units(spans_after_merge),
        "hard_max_satisfied": not overlong_spans,
        "overlong_spans": overlong_spans,
        "short_spans": short_spans,
        "token_count_mode": "custom_whole_span" if token_counter is not None else "regex_proxy",
        "text_mode": "source_spans" if source_text is not None else "legacy_newline_strip",
    }


def rebuild_chunks_from_boundary_vector(
    atoms_text: Sequence[str], b: Sequence[int], cfg: ProjectorConfig | None = None,
    gap_scores: Optional[Sequence[float]] = None, *,
    token_counter: Optional[TokenCounter] = None,
    source_text: Optional[str] = None,
    atom_char_spans: Optional[Sequence[Span]] = None,
    strict: bool = False,
) -> Dict:
    return project_boundary_vector(
        atoms_text=atoms_text, b=b, cfg=cfg, gap_scores=gap_scores,
        token_counter=token_counter, source_text=source_text,
        atom_char_spans=atom_char_spans, strict=strict,
    )
