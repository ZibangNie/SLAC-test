"""
Refiner input builder.

Main responsibility:
- build atoms
- build unit2atom_span
- build b0
- export refiner-standard atoms_b0 JSONL
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------
# Config
# -----------------------------------


@dataclass
class RefinerInputBuildConfig:
    # Atomizer defaults are aligned with the refiner design doc:
    # atom_max_tokens / atom_max_chars ~= 120 / 480
    atom_max_tokens: int = 120
    atom_max_chars: int = 480

    # Merge tiny atoms to avoid pathological fragmentation.
    atom_min_tokens: int = 8
    atom_min_chars: int = 20

    # Sentence splitting behavior
    split_cjk_sentences: bool = True
    split_en_sentences: bool = True
    dot_in_cjk: bool = True
    line_fallback: bool = True

    # Empty-unit handling
    fix_empty_units: bool = True
    prefer_merge_empty_to_right: bool = True

    # Validation / behavior
    require_nonempty_atoms: bool = True
    strict_validate: bool = True


# -----------------------------------
# Basic text utils
# -----------------------------------


def normalize_spaces(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def normalize_text_light(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ").replace("\u3000", " ")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("【", "[").replace("】", "]")
    s = s.replace("—", "-").replace("–", "-").replace("－", "-")
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))


def approx_token_count(text: str) -> int:
    """
    Lightweight token estimate for rule-based atomization.
    - English-like text: whitespace token count
    - CJK-heavy text: char-based coarse estimate
    """
    s = (text or "").strip()
    if not s:
        return 0

    if has_cjk(s):
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", s))
        latin_words = len(re.findall(r"[A-Za-z0-9]+", s))
        punct = len(re.findall(r"[,.!?;:，。！？；：、]", s))
        return max(1, cjk_chars + latin_words + punct // 2)

    words = re.findall(r"\S+", s)
    return max(1, len(words))


def is_blank_text(text: str) -> bool:
    return not (text or "").strip()


# -----------------------------------
# Sentence splitting
# -----------------------------------


_SENT_END_CHARS_ZH = "。！？；"
_SENT_END_CHARS_EN = ".!?;"
_SENT_END_ALL = _SENT_END_CHARS_ZH + _SENT_END_CHARS_EN


def _split_by_newlines(text: str) -> List[str]:
    parts = []
    for line in (text or "").split("\n"):
        s = normalize_spaces(line)
        if s:
            parts.append(s)
    return parts


def _is_probably_heading_like(line: str) -> bool:
    s = normalize_spaces(line)
    if not s:
        return False
    if len(s) <= 80 and not re.search(r"[。！？.!?;；]$", s):
        if re.match(r"^(\d+(\.\d+){0,6}|第.+?[章节条款项]|[A-Z][A-Z0-9 .\-]{0,40})\b", s):
            return True
    return False


def _safe_sentence_boundaries(text: str, dot_in_cjk: bool = True) -> List[int]:
    """
    Return sentence boundary indices (exclusive end positions).
    Conservative heuristic:
    - always split on Chinese sentence punctuation
    - split on English punctuation if followed by space/newline/CJK/quote/end
    - optionally allow '.' in CJK-heavy context
    """
    s = text or ""
    if not s:
        return []

    ends: List[int] = []
    n = len(s)

    for i, ch in enumerate(s):
        if ch in _SENT_END_CHARS_ZH:
            ends.append(i + 1)
            continue

        if ch in "!?;":
            # more universal
            nxt = s[i + 1] if i + 1 < n else ""
            if (not nxt) or nxt.isspace() or nxt in "\"')]}”’" or ("\u4e00" <= nxt <= "\u9fff"):
                ends.append(i + 1)
            continue

        if ch == ".":
            prev = s[i - 1] if i - 1 >= 0 else ""
            nxt = s[i + 1] if i + 1 < n else ""

            # decimal number: 3.14
            if prev.isdigit() and nxt.isdigit():
                continue

            # abbreviation-like protection: e.g. "e.g." / "U.S."
            if prev.isalpha() and nxt.isalpha():
                continue

            # English sentence end
            if (not nxt) or nxt.isspace() or nxt in "\"')]}”’":
                ends.append(i + 1)
                continue

            # Dot in CJK context
            if dot_in_cjk and ("\u4e00" <= prev <= "\u9fff" or "\u4e00" <= nxt <= "\u9fff"):
                ends.append(i + 1)
                continue

    # unique and sorted
    ends = sorted(set(x for x in ends if 0 < x <= n))
    return ends


def sentence_split(text: str, *, dot_in_cjk: bool = True) -> List[str]:
    """
    Sentence-first split. If no reliable sentence boundary is found,
    return line-based segments as fallback.
    """
    s = normalize_text_light(text)
    if not s:
        return []

    # If the text is already highly line-structured / heading-like, keep line granularity.
    line_parts = _split_by_newlines(s)
    if len(line_parts) > 1:
        heading_like_count = sum(1 for ln in line_parts if _is_probably_heading_like(ln))
        if heading_like_count >= max(1, len(line_parts) // 2):
            return line_parts

    ends = _safe_sentence_boundaries(s, dot_in_cjk=dot_in_cjk)
    if not ends:
        return line_parts if line_parts else [s]

    out: List[str] = []
    start = 0
    for end in ends:
        seg = normalize_spaces(s[start:end])
        if seg:
            out.append(seg)
        start = end
    tail = normalize_spaces(s[start:])
    if tail:
        out.append(tail)

    # If segmentation became too coarse or trivial, fallback to line split.
    if len(out) <= 1:
        line_parts = _split_by_newlines(s)
        if len(line_parts) > 1:
            return line_parts

    return out if out else [s]


# -----------------------------------
# Long atom splitting
# -----------------------------------


def _best_split_positions(text: str) -> List[int]:
    """
    Candidate split positions for long atoms, ordered by preference.
    """
    s = text or ""
    positions: List[int] = []

    # Stronger separators first
    for m in re.finditer(r"[。！？；.!?;:：]\s*", s):
        positions.append(m.end())

    for m in re.finditer(r"[，,、]\s*", s):
        positions.append(m.end())

    for m in re.finditer(r"\)\s+|\]\s+|\}\s+|\s+-\s+|\s+", s):
        positions.append(m.end())

    # unique sorted
    return sorted(set(p for p in positions if 0 < p < len(s)))


def _split_long_text_once(text: str, cfg: RefinerInputBuildConfig) -> List[str]:
    s = normalize_spaces(text)
    if not s:
        return []

    too_long = (
        len(s) > cfg.atom_max_chars
        or approx_token_count(s) > cfg.atom_max_tokens
    )
    if not too_long:
        return [s]

    candidates = _best_split_positions(s)
    if not candidates:
        mid = len(s) // 2
        return [normalize_spaces(s[:mid]), normalize_spaces(s[mid:])]

    half = len(s) / 2.0
    best = min(candidates, key=lambda x: abs(x - half))

    left = normalize_spaces(s[:best])
    right = normalize_spaces(s[best:])
    out = []
    if left:
        out.append(left)
    if right:
        out.append(right)
    return out if out else [s]


def _split_long_text_recursive(text: str, cfg: RefinerInputBuildConfig) -> List[str]:
    queue = [normalize_spaces(text)]
    out: List[str] = []

    while queue:
        cur = queue.pop(0)
        if not cur:
            continue

        too_long = (
            len(cur) > cfg.atom_max_chars
            or approx_token_count(cur) > cfg.atom_max_tokens
        )
        if not too_long:
            out.append(cur)
            continue

        parts = _split_long_text_once(cur, cfg)
        if len(parts) <= 1:
            out.append(cur)
            continue

        # breadth-first split
        for p in parts:
            if p:
                queue.append(p)

    return out


# -----------------------------------
# Short atom merging
# -----------------------------------


def _is_short_atom(atom: str, cfg: RefinerInputBuildConfig) -> bool:
    if not atom:
        return True
    return (
        len(atom) < cfg.atom_min_chars
        or approx_token_count(atom) < cfg.atom_min_tokens
    )


def merge_short_atoms(atoms: List[str], cfg: RefinerInputBuildConfig) -> List[str]:
    """
    Merge overly short atoms with adjacent atoms when safe.
    """
    if not atoms:
        return []

    out: List[str] = []
    i = 0
    while i < len(atoms):
        cur = normalize_spaces(atoms[i])
        if not cur:
            i += 1
            continue

        if not _is_short_atom(cur, cfg):
            out.append(cur)
            i += 1
            continue

        # Prefer merging into next if possible
        if i + 1 < len(atoms):
            nxt = normalize_spaces(atoms[i + 1])
            merged = normalize_spaces(cur + " " + nxt) if not has_cjk(cur + nxt) else cur + nxt
            if (
                len(merged) <= cfg.atom_max_chars
                and approx_token_count(merged) <= cfg.atom_max_tokens
            ):
                out.append(merged)
                i += 2
                continue

        # Otherwise merge backward
        if out:
            prev = out.pop()
            merged = normalize_spaces(prev + " " + cur) if not has_cjk(prev + cur) else prev + cur
            if (
                len(merged) <= cfg.atom_max_chars
                and approx_token_count(merged) <= cfg.atom_max_tokens
            ):
                out.append(merged)
            else:
                out.append(prev)
                out.append(cur)
            i += 1
            continue

        out.append(cur)
        i += 1

    return [x for x in out if x.strip()]


# -----------------------------------
# Atomization
# -----------------------------------


def atomize_unit_text(text: str, cfg: Optional[RefinerInputBuildConfig] = None) -> List[str]:
    cfg = cfg or RefinerInputBuildConfig()

    s = normalize_text_light(text)
    if not s:
        return []

    segments = sentence_split(s, dot_in_cjk=cfg.dot_in_cjk)
    if not segments and cfg.line_fallback:
        segments = _split_by_newlines(s)
    if not segments:
        segments = [s]

    atoms: List[str] = []
    for seg in segments:
        atoms.extend(_split_long_text_recursive(seg, cfg))

    atoms = [normalize_spaces(a) for a in atoms if normalize_spaces(a)]
    atoms = merge_short_atoms(atoms, cfg)

    # Final defensive cleanup
    final_atoms: List[str] = []
    for a in atoms:
        a = normalize_spaces(a)
        if not a:
            continue
        # re-split again if still too long after merge path
        if len(a) > cfg.atom_max_chars or approx_token_count(a) > cfg.atom_max_tokens:
            final_atoms.extend(_split_long_text_recursive(a, cfg))
        else:
            final_atoms.append(a)

    return [a for a in final_atoms if a.strip()]


# -----------------------------------
# Empty-unit repair
# -----------------------------------


def _find_nearest_nonempty_right(unit_texts: List[str], start_idx: int, cfg: RefinerInputBuildConfig) -> Optional[int]:
    for j in range(start_idx + 1, len(unit_texts)):
        if atomize_unit_text(unit_texts[j], cfg):
            return j
    return None


def _find_nearest_nonempty_left(unit_texts: List[str], start_idx: int, cfg: RefinerInputBuildConfig) -> Optional[int]:
    for j in range(start_idx - 1, -1, -1):
        if atomize_unit_text(unit_texts[j], cfg):
            return j
    return None


def repair_empty_units(
    chunk0_units: List[Dict[str, Any]],
    cfg: Optional[RefinerInputBuildConfig] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    If a unit atomizes to empty, merge its raw text into nearest non-empty unit.
    Default policy: prefer right, then left.
    """
    cfg = cfg or RefinerInputBuildConfig()
    units = copy.deepcopy(chunk0_units)

    stats = {
        "empty_units_before_fix": 0,
        "empty_units_fixed": 0,
        "empty_units_unfixable": 0,
        "projection_fixes": [],
    }

    if not cfg.fix_empty_units:
        return units, stats

    unit_texts = [str(u.get("text", "") or "") for u in units]

    # detect empty-after-atomize
    empty_idxs: List[int] = []
    for i, txt in enumerate(unit_texts):
        if not atomize_unit_text(txt, cfg):
            empty_idxs.append(i)

    stats["empty_units_before_fix"] = len(empty_idxs)
    if not empty_idxs:
        return units, stats

    for i in empty_idxs:
        src_text = unit_texts[i]
        if not src_text.strip():
            stats["projection_fixes"].append(
                {"unit_id": units[i].get("unit_id"), "fix": "empty_text_unit"}
            )
        else:
            stats["projection_fixes"].append(
                {"unit_id": units[i].get("unit_id"), "fix": "atomize_empty_unit"}
            )

        target_idx: Optional[int] = None

        if cfg.prefer_merge_empty_to_right:
            target_idx = _find_nearest_nonempty_right(unit_texts, i, cfg)
            if target_idx is None:
                target_idx = _find_nearest_nonempty_left(unit_texts, i, cfg)
        else:
            target_idx = _find_nearest_nonempty_left(unit_texts, i, cfg)
            if target_idx is None:
                target_idx = _find_nearest_nonempty_right(unit_texts, i, cfg)

        if target_idx is None:
            stats["empty_units_unfixable"] += 1
            continue

        src = unit_texts[i].strip()
        tgt = unit_texts[target_idx].strip()

        if src:
            merged = (tgt + "\n" + src).strip() if target_idx > i else (src + "\n" + tgt).strip()
            unit_texts[target_idx] = merged
            units[target_idx]["text"] = merged

        # mark source unit as effectively absorbed
        unit_texts[i] = ""
        units[i]["text"] = ""
        units[i]["_absorbed_into"] = units[target_idx].get("unit_id")
        stats["empty_units_fixed"] += 1

    return units, stats


# -----------------------------------
# Mapping: chunk0_units -> atoms, spans, b0
# -----------------------------------


def build_atoms_and_spans(
    chunk0_units: List[Dict[str, Any]],
    cfg: Optional[RefinerInputBuildConfig] = None,
) -> Tuple[List[str], List[Dict[str, int]], Dict[str, Any]]:
    """
    Build:
      atoms: flattened atom sequence
      unit2atom_span: span for each effective non-empty unit

    Units that remain empty after repair are skipped from span output.
    """
    cfg = cfg or RefinerInputBuildConfig()

    repaired_units, repair_stats = repair_empty_units(chunk0_units, cfg=cfg)

    atoms: List[str] = []
    unit2atom_span: List[Dict[str, int]] = []

    span_start = 0
    skipped_units = 0

    for u in repaired_units:
        uid = int(u.get("unit_id"))
        text = str(u.get("text", "") or "")
        unit_atoms = atomize_unit_text(text, cfg)

        if not unit_atoms:
            skipped_units += 1
            continue

        start_atom = span_start
        atoms.extend(unit_atoms)
        span_start = len(atoms)
        end_atom = span_start

        unit2atom_span.append(
            {
                "unit_id": uid,
                "start_atom": start_atom,
                "end_atom": end_atom,
            }
        )

    meta = {
        "num_input_units": len(chunk0_units),
        "num_effective_units": len(unit2atom_span),
        "num_skipped_units_after_fix": skipped_units,
        "num_atoms": len(atoms),
        "repair_stats": repair_stats,
    }

    if cfg.require_nonempty_atoms and not atoms:
        raise ValueError("All chunk0_units atomized to empty; cannot build refiner input.")

    return atoms, unit2atom_span, meta


def build_b0_from_unit_spans(
    unit2atom_span: List[Dict[str, int]],
    num_atoms: int,
    *,
    strict_validate: bool = True,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Given effective non-empty unit spans [s_j, e_j), project chunk0 boundaries
    into atom gap space:
        g_j = e_j - 1
    for each adjacent effective unit pair.
    """
    if num_atoms < 1:
        raise ValueError("num_atoms must be >= 1")

    if not unit2atom_span:
        raise ValueError("unit2atom_span is empty")

    # validate monotonicity / coverage sanity
    prev_end = 0
    for idx, sp in enumerate(unit2atom_span):
        s = int(sp["start_atom"])
        e = int(sp["end_atom"])
        if not (0 <= s < e <= num_atoms):
            raise ValueError(f"Invalid span at index {idx}: [{s}, {e}) with num_atoms={num_atoms}")
        if idx == 0:
            if s != 0 and strict_validate:
                raise ValueError(f"First span must start at 0, got {s}")
        if s < prev_end:
            raise ValueError(
                f"Overlapping or out-of-order spans: prev_end={prev_end}, current=[{s}, {e})"
            )
        prev_end = e

    if strict_validate and unit2atom_span[-1]["end_atom"] != num_atoms:
        raise ValueError(
            f"Last span must end at num_atoms={num_atoms}, got {unit2atom_span[-1]['end_atom']}"
        )

    b0 = [0] * (num_atoms - 1)
    projection_fix_count = 0
    projected_gaps: List[int] = []

    for j in range(len(unit2atom_span) - 1):
        left = unit2atom_span[j]
        right = unit2atom_span[j + 1]

        s_l = int(left["start_atom"])
        e_l = int(left["end_atom"])
        s_r = int(right["start_atom"])
        e_r = int(right["end_atom"])

        if not (s_l < e_l and s_r < e_r):
            projection_fix_count += 1
            continue

        # effective units must be consecutive in atom sequence
        if e_l != s_r:
            if strict_validate:
                raise ValueError(
                    f"Non-consecutive spans around boundary j={j}: left=[{s_l},{e_l}), right=[{s_r},{e_r})"
                )
            projection_fix_count += 1
            continue

        g = e_l - 1
        if not (0 <= g < len(b0)):
            projection_fix_count += 1
            continue

        b0[g] = 1
        projected_gaps.append(g)

    meta = {
        "num_seed_boundaries": int(sum(b0)),
        "projection_fix_count": projection_fix_count,
        "projected_gaps": projected_gaps,
    }
    return b0, meta


# -----------------------------------
# Validation
# -----------------------------------


def validate_refiner_input_record(record: Dict[str, Any]) -> None:
    required = ["doc_id", "atoms", "b0"]
    missing = [k for k in required if k not in record]
    if missing:
        raise KeyError(f"Refiner input missing required fields: {missing}")

    doc_id = record["doc_id"]
    atoms = record["atoms"]
    b0 = record["b0"]
    chunk0_units = record.get("chunk0_units")
    unit2atom_span = record.get("unit2atom_span")

    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string")

    if not isinstance(atoms, list) or len(atoms) < 1:
        raise ValueError("atoms must be a non-empty list")

    if not all(isinstance(x, str) and x.strip() for x in atoms):
        raise ValueError("All atoms must be non-empty strings")

    if not isinstance(b0, list):
        raise ValueError("b0 must be a list")

    if len(b0) != len(atoms) - 1:
        raise ValueError(f"len(b0) must equal len(atoms)-1, got {len(b0)} vs {len(atoms)-1}")

    if not all(x in {0, 1} for x in b0):
        raise ValueError("b0 values must be in {0,1}")

    if chunk0_units is not None and not isinstance(chunk0_units, list):
        raise ValueError("chunk0_units must be a list when provided")

    if unit2atom_span is not None:
        if not isinstance(unit2atom_span, list) or not unit2atom_span:
            raise ValueError("unit2atom_span must be a non-empty list when provided")

        prev_end = 0
        for idx, sp in enumerate(unit2atom_span):
            if not isinstance(sp, dict):
                raise ValueError(f"unit2atom_span[{idx}] must be a dict")
            for k in ("unit_id", "start_atom", "end_atom"):
                if k not in sp:
                    raise ValueError(f"unit2atom_span[{idx}] missing key: {k}")
            s = int(sp["start_atom"])
            e = int(sp["end_atom"])
            if not (0 <= s < e <= len(atoms)):
                raise ValueError(
                    f"Invalid unit2atom_span[{idx}] = [{s}, {e}) with len(atoms)={len(atoms)}"
                )
            if s < prev_end:
                raise ValueError(
                    f"unit2atom_span must be monotonic and non-overlapping, got prev_end={prev_end}, s={s}"
                )
            prev_end = e

        if prev_end != len(atoms):
            raise ValueError(
                f"unit2atom_span must cover atoms contiguously; last end_atom={prev_end}, len(atoms)={len(atoms)}"
            )


# -----------------------------------
# Public APIs
# -----------------------------------


def build_refiner_input_from_chunk0(
    doc_id: str,
    chunk0_units: List[Dict[str, Any]],
    *,
    domain: Optional[str] = None,
    cfg: Optional[RefinerInputBuildConfig] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main builder:
      chunk0_units -> refiner standard input JSON object
    """
    cfg = cfg or RefinerInputBuildConfig()

    if not isinstance(chunk0_units, list) or not chunk0_units:
        raise ValueError("chunk0_units must be a non-empty list")

    atoms, unit2atom_span, atom_meta = build_atoms_and_spans(chunk0_units, cfg=cfg)
    b0, b0_meta = build_b0_from_unit_spans(
        unit2atom_span=unit2atom_span,
        num_atoms=len(atoms),
        strict_validate=cfg.strict_validate,
    )

    out: Dict[str, Any] = {
        "doc_id": doc_id,
        "chunk0_units": copy.deepcopy(chunk0_units),
        "atoms": atoms,
        "unit2atom_span": unit2atom_span,
        "b0": b0,
        "meta": {
            "source": "chunk0_seed",
            "builder": "build_refiner_input",
            "atomizer": {
                "atom_max_tokens": cfg.atom_max_tokens,
                "atom_max_chars": cfg.atom_max_chars,
                "atom_min_tokens": cfg.atom_min_tokens,
                "atom_min_chars": cfg.atom_min_chars,
                "dot_in_cjk": cfg.dot_in_cjk,
                "line_fallback": cfg.line_fallback,
            },
            "stats": {
                **atom_meta,
                **b0_meta,
            },
        },
    }

    if domain is not None:
        out["domain"] = domain

    if meta:
        out["meta"].update(meta)

    validate_refiner_input_record(out)
    return out


def build_refiner_input_from_structure_doc(
    structure_doc: Dict[str, Any],
    *,
    domain: Optional[str] = None,
    cfg: Optional[RefinerInputBuildConfig] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for structure_doc that already contains chunk0_units.
    """
    if "doc_id" not in structure_doc:
        raise KeyError("structure_doc missing doc_id")
    if "chunk0_units" not in structure_doc:
        raise KeyError("structure_doc missing chunk0_units")

    return build_refiner_input_from_chunk0(
        doc_id=structure_doc["doc_id"],
        chunk0_units=structure_doc["chunk0_units"],
        domain=domain,
        cfg=cfg,
        meta=meta,
    )


def build_refiner_input_from_segment_record(
    segment_record: Dict[str, Any],
    *,
    domain: Optional[str] = None,
    cfg: Optional[RefinerInputBuildConfig] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Pipeline wrapper:
      segment_document_record(...) output
      or structure_doc-like record with chunk0_units
      -> refiner standard input
    """
    if "structure_doc" in segment_record:
        structure_doc = segment_record["structure_doc"]
    else:
        structure_doc = segment_record

    final_meta = dict(meta or {})
    if "source_path" in segment_record:
        final_meta.setdefault("source_path", segment_record.get("source_path"))
    if "source_type" in segment_record:
        final_meta.setdefault("source_type", segment_record.get("source_type"))
    if "diagnostics" in segment_record:
        final_meta.setdefault("segment_diagnostics", segment_record.get("diagnostics"))

    return build_refiner_input_from_structure_doc(
        structure_doc=structure_doc,
        domain=domain,
        cfg=cfg,
        meta=final_meta,
    )