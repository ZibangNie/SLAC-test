#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Boundary Refiner denoising samples from the unified real_dataset JSONL files.

Input schema (from real_dataset/*.jsonl):
{
  "sample_id": "...",
  "doc_id": "...",
  "doc_name": "...",
  "source_family": "llm_structured|railway_parsed",
  "domain": "papers|laws|standards|b|railway",
  "language": "en|zh|...",
  "atoms": [   # note: in real_dataset these are normalized source units/chunks
    {"atom_id": "0", "text": "...", "type": "heading", "level": 0, "meta": {...}},
    ...
  ]
}

Output schema (for Refiner denoising pretraining):
{
  "sample_id": "...",
  "doc_id": "...",
  "domain": "llm_gold|rail",
  "chunk0_units": [{"unit_id": 0, "text": "..."}, ...],
  "atoms": [{"aid": 0, "text": "..."}, ...],
  "unit2atom_span": [{"unit_id": 0, "s": 0, "e": 5}, ...],
  "b0": [0,1,0,...],                 # noisy boundaries (training input)
  "b_gold": [0,1,0,...],             # optional gold boundaries (evaluation/debug)
  "labels": {
    "edit": [{"g": 12, "y": "DEL"}, {"g": 57, "y": "SHIFT:-2"}, ...],
    "insert": [0,0,1,0,...]
  },
  "meta": {...}
}

Design alignment:
- chunk0 units -> fine-grained atoms
- b_gold projected from unit boundaries via unit2atom_span
- training b0 is obtained by applying DELETE / INSERT / SHIFT noise to b_gold
- labels are derived for pointer-style edit choice and gap-level insert supervision
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Basic IO helpers
# -----------------------------


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL line {line_no} in {path}: {e}") from e


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# -----------------------------
# Text normalization and atomization
# -----------------------------

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
BLANKLINE_RE = re.compile(r"\n\s*\n+")
MULTISPACE_RE = re.compile(r"[ \t\x0b\f]+")
REPEATED_PUNCT_RE = re.compile(r"([。！？；,.!?;:：、])\1+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([A-Za-z])\-\n([A-Za-z])", r"\1\2", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = "\n".join(MULTISPACE_RE.sub(" ", ln).strip() for ln in text.split("\n"))
    text = BLANKLINE_RE.sub("\n\n", text)
    return text.strip()


def looks_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def estimate_tokens(text: str, lang: str = "unknown") -> int:
    if not text:
        return 0
    cjk_chars = len(CJK_RE.findall(text))
    en_words = len(EN_WORD_RE.findall(text))
    others = max(0, len(text) - cjk_chars)
    return cjk_chars + en_words + max(0, others // 8)


def split_by_blanklines(text: str) -> List[str]:
    return [s.strip() for s in BLANKLINE_RE.split(text) if s.strip()]


def split_lines_nonempty(text: str) -> List[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def sentence_split_simple(seg: str, language: str = "unknown", dot_in_cjk_as_terminator: bool = True) -> List[str]:
    """
    Conservative sentence splitter for bilingual/OCR-ish text.
    - Strong split on Chinese 。！？； and English ! ? ;
    - Dot handling:
      * don't split decimals 3.14
      * don't split common abbreviations like e.g. U.S. Fig. Eq.
      * if dot neighbors CJK, treat as terminator when enabled
    """
    seg = seg.strip()
    if not seg:
        return []

    parts: List[str] = []
    buf: List[str] = []
    n = len(seg)
    abbr_re = re.compile(
        r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Fig|Eq|Ref|No|Art|Sec|Vol|Inc|Ltd|U\.S|U\.K|e\.g|i\.e)$",
        re.IGNORECASE,
    )

    def flush():
        s = "".join(buf).strip()
        if s:
            parts.append(s)
        buf.clear()

    for i, ch in enumerate(seg):
        buf.append(ch)
        prev_ch = seg[i - 1] if i > 0 else ""
        next_ch = seg[i + 1] if i + 1 < n else ""
        should_cut = False

        if ch in "。！？；!?;":
            should_cut = True
        elif ch == ".":
            if prev_ch.isdigit() and next_ch.isdigit():
                should_cut = False
            else:
                left = "".join(buf[:-1]).strip()
                if abbr_re.search(left[-12:]):
                    should_cut = False
                elif dot_in_cjk_as_terminator and (looks_cjk(prev_ch) or looks_cjk(next_ch)):
                    should_cut = True
                elif next_ch == "" or next_ch.isspace():
                    should_cut = True
        elif ch == "\n":
            should_cut = True

        if should_cut:
            flush()

    flush()
    return parts


def split_by_weak_separators(text: str) -> List[str]:
    pieces = re.split(r"([,，、:：])", text)
    if len(pieces) <= 1:
        return [text.strip()] if text.strip() else []
    out: List[str] = []
    cur = ""
    for p in pieces:
        if not p:
            continue
        if re.fullmatch(r"[,，、:：]", p):
            cur += p
            out.append(cur.strip())
            cur = ""
        else:
            cur += p
    if cur.strip():
        out.append(cur.strip())
    return [x for x in out if x]


def hard_cut(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        out.append(text[start:end].strip())
        start = end
    return [x for x in out if x]


def enforce_max_len(atoms: List[str], max_tokens: int, max_chars: int, language: str) -> List[str]:
    out: List[str] = []
    for atom in atoms:
        atom = atom.strip()
        if not atom:
            continue
        if len(atom) <= max_chars and estimate_tokens(atom, language) <= max_tokens:
            out.append(atom)
            continue

        if "\n" in atom:
            pieces = [p.strip() for p in atom.split("\n") if p.strip()]
        else:
            pieces = split_by_weak_separators(atom)
        if len(pieces) == 1:
            pieces = hard_cut(atom, max_chars)

        refined: List[str] = []
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            if len(p) <= max_chars and estimate_tokens(p, language) <= max_tokens:
                refined.append(p)
            else:
                refined.extend(hard_cut(p, max_chars))
        out.extend(refined)
    return out


def merge_short_atoms(
    atoms: List[str],
    min_tokens: int,
    min_chars: int,
    max_tokens: int,
    max_chars: int,
    language: str,
) -> List[str]:
    merged: List[str] = []
    for atom in atoms:
        atom = atom.strip()
        if not atom:
            continue
        if not merged:
            merged.append(atom)
            continue
        tok = estimate_tokens(atom, language)
        if tok < min_tokens or len(atom) < min_chars:
            candidate = merged[-1] + (" " if not merged[-1].endswith("\n") else "") + atom
            if len(candidate) <= max_chars and estimate_tokens(candidate, language) <= max_tokens:
                merged[-1] = candidate
            else:
                merged.append(atom)
        else:
            merged.append(atom)
    return merged


def atomize_text(
    text: str,
    language: str,
    line_first_min_lines: int,
    atom_max_tokens: int,
    atom_max_chars: int,
    merge_short_tokens: int,
    merge_short_chars: int,
    dot_in_cjk_as_terminator: bool,
    blankline_split: bool,
) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    nonempty_lines = split_lines_nonempty(text)
    if len(nonempty_lines) >= line_first_min_lines:
        segments = nonempty_lines
    else:
        segments = split_by_blanklines(text) if blankline_split else [text]

    atoms: List[str] = []
    for seg in segments:
        atoms.extend(sentence_split_simple(seg, language=language, dot_in_cjk_as_terminator=dot_in_cjk_as_terminator))

    atoms = enforce_max_len(atoms, max_tokens=atom_max_tokens, max_chars=atom_max_chars, language=language)
    atoms = merge_short_atoms(
        atoms,
        min_tokens=merge_short_tokens,
        min_chars=merge_short_chars,
        max_tokens=atom_max_tokens,
        max_chars=atom_max_chars,
        language=language,
    )
    return [a for a in atoms if a.strip()]


# -----------------------------
# Boundary helpers
# -----------------------------


def boundaries_from_spans(spans: Sequence[Dict[str, int]], num_atoms: int) -> List[int]:
    if num_atoms <= 1:
        return []
    b = [0] * (num_atoms - 1)
    for i in range(len(spans) - 1):
        s = spans[i]
        t = spans[i + 1]
        if s["e"] > s["s"] and t["e"] > t["s"]:
            g = s["e"] - 1
            if 0 <= g < len(b):
                b[g] = 1
    return b


def indices_from_boundary_vector(b: Sequence[int]) -> List[int]:
    return [i for i, x in enumerate(b) if int(x) == 1]


@dataclass
class NoiseConfig:
    K: int = 6
    p_shift: float = 0.50
    p_insert: float = 0.15
    p_delete: float = 0.40
    discrete_laplace_scale: float = 2.0
    max_insert_frac: float = 0.15
    max_delete_frac: float = 0.40
    max_shift_frac: float = 0.50

    burst_delete_prob: float = 0.35
    burst_delete_min: int = 2
    burst_delete_max: int = 4


@dataclass
class NoiseResult:
    b_noisy: List[int]
    edit_labels: List[Dict[str, Any]]
    insert_labels: List[int]
    gold_boundaries: List[int]
    noisy_boundaries: List[int]
    ops: Dict[str, Any]


def sample_discrete_laplace_nonzero(rng: random.Random, scale: float, K: int) -> int:
    for _ in range(50):
        u = rng.random() - 0.5
        k = int(
            math.copysign(
                math.floor(math.log1p(-2 * abs(u)) / math.log(math.exp(-1.0 / max(scale, 1e-6)))),
                u,
            )
        )
        if k == 0:
            continue
        k = max(-K, min(K, k))
        if k != 0:
            return k
    return rng.choice([k for k in range(-K, K + 1) if k != 0])


def choose_shift_target(
    gold_pos: int,
    occupied: set,
    num_gaps: int,
    rng: random.Random,
    K: int,
    scale: float,
) -> Optional[int]:
    candidates: List[int] = []
    for _ in range(20):
        delta = sample_discrete_laplace_nonzero(rng, scale=scale, K=K)
        pos = gold_pos + delta
        if 0 <= pos < num_gaps and pos not in occupied:
            candidates.append(pos)
    if not candidates:
        for delta in list(range(-K, 0)) + list(range(1, K + 1)):
            pos = gold_pos + delta
            if 0 <= pos < num_gaps and pos not in occupied:
                candidates.append(pos)
    if not candidates:
        return None
    candidates.sort(key=lambda x: abs(x - gold_pos))
    top = candidates[: min(4, len(candidates))]
    return rng.choice(top)


def apply_noise_to_gold(b_gold: List[int], cfg: NoiseConfig, rng: random.Random) -> NoiseResult:
    num_gaps = len(b_gold)
    gold_positions = indices_from_boundary_vector(b_gold)
    gold_set = set(gold_positions)

    current: Dict[int, Optional[int]] = {g: g for g in gold_positions}
    occupied = set(current.keys())

    max_del = min(len(gold_positions), max(0, int(round(len(gold_positions) * cfg.max_delete_frac))))

    # 先做单点 delete，概率由 p_delete 控制
    delete_set = set()
    for g in gold_positions:
        if rng.random() < cfg.p_delete:
            delete_set.add(g)

    # 再做 burst delete：连续删 2~4 个相邻真边界，制造长块缺多个边界的场景
    if gold_positions and rng.random() < cfg.burst_delete_prob:
        num_gold = len(gold_positions)
        span = rng.randint(cfg.burst_delete_min, cfg.burst_delete_max)
        span = min(span, num_gold)
        if span > 0:
            start_idx = rng.randint(0, max(0, num_gold - span))
            burst = gold_positions[start_idx:start_idx + span]
            for g in burst:
                delete_set.add(g)

    # 控制总 delete 数量不要超过上限
    delete_list = sorted(delete_set)
    if len(delete_list) > max_del:
        rng.shuffle(delete_list)
        delete_list = delete_list[:max_del]

    for g in delete_list:
        current.pop(g, None)
        occupied.discard(g)

    remaining_gold_boundaries = [pos for pos, src in current.items() if src is not None and pos == src]
    max_shift = min(len(remaining_gold_boundaries), max(0, int(round(len(gold_positions) * cfg.max_shift_frac))))
    shift_count = min(
        max_shift,
        sum(1 for _ in remaining_gold_boundaries if rng.random() < cfg.p_shift)
    )

    shifted_ops: List[Tuple[int, int]] = []
    if shift_count > 0:
        shift_candidates = remaining_gold_boundaries[:]
        rng.shuffle(shift_candidates)
        for old_pos in shift_candidates:
            if len(shifted_ops) >= shift_count:
                break
            if old_pos not in current:
                continue
            target = choose_shift_target(
                old_pos,
                occupied - {old_pos},
                num_gaps=num_gaps,
                rng=rng,
                K=cfg.K,
                scale=cfg.discrete_laplace_scale,
            )
            if target is None or target == old_pos:
                continue
            src_gold = current.pop(old_pos)
            occupied.discard(old_pos)
            current[target] = src_gold
            occupied.add(target)
            shifted_ops.append((old_pos, target))

    candidate_insert_positions = [i for i in range(num_gaps) if i not in occupied]
    max_ins = min(len(candidate_insert_positions), max(0, int(round(max(1, num_gaps) * cfg.max_insert_frac))))
    insert_count = min(
        max_ins,
        sum(1 for _ in candidate_insert_positions if rng.random() < cfg.p_insert)
    )

    inserted_spurious: List[int] = []
    if insert_count > 0:
        rng.shuffle(candidate_insert_positions)
        for pos in candidate_insert_positions[:insert_count]:
            if pos not in occupied:
                current[pos] = None
                occupied.add(pos)
                inserted_spurious.append(pos)

    noisy_positions = sorted(current.keys())
    b_noisy = [1 if i in current else 0 for i in range(num_gaps)]

    edit_labels: List[Dict[str, Any]] = []
    aligned_gold_targets = set()
    for g in noisy_positions:
        src = current[g]
        if src is None:
            label = "DEL"
            target = None
        else:
            delta = src - g
            target = src
            aligned_gold_targets.add(src)
            label = "SHIFT:0" if delta == 0 else f"SHIFT:{delta}"
        edit_labels.append({"g": g, "y": label, "target": target})

    insert_labels = [0] * num_gaps
    for gold_g in gold_positions:
        if gold_g not in aligned_gold_targets:
            insert_labels[gold_g] = 1

    ops = {
        "gold_boundary_count": len(gold_positions),
        "noisy_boundary_count": len(noisy_positions),
        "deleted_gold": sorted(list(gold_set - aligned_gold_targets)),
        "shifted": [{"from": old, "to": new} for old, new in sorted(shifted_ops)],
        "spurious_inserted": sorted(inserted_spurious),
    }

    return NoiseResult(
        b_noisy=b_noisy,
        edit_labels=edit_labels,
        insert_labels=insert_labels,
        gold_boundaries=gold_positions,
        noisy_boundaries=noisy_positions,
        ops=ops,
    )


# -----------------------------
# Conversion core
# -----------------------------


def stable_sample_id(*parts: str) -> str:
    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"refiner_{h}"


def compact_unit_for_output(unit: Dict[str, Any], fallback_uid: int) -> Dict[str, Any]:
    meta = unit.get("meta") if isinstance(unit.get("meta"), dict) else {}
    out = {
        "unit_id": unit.get("atom_id", unit.get("unit_id", fallback_uid)),
        "text": unit.get("text", ""),
        "type": unit.get("type", "other"),
        "level": unit.get("level", 0),
        "page": unit.get("page"),
        "meta": meta,
    }
    return out


def infer_domain_label(source_family: str) -> str:
    return "rail" if source_family == "railway_parsed" else "llm_gold"


def build_refiner_sample(
    doc: Dict[str, Any],
    split_name: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    doc_id = str(doc.get("doc_id") or doc.get("sample_id") or "unknown_doc")
    source_family = str(doc.get("source_family") or "unknown")
    language = str(doc.get("language") or "unknown")
    domain = str(doc.get("domain") or "unknown")
    doc_name = str(doc.get("doc_name") or doc_id)
    source_path = doc.get("source_path")
    source_rel_path = doc.get("source_rel_path")
    orig_split = doc.get("orig_split")

    raw_units = doc.get("atoms") or doc.get("units") or []
    if not isinstance(raw_units, list) or len(raw_units) == 0:
        return None, {
            "split": split_name,
            "doc_id": doc_id,
            "reason": "empty_units",
            "source_path": source_path,
        }

    chunk0_units: List[Dict[str, Any]] = []
    for idx, unit in enumerate(raw_units):
        if not isinstance(unit, dict):
            unit = {"text": str(unit)}
        text = normalize_text(unit.get("text", ""))
        unit_copy = dict(unit)
        unit_copy["text"] = text
        chunk0_units.append(compact_unit_for_output(unit_copy, fallback_uid=idx))

    mutable_units = [dict(u) for u in chunk0_units]
    projection_fixes: List[Dict[str, Any]] = []

    def unit_text(i: int) -> str:
        return str(mutable_units[i].get("text") or "")

    atomized_units: List[List[str]] = [
        atomize_text(
            unit_text(i),
            language=language,
            line_first_min_lines=args.line_first_min_lines,
            atom_max_tokens=args.atom_max_tokens,
            atom_max_chars=args.atom_max_chars,
            merge_short_tokens=args.merge_short_tokens,
            merge_short_chars=args.merge_short_chars,
            dot_in_cjk_as_terminator=args.dot_in_cjk_as_terminator,
            blankline_split=args.blankline_split,
        )
        for i in range(len(mutable_units))
    ]

    for i in range(len(mutable_units)):
        if atomized_units[i]:
            continue
        txt = unit_text(i)
        if not txt:
            continue
        target = None
        for j in range(i + 1, len(mutable_units)):
            if unit_text(j).strip():
                target = j
                break
        if target is None:
            for j in range(i - 1, -1, -1):
                if unit_text(j).strip():
                    target = j
                    break
        if target is not None and target != i:
            glue = (txt + "\n" + unit_text(target)).strip() if target > i else (unit_text(target) + "\n" + txt).strip()
            mutable_units[target]["text"] = glue
            mutable_units[i]["text"] = ""
            projection_fixes.append({"kind": "empty_unit_merge", "from": i, "to": target})

    atomized_units = [
        atomize_text(
            unit_text(i),
            language=language,
            line_first_min_lines=args.line_first_min_lines,
            atom_max_tokens=args.atom_max_tokens,
            atom_max_chars=args.atom_max_chars,
            merge_short_tokens=args.merge_short_tokens,
            merge_short_chars=args.merge_short_chars,
            dot_in_cjk_as_terminator=args.dot_in_cjk_as_terminator,
            blankline_split=args.blankline_split,
        )
        for i in range(len(mutable_units))
    ]

    atoms: List[Dict[str, Any]] = []
    unit2atom_span: List[Dict[str, int]] = []
    nonempty_units_for_output: List[Dict[str, Any]] = []
    cursor = 0
    for i, (unit, unit_atoms) in enumerate(zip(mutable_units, atomized_units)):
        if not unit_atoms:
            projection_fixes.append({"kind": "skip_empty_unit", "unit_index": i})
            continue
        s = cursor
        for atom_text in unit_atoms:
            atoms.append(
                {
                    "aid": cursor,
                    "text": atom_text,
                    "type": unit.get("type", "other"),
                    "level": unit.get("level", 0),
                    "page": unit.get("page"),
                    "meta": {
                        "chunk0_unit_id": unit.get("unit_id", i),
                        **(unit.get("meta") if isinstance(unit.get("meta"), dict) else {}),
                    },
                }
            )
            cursor += 1
        e = cursor
        nonempty_units_for_output.append(unit)
        unit2atom_span.append({"unit_id": unit.get("unit_id", i), "s": s, "e": e})

    if len(atoms) < 2:
        return None, {
            "split": split_name,
            "doc_id": doc_id,
            "reason": "too_few_atoms_after_atomization",
            "num_atoms": len(atoms),
            "source_path": source_path,
        }

    if args.max_doc_atoms is not None and len(atoms) > args.max_doc_atoms:
        return None, {
            "split": split_name,
            "doc_id": doc_id,
            "reason": "exceeds_max_doc_atoms",
            "num_atoms": len(atoms),
            "max_doc_atoms": args.max_doc_atoms,
            "source_path": source_path,
        }

    b_gold = boundaries_from_spans(unit2atom_span, num_atoms=len(atoms))
    if sum(b_gold) == 0:
        return None, {
            "split": split_name,
            "doc_id": doc_id,
            "reason": "no_projectable_gold_boundaries",
            "num_atoms": len(atoms),
            "num_units": len(nonempty_units_for_output),
            "source_path": source_path,
        }

    noise = apply_noise_to_gold(
        b_gold,
        cfg=NoiseConfig(
            K=args.K,
            p_shift=args.p_shift,
            p_insert=args.p_insert,
            p_delete=args.p_delete,
            discrete_laplace_scale=args.discrete_laplace_scale,
            max_insert_frac=args.max_insert_frac,
            max_delete_frac=args.max_delete_frac,
            max_shift_frac=args.max_shift_frac,
            burst_delete_prob=args.burst_delete_prob,
            burst_delete_min=args.burst_delete_min,
            burst_delete_max=args.burst_delete_max,
        ),
        rng=rng,
    )

    if len(noise.b_noisy) != len(b_gold):
        return None, {
            "split": split_name,
            "doc_id": doc_id,
            "reason": "boundary_vector_length_mismatch",
            "len_b_gold": len(b_gold),
            "len_b_noisy": len(noise.b_noisy),
            "source_path": source_path,
        }
    if len(noise.insert_labels) != len(b_gold):
        return None, {
            "split": split_name,
            "doc_id": doc_id,
            "reason": "insert_label_length_mismatch",
            "len_b_gold": len(b_gold),
            "len_insert": len(noise.insert_labels),
            "source_path": source_path,
        }

    out = {
        "sample_id": stable_sample_id(split_name, doc_id, str(source_path or source_rel_path or doc_name)),
        "doc_id": doc_id,
        "doc_name": doc_name,
        "domain": infer_domain_label(source_family),
        "source_family": source_family,
        "source_domain": domain,
        "language": language,
        "source_path": source_path,
        "source_rel_path": source_rel_path,
        "orig_split": orig_split,
        "chunk0_units": nonempty_units_for_output,
        "atoms": atoms,
        "unit2atom_span": unit2atom_span,
        "b0": noise.b_noisy,
        "b_gold": b_gold,
        "labels": {
            "edit": noise.edit_labels,
            "insert": noise.insert_labels,
        },
        "meta": {
            "split": split_name,
            "K": args.K,
            "tokenizer": args.tokenizer_name,
            "atomizer": {
                "line_first_min_lines": args.line_first_min_lines,
                "atom_max_tokens": args.atom_max_tokens,
                "atom_max_chars": args.atom_max_chars,
                "merge_short_tokens": args.merge_short_tokens,
                "merge_short_chars": args.merge_short_chars,
                "dot_in_cjk_as_terminator": args.dot_in_cjk_as_terminator,
                "blankline_split": args.blankline_split,
            },
            "length_constraints": {
                "min_chunk_tokens": args.min_chunk_tokens,
                "max_chunk_tokens": args.max_chunk_tokens,
                "min_chunk_atoms": args.min_chunk_atoms,
                "max_chunk_atoms": args.max_chunk_atoms,
            },
            "noise": {
                "p_shift": args.p_shift,
                "p_insert": args.p_insert,
                "p_delete": args.p_delete,
                "discrete_laplace_scale": args.discrete_laplace_scale,
                "max_insert_frac": args.max_insert_frac,
                "max_delete_frac": args.max_delete_frac,
                "max_shift_frac": args.max_shift_frac,
                **noise.ops,
            },
            "projection_fix": projection_fixes,
            "num_chunk0_units": len(nonempty_units_for_output),
            "num_atoms": len(atoms),
            "num_gold_boundaries": sum(b_gold),
            "num_noisy_boundaries": sum(noise.b_noisy),
            "num_insert_labels": int(sum(noise.insert_labels)),
        },
    }
    return out, None


# -----------------------------
# Stats
# -----------------------------


def percentile(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(math.floor((len(sorted_vals) - 1) * p))))
    return sorted_vals[idx]


def summarize_lengths(vals: List[int]) -> Dict[str, Any]:
    if not vals:
        return {"count": 0}
    xs = sorted(vals)
    return {
        "count": len(xs),
        "min": xs[0],
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "max": xs[-1],
        "mean": statistics.mean(xs),
    }


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Boundary Refiner denoising JSONL from real_dataset JSONL files.")
    ap.add_argument("--input_root", type=str, required=True, help="Path to real_dataset root containing train.jsonl/dev.jsonl/test.jsonl")
    ap.add_argument("--output_root", type=str, required=True, help="Output directory for refiner dataset")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])

    ap.add_argument("--line_first_min_lines", type=int, default=3)
    ap.add_argument("--atom_max_tokens", type=int, default=120)
    ap.add_argument("--atom_max_chars", type=int, default=480)
    ap.add_argument("--merge_short_tokens", type=int, default=8)
    ap.add_argument("--merge_short_chars", type=int, default=20)
    ap.add_argument("--dot_in_cjk_as_terminator", action="store_true", default=True)
    ap.add_argument("--no_dot_in_cjk_as_terminator", dest="dot_in_cjk_as_terminator", action="store_false")
    ap.add_argument("--blankline_split", action="store_true", default=True)
    ap.add_argument("--no_blankline_split", dest="blankline_split", action="store_false")

    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--p_shift", type=float, default=0.50)
    ap.add_argument("--p_insert", type=float, default=0.15)
    ap.add_argument("--p_delete", type=float, default=0.40)
    ap.add_argument("--discrete_laplace_scale", type=float, default=2.0)
    ap.add_argument("--max_insert_frac", type=float, default=0.15)
    ap.add_argument("--max_delete_frac", type=float, default=0.40)
    ap.add_argument("--max_shift_frac", type=float, default=0.50)

    ap.add_argument("--burst_delete_prob", type=float, default=0.35)
    ap.add_argument("--burst_delete_min", type=int, default=2)
    ap.add_argument("--burst_delete_max", type=int, default=4)

    ap.add_argument("--min_chunk_tokens", type=int, default=48)
    ap.add_argument("--max_chunk_tokens", type=int, default=384)
    ap.add_argument("--min_chunk_atoms", type=int, default=2)
    ap.add_argument("--max_chunk_atoms", type=int, default=64)
    ap.add_argument("--tokenizer_name", type=str, default="bge-m3")

    ap.add_argument(
        "--max_doc_atoms",
        type=int,
        default=None,
        help="If set, skip documents whose atomized length exceeds this value",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "stats").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    failed_path = output_root / "logs" / "failed_records.jsonl"
    skipped_path = output_root / "logs" / "skipped_files.jsonl"

    summary: Dict[str, Any] = {
        "config": {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "seed": args.seed,
            "splits": args.splits,
            "atomizer": {
                "line_first_min_lines": args.line_first_min_lines,
                "atom_max_tokens": args.atom_max_tokens,
                "atom_max_chars": args.atom_max_chars,
                "merge_short_tokens": args.merge_short_tokens,
                "merge_short_chars": args.merge_short_chars,
                "dot_in_cjk_as_terminator": args.dot_in_cjk_as_terminator,
                "blankline_split": args.blankline_split,
            },
            "noise": {
                "K": args.K,
                "p_shift": args.p_shift,
                "p_insert": args.p_insert,
                "p_delete": args.p_delete,
                "discrete_laplace_scale": args.discrete_laplace_scale,
                "max_insert_frac": args.max_insert_frac,
                "max_delete_frac": args.max_delete_frac,
                "max_shift_frac": args.max_shift_frac,
            },
            "length_constraints": {
                "min_chunk_tokens": args.min_chunk_tokens,
                "max_chunk_tokens": args.max_chunk_tokens,
                "min_chunk_atoms": args.min_chunk_atoms,
                "max_chunk_atoms": args.max_chunk_atoms,
            },
            "max_doc_atoms": args.max_doc_atoms,
        }
    }

    with JsonlWriter(failed_path) as failed_writer, JsonlWriter(skipped_path) as skipped_writer:
        for split in args.splits:
            in_path = input_root / f"{split}.jsonl"
            out_path = output_root / f"refiner_{split}.jsonl"
            if not in_path.exists():
                skipped_writer.write({"split": split, "reason": "missing_input_jsonl", "path": str(in_path)})
                continue

            rng = random.Random(args.seed + sum(ord(c) for c in split))
            doc_lengths: List[int] = []
            gold_boundary_counts: List[int] = []
            noisy_boundary_counts: List[int] = []
            insert_counts: List[int] = []
            domain_counts: Dict[str, int] = {}
            language_counts: Dict[str, int] = {}
            source_family_counts: Dict[str, int] = {}
            failure_counts: Dict[str, int] = {}
            emitted = 0
            seen = 0

            with JsonlWriter(out_path) as writer:
                for doc in read_jsonl(in_path):
                    seen += 1
                    sample, failure = build_refiner_sample(doc, split_name=split, args=args, rng=rng)
                    if failure is not None:
                        failure_counts[failure["reason"]] = failure_counts.get(failure["reason"], 0) + 1
                        failed_writer.write(failure)
                        continue

                    assert sample is not None
                    writer.write(sample)
                    emitted += 1
                    doc_lengths.append(sample["meta"]["num_atoms"])
                    gold_boundary_counts.append(sample["meta"]["num_gold_boundaries"])
                    noisy_boundary_counts.append(sample["meta"]["num_noisy_boundaries"])
                    insert_counts.append(sample["meta"]["num_insert_labels"])
                    domain_counts[sample["source_domain"]] = domain_counts.get(sample["source_domain"], 0) + 1
                    language_counts[sample["language"]] = language_counts.get(sample["language"], 0) + 1
                    source_family_counts[sample["source_family"]] = source_family_counts.get(sample["source_family"], 0) + 1

            summary[split] = {
                "input_docs": seen,
                "emitted_docs": emitted,
                "failed_docs": seen - emitted,
                "failure_counts": dict(sorted(failure_counts.items())),
                "length": summarize_lengths(doc_lengths),
                "gold_boundary_count": summarize_lengths(gold_boundary_counts),
                "noisy_boundary_count": summarize_lengths(noisy_boundary_counts),
                "insert_label_count": summarize_lengths(insert_counts),
                "domain_counts": dict(sorted(domain_counts.items())),
                "language_counts": dict(sorted(language_counts.items())),
                "source_family_counts": dict(sorted(source_family_counts.items())),
                "output_path": str(out_path),
            }

    with (output_root / "stats" / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()