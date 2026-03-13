#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Week2 railway pseudo labels from best-of-N candidates."
    )
    p.add_argument("--atoms_b0_jsonl", type=str, required=True)
    p.add_argument("--candidates_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # 预留接口；当前 lexical jump 不实际使用 embedding cache
    p.add_argument("--embed_cache_dir", type=str, default=None)

    # scorer 权重
    p.add_argument("--w_edit", type=float, default=0.8)
    p.add_argument("--w_density", type=float, default=0.8)
    p.add_argument("--w_jump", type=float, default=1.5)
    p.add_argument("--w_conf", type=float, default=1.0)
    p.add_argument("--w_anchor", type=float, default=0.8)
    p.add_argument("--w_tiny_noise", type=float, default=0.6)
    p.add_argument("--w_oversize_hard", type=float, default=7.0)
    p.add_argument("--w_oversize_soft", type=float, default=2.0)

    # 伪标筛选阈值（已按铁路域放宽）
    p.add_argument("--min_gain_vs_b0", type=float, default=0.03)
    p.add_argument("--min_margin", type=float, default=0.015)
    p.add_argument("--min_conf", type=float, default=0.50)
    p.add_argument("--max_edit_rate", type=float, default=0.60)
    p.add_argument("--min_density_ratio", type=float, default=0.45)
    p.add_argument("--max_density_ratio", type=float, default=2.20)

    p.add_argument("--k_shift", type=int, default=6)

    # chunk 约束
    p.add_argument("--max_chunk_tokens", type=int, default=384)
    p.add_argument("--max_chunk_atoms", type=int, default=64)
    p.add_argument("--max_chunk_chars", type=int, default=1600)

    # semantic jump 当前只支持 lexical proxy
    p.add_argument(
        "--semantic_mode",
        type=str,
        default="lexical",
        choices=["lexical"],
    )
    return p.parse_args()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse {path} line {line_no}: {e}") from e


def dump_json(path: str | Path, obj: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dense_b0_to_sparse(b0: Sequence[Any]) -> List[int]:
    return [i for i, v in enumerate(b0) if int(v) == 1]


def sparse_to_dense(boundaries_sparse: Sequence[int], num_atoms: int) -> List[int]:
    dense = [0] * max(0, num_atoms - 1)
    for g in boundaries_sparse:
        g = int(g)
        if 0 <= g < len(dense):
            dense[g] = 1
    return dense


def boundaries_to_chunk_spans(boundaries_sparse: Sequence[int], num_atoms: int) -> List[Tuple[int, int]]:
    if num_atoms <= 0:
        return []

    cuts = sorted(set(int(g) for g in boundaries_sparse if 0 <= int(g) < num_atoms - 1))
    spans: List[Tuple[int, int]] = []
    s = 0
    for g in cuts:
        e = g + 1
        spans.append((s, e))
        s = e
    spans.append((s, num_atoms))
    return spans


def rough_token_len(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    if " " in text:
        return max(1, len(text.split()))
    return max(1, len(text) // 2)


def get_span_lengths(
    atoms: Sequence[str],
    boundaries_sparse: Sequence[int],
) -> Tuple[List[int], List[int], List[int]]:
    spans = boundaries_to_chunk_spans(boundaries_sparse, len(atoms))
    atom_lens: List[int] = []
    tok_lens: List[int] = []
    char_lens: List[int] = []

    for s, e in spans:
        text = "\n".join(atoms[s:e]).strip()
        atom_lens.append(e - s)
        tok_lens.append(rough_token_len(text))
        char_lens.append(len(text))

    return atom_lens, tok_lens, char_lens


def mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs)) / max(1, len(xs))


def oversize_hard_violation_count(
    atom_lens: Sequence[int],
    tok_lens: Sequence[int],
    char_lens: Sequence[int],
    max_chunk_atoms: int,
    max_chunk_tokens: int,
    max_chunk_chars: int,
) -> int:
    cnt = 0
    for a, t, c in zip(atom_lens, tok_lens, char_lens):
        if a > max_chunk_atoms or t > max_chunk_tokens or c > max_chunk_chars:
            cnt += 1
    return cnt


def oversize_soft_penalty(
    atom_lens: Sequence[int],
    tok_lens: Sequence[int],
    char_lens: Sequence[int],
    max_chunk_atoms: int,
    max_chunk_tokens: int,
    max_chunk_chars: int,
) -> float:
    vals: List[float] = []
    # 只看上界附近或超过上界，不看短
    soft_tok = min(max_chunk_tokens - 64, 320)
    soft_atoms = min(max_chunk_atoms - 16, 48)
    soft_chars = min(max_chunk_chars - 400, 1200)

    soft_tok = max(64, soft_tok)
    soft_atoms = max(16, soft_atoms)
    soft_chars = max(200, soft_chars)

    for a, t, c in zip(atom_lens, tok_lens, char_lens):
        v = 0.0
        v += max(0.0, t - soft_tok) / float(max(1, soft_tok))
        v += 0.5 * max(0.0, a - soft_atoms) / float(max(1, soft_atoms))
        v += 0.5 * max(0.0, c - soft_chars) / float(max(1, soft_chars))
        vals.append(v)
    return mean(vals)


def normalize_text_for_jump(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_features(text: str) -> set:
    """
    词 + 字符 3-gram 混合特征。
    对中英混杂 / OCR 噪声更稳。
    """
    text = normalize_text_for_jump(text)
    feats = set()

    toks = [t for t in re.split(r"[^\w]+", text) if t]
    feats.update(toks)

    raw = text.replace(" ", "")
    if len(raw) <= 3:
        if raw:
            feats.add(raw)
    else:
        for i in range(len(raw) - 2):
            feats.add(raw[i:i + 3])

    return feats


def lexical_jump_one(left_texts: Sequence[str], right_texts: Sequence[str]) -> float:
    left = "\n".join(left_texts).strip()
    right = "\n".join(right_texts).strip()
    if not left or not right:
        return 0.0

    f1 = make_features(left)
    f2 = make_features(right)
    if not f1 or not f2:
        return 0.0

    inter = len(f1 & f2)
    union = len(f1 | f2)
    sim = inter / max(1, union)
    return 1.0 - sim


def semantic_jump_bonus(atoms: Sequence[str], boundaries_sparse: Sequence[int]) -> float:
    """
    当前版本用 lexical proxy 代替 embedding jump。
    对每个边界 g：
      左窗口 = [g-1, g]
      右窗口 = [g+1, g+2]
    """
    vals: List[float] = []
    n = len(atoms)

    for g in boundaries_sparse:
        g = int(g)
        left_ids = [i for i in [g - 1, g] if 0 <= i < n]
        right_ids = [i for i in [g + 1, g + 2] if 0 <= i < n]
        if not left_ids or not right_ids:
            continue
        left_texts = [atoms[i] for i in left_ids]
        right_texts = [atoms[i] for i in right_ids]
        vals.append(lexical_jump_one(left_texts, right_texts))

    return mean(vals)


def looks_structural_anchor(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False

    low = text.lower()
    short = len(text) <= 80

    patterns = [
        r"^\d+(\.\d+){0,5}[\.\)]?\s*$",                     # 1 / 1.2 / 1.2.3
        r"^\d+(\.\d+){0,5}[\.\)]?\s+\S+.*$",               # 1.2 Scope
        r"^第[一二三四五六七八九十百千万0-9]+[章节条部分]\s*.*$",
        r"^(附录|附件)\s*[A-Z0-9一二三四五六七八九十]?\s*.*$",
        r"^(annex|appendix|table|figure|note|scope|definitions?|requirements?)\b.*$",
        r"^[-•·]\s+\S+.*$",                                # bullet
        r"^[A-Z]\.\d+.*$",                                 # A.1 ...
        r"^[（(]?[0-9一二三四五六七八九十]+[)）]\s*.*$",     # (1) / （一）
    ]

    if short:
        for pat in patterns:
            if re.match(pat, low, flags=re.IGNORECASE):
                return True

    # 全大写短标题 / 编号感很强的短行
    if short and text.upper() == text and len(text.split()) <= 8:
        return True

    return False


def structure_anchor_bonus(atoms: Sequence[str], boundaries_sparse: Sequence[int]) -> float:
    spans = boundaries_to_chunk_spans(boundaries_sparse, len(atoms))
    vals = []
    for s, e in spans:
        if s < len(atoms):
            first_atom = atoms[s]
            vals.append(1.0 if looks_structural_anchor(first_atom) else 0.0)
    return mean(vals)


def tiny_noise_penalty(atoms: Sequence[str], boundaries_sparse: Sequence[int]) -> float:
    spans = boundaries_to_chunk_spans(boundaries_sparse, len(atoms))
    vals = []

    for s, e in spans:
        text = "\n".join(atoms[s:e]).strip()
        tok = rough_token_len(text)
        char_len = len(text)

        # 极短，且不像结构锚点，才视为噪声
        if tok <= 6 and char_len <= 25 and not looks_structural_anchor(text):
            vals.append(1.0)
        else:
            vals.append(0.0)

    return mean(vals)


def confidence_bonus(cand: Dict[str, Any]) -> float:
    ts = cand.get("teacher_stats", {})
    me = ts.get("mean_edit_prob", None)
    mi = ts.get("mean_insert_prob", None)

    me = float(me) if me is not None else 0.0
    mi = float(mi) if mi is not None else 0.0
    return 0.5 * me + 0.5 * mi


def edit_cost_norm(cand: Dict[str, Any], num_seed_boundaries: int) -> float:
    ts = cand.get("teacher_stats", {})
    n_del = int(ts.get("num_deleted", 0) or 0)
    n_ins = int(ts.get("num_inserted", 0) or 0)
    sum_abs_shift = int(ts.get("sum_abs_shift", 0) or 0)
    return (
        1.0 * n_del
        + 1.0 * n_ins
        + 0.25 * sum_abs_shift
    ) / max(1, num_seed_boundaries)


def edit_rate(cand: Dict[str, Any], num_seed_boundaries: int) -> float:
    ts = cand.get("teacher_stats", {})
    n_del = int(ts.get("num_deleted", 0) or 0)
    n_ins = int(ts.get("num_inserted", 0) or 0)
    n_shift = int(ts.get("num_shifted", 0) or 0)
    return (n_del + n_ins + n_shift) / max(1, num_seed_boundaries)


def density_ratio(cand_num_chunks: int, seed_num_chunks: int) -> float:
    return (float(cand_num_chunks) + 1e-6) / (float(seed_num_chunks) + 1e-6)


def boundary_density_penalty(cand_num_chunks: int, seed_num_chunks: int) -> float:
    ratio = density_ratio(cand_num_chunks, seed_num_chunks)
    return abs(math.log(ratio))


def score_candidate(
    cand: Dict[str, Any],
    atoms_doc: Dict[str, Any],
    seed_num_chunks: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    atoms = atoms_doc["atoms"]
    b_pred_sparse = cand["prediction"]["b_pred_sparse"]

    atom_lens, tok_lens, char_lens = get_span_lengths(atoms, b_pred_sparse)

    oversize_hard = oversize_hard_violation_count(
        atom_lens=atom_lens,
        tok_lens=tok_lens,
        char_lens=char_lens,
        max_chunk_atoms=args.max_chunk_atoms,
        max_chunk_tokens=args.max_chunk_tokens,
        max_chunk_chars=args.max_chunk_chars,
    )

    oversize_soft = oversize_soft_penalty(
        atom_lens=atom_lens,
        tok_lens=tok_lens,
        char_lens=char_lens,
        max_chunk_atoms=args.max_chunk_atoms,
        max_chunk_tokens=args.max_chunk_tokens,
        max_chunk_chars=args.max_chunk_chars,
    )

    edit_cost = edit_cost_norm(cand, num_seed_boundaries=max(1, seed_num_chunks - 1))
    num_chunks = len(atom_lens)
    density_pen = boundary_density_penalty(num_chunks, seed_num_chunks)
    jump = semantic_jump_bonus(atoms, b_pred_sparse)
    conf = confidence_bonus(cand)
    anchor = structure_anchor_bonus(atoms, b_pred_sparse)
    tiny_noise = tiny_noise_penalty(atoms, b_pred_sparse)

    score = (
        - args.w_oversize_hard * oversize_hard
        - args.w_oversize_soft * oversize_soft
        - args.w_edit * edit_cost
        - args.w_density * density_pen
        + args.w_jump * jump
        + args.w_conf * conf
        + args.w_anchor * anchor
        - args.w_tiny_noise * tiny_noise
    )

    return {
        "score": score,
        "oversize_hard_penalty": float(oversize_hard),
        "oversize_soft_penalty": oversize_soft,
        "edit_cost_norm": edit_cost,
        "boundary_density_penalty": density_pen,
        "semantic_jump_bonus": jump,
        "confidence_bonus": conf,
        "structure_anchor_bonus": anchor,
        "tiny_noise_penalty": tiny_noise,
        "num_chunks": num_chunks,
    }


def label_to_edit_choice(label: str, K: int) -> int:
    if label == "DEL":
        return 0
    if label == "KEEP":
        return 1 + K
    if label.startswith("SHIFT:"):
        k = int(label.split(":", 1)[1])
        return 1 + K + k
    raise ValueError(f"Unknown edit label: {label}")


def build_training_record(
    atoms_doc: Dict[str, Any],
    best_cand: Dict[str, Any],
    score_terms: Dict[str, float],
    gain_vs_b0: float,
    margin: float,
    sample_weight: Optional[float],
    K: int,
) -> Dict[str, Any]:
    atoms = atoms_doc["atoms"]
    num_atoms = len(atoms)
    num_gaps = max(0, num_atoms - 1)

    b0_dense = [int(x) for x in atoms_doc["b0"]]
    b_gold_dense = sparse_to_dense(best_cand["prediction"]["b_pred_sparse"], num_atoms)
    g0_positions = dense_b0_to_sparse(b0_dense)

    trace_by_g = {
        int(x["g0"]): (
            "DEL"
            if str(x["action"]) == "DEL"
            else ("KEEP" if str(x["action"]) == "KEEP" else f"SHIFT:{int(x['shift'])}")
        )
        for x in best_cand["prediction"]["edit_trace"]
    }

    edit_choice: List[int] = []
    edit_labels: List[Dict[str, Any]] = []
    for g in g0_positions:
        y = trace_by_g.get(int(g), "KEEP")
        edit_choice.append(label_to_edit_choice(y, K))
        edit_labels.append({"g": int(g), "y": y})

    insert_dense = [0] * num_gaps
    for item in best_cand["prediction"]["insert_gaps"]:
        g = int(item["gap"])
        if 0 <= g < num_gaps:
            insert_dense[g] = 1

    record = {
        "sample_id": f"rail_pseudo::{atoms_doc['doc_id']}::r1",
        "doc_id": atoms_doc["doc_id"],
        "domain": "rail_pseudo",
        "chunk0_units": atoms_doc.get("chunk0_units", []),
        "atoms": atoms,
        "atoms_text": atoms,  # 兼容当前 train_loop / dataset 读取
        "unit2atom_span": atoms_doc.get("unit2atom_span", []),
        "b0": b0_dense,
        "b_gold": b_gold_dense,
        "g0_positions": g0_positions,
        "edit_choice": edit_choice,
        "insert_labels": insert_dense,
        "labels": {
            "edit": edit_labels,
            "insert": insert_dense,
        },
        "pseudo_meta": {
            "round": 1,
            "seed_source": atoms_doc.get("meta", {}).get("source", "rail_rule_seed_json"),
            "teacher_ckpt": best_cand.get("teacher_ckpt"),
            "teacher_score": score_terms["score"],
            "score_gain_vs_b0": gain_vs_b0,
            "score_margin": margin,
            "candidate_id": best_cand.get("candidate_id"),
            "candidate_type": best_cand.get("candidate_type"),
            "score_terms": score_terms,
        },
    }

    if sample_weight is not None:
        record["sample_weight"] = float(sample_weight)

    return record


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def build_sample_weight(gain_vs_b0: float, margin: float) -> float:
    w = sigmoid(2.0 * gain_vs_b0 + 4.0 * margin)
    return max(0.30, min(1.00, float(w)))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    atoms_docs: Dict[str, Dict[str, Any]] = {}
    duplicate_doc_ids_in_atoms = 0
    for obj in load_jsonl(args.atoms_b0_jsonl):
        doc_id = str(obj["doc_id"])
        if doc_id in atoms_docs:
            duplicate_doc_ids_in_atoms += 1
        atoms_docs[doc_id] = {
            "doc_id": doc_id,
            "chunk0_units": obj.get("chunk0_units", []),
            "atoms": obj["atoms"],
            "unit2atom_span": obj.get("unit2atom_span", []),
            "b0": obj["b0"],
            "meta": obj.get("meta", {}),
        }

    cand_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_candidates = 0
    for cand in load_jsonl(args.candidates_jsonl):
        doc_id = str(cand["doc_id"])
        cand_groups[doc_id].append(cand)
        total_candidates += 1

    pseudo_raw_path = output_dir / "pseudo_train_raw.jsonl"
    pseudo_weighted_path = output_dir / "pseudo_train_weighted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    best_cands_path = output_dir / "best_candidates.jsonl"
    summary_path = output_dir / "selection_summary.json"

    keep_count = 0
    reject_count = 0
    reason_counter: Counter = Counter()
    by_candidate_type: Counter = Counter()

    score_gain_vals: List[float] = []
    margin_vals: List[float] = []
    sample_weights: List[float] = []

    with pseudo_raw_path.open("w", encoding="utf-8") as f_raw, \
         pseudo_weighted_path.open("w", encoding="utf-8") as f_w, \
         rejected_path.open("w", encoding="utf-8") as f_r, \
         best_cands_path.open("w", encoding="utf-8") as f_best:

        for doc_id, atoms_doc in atoms_docs.items():
            cands = cand_groups.get(doc_id, [])
            if not cands:
                reject_count += 1
                reason_counter["missing_candidates"] += 1
                f_r.write(json.dumps({
                    "doc_id": doc_id,
                    "reason": "missing_candidates",
                }, ensure_ascii=False) + "\n")
                continue

            identity_cands = [
                c for c in cands
                if c.get("candidate_type") == "identity" or c.get("candidate_id") == "identity"
            ]
            if not identity_cands:
                reject_count += 1
                reason_counter["missing_identity"] += 1
                f_r.write(json.dumps({
                    "doc_id": doc_id,
                    "reason": "missing_identity",
                }, ensure_ascii=False) + "\n")
                continue

            identity = identity_cands[0]
            seed_num_chunks = int(identity["chunk_stats"]["num_chunks"])

            scored: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
            for cand in cands:
                s = score_candidate(
                    cand=cand,
                    atoms_doc=atoms_doc,
                    seed_num_chunks=seed_num_chunks,
                    args=args,
                )
                scored.append((cand, s))

            scored.sort(key=lambda x: x[1]["score"], reverse=True)
            best_cand, best_score_terms = scored[0]
            second_score_terms = scored[1][1] if len(scored) > 1 else None

            identity_score_terms = None
            for cand, s in scored:
                if cand.get("candidate_id") == "identity" or cand.get("candidate_type") == "identity":
                    identity_score_terms = s
                    break

            if identity_score_terms is None:
                reject_count += 1
                reason_counter["identity_score_missing"] += 1
                f_r.write(json.dumps({
                    "doc_id": doc_id,
                    "reason": "identity_score_missing",
                }, ensure_ascii=False) + "\n")
                continue

            gain_vs_b0 = float(best_score_terms["score"] - identity_score_terms["score"])
            margin = float(best_score_terms["score"] - second_score_terms["score"]) if second_score_terms else 0.0

            num_seed_boundaries = max(1, seed_num_chunks - 1)
            cand_num_chunks = int(best_score_terms["num_chunks"])
            dens_ratio = density_ratio(cand_num_chunks, seed_num_chunks)
            conf_bonus_val = float(best_score_terms["confidence_bonus"])
            e_rate = edit_rate(best_cand, num_seed_boundaries=num_seed_boundaries)
            hard_viol = int(best_score_terms["oversize_hard_penalty"])

            reject_reason: Optional[str] = None
            if hard_viol != 0:
                reject_reason = "oversize_hard_violation"
            elif gain_vs_b0 < args.min_gain_vs_b0:
                reject_reason = "low_gain_vs_b0"
            elif margin < args.min_margin:
                reject_reason = "low_margin"
            elif conf_bonus_val < args.min_conf:
                reject_reason = "low_confidence"
            elif e_rate > args.max_edit_rate:
                reject_reason = "edit_rate_too_high"
            elif dens_ratio < args.min_density_ratio:
                reject_reason = "density_too_sparse"
            elif dens_ratio > args.max_density_ratio:
                reject_reason = "density_too_dense"

            best_out = {
                "doc_id": doc_id,
                "candidate_id": best_cand.get("candidate_id"),
                "candidate_type": best_cand.get("candidate_type"),
                "score_terms": best_score_terms,
                "identity_score": identity_score_terms["score"],
                "gain_vs_b0": gain_vs_b0,
                "margin": margin,
                "density_ratio": dens_ratio,
                "confidence_bonus": conf_bonus_val,
                "edit_rate": e_rate,
                "oversize_hard_violations": hard_viol,
                "selected": reject_reason is None,
            }
            f_best.write(json.dumps(best_out, ensure_ascii=False) + "\n")

            if reject_reason is not None:
                reject_count += 1
                reason_counter[reject_reason] += 1
                f_r.write(json.dumps({
                    "doc_id": doc_id,
                    "reason": reject_reason,
                    "candidate_id": best_cand.get("candidate_id"),
                    "candidate_type": best_cand.get("candidate_type"),
                    "score_terms": best_score_terms,
                    "gain_vs_b0": gain_vs_b0,
                    "margin": margin,
                    "density_ratio": dens_ratio,
                    "confidence_bonus": conf_bonus_val,
                    "edit_rate": e_rate,
                    "oversize_hard_violations": hard_viol,
                }, ensure_ascii=False) + "\n")
                continue

            w = build_sample_weight(gain_vs_b0, margin)
            rec_raw = build_training_record(
                atoms_doc=atoms_doc,
                best_cand=best_cand,
                score_terms=best_score_terms,
                gain_vs_b0=gain_vs_b0,
                margin=margin,
                sample_weight=None,
                K=args.k_shift,
            )
            rec_w = build_training_record(
                atoms_doc=atoms_doc,
                best_cand=best_cand,
                score_terms=best_score_terms,
                gain_vs_b0=gain_vs_b0,
                margin=margin,
                sample_weight=w,
                K=args.k_shift,
            )

            f_raw.write(json.dumps(rec_raw, ensure_ascii=False) + "\n")
            f_w.write(json.dumps(rec_w, ensure_ascii=False) + "\n")

            keep_count += 1
            by_candidate_type[str(best_cand.get("candidate_type", "unknown"))] += 1
            score_gain_vals.append(gain_vs_b0)
            margin_vals.append(margin)
            sample_weights.append(w)

    summary = {
        "atoms_b0_jsonl": os.path.abspath(args.atoms_b0_jsonl),
        "candidates_jsonl": os.path.abspath(args.candidates_jsonl),
        "output_dir": os.path.abspath(str(output_dir)),
        "num_docs_in_atoms": len(atoms_docs),
        "duplicate_doc_ids_in_atoms": duplicate_doc_ids_in_atoms,
        "num_docs_with_candidates": len(cand_groups),
        "num_candidates_total": total_candidates,
        "num_kept": keep_count,
        "num_rejected": reject_count,
        "keep_rate": (keep_count / max(1, len(atoms_docs))),
        "reject_reasons": dict(reason_counter),
        "selected_candidate_types": dict(by_candidate_type),
        "avg_gain_vs_b0": mean(score_gain_vals),
        "avg_margin": mean(margin_vals),
        "avg_sample_weight": mean(sample_weights),
        "scorer_config": {
            "w_edit": args.w_edit,
            "w_density": args.w_density,
            "w_jump": args.w_jump,
            "w_conf": args.w_conf,
            "w_anchor": args.w_anchor,
            "w_tiny_noise": args.w_tiny_noise,
            "w_oversize_hard": args.w_oversize_hard,
            "w_oversize_soft": args.w_oversize_soft,
            "min_gain_vs_b0": args.min_gain_vs_b0,
            "min_margin": args.min_margin,
            "min_conf": args.min_conf,
            "max_edit_rate": args.max_edit_rate,
            "min_density_ratio": args.min_density_ratio,
            "max_density_ratio": args.max_density_ratio,
            "semantic_mode": args.semantic_mode,
            "k_shift": args.k_shift,
            "max_chunk_tokens": args.max_chunk_tokens,
            "max_chunk_atoms": args.max_chunk_atoms,
            "max_chunk_chars": args.max_chunk_chars,
        },
        "outputs": {
            "pseudo_train_raw_jsonl": str(pseudo_raw_path),
            "pseudo_train_weighted_jsonl": str(pseudo_weighted_path),
            "rejected_jsonl": str(rejected_path),
            "best_candidates_jsonl": str(best_cands_path),
        },
    }

    dump_json(summary_path, summary)

    print(json.dumps({
        "num_kept": keep_count,
        "num_rejected": reject_count,
        "avg_gain_vs_b0": mean(score_gain_vals),
        "avg_margin": mean(margin_vals),
        "avg_sample_weight": mean(sample_weights),
        "selection_summary": str(summary_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()