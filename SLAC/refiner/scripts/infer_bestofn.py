#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slac_refiner.models.refiner import BoundaryRefinerModel
from slac_refiner.decoding.dp_edit_decode import batch_decode
from slac_refiner.decoding.projector import (
    ProjectorConfig,
    rebuild_chunks_from_boundary_vector,
    boundary_vector_to_spans,
    token_len_proxy,
)

# 与 train_loop.py 对齐
LOCAL_BGE_M3_DIR = r"/root/autodl-tmp/models/bge-m3/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

DEFAULT_PROJECTOR_CFG = ProjectorConfig(
    max_chunk_atoms=64,
    min_chunk_atoms=2,
    max_chunk_chars=1600,
    min_chunk_chars=20,
    max_chunk_tokens=384,
    min_chunk_tokens=48,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Week2 best-of-N refinement candidates from railway_atoms_b0.jsonl."
    )
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--doc_layers", type=int, default=1)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--k_shift", type=int, default=6)
    p.add_argument("--refine_passes", type=int, default=2)

    p.add_argument("--temperatures", type=float, nargs="*", default=[0.90, 1.00])
    p.add_argument("--insert_thresholds", type=float, nargs="*", default=[0.45, 0.50])
    p.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33])

    p.add_argument("--include_identity", action="store_true")
    p.add_argument("--include_greedy", action="store_true")

    p.add_argument("--max_chunk_tokens", type=int, default=384)
    p.add_argument("--min_chunk_tokens", type=int, default=48)
    p.add_argument("--max_chunk_atoms", type=int, default=64)
    p.add_argument("--min_chunk_atoms", type=int, default=2)
    p.add_argument("--max_chunk_chars", type=int, default=1600)
    p.add_argument("--min_chunk_chars", type=int, default=20)

    p.add_argument("--insert_min_sep", type=int, default=1)
    p.add_argument("--lambda_del", type=float, default=1.0)
    p.add_argument("--lambda_ins", type=float, default=1.0)   # 保留接口，当前 sample decode 不显式用
    p.add_argument("--lambda_shift", type=float, default=0.25)

    p.add_argument("--max_docs", type=int, default=None)

    p.add_argument("--max_atoms_full_decode", type=int, default=2200)
    p.add_argument("--max_atoms_greedy_only", type=int, default=3200)
    p.add_argument("--disable_autocast", action="store_true")
    return p.parse_args()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify_doc_id(doc_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_id.strip())
    safe = safe.strip("._")
    if safe:
        return safe[:160]
    h = hashlib.md5(doc_id.encode("utf-8")).hexdigest()[:16]
    return f"doc_{h}"


def load_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
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


def parse_span(span_obj: Any) -> Tuple[int, int]:
    if isinstance(span_obj, (list, tuple)) and len(span_obj) == 2:
        return int(span_obj[0]), int(span_obj[1])

    if isinstance(span_obj, dict):
        candidate_keys = [
            ("start_atom", "end_atom"),
            ("s", "e"),
            ("start", "end"),
            ("atom_start", "atom_end"),
            ("start_idx", "end_idx"),
        ]
        for ks, ke in candidate_keys:
            if ks in span_obj and ke in span_obj:
                return int(span_obj[ks]), int(span_obj[ke])

    raise KeyError(f"Unsupported span format: {span_obj}")


def normalize_atoms(raw_atoms: Sequence[Any]) -> List[str]:
    atoms: List[str] = []
    for a in raw_atoms:
        if isinstance(a, str):
            atoms.append(a)
        elif isinstance(a, dict):
            atoms.append(str(a.get("text", "")))
        else:
            atoms.append(str(a))
    return atoms


def dense_b0_to_sparse(b0: Sequence[Any]) -> List[int]:
    return [i for i, v in enumerate(b0) if int(v) == 1]


def sparse_to_dense(boundaries_sparse: Sequence[int], num_atoms: int) -> List[int]:
    dense = [0] * max(0, num_atoms - 1)
    for g in boundaries_sparse:
        g = int(g)
        if 0 <= g < len(dense):
            dense[g] = 1
    return dense


def validate_doc_schema(doc: Dict[str, Any]) -> None:
    required = ["doc_id", "chunk0_units", "atoms", "unit2atom_span", "b0"]
    for k in required:
        if k not in doc:
            raise KeyError(f"Missing required field '{k}' in doc {doc.get('doc_id', '<unknown>')}")

    atoms = normalize_atoms(doc["atoms"])
    b0 = doc["b0"]
    spans = doc["unit2atom_span"]

    if len(b0) != max(0, len(atoms) - 1):
        raise ValueError(
            f"Invalid b0 length for doc={doc['doc_id']}: len(b0)={len(b0)}, len(atoms)={len(atoms)}"
        )

    last_e = 0
    for i, sp in enumerate(spans):
        s, e = parse_span(sp)
        if i == 0 and s != 0:
            raise ValueError(f"Span coverage broken at first span for doc={doc['doc_id']}: s={s}")
        if s != last_e:
            raise ValueError(f"Span coverage broken for doc={doc['doc_id']}: expected s={last_e}, got {s}")
        if e < s:
            raise ValueError(f"Span monotonicity broken for doc={doc['doc_id']}: s={s}, e={e}")
        last_e = e
    if spans and last_e != len(atoms):
        raise ValueError(
            f"Final span coverage broken for doc={doc['doc_id']}: last_e={last_e}, num_atoms={len(atoms)}"
        )


def clamp_prob(p: float) -> float:
    return max(1e-8, min(1.0 - 1e-8, float(p)))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax_from_logits(row: torch.Tensor, temperature: float = 1.0) -> List[float]:
    row = row.detach().cpu().float() / max(1e-6, float(temperature))
    probs = torch.softmax(row, dim=-1).tolist()
    return [clamp_prob(p) for p in probs]


def pctl90(xs: Sequence[int]) -> int:
    if not xs:
        return 0
    ys = sorted(int(x) for x in xs)
    return ys[int(0.9 * (len(ys) - 1))]


def compute_chunk_stats(atoms: Sequence[str], b_dense: Sequence[int]) -> Dict[str, Any]:
    spans = boundary_vector_to_spans(len(atoms), b_dense)
    chunk_atom_lens: List[int] = []
    chunk_tok_lens: List[int] = []
    chunk_char_lens: List[int] = []

    for s, e in spans:
        text = "\n".join(atoms[s:e]).strip()
        chunk_atom_lens.append(e - s)
        chunk_tok_lens.append(token_len_proxy(text))
        chunk_char_lens.append(len(text))

    def avg(xs: Sequence[int]) -> float:
        return sum(xs) / max(1, len(xs))

    hard_viol = 0
    for a_len, t_len, c_len in zip(chunk_atom_lens, chunk_tok_lens, chunk_char_lens):
        if (
            a_len > DEFAULT_PROJECTOR_CFG.max_chunk_atoms
            or a_len < DEFAULT_PROJECTOR_CFG.min_chunk_atoms
            or t_len > DEFAULT_PROJECTOR_CFG.max_chunk_tokens
            or t_len < DEFAULT_PROJECTOR_CFG.min_chunk_tokens
            or c_len > DEFAULT_PROJECTOR_CFG.max_chunk_chars
            or c_len < DEFAULT_PROJECTOR_CFG.min_chunk_chars
        ):
            hard_viol += 1

    return {
        "num_chunks": len(spans),
        "avg_chunk_atoms": avg(chunk_atom_lens),
        "p90_chunk_atoms": pctl90(chunk_atom_lens),
        "max_chunk_atoms": max(chunk_atom_lens) if chunk_atom_lens else 0,
        "min_chunk_atoms": min(chunk_atom_lens) if chunk_atom_lens else 0,
        "avg_chunk_tokens": avg(chunk_tok_lens),
        "p90_chunk_tokens": pctl90(chunk_tok_lens),
        "max_chunk_tokens": max(chunk_tok_lens) if chunk_tok_lens else 0,
        "min_chunk_tokens": min(chunk_tok_lens) if chunk_tok_lens else 0,
        "avg_chunk_chars": avg(chunk_char_lens),
        "p90_chunk_chars": pctl90(chunk_char_lens),
        "max_chunk_chars": max(chunk_char_lens) if chunk_char_lens else 0,
        "min_chunk_chars": min(chunk_char_lens) if chunk_char_lens else 0,
        "num_hard_violations": hard_viol,
    }


def build_model(args: argparse.Namespace) -> BoundaryRefinerModel:
    model = BoundaryRefinerModel(
        atom_model_name=LOCAL_BGE_M3_DIR,
        atom_max_length=64,
        atom_freeze=True,
        doc_hidden_size=768,
        doc_layers=args.doc_layers,
        doc_heads=12,
        doc_dropout=0.1,
        window_size=args.window_size,
    )
    return model


def load_model_from_ckpt(args: argparse.Namespace) -> BoundaryRefinerModel:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model = build_model(args)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing[:20]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected[:20]}")

    model.eval()
    return model


def make_g0_positions_from_b_dense(b_dense: Sequence[int]) -> List[int]:
    return [i for i, x in enumerate(b_dense) if int(x) == 1]


def build_single_doc_batch(atoms: Sequence[str], b_dense: Sequence[int], device: torch.device) -> Tuple[Dict[str, Any], List[int]]:
    g0_positions = make_g0_positions_from_b_dense(b_dense)
    if len(g0_positions) == 0:
        g0_tensor = torch.full((1, 1), -1, dtype=torch.long, device=device)
    else:
        g0_tensor = torch.tensor([g0_positions], dtype=torch.long, device=device)

    batch = {
        "atoms_text": [list(atoms)],
        "g0_positions": g0_tensor,
    }
    return batch, g0_positions


@torch.no_grad()
def forward_one_doc(
    model: BoundaryRefinerModel,
    atoms: Sequence[str],
    b_dense: Sequence[int],
    use_autocast: bool = True,
) -> Dict[str, Any]:
    batch, g0_positions = build_single_doc_batch(atoms, b_dense, model.device)

    with torch.inference_mode():
        if use_autocast and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(batch)
        else:
            outputs = model(batch)

    num_gaps = max(0, len(atoms) - 1)
    valid_b0 = len(g0_positions)

    insert_logits = outputs.insert_logits[0, :num_gaps].detach().cpu()
    edit_choice_logits = outputs.edit_choice_logits[0, :valid_b0].detach().cpu()

    # 尽快释放 GPU 上的 forward outputs 引用
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "g0_positions": g0_positions,
        "insert_logits": insert_logits,
        "edit_choice_logits": edit_choice_logits,
    }


def label_to_class_idx(label: str, K: int) -> int:
    if label == "DEL":
        return 0
    if label == "KEEP":
        return 1 + K
    if label.startswith("SHIFT:"):
        k = int(label.split(":", 1)[1])
        return 1 + K + k
    raise ValueError(f"Unknown edit label: {label}")


def class_idx_to_label(cls_idx: int, K: int) -> str:
    if cls_idx == 0:
        return "DEL"
    k = cls_idx - 1 - K
    return "KEEP" if k == 0 else f"SHIFT:{k}"


def build_edit_trace_from_labels(
    pred_edit_labels: Sequence[Dict[str, Any]],
    g0_positions: Sequence[int],
    edit_choice_logits: torch.Tensor,
    K: int,
) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []
    g_to_row = {int(g): idx for idx, g in enumerate(g0_positions)}

    for item in pred_edit_labels:
        g = int(item["g"])
        label = str(item["y"])
        row_idx = g_to_row[g]
        probs = softmax_from_logits(edit_choice_logits[row_idx], temperature=1.0)
        cls_idx = label_to_class_idx(label, K)
        prob = probs[cls_idx]

        if label == "DEL":
            trace.append(
                {
                    "g0": g,
                    "action": "DEL",
                    "shift": None,
                    "to": None,
                    "prob": prob,
                }
            )
        elif label == "KEEP":
            trace.append(
                {
                    "g0": g,
                    "action": "KEEP",
                    "shift": 0,
                    "to": g,
                    "prob": prob,
                }
            )
        else:
            k = int(label.split(":", 1)[1])
            trace.append(
                {
                    "g0": g,
                    "action": "SHIFT",
                    "shift": k,
                    "to": g + k,
                    "prob": prob,
                }
            )
    return trace


def build_insert_gaps_from_binary(
    pred_insert_labels: Sequence[int],
    insert_logits: torch.Tensor,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    probs = torch.sigmoid(insert_logits).detach().cpu().tolist()
    for g, y in enumerate(pred_insert_labels):
        if int(y) == 1:
            out.append({"gap": int(g), "prob": clamp_prob(float(probs[g]))})
    return out


def compute_teacher_stats(edit_trace: Sequence[Dict[str, Any]], insert_gaps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    edit_probs = [float(tr.get("prob", 0.0)) for tr in edit_trace]
    insert_probs = [float(x.get("prob", 0.0)) for x in insert_gaps]

    num_deleted = sum(1 for tr in edit_trace if tr.get("action") == "DEL")
    num_shifted = sum(1 for tr in edit_trace if tr.get("action") == "SHIFT")
    num_inserted = len(insert_gaps)
    sum_abs_shift = sum(abs(int(tr.get("shift", 0) or 0)) for tr in edit_trace if tr.get("action") == "SHIFT")

    return {
        "mean_edit_prob": (sum(edit_probs) / max(1, len(edit_probs))) if edit_probs else None,
        "mean_insert_prob": (sum(insert_probs) / max(1, len(insert_probs))) if insert_probs else None,
        "num_deleted": num_deleted,
        "num_inserted": num_inserted,
        "num_shifted": num_shifted,
        "sum_abs_shift": sum_abs_shift,
    }


def build_identity_candidate(doc: Dict[str, Any], ckpt_path: str) -> Dict[str, Any]:
    atoms = normalize_atoms(doc["atoms"])
    b0_sparse = dense_b0_to_sparse(doc["b0"])

    return {
        "doc_id": doc["doc_id"],
        "candidate_id": "identity",
        "candidate_type": "identity",
        "teacher_ckpt": ckpt_path,
        "decode_cfg": {
            "temperature": None,
            "insert_threshold": None,
            "seed": None,
            "k_shift": None,
            "refine_passes": 0,
        },
        "input_stats": {
            "num_atoms": len(atoms),
            "num_units": len(doc["chunk0_units"]),
            "num_seed_boundaries": len(b0_sparse),
        },
        "prediction": {
            "b0_sparse": b0_sparse,
            "b_pred_sparse": b0_sparse,
            "edit_trace": [],
            "insert_gaps": [],
        },
        "teacher_stats": {
            "mean_edit_prob": None,
            "mean_insert_prob": None,
            "num_deleted": 0,
            "num_inserted": 0,
            "num_shifted": 0,
            "sum_abs_shift": 0,
        },
        "chunk_stats": compute_chunk_stats(atoms, doc["b0"]),
        "score_terms": {},
        "flags": {},
    }


def gumbel_noise(rng: random.Random) -> float:
    u = max(1e-9, min(1.0 - 1e-9, rng.random()))
    return -math.log(-math.log(u))


def build_candidates_for_boundary_row(
    g0: int,
    num_gaps: int,
    edit_choice_logit_row: torch.Tensor,
    K: int,
    temperature: float,
    lambda_del: float,
    lambda_shift: float,
) -> List[Dict[str, Any]]:
    probs = softmax_from_logits(edit_choice_logit_row, temperature=temperature)
    candidates: List[Dict[str, Any]] = []

    # class 0 -> DEL
    candidates.append(
        {
            "kind": "DEL",
            "pos": None,
            "score": float(math.log(probs[0]) - lambda_del),
            "label": "DEL",
            "prob": float(probs[0]),
        }
    )

    # classes 1..2K+1 -> k in [-K..K]
    for cls_idx in range(1, 2 * K + 2):
        k = cls_idx - 1 - K
        pos = g0 + k
        if not (0 <= pos < num_gaps):
            continue

        score = float(math.log(probs[cls_idx]) - lambda_shift * abs(k))
        label = "KEEP" if k == 0 else f"SHIFT:{k}"
        kind = "KEEP" if k == 0 else "SHIFT"

        candidates.append(
            {
                "kind": kind,
                "pos": pos,
                "score": score,
                "label": label,
                "prob": float(probs[cls_idx]),
                "shift": k,
            }
        )

    return candidates


def sample_monotonic_edit_decode(
    b_cur_dense: Sequence[int],
    g0_positions: Sequence[int],
    edit_choice_logits: torch.Tensor,
    K: int,
    temperature: float,
    seed: int,
    lambda_del: float,
    lambda_shift: float,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    num_gaps = len(b_cur_dense)
    n = len(g0_positions)
    if n == 0:
        return [], []

    rng = random.Random(seed)

    all_candidates: List[List[Dict[str, Any]]] = []
    for j, g in enumerate(g0_positions):
        cand_j = build_candidates_for_boundary_row(
            g0=g,
            num_gaps=num_gaps,
            edit_choice_logit_row=edit_choice_logits[j],
            K=K,
            temperature=temperature,
            lambda_del=lambda_del,
            lambda_shift=lambda_shift,
        )
        all_candidates.append(cand_j)

    NEG_INF = -1e18
    dp: List[List[float]] = []
    back: List[List[Optional[int]]] = []

    first = all_candidates[0]
    dp.append([c["score"] + gumbel_noise(rng) for c in first])
    back.append([None] * len(first))

    for j in range(1, n):
        prev = all_candidates[j - 1]
        cur = all_candidates[j]

        dp_j = [NEG_INF] * len(cur)
        back_j: List[Optional[int]] = [None] * len(cur)

        for c_idx, c in enumerate(cur):
            c_pos = c["pos"]
            c_score = c["score"] + gumbel_noise(rng)

            for p_idx, p in enumerate(prev):
                p_pos = p["pos"]

                ok = False
                if p_pos is None:
                    ok = True
                elif c_pos is None:
                    ok = True
                else:
                    ok = p_pos < c_pos

                if not ok:
                    continue

                cand_score = dp[j - 1][p_idx] + c_score
                if cand_score > dp_j[c_idx]:
                    dp_j[c_idx] = cand_score
                    back_j[c_idx] = p_idx

        dp.append(dp_j)
        back.append(back_j)

    last_idx = max(range(len(dp[-1])), key=lambda i: dp[-1][i])
    chosen: List[Tuple[int, Dict[str, Any]]] = []

    cur_idx = last_idx
    for j in range(n - 1, -1, -1):
        chosen.append((j, all_candidates[j][cur_idx]))
        prev_idx = back[j][cur_idx]
        if prev_idx is None:
            break
        cur_idx = prev_idx

    chosen.reverse()

    pred_boundary_positions: List[int] = []
    pred_edit_labels: List[Dict[str, Any]] = []

    for j, cand in chosen:
        g = g0_positions[j]
        pred_edit_labels.append({"g": g, "y": cand["label"]})
        if cand["pos"] is not None:
            pred_boundary_positions.append(int(cand["pos"]))

    pred_boundary_positions = sorted(set(pred_boundary_positions))
    edit_trace: List[Dict[str, Any]] = []
    for item, (_, cand) in zip(pred_edit_labels, chosen):
        g = int(item["g"])
        label = item["y"]
        if label == "DEL":
            edit_trace.append({"g0": g, "action": "DEL", "shift": None, "to": None, "prob": cand["prob"]})
        elif label == "KEEP":
            edit_trace.append({"g0": g, "action": "KEEP", "shift": 0, "to": g, "prob": cand["prob"]})
        else:
            k = int(label.split(":", 1)[1])
            edit_trace.append({"g0": g, "action": "SHIFT", "shift": k, "to": g + k, "prob": cand["prob"]})
    return pred_boundary_positions, edit_trace


def sample_insert_decode(
    insert_logits: torch.Tensor,
    edit_boundary_positions: Sequence[int],
    temperature: float,
    insert_threshold: float,
    min_sep: int,
    seed: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    rng = random.Random(seed + 7919)
    probs: List[float] = []
    for logit in insert_logits.detach().cpu().tolist():
        adj_p = sigmoid(float(logit) / max(1e-6, temperature))
        probs.append(clamp_prob(adj_p))

    pred_insert_labels = [0] * len(probs)
    accepted: List[Tuple[int, float]] = []

    for g, p in enumerate(probs):
        if p < insert_threshold:
            continue
        too_close = any(abs(g - b) <= min_sep for b in edit_boundary_positions)
        if too_close:
            continue
        if rng.random() < p:
            accepted.append((g, p))

    # neighborhood suppression
    accepted = sorted(accepted, key=lambda x: (x[0], -x[1]))
    kept: List[Tuple[int, float]] = []
    for g, p in accepted:
        if not kept:
            kept.append((g, p))
            continue
        prev_g, prev_p = kept[-1]
        if g - prev_g <= min_sep:
            if p > prev_p:
                kept[-1] = (g, p)
        else:
            kept.append((g, p))

    insert_gaps: List[Dict[str, Any]] = []
    for g, p in kept:
        pred_insert_labels[g] = 1
        insert_gaps.append({"gap": int(g), "prob": float(p)})

    return pred_insert_labels, insert_gaps


def run_greedy_candidate(
    model: BoundaryRefinerModel,
    doc: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    atoms = normalize_atoms(doc["atoms"])
    b_original = [int(x) for x in doc["b0"]]
    b_cur = list(b_original)

    final_edit_trace: List[Dict[str, Any]] = []
    final_insert_gaps: List[Dict[str, Any]] = []

    for _ in range(max(1, int(args.refine_passes))):
        raw = forward_one_doc(
            model=model,
            atoms=atoms,
            b_dense=b_cur,
            use_autocast=(not args.disable_autocast),
        )
        g0_positions = raw["g0_positions"]
        edit_choice_logits = raw["edit_choice_logits"]
        insert_logits = raw["insert_logits"]

        b0_tensor = torch.tensor([b_cur], dtype=torch.long)
        if len(g0_positions) == 0:
            g0_tensor = torch.full((1, 1), -1, dtype=torch.long)
        else:
            g0_tensor = torch.tensor([g0_positions], dtype=torch.long)

        dec = batch_decode(
            b0=b0_tensor,
            g0_positions=g0_tensor,
            edit_choice_logits=edit_choice_logits.unsqueeze(0),
            insert_logits=insert_logits.unsqueeze(0),
            K=args.k_shift,
            insert_threshold=0.50,
            min_sep=args.insert_min_sep,
            lambda_del=args.lambda_del,
            lambda_ins=args.lambda_ins,
            lambda_shift=args.lambda_shift,
        )

        raw_pred_b = dec.pred_b[0]
        projected = rebuild_chunks_from_boundary_vector(
            atoms_text=atoms,
            b=raw_pred_b,
            cfg=DEFAULT_PROJECTOR_CFG,
            gap_scores=insert_logits.tolist(),
        )
        b_cur = projected["projected_b"]

        final_edit_trace = build_edit_trace_from_labels(
            pred_edit_labels=dec.pred_edit_labels[0],
            g0_positions=g0_positions,
            edit_choice_logits=edit_choice_logits,
            K=args.k_shift,
        )
        final_insert_gaps = build_insert_gaps_from_binary(
            pred_insert_labels=dec.pred_insert_labels[0],
            insert_logits=insert_logits,
        )

    b0_sparse = dense_b0_to_sparse(b_original)
    b_pred_sparse = dense_b0_to_sparse(b_cur)

    return {
        "doc_id": doc["doc_id"],
        "candidate_id": "greedy",
        "candidate_type": "greedy",
        "teacher_ckpt": args.ckpt,
        "decode_cfg": {
            "temperature": 1.0,
            "insert_threshold": 0.50,
            "seed": 0,
            "k_shift": args.k_shift,
            "refine_passes": args.refine_passes,
        },
        "input_stats": {
            "num_atoms": len(atoms),
            "num_units": len(doc["chunk0_units"]),
            "num_seed_boundaries": len(b0_sparse),
        },
        "prediction": {
            "b0_sparse": b0_sparse,
            "b_pred_sparse": b_pred_sparse,
            "edit_trace": final_edit_trace,
            "insert_gaps": final_insert_gaps,
        },
        "teacher_stats": compute_teacher_stats(final_edit_trace, final_insert_gaps),
        "chunk_stats": compute_chunk_stats(atoms, b_cur),
        "score_terms": {},
        "flags": {},
    }


def run_sample_candidate(
    model: BoundaryRefinerModel,
    doc: Dict[str, Any],
    args: argparse.Namespace,
    temperature: float,
    insert_threshold: float,
    seed: int,
) -> Dict[str, Any]:
    atoms = normalize_atoms(doc["atoms"])
    b_original = [int(x) for x in doc["b0"]]
    b_cur = list(b_original)

    final_edit_trace: List[Dict[str, Any]] = []
    final_insert_gaps: List[Dict[str, Any]] = []

    doc_seed = (int(hashlib.md5(str(doc["doc_id"]).encode("utf-8")).hexdigest()[:8], 16) + int(seed)) % (2**31)

    for pass_idx in range(max(1, int(args.refine_passes))):
        raw = forward_one_doc(
            model=model,
            atoms=atoms,
            b_dense=b_cur,
            use_autocast=(not args.disable_autocast),
        )
        g0_positions = raw["g0_positions"]
        edit_choice_logits = raw["edit_choice_logits"]
        insert_logits = raw["insert_logits"]

        edit_positions, edit_trace = sample_monotonic_edit_decode(
            b_cur_dense=b_cur,
            g0_positions=g0_positions,
            edit_choice_logits=edit_choice_logits,
            K=args.k_shift,
            temperature=temperature,
            seed=doc_seed + pass_idx * 37,
            lambda_del=args.lambda_del,
            lambda_shift=args.lambda_shift,
        )

        pred_insert_labels, insert_gaps = sample_insert_decode(
            insert_logits=insert_logits,
            edit_boundary_positions=edit_positions,
            temperature=temperature,
            insert_threshold=insert_threshold,
            min_sep=args.insert_min_sep,
            seed=doc_seed + pass_idx * 101,
        )

        merged_sparse = sorted(set(edit_positions) | set(int(x["gap"]) for x in insert_gaps))
        merged_dense = sparse_to_dense(merged_sparse, len(atoms))

        projected = rebuild_chunks_from_boundary_vector(
            atoms_text=atoms,
            b=merged_dense,
            cfg=DEFAULT_PROJECTOR_CFG,
            gap_scores=insert_logits.tolist(),
        )
        b_cur = projected["projected_b"]

        final_edit_trace = edit_trace
        final_insert_gaps = insert_gaps

    b0_sparse = dense_b0_to_sparse(b_original)
    b_pred_sparse = dense_b0_to_sparse(b_cur)

    return {
        "doc_id": doc["doc_id"],
        "candidate_id": f"sample_T{temperature:.2f}_tau{insert_threshold:.2f}_seed{seed}",
        "candidate_type": "sample",
        "teacher_ckpt": args.ckpt,
        "decode_cfg": {
            "temperature": float(temperature),
            "insert_threshold": float(insert_threshold),
            "seed": int(seed),
            "k_shift": args.k_shift,
            "refine_passes": args.refine_passes,
        },
        "input_stats": {
            "num_atoms": len(atoms),
            "num_units": len(doc["chunk0_units"]),
            "num_seed_boundaries": len(b0_sparse),
        },
        "prediction": {
            "b0_sparse": b0_sparse,
            "b_pred_sparse": b_pred_sparse,
            "edit_trace": final_edit_trace,
            "insert_gaps": final_insert_gaps,
        },
        "teacher_stats": compute_teacher_stats(final_edit_trace, final_insert_gaps),
        "chunk_stats": compute_chunk_stats(atoms, b_cur),
        "score_terms": {},
        "flags": {},
    }


def main() -> None:
    args = parse_args()

    global DEFAULT_PROJECTOR_CFG
    DEFAULT_PROJECTOR_CFG = ProjectorConfig(
        max_chunk_atoms=args.max_chunk_atoms,
        min_chunk_atoms=args.min_chunk_atoms,
        max_chunk_chars=args.max_chunk_chars,
        min_chunk_chars=args.min_chunk_chars,
        max_chunk_tokens=args.max_chunk_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
    )

    output_dir = ensure_dir(args.output_dir)
    docwise_dir = ensure_dir(output_dir / "docwise")

    docs = list(load_jsonl(args.input_jsonl))
    if args.max_docs is not None:
        docs = docs[: int(args.max_docs)]

    for doc in docs:
        validate_doc_schema(doc)

    model = load_model_from_ckpt(args)

    all_candidates_path = output_dir / "all_candidates.jsonl"
    num_docs = 0
    num_candidates = 0

    with all_candidates_path.open("w", encoding="utf-8") as fout:
        for doc in docs:
            doc_id = str(doc["doc_id"])
            atoms = normalize_atoms(doc["atoms"])
            doc_candidates: List[Dict[str, Any]] = []

            if args.include_identity:
                doc_candidates.append(build_identity_candidate(doc, args.ckpt))

            if args.include_greedy:
                doc_candidates.append(run_greedy_candidate(model, doc, args))

            for T in args.temperatures:
                for tau in args.insert_thresholds:
                    for seed in args.seeds:
                        doc_candidates.append(
                            run_sample_candidate(
                                model=model,
                                doc=doc,
                                args=args,
                                temperature=float(T),
                                insert_threshold=float(tau),
                                seed=int(seed),
                            )
                        )

            for cand in doc_candidates:
                fout.write(json.dumps(cand, ensure_ascii=False) + "\n")
                num_candidates += 1

            docwise_obj = {
                "doc_id": doc_id,
                "num_atoms": len(atoms),
                "num_units": len(doc["chunk0_units"]),
                "num_candidates": len(doc_candidates),
                "candidates": doc_candidates,
            }
            dump_json(docwise_dir / f"{slugify_doc_id(doc_id)}.json", docwise_obj)

            num_docs += 1
            if num_docs % 10 == 0 or num_docs == len(docs):
                print(f"[infer_bestofn] processed {num_docs}/{len(docs)} docs, total_candidates={num_candidates}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    score_summary = {
        "num_docs": num_docs,
        "num_candidates": num_candidates,
        "include_identity": bool(args.include_identity),
        "include_greedy": bool(args.include_greedy),
        "temperatures": list(args.temperatures),
        "insert_thresholds": list(args.insert_thresholds),
        "seeds": list(args.seeds),
        "k_shift": args.k_shift,
        "refine_passes": args.refine_passes,
        "projector_cfg": {
            "max_chunk_atoms": args.max_chunk_atoms,
            "min_chunk_atoms": args.min_chunk_atoms,
            "max_chunk_chars": args.max_chunk_chars,
            "min_chunk_chars": args.min_chunk_chars,
            "max_chunk_tokens": args.max_chunk_tokens,
            "min_chunk_tokens": args.min_chunk_tokens,
        },
        "ckpt": os.path.abspath(args.ckpt),
        "input_jsonl": os.path.abspath(args.input_jsonl),
        "output_dir": os.path.abspath(str(output_dir)),
    }
    dump_json(output_dir / "score_summary.json", score_summary)

    print(
        json.dumps(
            {
                "num_docs": num_docs,
                "num_candidates": num_candidates,
                "all_candidates_jsonl": str(all_candidates_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()