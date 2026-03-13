#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 允许从项目根目录直接运行
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slac_refiner.atomize.normalize import normalize_text
from slac_refiner.atomize.splitter import split_text_to_atoms
from slac_refiner.atomize.mapping import build_atoms_b0_from_units


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Refiner train/dev dataset directly from validated LLM tree JSONs."
    )
    p.add_argument("--validated_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--glob", type=str, default="*.json")
    p.add_argument("--dev_ratio", type=float, default=0.12)
    p.add_argument("--k_shift", type=int, default=6)

    # 树 -> 线性 gold 单元策略
    p.add_argument("--keep_nonleaf_anchor_max_chars", type=int, default=120)

    # 每篇生成多少个不同噪声版本
    p.add_argument("--variants_per_doc", type=int, default=1)

    # Week2: 轻噪声，更适合高质量 gold 上的 domain adaptation
    p.add_argument("--p_shift", type=float, default=0.30)
    p.add_argument("--p_insert", type=float, default=0.08)
    p.add_argument("--p_delete", type=float, default=0.20)

    p.add_argument("--max_shift_frac", type=float, default=0.30)
    p.add_argument("--max_insert_frac", type=float, default=0.08)
    p.add_argument("--max_delete_frac", type=float, default=0.20)

    p.add_argument("--burst_delete_prob", type=float, default=0.10)
    p.add_argument("--burst_delete_min", type=int, default=2)
    p.add_argument("--burst_delete_max", type=int, default=2)

    # 最小限度保留一些 seed 结构，避免过于极端
    p.add_argument("--min_seed_boundaries_keep", type=int, default=1)

    return p.parse_args()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path: str | Path, obj: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stable_dev_split(doc_id: str, dev_ratio: float) -> bool:
    h = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return v < dev_ratio


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


def looks_structural_anchor(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False

    low = text.lower()
    short = len(text) <= 80

    patterns = [
        r"^\d+(\.\d+){0,5}[\.\)]?\s*$",
        r"^\d+(\.\d+){0,5}[\.\)]?\s+\S+.*$",
        r"^第[一二三四五六七八九十百千万0-9]+[章节条部分]\s*.*$",
        r"^(附录|附件)\s*[A-Z0-9一二三四五六七八九十]?\s*.*$",
        r"^(annex|appendix|table|figure|note|scope|definitions?|requirements?)\b.*$",
        r"^[-•·]\s+\S+.*$",
        r"^[A-Z]\.\d+.*$",
        r"^[（(]?[0-9一二三四五六七八九十]+[)）]\s*.*$",
    ]

    if short:
        for pat in patterns:
            if re.match(pat, low, flags=re.IGNORECASE):
                return True

    if short and text.upper() == text and len(text.split()) <= 8:
        return True

    return False


def load_validated_jsons(validated_dir: Path, pattern: str) -> List[Tuple[Path, Dict[str, Any]]]:
    items = []
    for p in sorted(validated_dir.rglob(pattern)):
        if not p.is_file():
            continue
        if p.name.endswith(".meta.json"):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append((p, obj))
    return items


def validate_min_tree(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    if not isinstance(obj, dict):
        return False, ["top-level is not object"]

    for k in ["doc_id", "doc_name", "language", "units"]:
        if k not in obj:
            errors.append(f"missing top-level key: {k}")
    if errors:
        return False, errors

    units = obj.get("units")
    if not isinstance(units, list) or not units:
        return False, ["units missing or empty"]

    seen = set()
    roots = 0
    by_id = {}

    for i, u in enumerate(units):
        if not isinstance(u, dict):
            errors.append(f"units[{i}] not object")
            continue
        for k in ["unit_id", "text", "type", "level", "parent_id"]:
            if k not in u:
                errors.append(f"units[{i}] missing {k}")
        if errors:
            continue

        try:
            uid = int(u["unit_id"])
            lvl = int(u["level"])
        except Exception:
            errors.append(f"units[{i}] bad unit_id/level")
            continue

        if uid in seen:
            errors.append(f"duplicate unit_id={uid}")
        seen.add(uid)
        by_id[uid] = u

        if u["parent_id"] is None:
            roots += 1

        if not str(u["text"] or "").strip():
            errors.append(f"unit {uid} empty text")

    if roots != 1:
        errors.append(f"root count must be 1, got {roots}")

    expected = list(range(len(units)))
    actual = sorted(seen)
    if actual != expected:
        errors.append("unit_id not contiguous from 0")

    for uid, u in by_id.items():
        if uid == 0:
            continue
        pid = u["parent_id"]
        if pid is None:
            errors.append(f"non-root {uid} has null parent")
            continue
        try:
            pid = int(pid)
        except Exception:
            errors.append(f"bad parent_id for {uid}")
            continue
        if pid not in by_id:
            errors.append(f"parent {pid} missing for {uid}")
            continue
        try:
            cl = int(u["level"])
            pl = int(by_id[pid]["level"])
            if cl != pl + 1:
                errors.append(f"bad level chain {uid}: child={cl}, parent={pl}")
        except Exception:
            errors.append(f"bad level parse for {uid}")

    return len(errors) == 0, errors


def build_children_map(units: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    ch = defaultdict(list)
    for u in units:
        pid = u.get("parent_id")
        uid = u.get("unit_id")
        if pid is not None:
            ch[int(pid)].append(int(uid))
    return ch


def extract_linear_gold_units(
    units: List[Dict[str, Any]],
    keep_nonleaf_anchor_max_chars: int,
) -> List[Dict[str, Any]]:
    """
    从 validated 树中提取线性 gold chunks：
    - root 不保留
    - 所有叶子节点保留
    - 短的非叶锚点（标题/编号/附录/列表锚点）保留
    """
    children = build_children_map(units)
    kept = []

    for u in sorted(units, key=lambda x: int(x["unit_id"])):
        uid = int(u["unit_id"])
        if uid == 0:
            continue

        text = str(u.get("text") or "").strip()
        if not text:
            continue

        has_children = uid in children and len(children[uid]) > 0

        if not has_children:
            kept.append({
                "unit_id": len(kept),
                "text": text,
                "type": str(u.get("type") or "other"),
                "level": int(u.get("level") or 0),
                "parent_id": None,
            })
            continue

        if len(text) <= keep_nonleaf_anchor_max_chars and looks_structural_anchor(text):
            kept.append({
                "unit_id": len(kept),
                "text": text,
                "type": str(u.get("type") or "other"),
                "level": int(u.get("level") or 0),
                "parent_id": None,
            })

    # 去掉连续重复
    dedup = []
    prev_norm = None
    for u in kept:
        cur = re.sub(r"\s+", "", u["text"]).lower()
        if cur and cur == prev_norm:
            continue
        dedup.append({**u, "unit_id": len(dedup)})
        prev_norm = cur

    return dedup


def reconstruct_chunk0_from_seed(
    atoms: Sequence[str],
    seed_boundaries_sparse: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    spans = boundaries_to_chunk_spans(seed_boundaries_sparse, len(atoms))
    chunk0_units: List[Dict[str, Any]] = []
    unit2atom_span: List[Dict[str, Any]] = []

    for idx, (s, e) in enumerate(spans):
        text = "\n".join(atoms[s:e]).strip()
        chunk0_units.append({
            "unit_id": idx,
            "text": text,
            "type": "paragraph",
            "level": 1,
            "parent_id": 0 if idx > 0 else None,
        })
        unit2atom_span.append({
            "unit_id": idx,
            "start_atom": s,
            "end_atom": e,
        })
    return chunk0_units, unit2atom_span


def generate_noisy_seed_from_gold(
    gold_boundaries_sparse: Sequence[int],
    num_atoms: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> List[int]:
    gold = sorted(set(int(x) for x in gold_boundaries_sparse if 0 <= int(x) < num_atoms - 1))
    num_gaps = max(0, num_atoms - 1)

    if num_gaps == 0:
        return []

    cur = list(gold)

    # 1) shift 一部分 gold boundaries
    max_shift_n = int(round(len(gold) * args.max_shift_frac))
    shift_candidates = [g for g in gold if rng.random() < args.p_shift]
    rng.shuffle(shift_candidates)
    shift_candidates = shift_candidates[:max_shift_n]

    shifted = set(cur)
    for g in shift_candidates:
        shifted.discard(g)
        ks = [k for k in range(-args.k_shift, args.k_shift + 1) if k != 0]
        if not ks:
            shifted.add(g)
            continue
        k = rng.choice(ks)
        ng = max(0, min(num_gaps - 1, g + k))
        shifted.add(ng)
    cur = sorted(shifted)

    # 2) delete 一部分
    max_del_n = int(round(len(gold) * args.max_delete_frac))
    del_candidates = [g for g in cur if rng.random() < args.p_delete]
    rng.shuffle(del_candidates)
    del_candidates = del_candidates[:max_del_n]
    cur = [g for g in cur if g not in set(del_candidates)]

    # 3) burst delete（轻量）
    if len(cur) >= args.burst_delete_min and rng.random() < args.burst_delete_prob:
        burst_len = rng.randint(args.burst_delete_min, args.burst_delete_max)
        burst_len = min(burst_len, len(cur))
        if burst_len > 0:
            st = rng.randint(0, len(cur) - burst_len)
            burst_set = set(cur[st:st + burst_len])
            cur = [g for g in cur if g not in burst_set]

    # 4) insert 一些额外边界
    max_ins_n = int(round(max(1, len(gold)) * args.max_insert_frac))
    max_ins_n = max(1, max_ins_n) if num_gaps > 1 else 0
    available = [g for g in range(num_gaps) if g not in set(cur)]
    ins_candidates = [g for g in available if rng.random() < args.p_insert]
    rng.shuffle(ins_candidates)
    ins_candidates = ins_candidates[:max_ins_n]
    cur = sorted(set(cur) | set(ins_candidates))

    # 5) 至少保留一点 seed 结构
    if len(cur) < args.min_seed_boundaries_keep and len(gold) > 0:
        fallback = gold[: args.min_seed_boundaries_keep]
        cur = sorted(set(cur) | set(fallback))

    cur = sorted(set(g for g in cur if 0 <= g < num_gaps))
    return cur


def oracle_actions_dp(
    g0: Sequence[int],
    g: Sequence[int],
    K: int,
    lambda_del: float = 1.0,
    lambda_ins: float = 1.0,
    lambda_shift: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    在顺序约束下，求从 seed boundaries g0 到 gold boundaries g 的最优编辑：
    - MATCH with |shift|<=K => KEEP / SHIFT
    - unmatched g0 => DEL
    - unmatched g => INSERT
    """
    g0 = list(sorted(int(x) for x in g0))
    g = list(sorted(int(x) for x in g))

    n = len(g0)
    m = len(g)
    INF = 1e18

    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur >= INF:
                continue

            if i < n:
                nd = cur + lambda_del
                if nd < dp[i + 1][j]:
                    dp[i + 1][j] = nd
                    bt[i + 1][j] = ("DEL", i, j)

            if j < m:
                ni = cur + lambda_ins
                if ni < dp[i][j + 1]:
                    dp[i][j + 1] = ni
                    bt[i][j + 1] = ("INS", i, j)

            if i < n and j < m:
                shift = g[j] - g0[i]
                if abs(shift) <= K:
                    nm = cur + lambda_shift * abs(shift)
                    if nm < dp[i + 1][j + 1]:
                        dp[i + 1][j + 1] = nm
                        bt[i + 1][j + 1] = ("MATCH", i, j)

    i, j = n, m
    matched_seed_to_gold = {}
    inserted_gold = set()

    while i > 0 or j > 0:
        step = bt[i][j]
        if step is None:
            break
        op, pi, pj = step
        if op == "MATCH":
            matched_seed_to_gold[pi] = pj
        elif op == "INS":
            inserted_gold.add(pj)
        i, j = pi, pj

    edit_labels: List[Dict[str, Any]] = []
    for i, g0_pos in enumerate(g0):
        if i not in matched_seed_to_gold:
            edit_labels.append({"g": int(g0_pos), "y": "DEL"})
            continue
        j = matched_seed_to_gold[i]
        shift = int(g[j] - g0_pos)
        if shift == 0:
            edit_labels.append({"g": int(g0_pos), "y": "KEEP"})
        else:
            edit_labels.append({"g": int(g0_pos), "y": f"SHIFT:{shift}"})

    insert_positions = sorted(g[j] for j in inserted_gold)
    return edit_labels, insert_positions


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
    doc_id: str,
    doc_name: str,
    language: str,
    gold_units: Sequence[Dict[str, Any]],
    atoms: Sequence[str],
    seed_boundaries_sparse: Sequence[int],
    gold_boundaries_sparse: Sequence[int],
    chunk0_units: Sequence[Dict[str, Any]],
    seed_unit2atom_span: Sequence[Dict[str, Any]],
    edit_labels: Sequence[Dict[str, Any]],
    insert_positions: Sequence[int],
    variant_idx: int,
    K: int,
) -> Dict[str, Any]:
    num_atoms = len(atoms)
    num_gaps = max(0, num_atoms - 1)

    b0_dense = sparse_to_dense(seed_boundaries_sparse, num_atoms)
    b_gold_dense = sparse_to_dense(gold_boundaries_sparse, num_atoms)
    g0_positions = dense_b0_to_sparse(b0_dense)

    edit_choice = [label_to_edit_choice(item["y"], K) for item in edit_labels]

    insert_dense = [0] * num_gaps
    for g in insert_positions:
        if 0 <= int(g) < num_gaps:
            insert_dense[int(g)] = 1

    return {
        "sample_id": f"rail_llm_gold::{doc_id}::v{variant_idx}",
        "doc_id": doc_id,
        "doc_name": doc_name,
        "language": language,
        "domain": "rail_llm_gold",
        "chunk0_units": list(chunk0_units),
        "atoms": list(atoms),
        "atoms_text": list(atoms),
        "unit2atom_span": list(seed_unit2atom_span),
        "b0": b0_dense,
        "b_gold": b_gold_dense,
        "g0_positions": g0_positions,
        "edit_choice": edit_choice,
        "insert_labels": insert_dense,
        "labels": {
            "edit": list(edit_labels),
            "insert": insert_dense,
        },
        "sample_weight": 1.0,
        "gold_meta": {
            "source": "llm_validated",
            "variant_idx": variant_idx,
            "num_gold_units": len(gold_units),
            "num_atoms": len(atoms),
            "num_seed_boundaries": len(seed_boundaries_sparse),
            "num_gold_boundaries": len(gold_boundaries_sparse),
        },
    }


def main() -> None:
    args = parse_args()

    validated_dir = Path(args.validated_dir)
    output_dir = ensure_dir(args.output_dir)

    train_out = output_dir / "rail_llm_gold_train.jsonl"
    dev_out = output_dir / "rail_llm_gold_dev.jsonl"
    rejects_out = output_dir / "rejects.jsonl"
    summary_out = output_dir / "conversion_summary.json"

    items = load_validated_jsons(validated_dir, args.glob)

    num_seen = 0
    num_train = 0
    num_dev = 0
    num_reject = 0
    reject_reasons = Counter()

    with train_out.open("w", encoding="utf-8") as f_train, \
         dev_out.open("w", encoding="utf-8") as f_dev, \
         rejects_out.open("w", encoding="utf-8") as f_rej:

        for path, obj in items:
            num_seen += 1

            ok, tree_errors = validate_min_tree(obj)
            if not ok:
                num_reject += 1
                reject_reasons["invalid_validated_tree"] += 1
                f_rej.write(json.dumps({
                    "path": str(path),
                    "doc_id": str(obj.get("doc_id", path.stem)),
                    "reason": "invalid_validated_tree",
                    "errors": tree_errors,
                }, ensure_ascii=False) + "\n")
                continue

            doc_id = str(obj["doc_id"])
            doc_name = str(obj.get("doc_name") or "unknown_title")
            language = str(obj.get("language") or "other")
            units = obj["units"]

            gold_units = extract_linear_gold_units(
                units=units,
                keep_nonleaf_anchor_max_chars=args.keep_nonleaf_anchor_max_chars,
            )
            if not gold_units:
                num_reject += 1
                reject_reasons["no_gold_units_after_flatten"] += 1
                f_rej.write(json.dumps({
                    "path": str(path),
                    "doc_id": doc_id,
                    "reason": "no_gold_units_after_flatten",
                }, ensure_ascii=False) + "\n")
                continue

            normalized_units = []
            for u in gold_units:
                normalized_units.append({
                    "unit_id": int(u["unit_id"]),
                    "text": normalize_text(str(u["text"])),
                })

            try:
                result = build_atoms_b0_from_units(
                    normalized_units=normalized_units,
                    splitter_fn=split_text_to_atoms,
                )
            except Exception as e:
                num_reject += 1
                reject_reasons["atomize_failed"] += 1
                f_rej.write(json.dumps({
                    "path": str(path),
                    "doc_id": doc_id,
                    "reason": "atomize_failed",
                    "error": repr(e),
                }, ensure_ascii=False) + "\n")
                continue

            if result["meta"].get("all_units_empty", False):
                num_reject += 1
                reject_reasons["all_units_empty_after_atomize"] += 1
                f_rej.write(json.dumps({
                    "path": str(path),
                    "doc_id": doc_id,
                    "reason": "all_units_empty_after_atomize",
                }, ensure_ascii=False) + "\n")
                continue

            atoms = result["atoms"]
            gold_unit2atom_span = result["unit2atom_span"]
            b_gold_dense = result["b0"]
            gold_boundaries_sparse = dense_b0_to_sparse(b_gold_dense)

            if len(atoms) == 0:
                num_reject += 1
                reject_reasons["zero_atoms"] += 1
                f_rej.write(json.dumps({
                    "path": str(path),
                    "doc_id": doc_id,
                    "reason": "zero_atoms",
                }, ensure_ascii=False) + "\n")
                continue

            num_atoms = len(atoms)

            for variant_idx in range(args.variants_per_doc):
                seed_hash = hashlib.md5(f"{doc_id}::v{variant_idx}".encode("utf-8")).hexdigest()
                rng = random.Random(int(seed_hash[:8], 16))

                seed_boundaries_sparse = generate_noisy_seed_from_gold(
                    gold_boundaries_sparse=gold_boundaries_sparse,
                    num_atoms=num_atoms,
                    args=args,
                    rng=rng,
                )

                chunk0_units, seed_unit2atom_span = reconstruct_chunk0_from_seed(
                    atoms=atoms,
                    seed_boundaries_sparse=seed_boundaries_sparse,
                )

                g0 = list(seed_boundaries_sparse)
                g = list(gold_boundaries_sparse)

                edit_labels, insert_positions = oracle_actions_dp(
                    g0=g0,
                    g=g,
                    K=args.k_shift,
                )

                rec = build_training_record(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    language=language,
                    gold_units=gold_units,
                    atoms=atoms,
                    seed_boundaries_sparse=seed_boundaries_sparse,
                    gold_boundaries_sparse=gold_boundaries_sparse,
                    chunk0_units=chunk0_units,
                    seed_unit2atom_span=seed_unit2atom_span,
                    edit_labels=edit_labels,
                    insert_positions=insert_positions,
                    variant_idx=variant_idx,
                    K=args.k_shift,
                )

                if stable_dev_split(doc_id, args.dev_ratio):
                    f_dev.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    num_dev += 1
                else:
                    f_train.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    num_train += 1

    summary = {
        "validated_dir": str(validated_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_validated_seen": num_seen,
        "num_train": num_train,
        "num_dev": num_dev,
        "num_reject": num_reject,
        "reject_reasons": dict(reject_reasons),
        "outputs": {
            "train_jsonl": str(train_out),
            "dev_jsonl": str(dev_out),
            "rejects_jsonl": str(rejects_out),
        },
        "config": {
            "dev_ratio": args.dev_ratio,
            "k_shift": args.k_shift,
            "keep_nonleaf_anchor_max_chars": args.keep_nonleaf_anchor_max_chars,
            "variants_per_doc": args.variants_per_doc,
            "p_shift": args.p_shift,
            "p_insert": args.p_insert,
            "p_delete": args.p_delete,
            "max_shift_frac": args.max_shift_frac,
            "max_insert_frac": args.max_insert_frac,
            "max_delete_frac": args.max_delete_frac,
            "burst_delete_prob": args.burst_delete_prob,
            "burst_delete_min": args.burst_delete_min,
            "burst_delete_max": args.burst_delete_max,
            "min_seed_boundaries_keep": args.min_seed_boundaries_keep,
        },
    }
    dump_json(summary_out, summary)

    print(json.dumps({
        "num_train": num_train,
        "num_dev": num_dev,
        "num_reject": num_reject,
        "summary": str(summary_out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()