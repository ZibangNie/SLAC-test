#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =========================================================
# Helpers
# =========================================================

def read_jsonl_head(path: Path, limit: int = 5) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                break
            if len(out) >= limit:
                break
    return out


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def ensure_report_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rel_str(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def ok(flag: bool) -> str:
    return "OK" if flag else "FAIL"


# =========================================================
# Validation
# =========================================================

def check_raw_sources_flat(raw_root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(raw_root),
        "exists": raw_root.exists(),
        "is_valid": False,
        "required_splits": {},
        "notes": [],
    }
    if not raw_root.exists():
        result["notes"].append("raw_sources_flat 不存在")
        return result

    required = ["train", "dev", "test"]
    split_ok = True
    for split in required:
        split_dir = raw_root / split
        exists = split_dir.exists() and split_dir.is_dir()
        result["required_splits"][split] = {
            "path": str(split_dir),
            "exists": exists,
        }
        split_ok = split_ok and exists

    # 更细一点检查 bucket
    expected = {
        "train": [
            raw_root / "train" / "llm_structured" / "A",
            raw_root / "train" / "llm_structured" / "B",
            raw_root / "train" / "railway_parsed",
        ],
        "dev": [
            raw_root / "dev" / "llm_structured" / "A",
            raw_root / "dev" / "llm_structured" / "B",
            raw_root / "dev" / "railway_parsed",
        ],
        "test": [
            raw_root / "test" / "llm_structured" / "A",
            raw_root / "test" / "llm_structured" / "B",
            raw_root / "test" / "railway_parsed",
        ],
    }

    bucket_checks = {}
    for split, paths in expected.items():
        bucket_checks[split] = []
        for p in paths:
            bucket_checks[split].append({"path": str(p), "exists": p.exists()})
    result["bucket_checks"] = bucket_checks

    file_count = 0
    for ext in ("*.json", "*.jsonl"):
        file_count += sum(1 for _ in raw_root.rglob(ext))
    result["num_files"] = file_count

    result["is_valid"] = split_ok and file_count > 0
    if result["is_valid"]:
        result["notes"].append("raw_sources_flat 结构完整，可作为唯一原始真源")
    return result


def check_canonical_dataset(canonical_root: Path, raw_root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(canonical_root),
        "exists": canonical_root.exists(),
        "is_valid": False,
        "required_files": {},
        "summary_points_to_raw_sources_flat": False,
        "sample_source_path_matches_raw_root": False,
        "content_checks": [],
        "notes": [],
    }
    if not canonical_root.exists():
        result["notes"].append("real_dataset_canonical 不存在")
        return result

    required_files = [
        canonical_root / "all_docs.jsonl",
        canonical_root / "train.jsonl",
        canonical_root / "dev.jsonl",
        canonical_root / "test.jsonl",
        canonical_root / "stats" / "summary.json",
        canonical_root / "logs" / "failed_records.jsonl",
        canonical_root / "logs" / "skipped_files.jsonl",
    ]
    rf_ok = True
    for p in required_files:
        exists = p.exists()
        result["required_files"][rel_str(p, canonical_root)] = exists
        rf_ok = rf_ok and exists

    summary = load_json(canonical_root / "stats" / "summary.json")
    if summary:
        raw_root_in_summary = (
            summary.get("input", {}).get("raw_root")
            if isinstance(summary.get("input"), dict)
            else None
        )
        result["summary_raw_root"] = raw_root_in_summary
        result["summary_points_to_raw_sources_flat"] = (
            raw_root_in_summary == str(raw_root)
        )

    # 抽样检查 canonical records 的 source_path 是否指向 raw_sources_flat
    sample_records = []
    for split in ["train", "dev", "test"]:
        sample_records.extend(read_jsonl_head(canonical_root / f"{split}.jsonl", limit=3))

    source_path_match_count = 0
    schema_ok_count = 0
    for rec in sample_records:
        sp = str(rec.get("source_path", ""))
        if sp.startswith(str(raw_root)):
            source_path_match_count += 1

        atoms = rec.get("atoms")
        num_atoms = rec.get("num_atoms")
        base_fields_ok = all(
            k in rec
            for k in [
                "sample_id",
                "doc_id",
                "doc_name",
                "source_family",
                "source_path",
                "source_rel_path",
                "orig_split",
                "domain",
                "language",
                "num_atoms",
                "atoms",
            ]
        )
        atoms_ok = isinstance(atoms, list)
        num_ok = isinstance(num_atoms, int) and atoms_ok and len(atoms) == num_atoms
        if base_fields_ok and atoms_ok and num_ok:
            schema_ok_count += 1

    if sample_records:
        result["sample_source_path_matches_raw_root"] = (
            source_path_match_count == len(sample_records)
        )
        result["sample_schema_ok_ratio"] = f"{schema_ok_count}/{len(sample_records)}"

    valid = (
        rf_ok
        and result["summary_points_to_raw_sources_flat"]
        and result["sample_source_path_matches_raw_root"]
    )
    result["is_valid"] = valid
    if valid:
        result["notes"].append("real_dataset_canonical 已经正确以内聚数据源 raw_sources_flat 为基础")
    else:
        result["notes"].append("real_dataset_canonical 需要进一步检查")
    return result


def validate_refiner_record(rec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    atoms = rec.get("atoms")
    b0 = rec.get("b0")
    labels = rec.get("labels")
    chunk0_units = rec.get("chunk0_units")
    u2a = rec.get("unit2atom_span")

    if not isinstance(atoms, list) or len(atoms) < 2:
        errs.append("atoms 缺失或长度 < 2")
    if not isinstance(chunk0_units, list) or len(chunk0_units) == 0:
        errs.append("chunk0_units 缺失")
    if not isinstance(u2a, list) or len(u2a) == 0:
        errs.append("unit2atom_span 缺失")
    if not isinstance(b0, list):
        errs.append("b0 缺失")
    if not isinstance(labels, dict):
        errs.append("labels 缺失")
        return len(errs) == 0, errs

    if isinstance(atoms, list) and isinstance(b0, list):
        if len(b0) != max(0, len(atoms) - 1):
            errs.append("len(b0) != len(atoms)-1")

    insert = labels.get("insert")
    edit = labels.get("edit")
    if not isinstance(insert, list):
        errs.append("labels.insert 缺失")
    elif isinstance(b0, list) and len(insert) != len(b0):
        errs.append("len(labels.insert) != len(b0)")

    if not isinstance(edit, list):
        errs.append("labels.edit 缺失")
    else:
        num_gaps = len(atoms) - 1 if isinstance(atoms, list) else -1
        for x in edit[:20]:
            if not isinstance(x, dict):
                errs.append("labels.edit 元素不是 dict")
                break
            if "g" not in x or "y" not in x:
                errs.append("labels.edit 元素缺少 g/y")
                break
            g = x.get("g")
            if not isinstance(g, int) or g < 0 or g >= num_gaps:
                errs.append("labels.edit.g 越界")
                break

    return len(errs) == 0, errs


def check_refiner_dataset(refiner_root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(refiner_root),
        "exists": refiner_root.exists(),
        "is_valid": False,
        "required_files": {},
        "content_checks": [],
        "notes": [],
    }
    if not refiner_root.exists():
        result["notes"].append("refiner_real_dataset_canonical 不存在")
        return result

    required_files = [
        refiner_root / "refiner_train.jsonl",
        refiner_root / "refiner_dev.jsonl",
        refiner_root / "refiner_test.jsonl",
        refiner_root / "stats" / "summary.json",
        refiner_root / "logs" / "failed_records.jsonl",
        refiner_root / "logs" / "skipped_files.jsonl",
    ]
    rf_ok = True
    for p in required_files:
        exists = p.exists()
        result["required_files"][rel_str(p, refiner_root)] = exists
        rf_ok = rf_ok and exists

    # 样本 schema 检查
    sample_records = []
    for split in ["train", "dev", "test"]:
        sample_records.extend(read_jsonl_head(refiner_root / f"refiner_{split}.jsonl", limit=2))

    good = 0
    bad_details = []
    for rec in sample_records:
        is_ok, errs = validate_refiner_record(rec)
        if is_ok:
            good += 1
        else:
            bad_details.append(
                {
                    "sample_id": rec.get("sample_id"),
                    "doc_id": rec.get("doc_id"),
                    "errors": errs,
                }
            )
    result["sample_schema_ok_ratio"] = f"{good}/{len(sample_records)}" if sample_records else "0/0"
    result["sample_schema_errors"] = bad_details[:5]

    result["is_valid"] = rf_ok and (good == len(sample_records)) and len(sample_records) > 0
    if result["is_valid"]:
        result["notes"].append("refiner_real_dataset_canonical 结构可用，可作为训练母集")
    else:
        result["notes"].append("refiner_real_dataset_canonical 需要进一步检查")
    return result


# =========================================================
# Cleanup
# =========================================================

def collect_legacy_targets(data_root: Path) -> List[Path]:
    """
    只收集“建议删除”的旧版本内容。
    保留：
    - real_dataset/raw_sources_flat
    - real_dataset_canonical
    - refiner_real_dataset_canonical
    """
    targets: List[Path] = []

    real_dataset = data_root / "real_dataset"
    if real_dataset.exists():
        # 明确删除旧 raw_sources
        p = real_dataset / "raw_sources"
        if p.exists():
            targets.append(p)

        # 删除旧版 real_dataset 产物，但保留 raw_sources_flat
        for name in [
            "all_docs.jsonl",
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "stats",
            "manifests",
            "logs",
        ]:
            p = real_dataset / name
            if p.exists():
                targets.append(p)

    # 删除旧版非 canonical refiner 数据集
    old_refiner = data_root / "refiner_real_dataset"
    if old_refiner.exists():
        targets.append(old_refiner)

    return targets


def delete_path(p: Path) -> Tuple[bool, str]:
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return True, "deleted"
    except Exception as e:
        return False, repr(e)


# =========================================================
# Main
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and optionally clean up dataset directories.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"D:\code\Github\SLAC-test\SLAC\refiner\data",
        help="refiner data root",
    )
    parser.add_argument(
        "--apply_delete",
        action="store_true",
        help="Actually delete legacy targets. Default is dry-run.",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="dataset_audit_report.json",
        help="Report filename under data_root",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    real_dataset = data_root / "real_dataset"
    raw_sources_flat = real_dataset / "raw_sources_flat"
    canonical_root = data_root / "real_dataset_canonical"
    refiner_canonical_root = data_root / "refiner_real_dataset_canonical"

    report: Dict[str, Any] = {
        "data_root": str(data_root),
        "authoritative_source_dataset": str(raw_sources_flat),
        "canonical_dataset": str(canonical_root),
        "refiner_training_dataset": str(refiner_canonical_root),
        "checks": {},
        "cleanup": {},
        "final_recommendation": {},
    }

    raw_check = check_raw_sources_flat(raw_sources_flat)
    canonical_check = check_canonical_dataset(canonical_root, raw_sources_flat)
    refiner_check = check_refiner_dataset(refiner_canonical_root)

    report["checks"]["raw_sources_flat"] = raw_check
    report["checks"]["real_dataset_canonical"] = canonical_check
    report["checks"]["refiner_real_dataset_canonical"] = refiner_check

    # 最终判定
    authoritative_ok = raw_check["is_valid"]
    canonical_ok = canonical_check["is_valid"]
    refiner_ok = refiner_check["is_valid"]

    if authoritative_ok and canonical_ok:
        current_dataset_msg = "当前唯一原始真源是 raw_sources_flat；标准化可用数据集是 real_dataset_canonical。"
    elif authoritative_ok:
        current_dataset_msg = "当前唯一原始真源是 raw_sources_flat，但 canonical 还需要修复。"
    else:
        current_dataset_msg = "当前 raw_sources_flat 本身还需要修复。"

    if refiner_ok:
        train_dataset_msg = "当前可直接用于 Boundary Refiner 训练的数据集是 refiner_real_dataset_canonical。"
    else:
        train_dataset_msg = "当前 Refiner 训练集还未通过检查，需重建 refiner_real_dataset_canonical。"

    report["final_recommendation"] = {
        "current_source_of_truth": str(raw_sources_flat),
        "current_canonical_dataset": str(canonical_root),
        "current_refiner_training_dataset": str(refiner_canonical_root),
        "source_dataset_ok": authoritative_ok,
        "canonical_dataset_ok": canonical_ok,
        "refiner_dataset_ok": refiner_ok,
        "message_source": current_dataset_msg,
        "message_training": train_dataset_msg,
    }

    # 清理目标
    targets = collect_legacy_targets(data_root)
    cleanup_entries = []
    for p in targets:
        entry = {
            "path": str(p),
            "exists": p.exists(),
            "action": "delete" if args.apply_delete else "would_delete",
            "status": "pending",
        }
        if args.apply_delete and p.exists():
            ok_flag, msg = delete_path(p)
            entry["status"] = "deleted" if ok_flag else "failed"
            entry["detail"] = msg
        else:
            entry["status"] = "dry_run"
        cleanup_entries.append(entry)

    report["cleanup"] = {
        "apply_delete": args.apply_delete,
        "targets": cleanup_entries,
    }

    report_path = data_root / args.report_name
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 控制台输出简洁摘要
    print("=" * 80)
    print("DATASET AUDIT SUMMARY")
    print("=" * 80)
    print(f"[raw_sources_flat]          {ok(raw_check['is_valid'])}  -> {raw_sources_flat}")
    print(f"[real_dataset_canonical]   {ok(canonical_check['is_valid'])}  -> {canonical_root}")
    print(f"[refiner_real_dataset_canonical] {ok(refiner_check['is_valid'])}  -> {refiner_canonical_root}")
    print("-" * 80)
    print(report["final_recommendation"]["message_source"])
    print(report["final_recommendation"]["message_training"])
    print("-" * 80)
    print(f"Report written to: {report_path}")
    print("-" * 80)
    print("Cleanup targets:")
    for x in cleanup_entries:
        print(f"  - {x['action']}: {x['path']} [{x['status']}]")
    print("=" * 80)


if __name__ == "__main__":
    main()