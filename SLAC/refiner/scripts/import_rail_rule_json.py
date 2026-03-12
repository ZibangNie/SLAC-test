#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import railway rule-seeded JSON docs into unified Week2 chunk0 jsonl format."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root directory containing per-document railway JSON files.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output JSONL path for imported chunk0 samples.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        required=True,
        help="Output JSON path for dataset summary.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input_root for .json files.",
    )
    parser.add_argument(
        "--drop_empty_units",
        action="store_true",
        help="Drop units whose text is empty/blank after strip.",
    )
    parser.add_argument(
        "--strict_doc_id",
        action="store_true",
        help="If set, documents without doc_id will raise an error instead of fallback.",
    )
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def find_json_files(input_root: str, recursive: bool) -> List[str]:
    paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(input_root):
            for fn in files:
                if fn.lower().endswith(".json"):
                    paths.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(input_root):
            fp = os.path.join(input_root, fn)
            if os.path.isfile(fp) and fn.lower().endswith(".json"):
                paths.append(fp)
    paths.sort()
    return paths


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def normalize_doc_id(obj: Dict[str, Any], path: str, strict_doc_id: bool) -> str:
    doc_id = safe_str(obj.get("doc_id")).strip()
    if doc_id:
        return doc_id
    if strict_doc_id:
        raise ValueError(f"Missing doc_id in {path}")
    base = os.path.splitext(os.path.basename(path))[0]
    return f"fallback::{base}"


def normalize_doc_name(obj: Dict[str, Any]) -> str:
    doc_name = safe_str(obj.get("doc_name")).strip()
    return doc_name if doc_name else "unknown_title"


def normalize_language(obj: Dict[str, Any]) -> str:
    lang = safe_str(obj.get("language")).strip().lower()
    return lang if lang else "unknown"


def unit_sort_key(unit: Dict[str, Any], idx: int) -> Tuple[int, int]:
    uid = unit.get("unit_id", idx)
    if isinstance(uid, int):
        return (0, uid)
    try:
        return (0, int(uid))
    except Exception:
        return (1, idx)


def normalize_units(
    units: Any,
    drop_empty_units: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not isinstance(units, list):
        raise ValueError("Top-level field 'units' is not a list")

    stats = {
        "num_input_units": len(units),
        "num_empty_dropped": 0,
        "had_non_monotonic_unit_ids": False,
        "had_duplicate_unit_ids": False,
        "had_missing_unit_ids": False,
    }

    indexed_units = list(enumerate(units))
    indexed_units.sort(key=lambda x: unit_sort_key(x[1], x[0]))

    seen_ids = set()
    norm_units: List[Dict[str, Any]] = []

    last_numeric_uid = None
    for new_idx, (orig_idx, u) in enumerate(indexed_units):
        if not isinstance(u, dict):
            continue

        raw_uid = u.get("unit_id", orig_idx)
        unit_id: int
        if isinstance(raw_uid, int):
            unit_id = raw_uid
        else:
            try:
                unit_id = int(raw_uid)
            except Exception:
                stats["had_missing_unit_ids"] = True
                unit_id = orig_idx

        if unit_id in seen_ids:
            stats["had_duplicate_unit_ids"] = True
        seen_ids.add(unit_id)

        if last_numeric_uid is not None and unit_id < last_numeric_uid:
            stats["had_non_monotonic_unit_ids"] = True
        last_numeric_uid = unit_id

        text = safe_str(u.get("text"))
        if drop_empty_units and not text.strip():
            stats["num_empty_dropped"] += 1
            continue

        norm_u: Dict[str, Any] = {
            "unit_id": len(norm_units),
            "orig_unit_id": unit_id,
            "text": text,
            "type": safe_str(u.get("type")).strip() or "unknown",
            "level": u.get("level", 0),
            "parent_id": u.get("parent_id", None),
            "unit_hash": safe_str(u.get("unit_hash")).strip() or "",
        }

        if "num_prefix" in u:
            norm_u["num_prefix"] = u["num_prefix"]
        if "marker_type" in u:
            norm_u["marker_type"] = u["marker_type"]

        norm_units.append(norm_u)

    stats["num_output_units"] = len(norm_units)
    return norm_units, stats


def build_record(path: str, obj: Dict[str, Any], args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    doc_id = normalize_doc_id(obj, path, args.strict_doc_id)
    doc_name = normalize_doc_name(obj)
    language = normalize_language(obj)
    units, unit_stats = normalize_units(obj.get("units"), args.drop_empty_units)

    if len(units) == 0:
        raise ValueError(f"No valid units after normalization: {path}")

    record = {
        "sample_id": f"rail::{doc_id}",
        "doc_id": doc_id,
        "doc_name": doc_name,
        "language": language,
        "domain": "rail_rule_chunk0",
        "source_path": os.path.abspath(path),
        "chunk0_units": units,
        "meta": {
            "source": "rail_rule_seed_json",
        },
    }

    doc_stats = {
        "doc_id": doc_id,
        "language": language,
        "num_units": len(units),
        **unit_stats,
    }
    return record, doc_stats


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.input_root):
        raise FileNotFoundError(f"input_root does not exist: {args.input_root}")

    ensure_parent(args.output_jsonl)
    ensure_parent(args.summary_json)

    files = find_json_files(args.input_root, args.recursive)
    if not files:
        raise RuntimeError(f"No .json files found under: {args.input_root}")

    lang_counter: Counter = Counter()
    num_units_list: List[int] = []
    failed_files: List[Dict[str, str]] = []
    doc_ids_seen = set()
    duplicate_doc_ids = 0

    total_empty_dropped = 0
    total_non_monotonic = 0
    total_duplicate_unit_ids = 0
    total_missing_unit_ids = 0

    kept = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for fp in files:
            try:
                obj = load_json(fp)
                record, doc_stats = build_record(fp, obj, args)

                doc_id = record["doc_id"]
                if doc_id in doc_ids_seen:
                    duplicate_doc_ids += 1
                doc_ids_seen.add(doc_id)

                lang_counter[doc_stats["language"]] += 1
                num_units_list.append(doc_stats["num_units"])
                total_empty_dropped += int(doc_stats["num_empty_dropped"])
                total_non_monotonic += int(doc_stats["had_non_monotonic_unit_ids"])
                total_duplicate_unit_ids += int(doc_stats["had_duplicate_unit_ids"])
                total_missing_unit_ids += int(doc_stats["had_missing_unit_ids"])

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

            except Exception as e:
                failed_files.append({
                    "path": os.path.abspath(fp),
                    "error": repr(e),
                })

    num_units_sorted = sorted(num_units_list)
    if num_units_sorted:
        p50 = num_units_sorted[len(num_units_sorted) // 2]
        p90 = num_units_sorted[min(len(num_units_sorted) - 1, int(0.9 * (len(num_units_sorted) - 1)))]
        units_min = num_units_sorted[0]
        units_max = num_units_sorted[-1]
    else:
        p50 = p90 = units_min = units_max = 0

    summary = {
        "input_root": os.path.abspath(args.input_root),
        "output_jsonl": os.path.abspath(args.output_jsonl),
        "num_json_files_found": len(files),
        "num_docs_imported": kept,
        "num_failed_files": len(failed_files),
        "failed_files": failed_files[:200],
        "languages": dict(lang_counter),
        "units_min": units_min,
        "units_p50": p50,
        "units_p90": p90,
        "units_max": units_max,
        "duplicate_doc_ids": duplicate_doc_ids,
        "total_empty_units_dropped": total_empty_dropped,
        "docs_with_non_monotonic_unit_ids": total_non_monotonic,
        "docs_with_duplicate_unit_ids": total_duplicate_unit_ids,
        "docs_with_missing_or_nonint_unit_ids": total_missing_unit_ids,
        "settings": {
            "recursive": args.recursive,
            "drop_empty_units": args.drop_empty_units,
            "strict_doc_id": args.strict_doc_id,
        },
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "num_docs_imported": kept,
        "num_failed_files": len(failed_files),
        "languages": dict(lang_counter),
        "units_p50": p50,
        "units_p90": p90,
        "units_max": units_max,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()