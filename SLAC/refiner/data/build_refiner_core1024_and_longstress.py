#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def update_counter(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def validate_refiner_record(rec: Dict[str, Any]) -> List[str]:
    errs = []
    atoms = rec.get("atoms")
    b0 = rec.get("b0")
    b_gold = rec.get("b_gold")
    labels = rec.get("labels", {})
    insert = labels.get("insert")
    edit = labels.get("edit")

    if not isinstance(atoms, list) or len(atoms) < 2:
        errs.append("atoms_missing_or_too_short")
        return errs

    n_gaps = len(atoms) - 1

    if not isinstance(b0, list) or len(b0) != n_gaps:
        errs.append("b0_length_mismatch")
    if not isinstance(b_gold, list) or len(b_gold) != n_gaps:
        errs.append("b_gold_length_mismatch")
    if not isinstance(insert, list) or len(insert) != n_gaps:
        errs.append("insert_length_mismatch")
    if not isinstance(edit, list):
        errs.append("edit_missing")
    else:
        for x in edit[:50]:
            if not isinstance(x, dict):
                errs.append("edit_item_not_dict")
                break
            if "g" not in x or "y" not in x:
                errs.append("edit_item_missing_keys")
                break
            g = x["g"]
            if not isinstance(g, int) or g < 0 or g >= n_gaps:
                errs.append("edit_gap_out_of_range")
                break

    return errs


def process_split(
    split: str,
    input_path: Path,
    core_writer: JsonlWriter,
    long_writer: JsonlWriter | None,
    max_atoms: int,
    keep_long_in_train: bool,
) -> Dict[str, Any]:
    seen = 0
    invalid = 0
    core_count = 0
    long_count = 0

    input_lengths: List[int] = []
    core_lengths: List[int] = []
    long_lengths: List[int] = []

    invalid_reasons: Dict[str, int] = {}
    core_domain_counts: Dict[str, int] = {}
    core_language_counts: Dict[str, int] = {}
    long_domain_counts: Dict[str, int] = {}
    long_language_counts: Dict[str, int] = {}

    for rec in read_jsonl(input_path):
        seen += 1
        errs = validate_refiner_record(rec)
        if errs:
            invalid += 1
            for e in errs:
                update_counter(invalid_reasons, e)
            continue

        n_atoms = len(rec["atoms"])
        input_lengths.append(n_atoms)

        domain = str(rec.get("source_domain") or rec.get("domain") or "unknown")
        lang = str(rec.get("language") or "unknown")

        if n_atoms <= max_atoms:
            core_writer.write(rec)
            core_count += 1
            core_lengths.append(n_atoms)
            update_counter(core_domain_counts, domain)
            update_counter(core_language_counts, lang)
        else:
            # train 默认不输出 long，避免误训练
            if split != "train" or keep_long_in_train:
                if long_writer is not None:
                    long_writer.write(rec)
            long_count += 1
            long_lengths.append(n_atoms)
            update_counter(long_domain_counts, domain)
            update_counter(long_language_counts, lang)

    return {
        "input_docs": seen,
        "invalid_docs": invalid,
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "core_docs": core_count,
        "long_docs": long_count,
        "input_length": summarize_lengths(input_lengths),
        "core_length": summarize_lengths(core_lengths),
        "long_length": summarize_lengths(long_lengths),
        "core_domain_counts": dict(sorted(core_domain_counts.items())),
        "core_language_counts": dict(sorted(core_language_counts.items())),
        "long_domain_counts": dict(sorted(long_domain_counts.items())),
        "long_language_counts": dict(sorted(long_language_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build core1024 training set and long-stress set from refiner_real_dataset_canonical.")
    parser.add_argument("--input_root", type=str, required=True, help="Input refiner dataset root")
    parser.add_argument("--core_output_root", type=str, required=True, help="Output root for core dataset")
    parser.add_argument("--long_output_root", type=str, required=True, help="Output root for long-stress dataset")
    parser.add_argument("--max_atoms", type=int, default=1024, help="Length cutoff for core set")
    parser.add_argument(
        "--keep_long_in_train",
        action="store_true",
        help="If set, also write long train samples to long_output_root/refiner_train.jsonl",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    core_root = Path(args.core_output_root)
    long_root = Path(args.long_output_root)

    core_root.mkdir(parents=True, exist_ok=True)
    long_root.mkdir(parents=True, exist_ok=True)
    (core_root / "stats").mkdir(parents=True, exist_ok=True)
    (long_root / "stats").mkdir(parents=True, exist_ok=True)

    overall_summary = {
        "config": {
            "input_root": str(input_root),
            "core_output_root": str(core_root),
            "long_output_root": str(long_root),
            "max_atoms": args.max_atoms,
            "keep_long_in_train": args.keep_long_in_train,
        }
    }

    for split in ["train", "dev", "test"]:
        input_path = input_root / f"refiner_{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input split file: {input_path}")

        core_path = core_root / f"refiner_{split}.jsonl"

        if split == "train" and not args.keep_long_in_train:
            long_path = None
        else:
            long_path = long_root / f"refiner_{split}.jsonl"

        with JsonlWriter(core_path) as core_writer:
            if long_path is not None:
                with JsonlWriter(long_path) as long_writer:
                    stats = process_split(
                        split=split,
                        input_path=input_path,
                        core_writer=core_writer,
                        long_writer=long_writer,
                        max_atoms=args.max_atoms,
                        keep_long_in_train=args.keep_long_in_train,
                    )
            else:
                stats = process_split(
                    split=split,
                    input_path=input_path,
                    core_writer=core_writer,
                    long_writer=None,
                    max_atoms=args.max_atoms,
                    keep_long_in_train=args.keep_long_in_train,
                )

        overall_summary[split] = stats

    with (core_root / "stats" / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    with (long_root / "stats" / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(overall_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()