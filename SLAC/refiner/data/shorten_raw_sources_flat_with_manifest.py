#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8")

    def write(self, obj: dict) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def parse_old_flat_filename(path: Path) -> Dict[str, str]:
    """
    解析旧版长文件名：
    llm_structured__laws__en__dev__31984D0557__31984D0557.tree__4fb8cf8b45.json
    """
    stem = path.stem
    parts = stem.split("__")
    if len(parts) >= 7:
        return {
            "source_family": parts[0],
            "domain": parts[1],
            "language": parts[2],
            "orig_split": parts[3],
            "doc_id": parts[4],
            "orig_stem": "__".join(parts[5:-1]) if len(parts) > 7 else parts[5],
            "hash": parts[-1],
        }

    return {
        "source_family": infer_source_family_from_path(path),
        "domain": infer_domain_from_path(path),
        "language": "unknown",
        "orig_split": "unknown",
        "doc_id": path.stem,
        "orig_stem": path.stem,
        "hash": "unknown",
    }


def infer_source_family_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "railway_parsed" in parts:
        return "railway_parsed"
    return "llm_structured"


def infer_domain_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "railway_parsed" in parts:
        return "railway"
    if "a" in parts:
        return "unknown_A"
    if "b" in parts:
        return "b"
    return "unknown"


def detect_bucket_and_dest_rel(src_path: Path, input_root: Path, split: str) -> Tuple[str, Path]:
    parts = [p.lower() for p in src_path.relative_to(input_root).parts]

    if "railway_parsed" in parts:
        return "R", Path(split) / "railway_parsed"

    if "llm_structured" in parts and "a" in parts:
        return "A", Path(split) / "llm_structured" / "A"

    if "llm_structured" in parts and "b" in parts:
        return "B", Path(split) / "llm_structured" / "B"

    return "U", Path(split) / "unknown"


def split_code(split: str) -> str:
    return {"train": "tr", "dev": "dv", "test": "te"}.get(split, "uk")


def iter_split_files(input_root: Path, split: str) -> List[Path]:
    split_dir = input_root / split
    if not split_dir.exists():
        return []
    files: List[Path] = []
    for ext in ("*.json", "*.jsonl"):
        files.extend(split_dir.rglob(ext))
    files = sorted(files, key=lambda p: str(p).lower())
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy raw_sources_flat into short-file-name version with manifest.")
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help=r'Old raw_sources_flat root, e.g. D:\code\Github\SLAC-test\SLAC\refiner\data\real_dataset\raw_sources_flat',
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help=r'New short-name raw source root, e.g. D:\code\Github\SLAC-test\SLAC\refiner\data\real_dataset\raw_sources_flat_short',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete output_root first.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_files_by_split": {},
        "num_files_by_bucket": {},
        "num_total_files": 0,
        "num_failed": 0,
    }

    bucket_counts: Dict[str, int] = {}
    counters: Dict[Tuple[str, str], int] = {}
    failures: List[dict] = []

    with JsonlWriter(output_root / "file_manifest.jsonl") as manifest_writer, \
         JsonlWriter(output_root / "copy_failures.jsonl") as failure_writer:

        for split in ["train", "dev", "test"]:
            files = iter_split_files(input_root, split)
            summary["num_files_by_split"][split] = len(files)

            for src_path in files:
                bucket, dest_rel_dir = detect_bucket_and_dest_rel(src_path, input_root, split)
                counters[(split, bucket)] = counters.get((split, bucket), 0) + 1
                idx = counters[(split, bucket)]

                ext = src_path.suffix.lower() if src_path.suffix else ".json"
                short_name = f"{bucket}_{split_code(split)}_{idx:06d}{ext}"
                dst_dir = output_root / dest_rel_dir
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / short_name

                meta = parse_old_flat_filename(src_path)

                manifest = {
                    "split": split,
                    "bucket": bucket,
                    "short_file_name": short_name,
                    "new_path": str(dst_path),
                    "new_rel_path": str(dst_path.relative_to(output_root)).replace("\\", "/"),
                    "old_path": str(src_path),
                    "old_rel_path": str(src_path.relative_to(input_root)).replace("\\", "/"),
                    "old_file_name": src_path.name,
                    "source_family": meta["source_family"],
                    "domain": meta["domain"],
                    "language": meta["language"],
                    "orig_split": meta["orig_split"],
                    "doc_id": meta["doc_id"],
                    "orig_stem": meta["orig_stem"],
                    "hash": meta["hash"],
                    "extension": ext,
                }

                try:
                    shutil.copy2(src_path, dst_path)
                    manifest_writer.write(manifest)
                    key = f"{split}|{bucket}"
                    bucket_counts[key] = bucket_counts.get(key, 0) + 1
                except Exception as e:
                    failure = {
                        "split": split,
                        "source_path": str(src_path),
                        "dest_path": str(dst_path),
                        "error": repr(e),
                    }
                    failures.append(failure)
                    failure_writer.write(failure)

    summary["num_files_by_bucket"] = dict(sorted(bucket_counts.items()))
    summary["num_total_files"] = sum(summary["num_files_by_split"].values())
    summary["num_failed"] = len(failures)

    write_json(summary, output_root / "summary.json")

    print("=" * 80)
    print("SHORTEN RAW SOURCES DONE")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()