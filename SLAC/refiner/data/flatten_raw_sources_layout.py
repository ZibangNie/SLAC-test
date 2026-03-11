#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flatten raw source file layout for real_dataset.

目标：
1. 读取 real_dataset/train.jsonl, dev.jsonl, test.jsonl
2. 根据其中每条样本的 source_path / source_family / source_rel_path / orig_split 等信息
   将原始 json/jsonl 文件复制到一个更扁平的目录结构中：
      raw_sources_flat/
        train/
          llm_structured/
            A/
            B/
            unknown/
          railway_parsed/
        dev/
          ...
        test/
          ...
3. 不修改现有 train.jsonl/dev.jsonl/test.jsonl
4. 生成 summary.json 与 copy_manifest.jsonl 方便审计

目录规则：
- split 以“新 split”为准：train/dev/test
- llm_structured 下最多到 A/B 这一层
- railway_parsed 不再额外细分
- 为避免重名，复制后的文件名会编码来源信息

用法：
python flatten_raw_sources_layout.py ^
  --input_root "D:\\code\\Github\\SLAC-test\\SLAC\\refiner\\data\\real_dataset"

PowerShell:
python "D:\\code\\Github\\SLAC-test\\SLAC\\refiner\\data\\flatten_raw_sources_layout.py" `
  --input_root "D:\\code\\Github\\SLAC-test\\SLAC\\refiner\\data\\real_dataset"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL line {lineno} in {path}: {e}") from e


def sanitize_component(s: Optional[str], max_len: int = 80) -> str:
    if s is None:
        s = "unknown"
    s = str(s).strip()
    if not s:
        s = "unknown"
    # 替换 Windows 非法字符和过于奇怪的字符
    s = s.replace(":", "_")
    s = s.replace("\\", "_")
    s = s.replace("/", "_")
    s = s.replace("*", "_")
    s = s.replace("?", "_")
    s = s.replace('"', "_")
    s = s.replace("<", "_")
    s = s.replace(">", "_")
    s = s.replace("|", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._ ")
    if not s:
        s = "unknown"
    if len(s) > max_len:
        s = s[:max_len].rstrip("._ ")
    return s or "unknown"


def detect_llm_bucket(rec: dict) -> str:
    """
    只决定 llm_structured 下放 A / B / unknown
    """
    rel = str(rec.get("source_rel_path") or "").replace("\\", "/").strip("/")
    if rel.startswith("A/"):
        return "A"
    if rel.startswith("B/"):
        return "B"

    src_path = str(rec.get("source_path") or "").replace("\\", "/")
    # 宽松兜底
    if "/A/" in src_path or src_path.endswith("/A"):
        return "A"
    if "/B/" in src_path or src_path.endswith("/B"):
        return "B"
    return "unknown"


def guess_source_family(rec: dict) -> str:
    sf = str(rec.get("source_family") or "").strip()
    if sf:
        return sf
    src = str(rec.get("source_path") or "").replace("\\", "/").lower()
    if "parsed_json" in src or "railway" in src:
        return "railway_parsed"
    return "llm_structured"


def build_dest_dir(flat_root: Path, split_name: str, rec: dict) -> Path:
    sf = guess_source_family(rec)
    if sf == "llm_structured":
        bucket = detect_llm_bucket(rec)
        return flat_root / split_name / "llm_structured" / bucket
    if sf == "railway_parsed":
        return flat_root / split_name / "railway_parsed"
    return flat_root / split_name / sanitize_component(sf)


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def build_flat_filename(rec: dict, src_path: Path) -> str:
    """
    文件名中保留足够 provenance，避免目录拍扁后重名。
    """
    source_family = sanitize_component(guess_source_family(rec), 32)
    domain = sanitize_component(rec.get("domain"), 48)
    language = sanitize_component(rec.get("language"), 16)
    orig_split = sanitize_component(rec.get("orig_split"), 16)
    doc_id = sanitize_component(rec.get("doc_id"), 64)
    stem = sanitize_component(src_path.stem, 80)

    # 用 source_rel_path 或 source_path 做短 hash，避免重复
    rel_or_src = str(rec.get("source_rel_path") or rec.get("source_path") or src_path)
    h = short_hash(rel_or_src, 10)

    suffix = src_path.suffix if src_path.suffix else ".json"
    filename = f"{source_family}__{domain}__{language}__{orig_split}__{doc_id}__{stem}__{h}{suffix}"

    # Windows 路径长度与文件名长度稳一点
    if len(filename) > 220:
        filename = (
            f"{source_family}__{domain}__{language}__{orig_split}__"
            f"{sanitize_component(rec.get('doc_id'), 32)}__"
            f"{sanitize_component(src_path.stem, 48)}__{h}{suffix}"
        )
    return filename


def ensure_unique_path(dst: Path) -> Path:
    """
    如果目标文件名重复，则自动加序号。
    """
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def copy_record_source(rec: dict, split_name: str, flat_root: Path) -> Tuple[bool, Dict]:
    source_path_str = rec.get("source_path")
    if not source_path_str:
        return False, {
            "reason": "missing_source_path",
            "split": split_name,
            "sample_id": rec.get("sample_id"),
            "doc_id": rec.get("doc_id"),
        }

    src_path = Path(source_path_str)
    if not src_path.exists():
        return False, {
            "reason": "source_file_not_found",
            "split": split_name,
            "sample_id": rec.get("sample_id"),
            "doc_id": rec.get("doc_id"),
            "source_path": str(src_path),
        }

    dest_dir = build_dest_dir(flat_root, split_name, rec)
    dest_dir.mkdir(parents=True, exist_ok=True)

    flat_name = build_flat_filename(rec, src_path)
    dst_path = ensure_unique_path(dest_dir / flat_name)

    try:
        shutil.copy2(src_path, dst_path)
    except Exception as e:
        return False, {
            "reason": "copy_failed",
            "split": split_name,
            "sample_id": rec.get("sample_id"),
            "doc_id": rec.get("doc_id"),
            "source_path": str(src_path),
            "dest_path": str(dst_path),
            "error": repr(e),
        }

    manifest = {
        "split": split_name,
        "sample_id": rec.get("sample_id"),
        "doc_id": rec.get("doc_id"),
        "doc_name": rec.get("doc_name"),
        "source_family": guess_source_family(rec),
        "domain": rec.get("domain"),
        "language": rec.get("language"),
        "orig_split": rec.get("orig_split"),
        "source_rel_path": rec.get("source_rel_path"),
        "source_path": str(src_path),
        "dest_path": str(dst_path),
        "dest_rel_path": str(dst_path).replace(str(flat_root), "").lstrip("\\/"),
    }
    return True, manifest


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def collect_split_records(input_root: Path, split_name: str) -> List[dict]:
    split_file = input_root / f"{split_name}.jsonl"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    return list(read_jsonl(split_file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten raw source file layout for real_dataset.")
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to real_dataset root, e.g. D:\\code\\Github\\SLAC-test\\SLAC\\refiner\\data\\real_dataset",
    )
    parser.add_argument(
        "--output_dirname",
        type=str,
        default="raw_sources_flat",
        help="Name of output directory under input_root. Default: raw_sources_flat",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete existing output dir before copying.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")

    flat_root = input_root / args.output_dirname

    if flat_root.exists() and args.overwrite:
        shutil.rmtree(flat_root)

    flat_root.mkdir(parents=True, exist_ok=True)

    all_failures: List[dict] = []
    all_manifest: List[dict] = []

    summary = {
        "input_root": str(input_root),
        "output_root": str(flat_root),
        "num_records_by_split": {},
        "num_copied_by_split": {},
        "num_failed_by_split": {},
        "num_copied_by_bucket": {},
        "num_unique_source_paths": 0,
        "num_manifest_records": 0,
    }

    copied_bucket_counter = Counter()
    unique_source_paths = set()

    for split_name in ["train", "dev", "test"]:
        records = collect_split_records(input_root, split_name)
        summary["num_records_by_split"][split_name] = len(records)

        copied = 0
        failed = 0

        for rec in records:
            ok, info = copy_record_source(rec, split_name, flat_root)
            if ok:
                copied += 1
                all_manifest.append(info)
                unique_source_paths.add(info["source_path"])

                sf = info.get("source_family", "unknown")
                if sf == "llm_structured":
                    bucket = detect_llm_bucket(rec)
                    key = f"{split_name}|llm_structured|{bucket}"
                elif sf == "railway_parsed":
                    key = f"{split_name}|railway_parsed"
                else:
                    key = f"{split_name}|{sanitize_component(sf)}"
                copied_bucket_counter[key] += 1
            else:
                failed += 1
                all_failures.append(info)

        summary["num_copied_by_split"][split_name] = copied
        summary["num_failed_by_split"][split_name] = failed

    summary["num_unique_source_paths"] = len(unique_source_paths)
    summary["num_manifest_records"] = len(all_manifest)
    summary["num_copied_by_bucket"] = dict(sorted(copied_bucket_counter.items()))

    write_json(summary, flat_root / "summary.json")
    write_jsonl(all_manifest, flat_root / "copy_manifest.jsonl")
    write_jsonl(all_failures, flat_root / "copy_failures.jsonl")

    print("=" * 80)
    print("Flatten raw source layout done.")
    print(f"Input root : {input_root}")
    print(f"Output root: {flat_root}")
    print("-" * 80)
    print("num_records_by_split =", summary["num_records_by_split"])
    print("num_copied_by_split  =", summary["num_copied_by_split"])
    print("num_failed_by_split  =", summary["num_failed_by_split"])
    print("num_unique_source_paths =", summary["num_unique_source_paths"])
    print("num_manifest_records    =", summary["num_manifest_records"])
    print("-" * 80)
    print("num_copied_by_bucket:")
    for k, v in summary["num_copied_by_bucket"].items():
        print(f"  {k}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()