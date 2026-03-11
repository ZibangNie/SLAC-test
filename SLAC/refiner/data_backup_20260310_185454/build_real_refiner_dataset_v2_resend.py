#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a unified real dataset for SLAC Boundary Refiner from:
1) LLM-labeled structured / semi-structured corpora
2) railway parsed_json corpus

Key behavior:
- Original train/dev/test folders are treated as source metadata only.
- First unify everything into one normalized document pool.
- Then re-split into new train/dev/test.
- Do NOT filter long documents at this stage.
- Copy original raw JSON/JSONL files into output_root/raw_sources/{train,dev,test}/...
  according to the NEW final split assignment.
  If one source file contributes records to multiple final splits, that file is copied to each split.

Output layout under output_root:
  all_docs.jsonl
  train.jsonl
  dev.jsonl
  test.jsonl
  stats/
    summary.json
    bucket_counts.json
  manifests/
    all_source_paths.txt
    train_source_paths.txt
    dev_source_paths.txt
    test_source_paths.txt
  raw_sources/
    train/
      llm_structured/... original relative path
      railway_parsed/... original relative path
    dev/
    test/
  logs/
    failed_records.jsonl
    skipped_files.jsonl

Example:
python build_real_refiner_dataset_v2.py \
  --llm_root "D:\\YOUR_LLM_ROOT" \
  --railway_root "D:\\code\\Github\\SLAC-test\\SLAC\\data\\parsed_json" \
  --output_root "D:\\code\\Github\\SLAC-test\\SLAC\\refiner\\data\\real_dataset"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TEXT_KEYS = [
    "text", "content", "raw_text", "value", "sentence", "line_text", "unit_text",
    "paragraph", "body", "normalized_text", "clean_text",
]
TITLE_KEYS = ["doc_name", "title", "name", "document_name", "file_name"]
LANG_KEYS = ["language", "lang"]
BOUNDARY_KEYS = ["gold_boundaries", "boundaries", "chunk_boundaries", "boundary_indices"]
UNIT_LIST_KEYS = ["units", "atoms", "lines", "paragraphs", "segments", "blocks"]
TYPE_KEYS = ["type", "unit_type", "category", "kind"]
LEVEL_KEYS = ["level", "depth", "heading_level"]
PAGE_KEYS = ["page", "page_id", "page_no", "page_num"]
ID_KEYS = ["unit_id", "atom_id", "id"]


def as_path(x: str) -> Path:
    return Path(x).expanduser().resolve()


def is_json_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".json"


def is_jsonl_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".jsonl"


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_dump(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(record: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_lang_from_text(text: str) -> str:
    if not text:
        return "unknown"
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if zh == 0 and en == 0:
        return "unknown"
    return "zh" if zh > en else "en"


def percentile(sorted_vals: List[int], p: float) -> Optional[int]:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round((len(sorted_vals) - 1) * p)))
    return sorted_vals[idx]


def bucket_key(rec: Dict[str, Any]) -> str:
    return "|".join([
        rec.get("source_family", "unknown"),
        rec.get("domain", "unknown"),
        rec.get("language", "unknown"),
    ])


def stable_hash_text(x: str) -> str:
    return hashlib.md5(x.encode("utf-8")).hexdigest()[:10]


# -----------------------------
# File readers
# -----------------------------

def iter_json_records(path: Path) -> Iterable[Tuple[int, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON file {path}: {e}")

    if isinstance(obj, dict):
        list_like = None
        for k in ["documents", "docs", "data", "items", "records"]:
            if isinstance(obj.get(k), list):
                list_like = obj[k]
                break
        if list_like is None:
            yield 0, obj
        else:
            for i, item in enumerate(list_like):
                yield i, item
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield i, item
    else:
        raise RuntimeError(f"Unsupported top-level JSON type in {path}: {type(obj)}")


def iter_jsonl_records(path: Path) -> Iterable[Tuple[int, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield i, json.loads(line)
                except Exception as e:
                    raise RuntimeError(f"Failed to parse line {i} in {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read JSONL file {path}: {e}")


def iter_records_from_file(path: Path) -> Iterable[Tuple[int, Any]]:
    if is_json_file(path):
        yield from iter_json_records(path)
    elif is_jsonl_file(path):
        yield from iter_jsonl_records(path)


# -----------------------------
# Source metadata inference
# -----------------------------

def detect_orig_split_from_parts(parts: Tuple[str, ...]) -> str:
    for p in parts:
        pl = p.lower()
        if pl in {"train", "dev", "test", "valid", "val"}:
            return "dev" if pl in {"dev", "valid", "val"} else pl
    return "unknown"


def detect_llm_domain_language(rel_parts: Tuple[str, ...]) -> Tuple[str, str]:
    domain = "unknown"
    language = "unknown"
    lowered = [p.lower() for p in rel_parts]

    if lowered:
        if lowered[0] == "a":
            if len(lowered) >= 2:
                domain = lowered[1]
            if len(lowered) >= 3 and lowered[2] in {"en", "zh"}:
                language = lowered[2]
            elif domain == "papers":
                language = "unknown"
        elif lowered[0] == "b":
            domain = "b"
            if len(lowered) >= 2 and lowered[1] in {"en", "zh"}:
                language = lowered[1]

    return domain, language


def gather_source_files(root: Path, allow_suffixes=(".json", ".jsonl")) -> List[Path]:
    files = []
    for suffix in allow_suffixes:
        files.extend(root.rglob(f"*{suffix}"))
    return sorted(set(p for p in files if p.is_file()))


# -----------------------------
# Normalization
# -----------------------------

def extract_text_from_any(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return normalize_whitespace(obj)
    if isinstance(obj, dict):
        text = safe_get(obj, TEXT_KEYS, "")
        if isinstance(text, str) and text.strip():
            return normalize_whitespace(text)
        for key in ["tokens", "spans"]:
            val = obj.get(key)
            if isinstance(val, list):
                pieces = [extract_text_from_any(v) for v in val]
                pieces = [p for p in pieces if p]
                if pieces:
                    return normalize_whitespace(" ".join(pieces))
    return ""


def normalize_atom(item: Any, idx: int) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        text = normalize_whitespace(item)
        if not text:
            return None
        return {
            "atom_id": f"atom_{idx:05d}",
            "text": text,
            "type": "text",
            "level": None,
            "page": None,
            "meta": {},
        }

    if not isinstance(item, dict):
        return None

    text = extract_text_from_any(item)
    if not text:
        return None

    atom_id = str(safe_get(item, ID_KEYS, f"atom_{idx:05d}"))
    atom_type = safe_get(item, TYPE_KEYS, "text")
    level = safe_get(item, LEVEL_KEYS, None)
    page = safe_get(item, PAGE_KEYS, None)

    meta = {}
    for k, v in item.items():
        if k in set(ID_KEYS + TEXT_KEYS + TYPE_KEYS + LEVEL_KEYS + PAGE_KEYS):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            meta[k] = v

    return {
        "atom_id": atom_id,
        "text": text,
        "type": atom_type,
        "level": level,
        "page": page,
        "meta": meta,
    }


def derive_atoms_from_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in UNIT_LIST_KEYS:
        if key in doc and isinstance(doc[key], list):
            atoms = []
            for i, item in enumerate(doc[key]):
                atom = normalize_atom(item, i)
                if atom is not None:
                    atoms.append(atom)
            if atoms:
                return atoms

    raw_text = extract_text_from_any(doc)
    if raw_text:
        atoms = []
        for i, line in enumerate(raw_text.splitlines()):
            line = normalize_whitespace(line)
            if not line:
                continue
            atoms.append({
                "atom_id": f"atom_{i:05d}",
                "text": line,
                "type": "line",
                "level": None,
                "page": None,
                "meta": {},
            })
        if atoms:
            return atoms

    return []


def normalize_boundaries(doc: Dict[str, Any]) -> Optional[List[int]]:
    b = safe_get(doc, BOUNDARY_KEYS, None)
    if b is None:
        return None
    if isinstance(b, list):
        out = []
        for x in b:
            if isinstance(x, bool):
                continue
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                out.append(int(x))
            elif isinstance(x, str) and x.strip().isdigit():
                out.append(int(x.strip()))
        return sorted(set(v for v in out if v >= 0))
    return None


def normalize_document(
    doc: Dict[str, Any],
    *,
    source_family: str,
    source_path: Path,
    source_rel_path: Path,
    rel_parts: Tuple[str, ...],
    record_idx: int,
) -> Optional[Dict[str, Any]]:
    if not isinstance(doc, dict):
        return None

    atoms = derive_atoms_from_document(doc)
    if not atoms:
        return None

    doc_id = str(doc.get("doc_id") or doc.get("id") or f"{source_path.stem}__{record_idx}")
    doc_name = str(safe_get(doc, TITLE_KEYS, source_path.stem))
    language = str(safe_get(doc, LANG_KEYS, "unknown"))

    if source_family == "llm_structured":
        domain, path_lang = detect_llm_domain_language(rel_parts)
    else:
        domain, path_lang = "railway", "unknown"

    if language == "unknown" or not language:
        if path_lang != "unknown":
            language = path_lang
        else:
            joined = "\n".join(a["text"] for a in atoms[:50])
            language = infer_lang_from_text(joined)

    orig_split = detect_orig_split_from_parts(rel_parts)
    gold_boundaries = normalize_boundaries(doc)

    top_meta = {}
    for k, v in doc.items():
        if k in set(TITLE_KEYS + LANG_KEYS + BOUNDARY_KEYS + UNIT_LIST_KEYS + ["doc_id", "id"]):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            top_meta[k] = v

    source_key = f"{source_family}::{source_rel_path.as_posix()}"
    sample_key = f"{source_key}::{doc_id}::{record_idx}"

    return {
        "sample_id": f"sample_{stable_hash_text(sample_key)}",
        "doc_id": doc_id,
        "doc_name": doc_name,
        "source_family": source_family,
        "source_path": str(source_path),
        "source_rel_path": source_rel_path.as_posix(),
        "orig_split": orig_split,
        "domain": domain,
        "language": language,
        "num_atoms": len(atoms),
        "atoms": atoms,
        "gold_boundaries": gold_boundaries,
        "meta": top_meta,
    }


# -----------------------------
# Splitting
# -----------------------------

def split_bucket(records: List[Dict[str, Any]], train_ratio: float, dev_ratio: float, test_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = len(records)
    if n == 0:
        return [], [], []

    n_train = int(round(n * train_ratio))
    n_dev = int(round(n * dev_ratio))
    n_test = n - n_train - n_dev

    while n_test < 0:
        if n_dev > 0:
            n_dev -= 1
        elif n_train > 0:
            n_train -= 1
        n_test = n - n_train - n_dev

    if n == 1:
        return records[:], [], []
    if n == 2:
        return records[:1], records[1:2], []
    if n == 3:
        return records[:1], records[1:2], records[2:3]

    if n >= 5 and n_dev == 0:
        n_dev = 1
        n_train = max(1, n_train - 1)
    if n >= 8 and n_test == 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    while n_train + n_dev + n_test > n:
        if n_train >= max(n_dev, n_test, 1):
            n_train -= 1
        elif n_dev >= max(n_test, 1):
            n_dev -= 1
        else:
            n_test -= 1
    while n_train + n_dev + n_test < n:
        n_train += 1

    train = records[:n_train]
    dev = records[n_train:n_train + n_dev]
    test = records[n_train + n_dev:]
    return train, dev, test


def stratified_split(
    records: List[Dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in records:
        buckets[bucket_key(r)].append(r)

    train_all, dev_all, test_all = [], [], []
    bucket_counts = {}

    for bkey, items in sorted(buckets.items()):
        rng.shuffle(items)
        tr, dv, te = split_bucket(items, train_ratio, dev_ratio, test_ratio)
        train_all.extend(tr)
        dev_all.extend(dv)
        test_all.extend(te)
        bucket_counts[bkey] = {
            "total": len(items),
            "train": len(tr),
            "dev": len(dv),
            "test": len(te),
        }

    rng.shuffle(train_all)
    rng.shuffle(dev_all)
    rng.shuffle(test_all)
    return train_all, dev_all, test_all, bucket_counts


# -----------------------------
# Stats and raw copy helpers
# -----------------------------

def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = sorted(r.get("num_atoms", 0) for r in records)
    source_family_counts = Counter(r.get("source_family", "unknown") for r in records)
    domain_counts = Counter(r.get("domain", "unknown") for r in records)
    lang_counts = Counter(r.get("language", "unknown") for r in records)
    orig_split_counts = Counter(r.get("orig_split", "unknown") for r in records)
    with_gold = sum(1 for r in records if r.get("gold_boundaries") is not None)

    return {
        "num_docs": len(records),
        "num_docs_with_gold_boundaries": with_gold,
        "length": {
            "min": lengths[0] if lengths else None,
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": lengths[-1] if lengths else None,
            "mean": (sum(lengths) / len(lengths)) if lengths else None,
        },
        "source_family_counts": dict(source_family_counts),
        "domain_counts": dict(domain_counts),
        "language_counts": dict(lang_counts),
        "orig_split_counts": dict(orig_split_counts),
    }


def copy_raw_files_for_split(records: List[Dict[str, Any]], split_name: str, output_root: Path, failed_record_log: Path) -> Dict[str, int]:
    copied = 0
    skipped_existing = 0
    failed = 0

    unique_files = {}
    for r in records:
        unique_files[(r["source_family"], r["source_path"], r["source_rel_path"])] = r

    for (_, src_path_str, src_rel_path_str), r in unique_files.items():
        src_path = Path(src_path_str)
        dst = output_root / "raw_sources" / split_name / r["source_family"] / Path(src_rel_path_str)
        ensure_dir(dst.parent)
        try:
            if dst.exists():
                skipped_existing += 1
                continue
            shutil.copy2(src_path, dst)
            copied += 1
        except Exception as e:
            failed += 1
            append_jsonl({
                "reason": "copy_raw_source_failed",
                "split": split_name,
                "source_family": r["source_family"],
                "source_path": str(src_path),
                "source_rel_path": src_rel_path_str,
                "dst": str(dst),
                "error": str(e),
            }, failed_record_log)

    return {
        "num_unique_source_files": len(unique_files),
        "copied": copied,
        "skipped_existing": skipped_existing,
        "failed": failed,
    }


# -----------------------------
# Main pipeline
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified real_dataset for SLAC Boundary Refiner")
    p.add_argument("--llm_root", type=str, required=True, help="Root containing A/, B/, index/ ...")
    p.add_argument("--railway_root", type=str, required=True, help="Root of railway parsed_json")
    p.add_argument("--output_root", type=str, required=True, help="Output directory for real_dataset")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--train_ratio", type=float, default=0.80)
    p.add_argument("--dev_ratio", type=float, default=0.10)
    p.add_argument("--test_ratio", type=float, default=0.10)
    p.add_argument(
        "--include_orig_splits",
        type=str,
        default="train,dev,test",
        help="Comma-separated original source subdirs to include. Default keeps train/dev/test as source pools.",
    )
    return p.parse_args()


def should_include_by_orig_split(path_parts: Tuple[str, ...], allowed: set) -> bool:
    split = detect_orig_split_from_parts(path_parts)
    return split in allowed or split == "unknown"


def main() -> None:
    args = parse_args()

    if not math.isclose(args.train_ratio + args.dev_ratio + args.test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_ratio + dev_ratio + test_ratio must equal 1.0")

    llm_root = as_path(args.llm_root)
    railway_root = as_path(args.railway_root)
    output_root = as_path(args.output_root)

    ensure_dir(output_root)
    ensure_dir(output_root / "stats")
    ensure_dir(output_root / "manifests")
    ensure_dir(output_root / "logs")
    ensure_dir(output_root / "raw_sources")

    allowed_orig_splits = {x.strip().lower() for x in args.include_orig_splits.split(",") if x.strip()}

    all_records: List[Dict[str, Any]] = []
    all_source_paths = []

    skipped_file_log = output_root / "logs" / "skipped_files.jsonl"
    failed_record_log = output_root / "logs" / "failed_records.jsonl"
    if skipped_file_log.exists():
        skipped_file_log.unlink()
    if failed_record_log.exists():
        failed_record_log.unlink()

    llm_files = gather_source_files(llm_root)
    for file_path in llm_files:
        rel_parts = file_path.relative_to(llm_root).parts
        if not should_include_by_orig_split(rel_parts, allowed_orig_splits):
            append_jsonl({
                "reason": "orig_split_filtered_out",
                "source_family": "llm_structured",
                "file": str(file_path),
            }, skipped_file_log)
            continue

        all_source_paths.append(str(file_path))
        source_rel_path = file_path.relative_to(llm_root)
        try:
            for rec_idx, raw_doc in iter_records_from_file(file_path):
                try:
                    norm = normalize_document(
                        raw_doc,
                        source_family="llm_structured",
                        source_path=file_path,
                        source_rel_path=source_rel_path,
                        rel_parts=rel_parts,
                        record_idx=rec_idx,
                    )
                    if norm is None:
                        append_jsonl({
                            "reason": "normalize_returned_none",
                            "source_family": "llm_structured",
                            "file": str(file_path),
                            "record_idx": rec_idx,
                        }, failed_record_log)
                        continue
                    all_records.append(norm)
                except Exception as e:
                    append_jsonl({
                        "reason": "record_exception",
                        "source_family": "llm_structured",
                        "file": str(file_path),
                        "record_idx": rec_idx,
                        "error": str(e),
                    }, failed_record_log)
        except Exception as e:
            append_jsonl({
                "reason": "file_exception",
                "source_family": "llm_structured",
                "file": str(file_path),
                "error": str(e),
            }, failed_record_log)

    railway_files = gather_source_files(railway_root)
    for file_path in railway_files:
        rel_parts = file_path.relative_to(railway_root).parts
        source_rel_path = file_path.relative_to(railway_root)
        all_source_paths.append(str(file_path))
        try:
            for rec_idx, raw_doc in iter_records_from_file(file_path):
                try:
                    norm = normalize_document(
                        raw_doc,
                        source_family="railway_parsed",
                        source_path=file_path,
                        source_rel_path=source_rel_path,
                        rel_parts=rel_parts,
                        record_idx=rec_idx,
                    )
                    if norm is None:
                        append_jsonl({
                            "reason": "normalize_returned_none",
                            "source_family": "railway_parsed",
                            "file": str(file_path),
                            "record_idx": rec_idx,
                        }, failed_record_log)
                        continue
                    all_records.append(norm)
                except Exception as e:
                    append_jsonl({
                        "reason": "record_exception",
                        "source_family": "railway_parsed",
                        "file": str(file_path),
                        "record_idx": rec_idx,
                        "error": str(e),
                    }, failed_record_log)
        except Exception as e:
            append_jsonl({
                "reason": "file_exception",
                "source_family": "railway_parsed",
                "file": str(file_path),
                "error": str(e),
            }, failed_record_log)

    dedup = {}
    for r in all_records:
        key = (r["source_family"], r["source_path"], r["doc_id"], r["sample_id"])
        if key not in dedup:
            dedup[key] = r
    all_records = list(dedup.values())

    write_jsonl(all_records, output_root / "all_docs.jsonl")

    train_records, dev_records, test_records, bucket_counts = stratified_split(
        all_records,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    write_jsonl(train_records, output_root / "train.jsonl")
    write_jsonl(dev_records, output_root / "dev.jsonl")
    write_jsonl(test_records, output_root / "test.jsonl")

    with (output_root / "manifests" / "all_source_paths.txt").open("w", encoding="utf-8") as f:
        for p in sorted(set(all_source_paths)):
            f.write(p + "\n")
    for name, recs in [("train", train_records), ("dev", dev_records), ("test", test_records)]:
        with (output_root / "manifests" / f"{name}_source_paths.txt").open("w", encoding="utf-8") as f:
            for p in sorted(set(r["source_path"] for r in recs)):
                f.write(p + "\n")

    raw_copy_stats = {
        "train": copy_raw_files_for_split(train_records, "train", output_root, failed_record_log),
        "dev": copy_raw_files_for_split(dev_records, "dev", output_root, failed_record_log),
        "test": copy_raw_files_for_split(test_records, "test", output_root, failed_record_log),
    }

    summary = {
        "config": {
            "llm_root": str(llm_root),
            "railway_root": str(railway_root),
            "output_root": str(output_root),
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "dev_ratio": args.dev_ratio,
            "test_ratio": args.test_ratio,
            "include_orig_splits": sorted(allowed_orig_splits),
        },
        "all_docs": summarize(all_records),
        "train": summarize(train_records),
        "dev": summarize(dev_records),
        "test": summarize(test_records),
        "raw_source_copy_stats": raw_copy_stats,
    }
    json_dump(summary, output_root / "stats" / "summary.json")
    json_dump(bucket_counts, output_root / "stats" / "bucket_counts.json")

    print("=" * 80)
    print("Build finished")
    print("all_docs:", len(all_records))
    print("train   :", len(train_records))
    print("dev     :", len(dev_records))
    print("test    :", len(test_records))
    print("output  :", output_root)
    print("summary :", output_root / "stats" / "summary.json")
    print("bucket  :", output_root / "stats" / "bucket_counts.json")


if __name__ == "__main__":
    main()
