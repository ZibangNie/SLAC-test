#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild canonical real_dataset JSONL files from raw_sources_flat.

目标：
1. 以 raw_sources_flat 作为唯一原始来源
2. 扫描：
      real_dataset/raw_sources_flat/train
      real_dataset/raw_sources_flat/dev
      real_dataset/raw_sources_flat/test
3. 重建标准化后的：
      train.jsonl
      dev.jsonl
      test.jsonl
      all_docs.jsonl
4. 输出中的 source_path / source_rel_path 全部指向 raw_sources_flat 内部路径
5. 兼容当前已有的扁平文件命名格式：
      source_family__domain__language__orig_split__doc_id__stem__hash.json

推荐输出目录：
D:\\code\\Github\\SLAC-test\\SLAC\\refiner\\data\\real_dataset_canonical
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# IO helpers
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# -----------------------------
# Utilities
# -----------------------------


MULTISPACE_RE = re.compile(r"[ \t\x0b\f]+")
BLANKLINE_RE = re.compile(r"\n\s*\n+")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = "\n".join(MULTISPACE_RE.sub(" ", ln).strip() for ln in text.split("\n"))
    text = BLANKLINE_RE.sub("\n\n", text)
    return text.strip()


def sanitize_component(s: Optional[str], max_len: int = 120) -> str:
    if s is None:
        s = "unknown"
    s = str(s).strip()
    if not s:
        s = "unknown"
    s = s.replace(":", "_").replace("\\", "_").replace("/", "_")
    s = s.replace("*", "_").replace("?", "_").replace('"', "_")
    s = s.replace("<", "_").replace(">", "_").replace("|", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._ ")
    if not s:
        s = "unknown"
    if len(s) > max_len:
        s = s[:max_len].rstrip("._ ")
    return s or "unknown"


def stable_sample_id(*parts: str) -> str:
    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"sample_{h}"


def percentile(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int((len(sorted_vals) - 1) * p)))
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
# Parse flat filename
# -----------------------------


def parse_flat_filename(path: Path) -> Dict[str, str]:
    """
    解析形如：
    llm_structured__papers__en__test__corpus_46827714__root__arxiv_1707.00311.tree__abcd1234ef.json

    由于 stem 里原本可能带下划线，但不会带双下划线（sanitize 时已压缩），
    所以这里按 "__" 切分是稳定的。
    """
    name = path.stem
    parts = name.split("__")
    if len(parts) < 7:
        return {
            "source_family": infer_source_family_from_path(path),
            "domain": infer_domain_from_path(path),
            "language": infer_language_from_path(path),
            "orig_split": "unknown",
            "doc_id": sanitize_component(path.stem, 80),
            "orig_stem": sanitize_component(path.stem, 80),
            "hash": "unknown",
        }

    source_family = parts[0]
    domain = parts[1]
    language = parts[2]
    orig_split = parts[3]
    doc_id = parts[4]
    hash_part = parts[-1]
    orig_stem = "__".join(parts[5:-1]) if len(parts) > 7 else parts[5]

    return {
        "source_family": source_family or infer_source_family_from_path(path),
        "domain": domain or infer_domain_from_path(path),
        "language": language or infer_language_from_path(path),
        "orig_split": orig_split or "unknown",
        "doc_id": doc_id or sanitize_component(path.stem, 80),
        "orig_stem": orig_stem or sanitize_component(path.stem, 80),
        "hash": hash_part or "unknown",
    }


def infer_source_family_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "railway_parsed" in parts:
        return "railway_parsed"
    return "llm_structured"


def infer_domain_from_path(path: Path) -> str:
    # 目录只有 llm_structured/A or B / railway_parsed
    parts = list(path.parts)
    lower = [p.lower() for p in parts]
    if "railway_parsed" in lower:
        return "railway"
    if "a" in lower:
        return "unknown_A"
    if "b" in lower:
        return "b"
    return "unknown"


def infer_language_from_path(path: Path) -> str:
    # 扁平目录本身没有语言层，优先靠文件名元信息，兜底 unknown
    return "unknown"


# -----------------------------
# Load raw source docs
# -----------------------------


def load_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_docs_from_file(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """
    支持：
    - .json 单对象
    - .json 对象列表
    - .jsonl 多对象
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for idx, obj in enumerate(read_jsonl(path)):
            if isinstance(obj, dict):
                yield idx, obj
        return

    data = load_json_any(path)
    if isinstance(data, dict):
        yield 0, data
    elif isinstance(data, list):
        for idx, obj in enumerate(data):
            if isinstance(obj, dict):
                yield idx, obj
    else:
        return


# -----------------------------
# Normalize source docs into real_dataset schema
# -----------------------------


def split_text_fallback(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if "\n\n" in text:
        segs = [s.strip() for s in BLANKLINE_RE.split(text) if s.strip()]
    else:
        segs = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not segs:
        segs = [text]
    return segs


def extract_units(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    尽量兼容：
    - units
    - atoms
    - blocks / segments / paragraphs / lines
    - 原始整段 text/content/raw_text
    """
    candidate_keys = ["units", "atoms", "blocks", "segments", "paragraphs", "lines"]
    raw_units = None
    for k in candidate_keys:
        if isinstance(obj.get(k), list):
            raw_units = obj[k]
            break

    units: List[Dict[str, Any]] = []

    if isinstance(raw_units, list):
        for idx, u in enumerate(raw_units):
            if isinstance(u, dict):
                text = normalize_text(
                    u.get("text")
                    or u.get("content")
                    or u.get("raw_text")
                    or u.get("value")
                    or ""
                )
                if not text:
                    continue
                meta = {
                    k: v
                    for k, v in u.items()
                    if k not in {"unit_id", "atom_id", "text", "content", "raw_text", "value", "type", "level", "page", "meta"}
                }
                if isinstance(u.get("meta"), dict):
                    meta = {**u["meta"], **meta}
                units.append(
                    {
                        "atom_id": str(u.get("atom_id", u.get("unit_id", idx))),
                        "text": text,
                        "type": u.get("type", "other"),
                        "level": u.get("level", 0),
                        "page": u.get("page"),
                        "meta": meta,
                    }
                )
        return units

    fallback_text = (
        obj.get("text")
        or obj.get("content")
        or obj.get("raw_text")
        or obj.get("value")
        or None
    )
    if fallback_text:
        pieces = split_text_fallback(str(fallback_text))
        for idx, piece in enumerate(pieces):
            units.append(
                {
                    "atom_id": str(idx),
                    "text": piece,
                    "type": "paragraph",
                    "level": 0,
                    "page": None,
                    "meta": {},
                }
            )
    return units


def extract_doc_name(obj: Dict[str, Any], meta_from_name: Dict[str, str], path: Path) -> str:
    for key in ["doc_name", "title", "name", "doc_title"]:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return normalize_text(val)
    if meta_from_name.get("orig_stem") and meta_from_name["orig_stem"] != "unknown":
        return meta_from_name["orig_stem"]
    return path.stem


def extract_language(obj: Dict[str, Any], meta_from_name: Dict[str, str]) -> str:
    val = obj.get("language")
    if isinstance(val, str) and val.strip():
        return val.strip().lower()
    if meta_from_name.get("language"):
        return meta_from_name["language"].lower()
    return "unknown"


def extract_doc_id(obj: Dict[str, Any], meta_from_name: Dict[str, str], path: Path, rec_idx: int) -> str:
    val = obj.get("doc_id")
    if isinstance(val, str) and val.strip():
        return val.strip()
    if meta_from_name.get("doc_id") and meta_from_name["doc_id"] != "unknown":
        return meta_from_name["doc_id"]
    return f"{sanitize_component(path.stem, 80)}__{rec_idx}"


def normalize_doc_record(
    obj: Dict[str, Any],
    split_name: str,
    src_path: Path,
    raw_root: Path,
    rec_idx: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    meta_name = parse_flat_filename(src_path)

    units = extract_units(obj)
    if not units:
        return None, {
            "split": split_name,
            "source_path": str(src_path),
            "reason": "no_units_extracted",
            "record_index": rec_idx,
        }

    doc_id = extract_doc_id(obj, meta_name, src_path, rec_idx)
    doc_name = extract_doc_name(obj, meta_name, src_path)
    language = extract_language(obj, meta_name)
    source_family = meta_name["source_family"]
    domain = meta_name["domain"]
    orig_split = meta_name["orig_split"]

    rel_path = src_path.relative_to(raw_root)

    sample = {
        "sample_id": stable_sample_id(split_name, str(rel_path), doc_id, str(rec_idx)),
        "doc_id": doc_id,
        "doc_name": doc_name,
        "source_family": source_family,
        "source_path": str(src_path),
        "source_rel_path": str(rel_path).replace("\\", "/"),
        "orig_split": orig_split,
        "domain": domain,
        "language": language,
        "num_atoms": len(units),
        "atoms": units,
        "meta": {
            "rebuilt_from": "raw_sources_flat",
            "split": split_name,
            "record_index_in_file": rec_idx,
            "source_file_name": src_path.name,
        },
    }
    return sample, None


# -----------------------------
# Scan raw_sources_flat
# -----------------------------


def iter_split_files(raw_root: Path, split_name: str) -> Iterable[Path]:
    split_dir = raw_root / split_name
    if not split_dir.exists():
        return []
    for ext in ("*.json", "*.jsonl"):
        yield from split_dir.rglob(ext)


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild canonical real_dataset from raw_sources_flat.")
    ap.add_argument(
        "--raw_root",
        type=str,
        required=True,
        help=r'Path to raw_sources_flat, e.g. D:\code\Github\SLAC-test\SLAC\refiner\data\real_dataset\raw_sources_flat',
    )
    ap.add_argument(
        "--output_root",
        type=str,
        required=True,
        help=r'Output root, e.g. D:\code\Github\SLAC-test\SLAC\refiner\data\real_dataset_canonical',
    )
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "stats").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    all_docs_writer = JsonlWriter(output_root / "all_docs.jsonl")
    split_writers = {
        "train": JsonlWriter(output_root / "train.jsonl"),
        "dev": JsonlWriter(output_root / "dev.jsonl"),
        "test": JsonlWriter(output_root / "test.jsonl"),
    }
    failed_writer = JsonlWriter(output_root / "logs" / "failed_records.jsonl")
    skipped_writer = JsonlWriter(output_root / "logs" / "skipped_files.jsonl")

    summary: Dict[str, Any] = {
        "input": {
            "raw_root": str(raw_root),
            "output_root": str(output_root),
        },
        "total_docs": 0,
        "split_counts": {},
        "source_family_counts": {},
        "domain_counts": {},
        "language_counts": {},
        "length": {},
    }

    global_lengths: List[int] = []
    global_source_family: Dict[str, int] = {}
    global_domain: Dict[str, int] = {}
    global_language: Dict[str, int] = {}

    for split_name in ["train", "dev", "test"]:
        files = list(iter_split_files(raw_root, split_name))
        if not files:
            skipped_writer.write(
                {
                    "split": split_name,
                    "reason": "no_files_found_under_split",
                    "path": str(raw_root / split_name),
                }
            )

        emitted = 0
        failure_counts: Dict[str, int] = {}
        lengths: List[int] = []
        sf_counts: Dict[str, int] = {}
        dom_counts: Dict[str, int] = {}
        lang_counts: Dict[str, int] = {}

        for src_path in files:
            try:
                had_doc = False
                for rec_idx, obj in iter_docs_from_file(src_path):
                    had_doc = True
                    sample, failure = normalize_doc_record(
                        obj=obj,
                        split_name=split_name,
                        src_path=src_path,
                        raw_root=raw_root,
                        rec_idx=rec_idx,
                    )
                    if failure is not None:
                        failure_counts[failure["reason"]] = failure_counts.get(failure["reason"], 0) + 1
                        failed_writer.write(failure)
                        continue

                    emitted += 1
                    split_writers[split_name].write(sample)
                    all_docs_writer.write(sample)

                    n = int(sample["num_atoms"])
                    lengths.append(n)
                    global_lengths.append(n)

                    sf = sample["source_family"]
                    dm = sample["domain"]
                    lg = sample["language"]

                    sf_counts[sf] = sf_counts.get(sf, 0) + 1
                    dom_counts[dm] = dom_counts.get(dm, 0) + 1
                    lang_counts[lg] = lang_counts.get(lg, 0) + 1

                    global_source_family[sf] = global_source_family.get(sf, 0) + 1
                    global_domain[dm] = global_domain.get(dm, 0) + 1
                    global_language[lg] = global_language.get(lg, 0) + 1

                if not had_doc:
                    skipped_writer.write(
                        {
                            "split": split_name,
                            "reason": "file_contains_no_valid_json_docs",
                            "path": str(src_path),
                        }
                    )
            except Exception as e:
                failed_writer.write(
                    {
                        "split": split_name,
                        "reason": "exception_during_file_parse",
                        "source_path": str(src_path),
                        "error": repr(e),
                    }
                )
                failure_counts["exception_during_file_parse"] = failure_counts.get("exception_during_file_parse", 0) + 1

        summary["split_counts"][split_name] = {
            "emitted_docs": emitted,
            "failure_counts": dict(sorted(failure_counts.items())),
            "source_family_counts": dict(sorted(sf_counts.items())),
            "domain_counts": dict(sorted(dom_counts.items())),
            "language_counts": dict(sorted(lang_counts.items())),
            "length": summarize_lengths(lengths),
        }

    all_docs_writer.close()
    for w in split_writers.values():
        w.close()
    failed_writer.close()
    skipped_writer.close()

    summary["total_docs"] = sum(v["emitted_docs"] for v in summary["split_counts"].values())
    summary["source_family_counts"] = dict(sorted(global_source_family.items()))
    summary["domain_counts"] = dict(sorted(global_domain.items()))
    summary["language_counts"] = dict(sorted(global_language.items()))
    summary["length"] = summarize_lengths(global_lengths)

    with (output_root / "stats" / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()