#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_sample_id(*parts: str) -> str:
    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"sample_{h}"


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


def iter_docs_from_file(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for idx, obj in enumerate(read_jsonl(path)):
            if isinstance(obj, dict):
                yield idx, obj
        return

    data = load_json(path)
    if isinstance(data, dict):
        yield 0, data
    elif isinstance(data, list):
        for idx, obj in enumerate(data):
            if isinstance(obj, dict):
                yield idx, obj


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


def load_manifest_map(raw_root: Path) -> Dict[str, Dict[str, Any]]:
    manifest_path = raw_root / "file_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    mp: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(manifest_path):
        new_rel = str(rec.get("new_rel_path", "")).replace("\\", "/")
        if new_rel:
            mp[new_rel] = rec
    return mp


def normalize_doc_record(
    obj: Dict[str, Any],
    split_name: str,
    src_path: Path,
    raw_root: Path,
    rec_idx: int,
    manifest_meta: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    units = extract_units(obj)
    if not units:
        return None, {
            "split": split_name,
            "source_path": str(src_path),
            "reason": "no_units_extracted",
            "record_index": rec_idx,
        }

    doc_id = obj.get("doc_id") or manifest_meta.get("doc_id") or f"{src_path.stem}__{rec_idx}"
    doc_name = (
        obj.get("doc_name")
        or obj.get("title")
        or obj.get("name")
        or manifest_meta.get("orig_stem")
        or src_path.stem
    )
    language = (obj.get("language") or manifest_meta.get("language") or "unknown").lower()
    source_family = manifest_meta.get("source_family", "unknown")
    domain = manifest_meta.get("domain", "unknown")
    orig_split = manifest_meta.get("orig_split", "unknown")

    rel_path = str(src_path.relative_to(raw_root)).replace("\\", "/")

    sample = {
        "sample_id": stable_sample_id(split_name, rel_path, str(doc_id), str(rec_idx)),
        "doc_id": str(doc_id),
        "doc_name": str(doc_name),
        "source_family": str(source_family),
        "source_path": str(src_path),
        "source_rel_path": rel_path,
        "orig_split": str(orig_split),
        "domain": str(domain),
        "language": str(language),
        "num_atoms": len(units),
        "atoms": units,
        "meta": {
            "rebuilt_from": "raw_sources_flat_short_with_manifest",
            "split": split_name,
            "record_index_in_file": rec_idx,
            "source_file_name": src_path.name,
            "manifest_old_path": manifest_meta.get("old_path"),
            "manifest_old_rel_path": manifest_meta.get("old_rel_path"),
        },
    }
    return sample, None


def iter_split_files(raw_root: Path, split_name: str) -> List[Path]:
    split_dir = raw_root / split_name
    if not split_dir.exists():
        return []
    files: List[Path] = []
    for ext in ("*.json", "*.jsonl"):
        files.extend(split_dir.rglob(ext))
    return sorted(files, key=lambda p: str(p).lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild canonical real_dataset from short raw_sources_flat + manifest.")
    parser.add_argument(
        "--raw_root",
        type=str,
        required=True,
        help=r'Path to short raw source root, e.g. D:\code\Github\SLAC-test\SLAC\refiner\data\real_dataset\raw_sources_flat_short',
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help=r'Output canonical root, e.g. D:\code\Github\SLAC-test\SLAC\refiner\data\real_dataset_canonical_short',
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)

    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root does not exist: {raw_root}")

    manifest_map = load_manifest_map(raw_root)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "stats").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

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

    with JsonlWriter(output_root / "all_docs.jsonl") as all_writer, \
         JsonlWriter(output_root / "train.jsonl") as train_writer, \
         JsonlWriter(output_root / "dev.jsonl") as dev_writer, \
         JsonlWriter(output_root / "test.jsonl") as test_writer, \
         JsonlWriter(output_root / "logs" / "failed_records.jsonl") as failed_writer, \
         JsonlWriter(output_root / "logs" / "skipped_files.jsonl") as skipped_writer:

        split_writer_map = {
            "train": train_writer,
            "dev": dev_writer,
            "test": test_writer,
        }

        for split_name in ["train", "dev", "test"]:
            files = iter_split_files(raw_root, split_name)

            emitted = 0
            failure_counts: Dict[str, int] = {}
            lengths: List[int] = []
            sf_counts: Dict[str, int] = {}
            dom_counts: Dict[str, int] = {}
            lang_counts: Dict[str, int] = {}

            if not files:
                skipped_writer.write(
                    {
                        "split": split_name,
                        "reason": "no_files_found_under_split",
                        "path": str(raw_root / split_name),
                    }
                )

            for src_path in files:
                rel_path = str(src_path.relative_to(raw_root)).replace("\\", "/")
                manifest_meta = manifest_map.get(rel_path)
                if manifest_meta is None:
                    failure = {
                        "split": split_name,
                        "source_path": str(src_path),
                        "reason": "missing_manifest_entry_for_file",
                    }
                    failed_writer.write(failure)
                    failure_counts["missing_manifest_entry_for_file"] = failure_counts.get("missing_manifest_entry_for_file", 0) + 1
                    continue

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
                            manifest_meta=manifest_meta,
                        )
                        if failure is not None:
                            failed_writer.write(failure)
                            failure_counts[failure["reason"]] = failure_counts.get(failure["reason"], 0) + 1
                            continue

                        split_writer_map[split_name].write(sample)
                        all_writer.write(sample)
                        emitted += 1

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
                    failure = {
                        "split": split_name,
                        "source_path": str(src_path),
                        "reason": "exception_during_file_parse",
                        "error": repr(e),
                    }
                    failed_writer.write(failure)
                    failure_counts["exception_during_file_parse"] = failure_counts.get("exception_during_file_parse", 0) + 1

            summary["split_counts"][split_name] = {
                "emitted_docs": emitted,
                "failure_counts": dict(sorted(failure_counts.items())),
                "source_family_counts": dict(sorted(sf_counts.items())),
                "domain_counts": dict(sorted(dom_counts.items())),
                "language_counts": dict(sorted(lang_counts.items())),
                "length": summarize_lengths(lengths),
            }

    summary["total_docs"] = sum(v["emitted_docs"] for v in summary["split_counts"].values())
    summary["source_family_counts"] = dict(sorted(global_source_family.items()))
    summary["domain_counts"] = dict(sorted(global_domain.items()))
    summary["language_counts"] = dict(sorted(global_language.items()))
    summary["length"] = summarize_lengths(global_lengths)

    with (output_root / "stats" / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("REBUILD CANONICAL DATASET DONE")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()