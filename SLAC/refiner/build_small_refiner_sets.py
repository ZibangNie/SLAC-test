from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LENGTH_BINS = [2000, 8000, 20000]
JSON_GLOBS = ["*.json", "*.tree.json"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build small file-based train/dev collections for Boundary Refiner."
    )
    p.add_argument("--structure_root", type=str, required=True)
    p.add_argument("--index_path", type=str, required=True)
    p.add_argument("--rail_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--train_size", type=int, default=40)
    p.add_argument("--dev_size", type=int, default=10)
    p.add_argument("--rail_size", type=int, default=20)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--copy_mode", type=str, default="copy", choices=["copy", "hardlink"])
    return p.parse_args()


def read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


_ws_re = re.compile(r"\s+")


def norm_text(x: str) -> str:
    x = str(x or "")
    x = _ws_re.sub(" ", x).strip()
    return x


def valid_units(data: dict) -> List[dict]:
    units = data.get("units")
    if not isinstance(units, list):
        return []
    out = []
    for u in units:
        if not isinstance(u, dict):
            continue
        text = norm_text(u.get("text", ""))
        if not text:
            continue
        out.append(u)
    return out


def detect_language(data: dict, fallback: str = "other") -> str:
    for k in ["language", "lang"]:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return fallback


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_RE = re.compile(r"[A-Za-z]")


def infer_lang_from_units(units: List[dict]) -> str:
    text = " ".join(norm_text(u.get("text", "")) for u in units[:20])
    has_cjk = bool(CJK_RE.search(text))
    has_ascii = bool(ASCII_RE.search(text))
    if has_cjk and not has_ascii:
        return "zh"
    if has_ascii and not has_cjk:
        return "en"
    if has_cjk:
        return "zh"
    if has_ascii:
        return "en"
    return "other"


def total_chars(units: List[dict]) -> int:
    return sum(len(norm_text(u.get("text", ""))) for u in units)


def length_bin(n_chars: int) -> str:
    if n_chars < LENGTH_BINS[0]:
        return "L0"
    if n_chars < LENGTH_BINS[1]:
        return "L1"
    if n_chars < LENGTH_BINS[2]:
        return "L2"
    return "L3"


def rel_parts_split(path: Path) -> Tuple[str, str, str, str]:
    # source_type, dataset_tag, lang, split
    parts = [p.lower() for p in path.parts]
    source_type = "unknown"
    dataset_tag = "unknown"
    lang = "other"
    split = "unknown"

    if parts:
        if parts[0] in {"a", "b"}:
            source_type = parts[0].upper()
    if "laws" in parts:
        dataset_tag = "laws"
    elif "papers" in parts:
        dataset_tag = "papers"
    elif "standards" in parts:
        dataset_tag = "standards"
    elif source_type == "B":
        dataset_tag = "semi_structured"

    for x in ["zh", "en"]:
        if x in parts:
            lang = x
            break

    for x in ["train", "dev", "test"]:
        if x in parts:
            split = x
            break

    return source_type, dataset_tag, lang, split


class Record(dict):
    pass


def build_llm_record(path: Path, structure_root: Path) -> Optional[Record]:
    data = read_json(path)
    if not data:
        return None
    units = valid_units(data)
    if len(units) < 2:
        return None

    try:
        rel = path.relative_to(structure_root)
    except Exception:
        rel = Path(path.name)

    source_type, dataset_tag, rel_lang, split = rel_parts_split(rel)
    lang = detect_language(data, fallback=rel_lang)
    if lang not in {"zh", "en"}:
        lang = infer_lang_from_units(units)

    doc_id = str(data.get("doc_id") or path.stem)
    chars = total_chars(units)

    return Record(
        doc_id=doc_id,
        abs_path=str(path),
        rel_path=str(rel),
        source_type=source_type,
        dataset_tag=dataset_tag,
        language=lang,
        orig_split=split,
        total_chars=chars,
        length_bin=length_bin(chars),
        llm_confidence=1.0,
    )


def discover_llm_records(structure_root: Path, index_path: Path) -> Tuple[List[Record], Dict[str, int]]:
    diag = {
        "index_rows": 0,
        "index_paths_seen": 0,
        "index_records_ok": 0,
        "scan_paths_seen": 0,
        "scan_records_ok": 0,
    }

    records: List[Record] = []
    seen = set()

    # try index first
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                diag["index_rows"] += 1
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                abs_path = item.get("abs_path")
                if not abs_path:
                    continue
                p = Path(abs_path)
                if not p.exists():
                    p = structure_root / str(item.get("rel_path", ""))
                if not p.exists():
                    continue
                diag["index_paths_seen"] += 1
                rec = build_llm_record(p, structure_root)
                if rec is None:
                    continue
                if rec["doc_id"] in seen:
                    continue
                seen.add(rec["doc_id"])
                records.append(rec)
                diag["index_records_ok"] += 1

    # fallback / supplement scan if too few
    if len(records) < 10:
        for g in JSON_GLOBS:
            for p in structure_root.rglob(g):
                if not p.is_file():
                    continue
                if "index" in [x.lower() for x in p.parts]:
                    continue
                diag["scan_paths_seen"] += 1
                rec = build_llm_record(p, structure_root)
                if rec is None:
                    continue
                if rec["doc_id"] in seen:
                    continue
                seen.add(rec["doc_id"])
                records.append(rec)
                diag["scan_records_ok"] += 1

    return records, diag


def build_rail_record(path: Path, rail_root: Path) -> Optional[Record]:
    data = read_json(path)
    if not data:
        return None
    units = valid_units(data)
    if len(units) < 2:
        return None

    lang = detect_language(data)
    if lang not in {"zh", "en"}:
        lang = infer_lang_from_units(units)

    try:
        rel = path.relative_to(rail_root)
    except Exception:
        rel = Path(path.name)

    doc_id = str(data.get("doc_id") or path.stem)
    chars = total_chars(units)

    return Record(
        doc_id=doc_id,
        abs_path=str(path),
        rel_path=str(rel),
        source_type="rail_hardcoded",
        dataset_tag="rail_hardcoded",
        language=lang,
        orig_split="unknown",
        total_chars=chars,
        length_bin=length_bin(chars),
        llm_confidence=0.35,
    )


def discover_rail_records(rail_root: Path) -> List[Record]:
    records: List[Record] = []
    seen = set()
    for g in JSON_GLOBS:
        for p in rail_root.rglob(g):
            if not p.is_file():
                continue
            rec = build_rail_record(p, rail_root)
            if rec is None:
                continue
            if rec["doc_id"] in seen:
                continue
            seen.add(rec["doc_id"])
            records.append(rec)
    return records


def stratified_sample(records: List[Record], target_n: int, rng: random.Random) -> List[Record]:
    if target_n <= 0 or not records:
        return []
    if len(records) <= target_n:
        return list(records)

    buckets = defaultdict(list)
    for r in records:
        key = (r["language"], r["source_type"], r["dataset_tag"], r["length_bin"])
        buckets[key].append(r)

    for k in buckets:
        rng.shuffle(buckets[k])

    selected: List[Record] = []
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)

    # round-robin stratified fill
    while len(selected) < target_n:
        progressed = False
        for k in bucket_keys:
            if buckets[k]:
                selected.append(buckets[k].pop())
                progressed = True
                if len(selected) >= target_n:
                    break
        if not progressed:
            break

    if len(selected) < target_n:
        remaining = []
        for v in buckets.values():
            remaining.extend(v)
        rng.shuffle(remaining)
        selected.extend(remaining[: target_n - len(selected)])

    return selected[:target_n]


def summarize(records: List[Record]) -> Dict:
    return {
        "count": len(records),
        "language": dict(Counter(r["language"] for r in records)),
        "dataset_tag": dict(Counter(r["dataset_tag"] for r in records)),
        "length_bin": dict(Counter(r["length_bin"] for r in records)),
        "orig_split": dict(Counter(r["orig_split"] for r in records)),
    }


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            if dst.exists():
                dst.unlink()
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def write_collection(records: List[Record], split_dir: Path, mode: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for r in records:
            src = Path(r["abs_path"])
            name = f"{r['doc_id']}{src.suffix}"
            dst = split_dir / name
            # avoid odd double suffix names like .tree.json vs .json collisions
            if dst.exists():
                stem = src.name.replace(".json", "").replace(".tree", "")
                dst = split_dir / f"{r['doc_id']}__{stem}.json"
            link_or_copy(src, dst, mode)
            out = dict(r)
            out["copied_to"] = str(dst)
            mf.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    structure_root = Path(args.structure_root)
    index_path = Path(args.index_path)
    rail_root = Path(args.rail_root)
    out_dir = Path(args.out_dir)

    llm_records, diag = discover_llm_records(structure_root, index_path)
    rail_records = discover_rail_records(rail_root)

    llm_train_pool = [r for r in llm_records if r["orig_split"] == "train"]
    llm_dev_pool = [r for r in llm_records if r["orig_split"] == "dev"]
    llm_test_pool = [r for r in llm_records if r["orig_split"] == "test"]

    # if dev insufficient, supplement from test; if still insufficient, supplement from train leftovers
    dev_candidates = list(llm_dev_pool) + list(llm_test_pool)
    train_selected = stratified_sample(llm_train_pool, args.train_size, rng)
    selected_ids = {r["doc_id"] for r in train_selected}
    dev_candidates = [r for r in dev_candidates if r["doc_id"] not in selected_ids]
    if len(dev_candidates) < args.dev_size:
        train_leftovers = [r for r in llm_train_pool if r["doc_id"] not in selected_ids]
        dev_candidates.extend(train_leftovers)
    dev_selected = stratified_sample(dev_candidates, args.dev_size, rng)

    rail_selected = stratified_sample(rail_records, args.rail_size, rng)

    # directory layout
    ensure_empty_dir(out_dir)
    llm_root = out_dir / "llm_gold"
    rail_out = out_dir / "rail_pool"
    write_collection(train_selected, llm_root / "train", args.copy_mode)
    write_collection(dev_selected, llm_root / "dev", args.copy_mode)
    if args.rail_size > 0:
        write_collection(rail_selected, rail_out, args.copy_mode)

    summary = {
        "diagnostics": {
            "llm_discovery": "index+fallback_scan",
            **diag,
            "llm_records_total": len(llm_records),
            "llm_train_pool": len(llm_train_pool),
            "llm_dev_pool": len(llm_dev_pool),
            "llm_test_pool": len(llm_test_pool),
            "rail_records_total": len(rail_records),
        },
        "train": summarize(train_selected),
        "dev": summarize(dev_selected),
        "notes": {
            "format": "file_collection",
            "sampling": "priority on LLM-annotated data; split-aware; length-stratified; language/source balanced as much as possible",
            "llm_confidence": 1.0,
            "rail_confidence": 0.35,
        },
    }
    if args.rail_size > 0:
        summary["rail_pool"] = summarize(rail_selected)

    with (out_dir / "selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Build small refiner file collections done ===")
    print(f"out_dir: {out_dir}")
    print(f"LLM records total: {len(llm_records)}")
    print(f"LLM pools: train={len(llm_train_pool)}, dev={len(llm_dev_pool)}, test={len(llm_test_pool)}")
    print(f"Selected train={len(train_selected)}, dev={len(dev_selected)}")
    print(f"Selected rail_pool={len(rail_selected)}")
    print(f"Summary -> {out_dir / 'selection_summary.json'}")


if __name__ == "__main__":
    main()
