"""Bounded, metadata-only lexical overlap screen of a frozen non-test Qasper pool.

This reports review candidates, not a leakage verdict or independent-evaluation
clearance. It does not change the pool or read QA, raw archives, test or credentials.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata

SHINGLE_SIZE = 5
MAX_DOCUMENT_CHARACTERS = 2_000_000
MAX_QUERY_SHINGLES = 500_000
WORD = re.compile(r"\w+", re.UNICODE)
ARXIV = re.compile(r"(?<![\w.])(\d{4}\.\d{4,5}|[a-z][a-z.\-]+/\d{7})(?:v\d+)?(?!\d)", re.I)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def jsonl(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                yield number, json.loads(line)
            except (ValueError, UnicodeError):
                raise ValueError(f"invalid input JSON at line {number}") from None


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def shingles(text):
    """Exact Unicode word 5-gram strings, with no hash collisions or sketch sampling."""
    words = WORD.findall(unicodedata.normalize("NFC", text).casefold())
    if not words:
        return set()
    if len(words) < SHINGLE_SIZE:
        return {" ".join(words)}
    return {" ".join(words[i:i + SHINGLE_SIZE]) for i in range(len(words) - SHINGLE_SIZE + 1)}


def body(row):
    text = row["canonical_text"]
    spans = [b["char_span"] for b in row["blocks"] if b["kind"] not in {"heading", "raw_wikitext_heading"}]
    if any(type(s) is not int or type(e) is not int or not 0 <= s <= e <= len(text) for s, e in spans):
        raise ValueError("invalid canonical span")
    return "\n".join(text[s:e] for s, e in spans)


def legacy_body(row):
    values = []
    for atom in row["atoms"]:
        if isinstance(atom, dict):
            if atom.get("type", "").casefold() in {"heading", "title", "section_heading", "h1", "h2", "h3", "h4", "h5", "h6"}:
                continue
            atom = atom["text"]
        if not isinstance(atom, str):
            raise ValueError("legacy atom text is invalid")
        values.append(atom)
    return "\n".join(values)


def identities(row):
    fields = ("doc_id", "source_id", "family_id", "doc_name", "source_rel_path", "source_path")
    paper_ids = {match.casefold() for key in fields for match in ARXIV.findall(str(row.get(key, "")))}
    family = row.get("family_id")
    # Legacy source_family is a corpus/domain, not a paper identity.
    return str(family) if family else None, paper_ids


def overlap(shared, query_size, reference_size):
    union = query_size + reference_size - shared
    return {"shared_shingles": shared, "query_shingles": query_size,
            "reference_shingles": reference_size,
            "jaccard": shared / union if union else 0.0,
            "query_containment": shared / query_size if query_size else 0.0,
            "reference_containment": shared / reference_size if reference_size else 0.0}


def lexical_flags(metrics):
    shared = metrics["shared_shingles"]
    maximum = max(metrics["query_containment"], metrics["reference_containment"])
    if shared >= 100 and (metrics["jaccard"] >= 0.8 or maximum >= 0.9):
        return ["high_lexical_overlap_review"]
    if shared >= 25 and (metrics["jaccard"] >= 0.1 or maximum >= 0.3):
        return ["moderate_lexical_overlap_review"]
    return []


def input_paths(manifest_path, manifest):
    """Only open named permitted input roles from the earlier frozen manifest."""
    recorded = manifest["input_sha256"]
    allowed = {}
    for raw, expected in recorded.items():
        path = Path(raw)
        if path.name in {"index.jsonl.gz", "refiner_train.jsonl", "refiner_dev.jsonl"} or re.fullmatch(r"documents-\d{5}\.jsonl\.gz", path.name):
            allowed[path.resolve(strict=True)] = expected
    indexes = [p for p in allowed if p.name == "index.jsonl.gz"]
    if len(indexes) != 1:
        raise ValueError("exactly one canonical index is required")
    canonical = indexes[0].parent
    shards = sorted(p for p in allowed if p.name.startswith("documents-"))
    if not shards or any(p.parent != canonical for p in shards):
        raise ValueError("canonical shards must be local to their index")
    legacy = {split: [p for p in allowed if p.name == f"refiner_{split}.jsonl"] for split in ("train", "dev")}
    if any(len(paths) != 1 for paths in legacy.values()):
        raise ValueError("explicit legacy train and dev are required")
    candidate = manifest_path.parent / "candidates.jsonl"
    return allowed, indexes[0], shards, {split: paths[0] for split, paths in legacy.items()}, candidate


def screen(pool_manifest, output_dir, max_seconds=300):
    if not 1 <= max_seconds <= 300:
        raise ValueError("runtime cap must be within 1..300 seconds")
    started = time.monotonic()
    deadline = started + max_seconds
    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError("screen runtime cap reached")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(pool_manifest).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths, index_path, shards, legacy, candidate_path = input_paths(manifest_path, manifest)
    report = {
        "schema": "slac-qasper-overlap-screen-v1", "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_calls": 0, "test_payload_read": False, "qa_payload_read": False,
        "pool_changed": False, "cleared_for_evaluation": False,
        "independent_confirmation_test": False, "near_duplicate_clearance": False,
        "method": "Exact unique Unicode-word 5-shingle inverted index over queries; all reference shingles compared without sampling. NFC and casefold; punctuation ignored; declared headings excluded.",
        "thresholds": {"high": "shared>=100 and (Jaccard>=0.8 or either containment>=0.9)",
                       "moderate": "shared>=25 and (Jaccard>=0.1 or either containment>=0.3)",
                       "identity": "exact declared family_id or arXiv base ID found in identity/path fields; version suffix ignored; legacy source_family excluded"},
        "limits": [
            "Flags are review candidates, not automatic leakage or same-paper verdicts; related papers may share definitions, methods or boilerplate.",
            "Lexical screening cannot exclude translations, paraphrases, OCR changes or unrecorded source families.",
            "The scope is only the frozen candidates, canonical Qasper train/validation and named legacy train/dev; test and other corpora are not inspected.",
            "No candidate is removed or replaced. A zero-flag result is not independent-evaluation clearance.",
            "Five-word sets ignore multiplicity, punctuation and sentence boundaries; below-threshold overlap is summarized by nearest neighbors, not declared absent.",
        ],
        "resource_bounds": {"max_seconds": max_seconds, "max_document_characters": MAX_DOCUMENT_CHARACTERS, "max_query_union_shingles": MAX_QUERY_SHINGLES},
        "input_sha256": {}, "source_sha256": digest(Path(__file__)),
    }
    counts, flagged, nearest, skipped = Counter(), [], {}, []
    completed = []
    input_hashes = report["input_sha256"]
    try:
        # Validate the metadata index before even hashing a canonical payload.
        # A mixed test shard is refused without opening its bytes.
        if digest(index_path) != paths[index_path]:
            raise ValueError("canonical index hash differs from frozen manifest")
        indexes = [r for _, r in jsonl(index_path)]
        known = {r["doc_id"]: r for r in indexes}
        if len(known) != len(indexes) or any(r.get("source") != "qasper" or r.get("original_split") not in {"train", "validation"} for r in indexes):
            raise ValueError("index contains duplicate or unsupported identities/splits; payload not opened")
        if {r["shard"] for r in indexes} != {p.name for p in shards}:
            raise ValueError("index/shard set mismatch")
        for path, expected in paths.items():
            check_time()
            actual = digest(path)
            if actual != expected:
                raise ValueError("source hash differs from frozen pool manifest")
            input_hashes[str(path)] = actual
        for path in (manifest_path, candidate_path):
            input_hashes[str(path)] = digest(path)
        if input_hashes[str(candidate_path)] != manifest["candidate_manifest_sha256"]:
            raise ValueError("candidate file differs from frozen manifest")
        candidates = [r for _, r in jsonl(candidate_path)]
        chosen = {r["doc_id"]: r for r in candidates}
        if not 1 <= len(chosen) <= 32 or len(chosen) != len(candidates) or len(chosen) != manifest["selected_documents"]:
            raise ValueError("invalid frozen candidate count or identities")
        for doc_id, candidate in chosen.items():
            row = known.get(doc_id, {})
            if row.get("original_split") != "validation" or candidate.get("official_split") != "validation" or any(candidate.get(k) != row.get(k) for k in ("family_id", "source_id", "normalized_body_sha256")):
                raise ValueError("candidate identity does not match validation index")
        queries, seen = [], set()
        for path in shards:
            for number, row in jsonl(path):
                check_time()
                doc_id = row.get("doc_id")
                index = known.get(doc_id, {})
                if doc_id in seen or not index or row.get("original_split") != index["original_split"] or index["shard"] != path.name or index["row_in_shard"] != number - 1:
                    raise ValueError("canonical row identity/location mismatch")
                seen.add(doc_id)
                if doc_id in chosen:
                    text = body(row)
                    if len(text) > MAX_DOCUMENT_CHARACTERS:
                        raise ValueError("query exceeds complete-document resource bound")
                    if hashlib.sha256(normalized(text).encode()).hexdigest() != chosen[doc_id]["normalized_body_sha256"]:
                        raise ValueError("candidate body hash mismatch")
                    family, paper_ids = identities(row)
                    queries.append({"doc_id": doc_id, "family": family, "paper_ids": paper_ids, "shingles": shingles(text)})
        if seen != set(known) or len(queries) != len(chosen):
            raise ValueError("incomplete canonical/query loading")
        queries.sort(key=lambda r: r["doc_id"])
        query_number = {r["doc_id"]: i for i, r in enumerate(queries)}
        inverted = {}
        for i, query in enumerate(queries):
            nearest[query["doc_id"]] = []
            for shingle in query["shingles"]:
                inverted[shingle] = inverted.get(shingle, 0) | (1 << i)
            if len(inverted) > MAX_QUERY_SHINGLES:
                raise ValueError("query shingles exceed memory bound")
        union = set(inverted)
        counts["query_documents"] = len(queries)
        counts["query_union_shingles"] = len(union)

        def compare(row, text, scope, line_number):
            check_time()
            doc_id = str(row["doc_id"])
            if len(text) > MAX_DOCUMENT_CHARACTERS:
                skipped.append({"scope": scope, "doc_id": doc_id, "line": line_number, "reason": "document_character_bound", "characters": len(text)})
                return
            reference = shingles(text)
            intersections = [0] * len(queries)
            for shingle in reference.intersection(union):
                mask = inverted[shingle]
                while mask:
                    lowest = mask & -mask
                    intersections[lowest.bit_length() - 1] += 1
                    mask ^= lowest
            family, paper_ids = identities(row)
            counts[f"{scope}_documents"] += 1
            counts["max_reference_shingles"] = max(counts["max_reference_shingles"], len(reference))
            for i, query in enumerate(queries):
                # Query/self is excluded; candidate pairs are reported once.
                if scope.startswith("qasper_") and doc_id in query_number and i >= query_number[doc_id]:
                    continue
                counts[f"{scope}_pairs"] += 1
                metrics = overlap(intersections[i], len(query["shingles"]), len(reference))
                reasons = lexical_flags(metrics)
                if family and family == query["family"]:
                    reasons.append("declared_family_identity_review")
                if paper_ids & query["paper_ids"]:
                    reasons.append("arxiv_base_identity_review")
                pair = {"query_doc_id": query["doc_id"], "reference_doc_id": doc_id,
                        "reference_scope": scope, "reference_line": line_number,
                        "reference_orig_split": row.get("orig_split", row.get("original_split")),
                        "metrics": metrics, "review_reasons": reasons}
                if reasons:
                    flagged.append(pair)
                best = nearest[query["doc_id"]]
                best.append(pair)
                best.sort(key=lambda p: (-p["metrics"]["jaccard"], -p["metrics"]["query_containment"], p["reference_scope"], p["reference_doc_id"], p["reference_line"]))
                del best[3:]

        for path in shards:
            for number, row in jsonl(path):
                compare(row, body(row), "qasper_" + row["original_split"], number)
        completed.append("canonical_qasper_train_validation_and_candidate_pairs")
        for split, path in legacy.items():
            for number, row in jsonl(path):
                check_time()
                if row.get("meta", {}).get("split", split) != split:
                    raise ValueError("legacy named split disagrees with row")
                counts[f"legacy_{split}_orig_split_{row.get('orig_split', 'missing')}"] += 1
                compare(row, legacy_body(row), "legacy_" + split, number)
            completed.append("legacy_" + split)
        report["status"] = "complete_lexical_screen_with_limits" if not skipped else "partial_document_size_limit"
    except TimeoutError:
        report["status"] = "partial_runtime_limit"
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        raise
    finally:
        # Hash verification is mandatory even for bounded/failed runs, outside the
        # scan cap. It reads bytes only and its time is separately visible.
        scan_elapsed = time.monotonic() - started
        report["all_input_hashes_unchanged"] = all(digest(Path(path)) == expected for path, expected in input_hashes.items()) if input_hashes else None
        if report["all_input_hashes_unchanged"] is False:
            report["status"] = "failed_source_changed"
        report.update(counts=dict(counts), completed_scopes=completed, skipped_documents=skipped,
                      flagged_pairs=len(flagged), flagged_reason_counts=dict(Counter(reason for pair in flagged for reason in pair["review_reasons"])),
                      scan_elapsed_seconds=round(scan_elapsed, 3), elapsed_seconds=round(time.monotonic() - started, 3))
        report["scope"] = {"candidate_pool": str(candidate_path), "canonical_index": str(index_path),
                           "canonical_declared_splits": ["train", "validation"], "legacy_inputs": {k: str(v) for k, v in legacy.items()},
                           "legacy_source_lineage_note": "Only named train/dev files are opened; retained orig_split=test rows inside these historical inputs are counted, not presented as a test evaluation."}
        for name, rows in (("flagged_pairs.jsonl", flagged), ("nearest_pairs.jsonl", [p for key in sorted(nearest) for p in nearest[key]])):
            with (output_dir / name).open("x", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            report.setdefault("output_sha256", {})[name] = digest(output_dir / name)
        with (output_dir / "overlap_report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seconds", type=int, default=300)
    args = parser.parse_args()
    result = screen(args.pool_manifest, args.output_dir, args.max_seconds)
    print(json.dumps({key: result[key] for key in ("status", "counts", "completed_scopes", "flagged_pairs", "flagged_reason_counts", "elapsed_seconds", "all_input_hashes_unchanged")}, ensure_ascii=False))
