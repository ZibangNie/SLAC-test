"""Freeze metadata-only Qasper validation candidates, never a cleared evaluation set.

Reads only canonical train/validation, their metadata ledgers, and explicitly
named legacy train/dev JSONL. Raw archives, QA gold and credentials are not read.
The output directory must not exist. Missing canonical QA remains a blocker.
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
import unicodedata

SELECTION_SALT = "SLAC-QASPER-CANDIDATE-v1"


def digest_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip()


def text_digests(text):
    normal = normalized(text)
    return {
        "normalized": hashlib.sha256(normal.encode("utf-8")).hexdigest(),
        "without_whitespace": hashlib.sha256(re.sub(r"\s+", "", normal).encode("utf-8")).hexdigest(),
    }


def jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield line_number, json.loads(line)
            except (ValueError, UnicodeError):
                raise ValueError(f"invalid JSON at line {line_number}") from None


def legacy_input(path, split):
    if any(char in str(path) for char in "*?[]"):
        raise ValueError("legacy inputs must be explicit files")
    path = Path(path).resolve(strict=True)
    tokens = re.split(r"[^a-z0-9]+", path.stem.casefold())
    if split not in tokens or "test" in tokens or path.suffix != ".jsonl":
        raise ValueError("legacy inputs must identify only the requested train/dev split")
    return path


def is_heading_atom(atom):
    # Do not infer headings from content. Only declared heading types are removed.
    return isinstance(atom, dict) and str(atom.get("type", "")).casefold() in {
        "heading", "title", "section_heading", "h1", "h2", "h3", "h4", "h5", "h6"
    }


def scan_legacy(paths):
    hashes = {kind: set() for kind in ("normalized", "without_whitespace")}
    ids = set()
    rows = {}
    for split, path in paths.items():
        count = 0
        for _, row in jsonl(path):
            if row.get("meta", {}).get("split", split) != split:
                raise ValueError("legacy row split disagrees with named input")
            atoms = row.get("atoms")
            if not isinstance(atoms, list):
                raise ValueError("legacy row is missing atoms")
            all_text, body_text = [], []
            for atom in atoms:
                text = atom.get("text") if isinstance(atom, dict) else atom
                if not isinstance(text, str):
                    raise ValueError("legacy atom is missing text")
                all_text.append(text)
                if not is_heading_atom(atom):
                    body_text.append(text)
            for texts in (all_text, body_text):
                for kind, value in text_digests("\n".join(texts)).items():
                    hashes[kind].add(value)
            ids.add(str(row.get("doc_id", "")))
            count += 1
        rows[split] = count
    return hashes, ids, rows


def freeze(handoff, train_input, dev_input, output_dir, max_docs=32):
    if not 1 <= max_docs <= 32:
        raise ValueError("max_docs must be between 1 and 32")
    handoff = Path(handoff).resolve(strict=True)
    data = handoff / "SLAC-datasets-v2"
    canonical_dir = data / "processed/v1/qasper"
    index_path = canonical_dir / "index.jsonl.gz"
    reserve_path = data / "processed/boundary-pilot-v1/development_reservations.json"
    exposure_path = handoff / "SLAC-review-20260924/phase1/public_pilot/public_pilot_manifest.jsonl"
    old_paths = {"train": legacy_input(train_input, "train"), "dev": legacy_input(dev_input, "dev")}
    if old_paths["train"] == old_paths["dev"]:
        raise ValueError("legacy inputs must differ")
    indexes = [row for _, row in jsonl(index_path)]
    if len({row["doc_id"] for row in indexes}) != len(indexes):
        raise ValueError("duplicate canonical index doc_id")
    # Refuse a mixed shard before opening its canonical payload.
    if any(row.get("source") != "qasper" or row.get("original_split") not in {"train", "validation"} for row in indexes):
        raise ValueError("index includes unsupported split/source; canonical payload not opened")
    shard_names = sorted({row["shard"] for row in indexes})
    if any(not re.fullmatch(r"documents-\d{5}\.jsonl\.gz", name) for name in shard_names):
        raise ValueError("invalid shard basename in canonical index")
    shard_paths = [canonical_dir / name for name in shard_names]
    inputs = [index_path, reserve_path, exposure_path, *old_paths.values(), *shard_paths]
    before_hashes = {str(path): digest_file(path) for path in inputs}
    reservations = json.loads(reserve_path.read_text(encoding="utf-8"))
    exposed_sources = {str(row["paper_id"]) for _, row in jsonl(exposure_path)}
    reserved_ids = set(reservations["doc_ids"])
    reserved_families = set(reservations["family_ids"])
    reserved_hashes = set(reservations["normalized_body_sha256"])
    known = {row["doc_id"]: row for row in indexes}
    for row in indexes:
        if row.get("development_exposed") or row["source_id"] in exposed_sources:
            reserved_ids.add(row["doc_id"])
            reserved_families.add(row["family_id"])
            reserved_hashes.add(row["normalized_body_sha256"])
    legacy_hashes, legacy_ids, old_rows = scan_legacy(old_paths)
    eligible, excluded = [], []
    counts = Counter()
    seen = set()
    for shard in shard_paths:
        for line_number, row in jsonl(shard):
            doc_id = row.get("doc_id")
            if doc_id not in known or doc_id in seen:
                raise ValueError("canonical/index identity mismatch")
            seen.add(doc_id)
            index = known[doc_id]
            if row.get("original_split") != index["original_split"] or index["shard"] != shard.name or index["row_in_shard"] != line_number - 1:
                raise ValueError("canonical/index split or row-location mismatch")
            if row["original_split"] not in {"train", "validation"}:
                raise ValueError("unexpected canonical split")
            text = row["canonical_text"]
            spans = [block["char_span"] for block in row["blocks"] if block["kind"] not in {"heading", "raw_wikitext_heading"}]
            if any(type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(text) for start, end in spans):
                raise ValueError("invalid canonical body span")
            body = "\n".join(text[start:end] for start, end in spans)
            body_hashes, full_hashes = text_digests(body), text_digests(text)
            if body_hashes["normalized"] != index["normalized_body_sha256"]:
                raise ValueError("canonical normalized body hash disagrees with index")
            counts[f"canonical_{row['original_split']}"] += 1
            extra = row.get("extra", {})
            qa_present = isinstance(row.get("qas"), list) and bool(row["qas"])
            counts["qa_field_present"] += int(qa_present)
            counts["qa_not_exported_metadata"] += int(extra.get("qa_exported") is False)
            reasons = []
            if row["original_split"] != "validation":
                reasons.append("not_official_validation")
            if doc_id in reserved_ids or row["source_id"] in exposed_sources or row.get("family_id") in reserved_families or body_hashes["normalized"] in reserved_hashes or extra.get("development_exposed"):
                reasons.append("known_development_exposure_or_family")
            exact_overlap = any(value in legacy_hashes[kind] for hashes in (body_hashes, full_hashes) for kind, value in hashes.items())
            if exact_overlap or doc_id in legacy_ids or row["source_id"] in legacy_ids:
                reasons.append("legacy_train_dev_exact_overlap")
            if not normalized(body):
                reasons.append("empty_body")
            metadata = {
                "doc_id": doc_id,
                "source_id": row["source_id"],
                "family_id": row["family_id"],
                "official_split": row["original_split"],
                "normalized_body_sha256": body_hashes["normalized"],
                "compact_body_sha256": body_hashes["without_whitespace"],
                "canonical_shard": shard.name,
                "row_in_shard": line_number - 1,
                "body_characters": index["body_characters"],
                "qa_field_present": qa_present,
                "qa_exported": extra.get("qa_exported"),
                "skipped_source_fields": extra.get("skipped_source_fields_without_value_decoding", []),
                "evidence_alignment_ready": False,
                "legacy_exact_overlap_detected": exact_overlap,
                "development_exposure_excluded": not bool(set(reasons) & {"known_development_exposure_or_family"}),
            }
            if reasons:
                excluded.append({"doc_id": doc_id, "reasons": reasons})
                counts.update(f"excluded_{reason}" for reason in reasons)
            else:
                metadata["selection_hash"] = hashlib.sha256(f"{SELECTION_SALT}|{row['family_id']}".encode()).hexdigest()
                eligible.append(metadata)
    if seen != set(known):
        raise ValueError("canonical/index document count mismatch")
    selected, selected_families, selected_bodies = [], set(), set()
    for row in sorted(eligible, key=lambda item: (item["selection_hash"], item["doc_id"])):
        if row["family_id"] in selected_families or row["normalized_body_sha256"] in selected_bodies:
            continue
        selected.append(row)
        selected_families.add(row["family_id"])
        selected_bodies.add(row["normalized_body_sha256"])
        if len(selected) == max_docs:
            break
    for path in inputs:
        if digest_file(path) != before_hashes[str(path)]:
            raise ValueError("an input changed during the audit")
    manifest = {
        "schema": "slac-qasper-candidate-pool-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_metadata_only_missing_qa",
        "selection_salt": SELECTION_SALT,
        "selection_policy": "Official validation only; exclude known exposure/family/body hashes and legacy exact overlaps; sort fixed family hash; unique family and body; take at most 32.",
        "selected_documents": len(selected),
        "eligible_metadata_documents": len(eligible),
        "max_documents": max_docs,
        "source_split": "validation",
        "test_payload_read": False,
        "raw_archives_opened": False,
        "api_calls": 0,
        "cleared_for_evaluation": False,
        "independent_confirmation_test": False,
        "near_duplicate_clearance": False,
        "all_input_hashes_unchanged": True,
        "input_sha256": before_hashes,
        "known_exposed_qasper_source_ids": sorted(exposed_sources),
        "legacy_rows_checked": old_rows,
        "legacy_exact_hash_basis": "NFC casefold, whitespace collapsed and separately removed; compare both Qasper body/full canonical text against legacy all-atoms and declared-heading-excluded atom concatenations. Representation differences remain possible.",
        "gates_remaining": [
            "Canonical Qasper intentionally omits qas: export original non-test QA/answers/evidence with source lineage before evaluation.",
            "Validate evidence spans, multiple answers, unanswerable and figure/table handling; do not fabricate missing gold.",
            "Complete near-duplicate, paper-version and family review; zero exact overlap does not establish independence.",
            "Freeze task/protocol and evaluator version; reserve this pool from any training or prompt examples.",
        ],
        "counts": dict(counts),
        "candidate_manifest_sha256": hashlib.sha256(b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8") for row in selected)).hexdigest(),
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "candidates.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    with (destination / "pool_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with (destination / "eligibility_audit.json").open("x", encoding="utf-8") as handle:
        json.dump({"excluded": excluded, "eligible_ids": [row["doc_id"] for row in eligible]}, handle, ensure_ascii=False, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--legacy-train", required=True)
    parser.add_argument("--legacy-dev", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-docs", type=int, default=32)
    args = parser.parse_args()
    result = freeze(args.handoff, args.legacy_train, args.legacy_dev, args.output_dir, args.max_docs)
    print(json.dumps({key: result[key] for key in ("status", "selected_documents", "eligible_metadata_documents", "legacy_rows_checked", "counts", "all_input_hashes_unchanged")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
