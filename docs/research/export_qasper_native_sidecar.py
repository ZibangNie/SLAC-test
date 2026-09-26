"""Export native validation QA for an existing candidate pool without inventing gold.

Only qasper-dev-v0.3.json is read from the named train/dev archive. Evidence
matches preserve all identical paragraph locations; ambiguous/figure/unmatched
evidence is reported, never silently mapped to the first or dropped.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile
import unicodedata

from freeze_qasper_candidate_pool import digest_file

MEMBER = "qasper-dev-v0.3.json"


def normalize_evidence(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def align_evidence(text, paragraph_locations):
    if text.lstrip().startswith("FLOAT SELECTED"):
        return {"status": "figure_or_table", "paragraph_candidates": []}
    if not normalize_evidence(text):
        return {"status": "empty_evidence", "paragraph_candidates": []}
    matches = paragraph_locations.get(normalize_evidence(text), [])
    return {
        "status": "unique_paragraph" if len(matches) == 1 else "ambiguous_repeated_paragraph" if matches else "unmatched_text",
        "paragraph_candidates": matches,
    }


def export(archive, canonical_shard, pool_dir):
    archive = Path(archive).resolve(strict=True)
    canonical_shard = Path(canonical_shard).resolve(strict=True)
    pool_dir = Path(pool_dir).resolve(strict=True)
    if archive.name != "qasper-train-dev-v0.3.tgz":
        raise ValueError("only the named official train/dev archive is allowed")
    if not re.fullmatch(r"documents-\d{5}\.jsonl\.gz", canonical_shard.name):
        raise ValueError("expected explicit canonical shard")
    sidecar = pool_dir / "native_qa_sidecar.jsonl"
    report_path = pool_dir / "native_qa_alignment.json"
    if sidecar.exists() or report_path.exists():
        raise FileExistsError("sidecar output already exists")
    candidates_path = pool_dir / "candidates.jsonl"
    candidates = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()]
    if not candidates or len(candidates) > 32 or any(row["official_split"] != "validation" for row in candidates):
        raise ValueError("pool must contain 1-32 validation candidates")
    expected = {row["doc_id"]: row for row in candidates}
    if len(expected) != len(candidates):
        raise ValueError("duplicate candidate doc IDs")
    input_hashes = {str(path): digest_file(path) for path in (archive, canonical_shard, candidates_path)}
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(MEMBER)
        if not member.isfile() or member.size > 32 * 1024 * 1024:
            raise ValueError("unexpected native dev member")
        with tar.extractfile(member) as handle:
            payload = handle.read()
    native = json.loads(payload)
    canonical = {}
    with gzip.open(canonical_shard, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["doc_id"] in expected:
                if row["original_split"] != "validation":
                    raise ValueError("candidate canonical split mismatch")
                canonical[row["doc_id"]] = row
    if canonical.keys() != expected.keys():
        raise ValueError("candidate missing from named canonical shard")
    output_rows, counts = [], Counter()
    per_document = []
    question_ids = set()
    for candidate in candidates:
        doc_id, source_id = candidate["doc_id"], candidate["source_id"]
        if source_id not in native:
            raise ValueError("candidate missing from official validation member")
        row, paper = canonical[doc_id], native[source_id]
        locator = row["raw_locator"]
        if locator.get("member") != MEMBER or locator.get("json_pointer") != f"/{source_id}":
            raise ValueError("canonical raw locator disagrees with native source")
        if row.get("extra", {}).get("raw_archive_sha256") != input_hashes[str(archive)]:
            raise ValueError("canonical archive provenance hash mismatch")
        block_map = {block["source_locator"].get("json_pointer"): block for block in row["blocks"]}
        locations = defaultdict(list)
        doc_counts = Counter()
        for section_index, section in enumerate(paper["full_text"]):
            for paragraph_index, text in enumerate(section["paragraphs"]):
                if not text.strip():
                    doc_counts["empty_native_paragraphs"] += 1
                    continue
                pointer = f"/{source_id}/full_text/{section_index}/paragraphs/{paragraph_index}"
                block = block_map.get(pointer)
                if block is None:
                    raise ValueError("nonempty native paragraph missing canonical locator")
                start, end = block["char_span"]
                if normalize_evidence(row["canonical_text"][start:end]) != normalize_evidence(text):
                    raise ValueError("native/canonical paragraph text mismatch")
                locations[normalize_evidence(text)].append({
                    "native_json_pointer": pointer,
                    "canonical_block_id": block["block_id"],
                    "canonical_char_span": [start, end],
                })
                doc_counts["native_paragraphs_validated"] += 1
        for qa in paper["qas"]:
            question_id = qa["question_id"]
            if (doc_id, question_id) in question_ids:
                raise ValueError("duplicate question ID within paper")
            question_ids.add((doc_id, question_id))
            answers = []
            question_all_unique = True
            question_any_unique = False
            for annotation in qa["answers"]:
                answer = annotation["answer"]
                if not isinstance(answer.get("evidence"), list):
                    raise ValueError("answer evidence must be a list")
                alignments = []
                for evidence_index, evidence in enumerate(answer["evidence"]):
                    alignment = {"evidence_index": evidence_index, **align_evidence(evidence, locations)}
                    alignments.append(alignment)
                    doc_counts[f"evidence_{alignment['status']}"] += 1
                statuses = [item["status"] for item in alignments]
                all_unique = bool(statuses) and all(status == "unique_paragraph" for status in statuses)
                question_any_unique |= all_unique
                question_all_unique &= all_unique
                doc_counts["answer_annotations"] += 1
                doc_counts["unanswerable_annotations"] += int(answer.get("unanswerable") is True)
                doc_counts["answer_annotations_no_evidence"] += int(not statuses)
                doc_counts["answer_annotations_all_evidence_unique_text"] += int(all_unique)
                answers.append({
                    "annotation_id": annotation.get("annotation_id"),
                    "native_answer": answer,
                    "evidence_alignment": alignments,
                    "all_evidence_unique_text_paragraphs": all_unique,
                })
            doc_counts["questions"] += 1
            doc_counts["questions_any_annotation_unique_text_evidence"] += int(question_any_unique)
            doc_counts["questions_all_annotations_unique_text_evidence"] += int(bool(answers) and question_all_unique)
            output_rows.append({
                "schema": "slac-qasper-native-qa-sidecar-v1",
                "doc_id": doc_id,
                "source_id": source_id,
                "family_id": candidate["family_id"],
                "official_split": "validation",
                "question_id": question_id,
                "question": qa["question"],
                "answer_annotations": answers,
                "annotation_origin": "Qasper v0.3 original answer providers; no new labels or semantic review",
                "original_archive_sha256": input_hashes[str(archive)],
                "original_member": MEMBER,
                "native_member_sha256": hashlib.sha256(payload).hexdigest(),
                "canonical_shard_sha256": input_hashes[str(canonical_shard)],
                "canonical_body_sha256": candidate["normalized_body_sha256"],
                "cleared_for_evaluation": False,
                "independent_confirmation_test": False,
            })
        counts.update(doc_counts)
        per_document.append({"doc_id": doc_id, "counts": dict(doc_counts)})
    for path_string, expected_hash in input_hashes.items():
        if digest_file(Path(path_string)) != expected_hash:
            raise ValueError("input changed during native QA export")
    sidecar_hash = hashlib.sha256()
    with sidecar.open("xb") as handle:
        for row in output_rows:
            encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            handle.write(encoded)
            sidecar_hash.update(encoded)
    report = {
        "schema": "slac-qasper-native-alignment-v1",
        "status": "candidate_native_qa_exported_mechanical_alignment_only",
        "documents": len(candidates),
        "counts": dict(counts),
        "input_sha256": input_hashes,
        "sidecar_sha256": sidecar_hash.hexdigest(),
        "all_input_hashes_unchanged": True,
        "raw_member_read": MEMBER,
        "test_payload_read": False,
        "api_calls": 0,
        "near_duplicate_clearance": False,
        "cleared_for_evaluation": False,
        "independent_confirmation_test": False,
        "normalization": "Unicode NFC and remove whitespace; case preserved. All repeated-paragraph matches retained.",
        "privacy_projection": "Question-writer/worker identifiers and background fields omitted; all original answer annotations retained without worker_id.",
        "relationship_to_pool_manifest": "The earlier pool manifest describes canonical QA absence. This sidecar supplies original native QA separately; other eligibility gates remain unmet.",
        "limits": [
            "Mechanical evidence alignment is not independent semantic verification or evidence sufficiency gold.",
            "Figure/table, ambiguous and unmatched evidence are retained with status, not discarded or guessed.",
            "Near-duplicate/version/family isolation, task protocol, evaluator adaptation and final test contamination clearance remain pending.",
            "All native questions for selected papers are exported; no model-outcome-based filtering occurred.",
        ],
        "per_document": per_document,
    }
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--canonical-shard", required=True)
    parser.add_argument("--pool-dir", required=True)
    args = parser.parse_args()
    result = export(args.archive, args.canonical_shard, args.pool_dir)
    print(json.dumps({key: result[key] for key in ("status", "documents", "counts", "all_input_hashes_unchanged")}, ensure_ascii=False))
