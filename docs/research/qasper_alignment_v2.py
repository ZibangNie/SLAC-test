"""Deterministic native-Qasper validation evidence alignment, with source spans.

All native text blocks are indexed before questions are examined. Matching is
whole-block only under the v1 NFC/whitespace normalization; no fuzzy, substring,
or cross-block heuristic is used. Figure captions retain a separate native
locator and never become text evidence merely because their caption is known.
Historical artifacts are read-only. This is development alignment, not semantic
gold validation or independent test clearance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile
import unicodedata

MEMBER = "qasper-dev-v0.3.json"
SCHEMA = "slac-qasper-native-qa-sidecar-v2"
TEXT_STATUSES = {"unique_paragraph", "unique_heading", "unique_abstract", "unique_title"}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_evidence(text):
    if not isinstance(text, str):
        raise ValueError("evidence and native text must be strings")
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def native_text_blocks(paper, source_id):
    """Yield every native title, abstract, section heading, and paragraph.

    A canonical section-heading locator addresses the section object; its exact
    native text is section_name. The artificial canonical 'Abstract' heading has
    no independent native evidence field and is deliberately not a candidate.
    """
    prefix = f"/{source_id}"
    yield "title", prefix + "/title", prefix + "/title", paper["title"]
    yield "abstract", prefix + "/abstract", prefix + "/abstract", paper["abstract"]
    for si, section in enumerate(paper["full_text"]):
        section_pointer = f"{prefix}/full_text/{si}"
        yield "heading", section_pointer + "/section_name", section_pointer, section["section_name"]
        for pi, text in enumerate(section["paragraphs"]):
            pointer = f"{section_pointer}/paragraphs/{pi}"
            yield "paragraph", pointer, pointer, text


def build_text_index(paper, canonical, source_id):
    """Validate native/canonical blocks and retain every exact-match location."""
    block_map = defaultdict(list)
    text = canonical["canonical_text"]
    for block in canonical["blocks"]:
        block_map[block["source_locator"].get("json_pointer")].append(block)
    index, counts = defaultdict(list), Counter()
    for kind, native_pointer, canonical_pointer, native in native_text_blocks(paper, source_id):
        normalized = normalize_evidence(native)
        if not normalized:
            counts[f"empty_native_{kind}"] += 1
            continue
        canonical_kind = "heading" if kind == "title" else kind
        blocks = [b for b in block_map[canonical_pointer] if b["kind"] == canonical_kind]
        if len(blocks) != 1:
            raise ValueError(f"expected one canonical block for native {kind}: {native_pointer}")
        block = blocks[0]
        start, end = block["char_span"]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
            raise ValueError("invalid canonical character span")
        if normalize_evidence(text[start:end]) != normalized:
            raise ValueError(f"native/canonical {kind} text mismatch: {native_pointer}")
        index[normalized].append({
            "kind": kind,
            "native_json_pointers": [native_pointer],
            "canonical_source_locators": [canonical_pointer],
            "canonical_block_ids": [block["block_id"]],
            "canonical_char_spans": [[start, end]],
            "native_text_sha256": hashlib.sha256(native.encode("utf-8")).hexdigest(),
        })
        counts[f"native_{kind}_validated"] += 1
    return dict(index), counts


def build_caption_index(paper, source_id):
    index = defaultdict(list)
    for fi, item in enumerate(paper["figures_and_tables"]):
        caption = item["caption"]
        normalized = normalize_evidence(caption)
        if normalized:
            index[normalized].append({
                "native_json_pointer": f"/{source_id}/figures_and_tables/{fi}/caption",
                "native_file_reference": item["file"],
                "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            })
    return dict(index)


def align_evidence(text, text_index, caption_index):
    if text.lstrip().startswith("FLOAT SELECTED"):
        match = re.fullmatch(r"\s*FLOAT SELECTED\s*:\s*(.*)", text, re.DOTALL)
        captions = caption_index.get(normalize_evidence(match.group(1)), []) if match else []
        return {
            "status": "figure_or_table",
            "match_method": "native_caption_exact_nfc_no_whitespace" if captions else "native_caption_unmatched",
            "candidates": [],
            "native_caption_status": "unique_caption" if len(captions) == 1 else "ambiguous_repeated_caption" if captions else "unmatched_caption",
            "native_caption_candidates": deepcopy(captions),
            "text_span_available": False,
        }
    normalized = normalize_evidence(text)
    candidates = text_index.get(normalized, []) if normalized else []
    status = (f"unique_{candidates[0]['kind']}" if len(candidates) == 1 else
              "ambiguous_repeated_text" if candidates else "unmatched_text" if normalized else "empty_evidence")
    return {
        "status": status,
        "match_method": "whole_native_block_exact_nfc_no_whitespace" if candidates else "none",
        "candidates": deepcopy(candidates),
        "text_span_available": bool(candidates),
    }


def validate_alignment_spans(evidence, alignment, canonical_text):
    """Validate every alternative, never silently select the first location."""
    checked = 0
    for candidate in alignment["candidates"]:
        spans = candidate["canonical_char_spans"]
        if len(spans) != 1:
            raise ValueError("v2 permits only exact whole-block alignment")
        start, end = spans[0]
        if not 0 <= start < end <= len(canonical_text):
            raise ValueError("alignment span outside canonical text")
        if normalize_evidence(canonical_text[start:end]) != normalize_evidence(evidence):
            raise ValueError("aligned source span does not reproduce evidence")
        checked += 1
    return checked


def export(pool_dir, output_dir):
    pool_dir = Path(pool_dir).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output directory must not exist")
    report_path = pool_dir / "native_qa_alignment.json"
    old_sidecar = pool_dir / "native_qa_sidecar.jsonl"
    candidate_path = pool_dir / "candidates.jsonl"
    previous = json.loads(report_path.read_text(encoding="utf-8"))
    source_paths = [Path(p).resolve(strict=True) for p in previous["input_sha256"]]
    archives = [p for p in source_paths if p.name == "qasper-train-dev-v0.3.tgz"]
    shards = [p for p in source_paths if re.fullmatch(r"documents-\d{5}\.jsonl\.gz", p.name)]
    if len(archives) != 1 or len(shards) != 1:
        raise ValueError("expected one official train/dev archive and canonical shard")
    archive, shard = archives[0], shards[0]
    for path_string, expected in previous["input_sha256"].items():
        if digest_file(path_string) != expected:
            raise ValueError("historical source input hash mismatch")
    if digest_file(old_sidecar) != previous["sidecar_sha256"]:
        raise ValueError("historical sidecar hash mismatch")
    tracked_inputs = {str(p): digest_file(p) for p in (*source_paths, report_path, old_sidecar, candidate_path)}
    candidates = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()]
    if not candidates or len(candidates) > 32 or any(c["official_split"] != "validation" for c in candidates):
        raise ValueError("candidate pool must contain 1-32 official validation documents")
    expected = {c["doc_id"]: c for c in candidates}
    if len(expected) != len(candidates):
        raise ValueError("duplicate candidate doc IDs")
    with tarfile.open(archive, "r:gz") as handle:
        member = handle.getmember(MEMBER)
        if not member.isfile() or member.size > 32 * 1024 * 1024:
            raise ValueError("unexpected native validation member")
        with handle.extractfile(member) as stream:
            native_payload = stream.read()
    native = json.loads(native_payload)
    native_member_hash = hashlib.sha256(native_payload).hexdigest()
    canonical = {}
    with gzip.open(shard, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["doc_id"] in expected:
                if row["doc_id"] in canonical:
                    raise ValueError("duplicate canonical candidate")
                canonical[row["doc_id"]] = row
    if canonical.keys() != expected.keys():
        raise ValueError("candidate missing from canonical shard")
    text_indices, caption_indices, counts, per_document = {}, {}, Counter(), []
    expected_questions = {}
    for candidate in candidates:
        doc_id, source_id = candidate["doc_id"], candidate["source_id"]
        row, paper = canonical[doc_id], native[source_id]
        if row["original_split"] != "validation" or row["source_id"] != source_id or row["family_id"] != candidate["family_id"]:
            raise ValueError("canonical identity or split mismatch")
        if row["raw_locator"].get("member") != MEMBER or row["raw_locator"].get("json_pointer") != f"/{source_id}":
            raise ValueError("canonical native locator mismatch")
        if row["extra"]["raw_archive_sha256"] != tracked_inputs[str(archive)]:
            raise ValueError("canonical native archive lineage mismatch")
        text_indices[doc_id], dc = build_text_index(paper, row, source_id)
        caption_indices[doc_id] = build_caption_index(paper, source_id)
        counts.update(dc)
        per_document.append({"doc_id": doc_id, "native_blocks": dict(dc)})
        for qa in paper["qas"]:
            key = (doc_id, qa["question_id"])
            if key in expected_questions:
                raise ValueError("duplicate native question")
            expected_questions[key] = qa
    old_rows = [json.loads(line) for line in old_sidecar.read_text(encoding="utf-8").splitlines()]
    new_rows, seen_questions, changes = [], set(), []
    original_fields_checked = span_matches_checked = 0
    for old in old_rows:
        key = (old["doc_id"], old["question_id"])
        if key not in expected_questions or key in seen_questions:
            raise ValueError("unexpected or duplicate historical QA row")
        seen_questions.add(key)
        qa, row = expected_questions[key], deepcopy(old)
        if old["official_split"] != "validation" or old["question"] != qa["question"]:
            raise ValueError("historical QA split/question mismatch")
        if old["source_id"] != expected[old["doc_id"]]["source_id"] or old["family_id"] != expected[old["doc_id"]]["family_id"]:
            raise ValueError("historical QA identity mismatch")
        if old["original_archive_sha256"] != tracked_inputs[str(archive)] or old["native_member_sha256"] != native_member_hash or old["canonical_shard_sha256"] != tracked_inputs[str(shard)]:
            raise ValueError("historical QA source hash mismatch")
        if len(old["answer_annotations"]) != len(qa["answers"]):
            raise ValueError("native answer annotation count mismatch")
        row["schema"] = SCHEMA
        row["alignment_schema_v2"] = "exact_native_blocks_and_separate_float_captions-v2"
        row["cleared_for_evaluation"] = False
        row["independent_confirmation_test"] = False
        any_unique, all_unique = False, True
        for ai, (answer, original_annotation) in enumerate(zip(row["answer_annotations"], qa["answers"], strict=True)):
            if answer["native_answer"] != original_annotation["answer"] or answer["annotation_id"] != original_annotation.get("annotation_id"):
                raise ValueError("historical answer content differs from native source")
            counts["answer_annotations"] += 1
            counts["unanswerable_annotations"] += int(answer["native_answer"].get("unanswerable") is True)
            alignments = []
            for ei, evidence in enumerate(answer["native_answer"]["evidence"]):
                alignment = {"evidence_index": ei, **align_evidence(evidence, text_indices[row["doc_id"]], caption_indices[row["doc_id"]])}
                alignments.append(alignment)
                counts[f"evidence_{alignment['status']}"] += 1
                if "native_caption_status" in alignment:
                    counts[f"figure_{alignment['native_caption_status']}"] += 1
                span_matches_checked += validate_alignment_spans(evidence, alignment, canonical[row["doc_id"]]["canonical_text"])
                prior = answer["evidence_alignment"][ei]
                if prior["evidence_index"] != ei:
                    raise ValueError("historical evidence index mismatch")
                if prior["status"] != alignment["status"] or prior["status"] == "figure_or_table":
                    changes.append({"doc_id": row["doc_id"], "question_id": row["question_id"], "answer_index": ai, "evidence_index": ei,
                                    "old_status": prior["status"], "new_status": alignment["status"],
                                    "native_caption_status": alignment.get("native_caption_status"),
                                    "evidence_sha256": hashlib.sha256(evidence.encode("utf-8")).hexdigest()})
            unique = bool(alignments) and all(a["status"] in TEXT_STATUSES for a in alignments)
            answer["evidence_alignment_v2"] = alignments
            answer["all_evidence_uniquely_located_text_v2"] = unique
            counts["answer_annotations_all_evidence_uniquely_located_text_v2"] += int(unique)
            counts["answer_annotations_no_evidence"] += int(not alignments)
            any_unique |= unique
            all_unique &= unique
        counts["questions"] += 1
        counts["questions_any_annotation_uniquely_located_text_v2"] += int(any_unique)
        counts["questions_all_annotations_uniquely_located_text_v2"] += int(bool(row["answer_annotations"]) and all_unique)
        # Verify every original field recursively except the declared schema label.
        restored = deepcopy(row)
        restored["schema"] = old["schema"]
        del restored["alignment_schema_v2"]
        for answer in restored["answer_annotations"]:
            del answer["evidence_alignment_v2"]
            del answer["all_evidence_uniquely_located_text_v2"]
        if restored != old:
            raise ValueError("an original sidecar field was changed")
        original_fields_checked += 1
        new_rows.append(row)
    if seen_questions != expected_questions.keys():
        raise ValueError("historical sidecar did not retain every candidate native QA")
    if any(digest_file(p) != expected_hash for p, expected_hash in tracked_inputs.items()):
        raise ValueError("historical input changed during alignment")
    output_dir.mkdir(parents=True)
    output_sidecar = output_dir / "native_qa_sidecar_v2.jsonl"
    with output_sidecar.open("x", encoding="utf-8", newline="\n") as handle:
        for row in new_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema": "slac-qasper-alignment-audit-v2", "status": "development_mechanical_alignment_complete",
        "documents": len(candidates), "counts": dict(counts), "per_document": per_document,
        "input_sha256": tracked_inputs, "sidecar_sha256": digest_file(output_sidecar),
        "script_sha256": digest_file(__file__), "raw_member_read": MEMBER,
        "all_input_hashes_unchanged": True, "all_original_sidecar_fields_unchanged_except_schema": True,
        "original_sidecar_rows_verified": original_fields_checked, "original_native_qa_and_answers_verified": True,
        "evidence_span_matches_verified": span_matches_checked, "alignment_changes": changes,
        "normalization": "Unicode NFC; remove Unicode whitespace; preserve case, punctuation, and all other characters.",
        "matching": "Every source-mapped title, abstract, section heading and paragraph is indexed before QA. Exact whole-block matches only; all alternatives retained.",
        "figure_policy": "Strip the FLOAT SELECTED: marker only to locate the native caption. Keep figure_or_table status; no canonical text span, no inferred figure contents.",
        "test_payload_read": False, "api_calls": 0, "near_duplicate_clearance": False,
        "cleared_for_evaluation": False, "independent_confirmation_test": False,
        "limits": ["Location is not semantic sufficiency validation.",
                   "NFC/whitespace span matches do not redefine official exact-string Qasper evidence scoring.",
                   "Text-location stratification includes native section headings and titles; it is not paragraph-only gold.",
                   "Unmatched, repeated, no-evidence, unanswerable and figure annotations remain present.",
                   "No fuzzy, substring or cross-block alignment is attempted.",
                   "Near-duplicate/version isolation and evaluator acceptance remain independent gates."],
    }
    with (output_dir / "alignment_audit_v2.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = export(args.pool_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "documents", "counts", "evidence_span_matches_verified", "all_input_hashes_unchanged")}, ensure_ascii=False))
