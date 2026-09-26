"""Offline fixtures for native evidence provenance and exact span alignment."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "docs/research/qasper_alignment_v2.py"
SPEC = importlib.util.spec_from_file_location("qasper_alignment_v2", SCRIPT)
alignment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alignment)


def fixture_paper():
    return {
        "title": "A title", "abstract": "An abstract.",
        "full_text": [
            {"section_name": "A heading", "paragraphs": ["Repeated paragraph.", "Caf\u00e9 text.", "   "]},
            {"section_name": "Another heading", "paragraphs": ["Repeated paragraph."]},
        ],
        "figures_and_tables": [{"file": "figures/f1.png", "caption": "Table 1: A caption."}],
        "qas": [],
    }


def fixture_canonical(paper, source_id="fixture"):
    parts, blocks = [], []

    def add(kind, pointer, text):
        if not text.strip():
            return
        prefix = "\n\n" if parts else ""
        start = sum(len(part) for part in parts) + len(prefix)
        parts.extend([prefix, text])
        blocks.append({"block_id": f"b{len(blocks)}", "kind": kind,
                       "source_locator": {"json_pointer": pointer}, "char_span": [start, start + len(text)]})

    add("heading", f"/{source_id}/title", paper["title"])
    add("heading", f"/{source_id}/abstract", "Abstract")
    add("abstract", f"/{source_id}/abstract", paper["abstract"])
    for si, section in enumerate(paper["full_text"]):
        add("heading", f"/{source_id}/full_text/{si}", section["section_name"])
        for pi, text in enumerate(section["paragraphs"]):
            add("paragraph", f"/{source_id}/full_text/{si}/paragraphs/{pi}", text)
    return {"doc_id": f"qasper:{source_id}", "source_id": source_id, "family_id": source_id,
            "original_split": "validation", "raw_locator": {"member": alignment.MEMBER, "json_pointer": f"/{source_id}"},
            "extra": {}, "canonical_text": "".join(parts), "blocks": blocks}


def indexes(paper=None):
    paper = paper or fixture_paper()
    canonical = fixture_canonical(paper)
    text_index, counts = alignment.build_text_index(paper, canonical, "fixture")
    return canonical, text_index, alignment.build_caption_index(paper, "fixture"), counts


def test_all_native_kinds_validated_and_synthetic_heading_excluded():
    canonical, text_index, captions, counts = indexes()
    assert counts == {"native_title_validated": 1, "native_abstract_validated": 1,
                      "native_heading_validated": 2, "native_paragraph_validated": 3,
                      "empty_native_paragraph": 1}
    for text, kind in [("A title", "title"), ("An abstract.", "abstract"), ("A heading", "heading")]:
        result = alignment.align_evidence(text, text_index, captions)
        assert result["status"] == f"unique_{kind}"
        assert alignment.validate_alignment_spans(text, result, canonical["canonical_text"]) == 1
    assert alignment.align_evidence("Abstract", text_index, captions)["status"] == "unmatched_text"


def test_section_name_pointer_distinguished_from_canonical_section_object():
    _, text_index, captions, _ = indexes()
    candidate = alignment.align_evidence("A heading", text_index, captions)["candidates"][0]
    assert candidate["native_json_pointers"] == ["/fixture/full_text/0/section_name"]
    assert candidate["canonical_source_locators"] == ["/fixture/full_text/0"]


def test_repeated_locations_preserved_not_first_match():
    canonical, text_index, captions, _ = indexes()
    result = alignment.align_evidence("Repeated paragraph.", text_index, captions)
    assert result["status"] == "ambiguous_repeated_text"
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["canonical_char_spans"] != result["candidates"][1]["canonical_char_spans"]
    assert alignment.validate_alignment_spans("Repeated paragraph.", result, canonical["canonical_text"]) == 2


def test_ambiguity_across_kinds_preserved():
    paper = fixture_paper()
    paper["title"] = "A heading"
    _, text_index, captions, _ = indexes(paper)
    result = alignment.align_evidence("A heading", text_index, captions)
    assert result["status"] == "ambiguous_repeated_text"
    assert {c["kind"] for c in result["candidates"]} == {"title", "heading"}


def test_nfc_and_whitespace_only_normalization():
    canonical, text_index, captions, _ = indexes()
    evidence = "Cafe\u0301\n  text."
    result = alignment.align_evidence(evidence, text_index, captions)
    assert result["status"] == "unique_paragraph"
    assert alignment.validate_alignment_spans(evidence, result, canonical["canonical_text"]) == 1


@pytest.mark.parametrize("text", ["heading", "a heading", "A heading!", "A headingRepeated paragraph."])
def test_no_substring_casefold_fuzzy_or_cross_block_matching(text):
    _, text_index, captions, _ = indexes()
    result = alignment.align_evidence(text, text_index, captions)
    assert result["status"] == "unmatched_text"
    assert result["candidates"] == []


def test_figure_caption_kept_separate_from_text():
    _, text_index, captions, _ = indexes()
    result = alignment.align_evidence("FLOAT SELECTED: Table 1: A caption.", text_index, captions)
    assert result["status"] == "figure_or_table"
    assert result["native_caption_status"] == "unique_caption"
    assert result["native_caption_candidates"][0]["native_json_pointer"] == "/fixture/figures_and_tables/0/caption"
    assert result["candidates"] == [] and result["text_span_available"] is False


def test_repeated_caption_and_missing_caption_not_silently_resolved():
    paper = fixture_paper()
    paper["figures_and_tables"].append({"file": "figures/f2.png", "caption": "Table 1: A caption."})
    _, text_index, captions, _ = indexes(paper)
    result = alignment.align_evidence("FLOAT SELECTED: Table 1: A caption.", text_index, captions)
    assert result["native_caption_status"] == "ambiguous_repeated_caption"
    assert len(result["native_caption_candidates"]) == 2
    for evidence in ["FLOAT SELECTED: missing", "FLOAT SELECTED without separator"]:
        result = alignment.align_evidence(evidence, text_index, captions)
        assert result["status"] == "figure_or_table"
        assert result["native_caption_status"] == "unmatched_caption"


def test_empty_evidence_retained():
    _, text_index, captions, _ = indexes()
    assert alignment.align_evidence(" \n\t", text_index, captions)["status"] == "empty_evidence"


def test_native_canonical_mismatch_rejected():
    paper = fixture_paper()
    canonical = fixture_canonical(paper)
    canonical["canonical_text"] = canonical["canonical_text"].replace("A heading", "B heading")
    with pytest.raises(ValueError, match="text mismatch"):
        alignment.build_text_index(paper, canonical, "fixture")


def test_span_validation_rejects_shifted_or_multiple_spans():
    canonical, text_index, captions, _ = indexes()
    result = alignment.align_evidence("A heading", text_index, captions)
    result["candidates"][0]["canonical_char_spans"][0][0] += 1
    with pytest.raises(ValueError, match="reproduce evidence"):
        alignment.validate_alignment_spans("A heading", result, canonical["canonical_text"])
    result["candidates"][0]["canonical_char_spans"].append([0, 1])
    with pytest.raises(ValueError, match="whole-block"):
        alignment.validate_alignment_spans("A heading", result, canonical["canonical_text"])


def write_export_fixture(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    paper = fixture_paper()
    answers = [
        {"annotation_id": "ann1", "answer": {"evidence": ["A heading", "Cafe\u0301 text."], "unanswerable": False}},
        {"annotation_id": "ann2", "answer": {"evidence": ["FLOAT SELECTED: Table 1: A caption."], "unanswerable": False}},
        {"annotation_id": "ann3", "answer": {"evidence": [], "unanswerable": True}},
    ]
    paper["qas"] = [{"question_id": "q1", "question": "A fixture question?", "answers": answers}]
    native_payload = json.dumps({"fixture": paper}).encode("utf-8")
    archive = tmp_path / "qasper-train-dev-v0.3.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo(alignment.MEMBER)
        member.size = len(native_payload)
        handle.addfile(member, io.BytesIO(native_payload))
    canonical = fixture_canonical(paper)
    canonical["extra"]["raw_archive_sha256"] = alignment.digest_file(archive)
    shard = tmp_path / "documents-00000.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(canonical) + "\n")
    candidate = {"doc_id": "qasper:fixture", "source_id": "fixture", "family_id": "fixture", "official_split": "validation"}
    candidate_path = pool / "candidates.jsonl"
    candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    row = {**candidate, "schema": "slac-qasper-native-qa-sidecar-v1", "question_id": "q1", "question": paper["qas"][0]["question"],
           "original_archive_sha256": alignment.digest_file(archive), "canonical_shard_sha256": alignment.digest_file(shard),
           "native_member_sha256": hashlib.sha256(native_payload).hexdigest(),
           "cleared_for_evaluation": False, "independent_confirmation_test": False,
           "answer_annotations": []}
    for answer in answers:
        row["answer_annotations"].append({"annotation_id": answer["annotation_id"], "native_answer": answer["answer"],
            "evidence_alignment": [{"evidence_index": i, "status": "unmatched_text", "paragraph_candidates": []}
                                   for i, _ in enumerate(answer["answer"]["evidence"])],
            "all_evidence_unique_text_paragraphs": False})
    sidecar = pool / "native_qa_sidecar.jsonl"
    sidecar.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = {"input_sha256": {str(p): alignment.digest_file(p) for p in (archive, shard, candidate_path)},
              "sidecar_sha256": alignment.digest_file(sidecar)}
    (pool / "native_qa_alignment.json").write_text(json.dumps(report), encoding="utf-8")
    return pool, row


def test_full_export_preserves_source_fields_multianswers_and_input_hashes(tmp_path):
    pool, old = write_export_fixture(tmp_path)
    output = tmp_path / "output"
    report = alignment.export(pool, output)
    row = json.loads((output / "native_qa_sidecar_v2.jsonl").read_text("utf-8"))
    assert report["documents"] == 1 and report["counts"]["questions"] == 1
    assert report["counts"]["answer_annotations"] == 3
    assert report["evidence_span_matches_verified"] == 2
    assert report["all_input_hashes_unchanged"] is True
    assert report["test_payload_read"] is False
    assert [a["native_answer"] for a in row["answer_annotations"]] == [a["native_answer"] for a in old["answer_annotations"]]
    assert [a["all_evidence_uniquely_located_text_v2"] for a in row["answer_annotations"]] == [True, False, False]
    assert row["cleared_for_evaluation"] is False
    with pytest.raises(FileExistsError):
        alignment.export(pool, output)


def test_full_export_rejects_modified_historical_input(tmp_path):
    pool, _ = write_export_fixture(tmp_path)
    (pool / "native_qa_sidecar.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar hash mismatch"):
        alignment.export(pool, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_native_answer_mismatch_rejected_even_if_sidecar_hash_updated(tmp_path):
    pool, row = write_export_fixture(tmp_path)
    row["answer_annotations"][0]["native_answer"]["evidence"] = ["different evidence"]
    sidecar = pool / "native_qa_sidecar.jsonl"
    sidecar.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report_path = pool / "native_qa_alignment.json"
    report = json.loads(report_path.read_text("utf-8"))
    report["sidecar_sha256"] = alignment.digest_file(sidecar)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="answer content differs"):
        alignment.export(pool, tmp_path / "output")
    assert not (tmp_path / "output").exists()
