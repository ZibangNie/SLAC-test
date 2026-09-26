import sys
import json
import hashlib
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
from run_qasper_evidence_baselines import (
    Unit, bm25_ranking, pack_ranked, render_pack, oracle_candidates,
    choose_oracle, score_selection, native_pointer, aggregate, build_units, load_frozen_pool, PackCounter,
)


def unit(index, text):
    return Unit(f"u{index}", index, "paragraph", index * 10, index * 10 + len(text), text, text)


def annotation(evidence):
    return {"native_answer": {"unanswerable": False, "extractive_spans": [],
        "free_form_answer": "answer", "yes_no": None, "evidence": evidence}}


def test_rank_and_pack_do_not_consume_gold_or_truncate():
    units = [unit(0, "irrelevant background"), unit(1, "target target research"), unit(2, "target")]
    ranking = bm25_ranking("target", units)
    assert ranking[0] != 0
    count = lambda selected: sum(len(units[i].text) for i in selected)
    selected = pack_ranked(units, ranking, 8, count)
    assert selected == [2]
    assert count(selected) <= 8
    assert units[2].text in render_pack(units, selected)


def test_packing_checks_whole_combination_not_additive_estimate():
    units = [unit(0, "a"), unit(1, "b")]
    # Individually each fits, but their rendered joint pack is over budget.
    count = lambda selected: 11 if len(selected) == 2 else 3 * len(selected)
    assert pack_ranked(units, [0, 1], 6, count) == [0]


def test_top_k_counts_fitting_distinct_units():
    units = [unit(0, "overlong"), unit(1, "a"), unit(2, "a"), unit(3, "b")]
    count = lambda selected: sum(len(units[i].text) for i in selected)
    assert pack_ranked(units, [0, 1, 2, 3], 4, count, max_units=1) == [1]
    assert pack_ranked(units, [0, 1, 2, 3], 4, count, max_units=2) == [1, 3]


def test_oracle_enumerates_subsets_and_keeps_unreachable_gold_in_score():
    units = [unit(0, "noise"), unit(1, "gold one"), unit(2, "gold two")]
    annotations = [annotation(["gold one", "gold two", "unreachable"])]
    count = lambda selected: sum({0: 4, 1: 4, 2: 7}[i] for i in selected)
    options = oracle_candidates(units, [annotations[0]["native_answer"]["evidence"]], count)
    assert {tuple(selected) for selected, _ in options} == {(), (1,), (2,), (1, 2)}
    selected = choose_oracle(options, 11, units, annotations)
    assert selected == [1, 2]
    result = score_selection(units, selected, annotations, count, 11)
    assert result["official_evidence_f1"] == pytest.approx(.8)
    assert result["reference_evidence_recall"] == pytest.approx(2 / 3)


def test_empty_reference_and_exact_duplicate_candidates():
    units = [unit(0, "same"), unit(1, "same")]
    count = lambda selected: 4 * len(selected)
    assert pack_ranked(units, [0, 1], 100, count) == [0]
    assert choose_oracle(oracle_candidates(units, [[]], count), 100, units, [annotation([])]) == []


def test_oracle_cap_is_not_silently_approximate():
    units = [unit(i, str(i)) for i in range(17)]
    with pytest.raises(ValueError, match="cap"):
        oracle_candidates(units, [[u.text for u in units]], len)


def test_model_visible_source_whitelist_excludes_gold():
    paper = {"title": "Title", "abstract": "Abstract", "qas": [{"answer": "secret gold"}],
             "full_text": [{"section_name": "Methods", "paragraphs": ["text"]}]}
    assert native_pointer(paper, "/p/full_text/0/paragraphs/0", "p") == "text"
    with pytest.raises(ValueError, match="whitelist"):
        native_pointer(paper, "/p/qas/0/answer", "p")


def test_native_units_skip_synthetic_abstract_heading_and_resolve_section_object():
    paper = {"title": "Title", "abstract": "Summary", "full_text": [
        {"section_name": "Methods", "paragraphs": ["Body"]}]}
    fields = [("heading", "/p/title", "Title"), ("heading", "/p/abstract", "Abstract"),
              ("abstract", "/p/abstract", "Summary"), ("heading", "/p/full_text/0", "Methods"),
              ("paragraph", "/p/full_text/0/paragraphs/0", "Body")]
    text, blocks = "", []
    for i, (kind, pointer, content) in enumerate(fields):
        start = len(text)
        text += content + "\n"
        blocks.append({"kind": kind, "source_locator": {"json_pointer": pointer},
                       "char_span": [start, start + len(content)], "block_id": str(i)})
    units = build_units({"source_id": "p", "canonical_text": text, "blocks": blocks}, paper)
    assert [u.native_text for u in units] == ["Title", "Summary", "Methods", "Body"]
    assert [u.unit_id for u in units] == ["0", "2", "3", "4"]


def test_document_macro_does_not_overweight_many_question_papers():
    records = []
    for doc, scores in (("a", [1, 1, 1]), ("b", [0])):
        for score in scores:
            records.append({"method": "m", "budget": 100, "doc_id": doc,
                "official_evidence_f1": score, "reference_evidence_recall": score,
                "official_text_only_evidence_f1": score, "actual_evidence_tokens": 10})
    result = aggregate(records)[0]
    assert result["official_evidence_f1_question_macro"] == .75
    assert result["official_evidence_f1_document_macro"] == .5


def pool_fixture(path, duplicate_doc=False, duplicate_question=False, bad_lineage=False):
    candidate = {"doc_id": "p", "source_id": "p", "family_id": "p", "official_split": "validation", "normalized_body_sha256": "body"}
    question = {"doc_id": "p", "source_id": "wrong" if bad_lineage else "p", "family_id": "p", "official_split": "validation",
                "canonical_body_sha256": "body", "question_id": "q"}
    candidate_path, sidecar = path / "candidates.jsonl", path / "sidecar.jsonl"
    candidate_path.write_text((json.dumps(candidate) + "\n") * (2 if duplicate_doc else 1), encoding="utf-8")
    sidecar.write_text((json.dumps(question) + "\n") * (2 if duplicate_question else 1), encoding="utf-8")
    chash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    (path / "pool_manifest.json").write_text(json.dumps({"candidate_manifest_sha256": chash}), encoding="utf-8")
    (path / "alignment_audit_v2.json").write_text(json.dumps({"sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "input_sha256": {str(candidate_path.resolve()): chash}}), encoding="utf-8")
    return sidecar


@pytest.mark.parametrize("mutation,match", [("candidate", "candidate manifest hash"), ("sidecar", "v2 sidecar hash"),
    ("duplicate_doc", "duplicate candidate"), ("duplicate_question", "duplicate question"), ("bad_lineage", "lineage mismatch")])
def test_frozen_pool_rejects_tampering_and_duplicate_identities(tmp_path, mutation, match):
    sidecar = pool_fixture(tmp_path, **({mutation: True} if mutation not in {"candidate", "sidecar"} else {}))
    if mutation in {"candidate", "sidecar"}:
        path = tmp_path / "candidates.jsonl" if mutation == "candidate" else sidecar
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    with pytest.raises(ValueError, match=match):
        load_frozen_pool(tmp_path, sidecar)


def test_deadline_checked_in_tokenization_and_oracle_scoring():
    units = [unit(0, "text")]
    with pytest.raises(TimeoutError):
        PackCounter(None, units, deadline=time.monotonic() - 1)([0])
    with pytest.raises(TimeoutError):
        choose_oracle([((), 0)], 100, units, [annotation([])], deadline=time.monotonic() - 1)
