"""Offline counterfactual intervention and provenance tests; no real provider access."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import analyze_qasper_relation_counterfactuals as counterfactual
import analyze_qasper_relation_pilot as analysis
from qasper_relation_replay import AdjacentRelation, replay_policy
from test_qasper_relation_analysis import completed
from test_qasper_relation_pilot import annotation
from test_qasper_relation_replay import CharacterTokenizer, document


def synthetic():
    units = document(["s" * 210, "n" * 210, "f" * 210, "e" * 210])
    ids = [unit.unit_id for unit in units]
    queries = [{"family_id": "family", "doc_id": "doc", "question_id": f"q{i}",
                "candidate_ids": ids, "ranked_ids": ["u0", "u2", "u3", "u1"]} for i in range(2)]
    static = [{"id": "edge", "doc_id": "doc", "left_id": "u0", "right_id": "u1"}]
    support = [{"id": f"q{i}-{uid}", "doc_id": "doc", "question_id": f"q{i}", "unit_id": uid}
               for i in range(2) for uid in ids]
    prepared = {"queries": queries, "static_tasks": static, "support_tasks": support}
    labels = {b: {task["id"]: "yes" for task in support} for b in counterfactual.BACKENDS}
    labels["jev"]["edge"], labels["general"]["edge"] = "dependent", "independent"
    annotations = {("doc", f"q{i}"): [annotation([units[1].native_text])] for i in range(2)}
    return prepared, {"doc": units}, annotations, labels


def test_uncapped_intervention_exposes_chunk_gate_without_changing_evidence_limits():
    prepared, documents, annotations, labels = synthetic()
    records, diagnostics = counterfactual.replay_counterfactuals(prepared, documents, annotations, labels, CharacterTokenizer())
    assert len(records) == 20 and len(diagnostics) == 16
    original = next(row for row in diagnostics if row["method"] == counterfactual.method_name("jev", "jev", "original"))
    uncapped = next(row for row in diagnostics if row["method"] == counterfactual.method_name("jev", "jev", counterfactual.VARIANTS[1]))
    assert original["funnel"]["counts"]["dependent"] == 1
    assert original["funnel"]["counts"]["accepted_merge"] == 0
    assert original["selected_ids"] == ["u0", "u2", "u3"]
    assert uncapped["selected_ids"] == ["u0", "u1", "u2"]
    assert set(uncapped["funnel"]["counts"].values()) == {1}
    assert uncapped["funnel"]["selected_set_changed"] is True
    assert original["candidate_information_sha256"] == uncapped["candidate_information_sha256"]
    assert all(row["actual_evidence_tokens"] <= 1024 and row["selected_units"] <= 3 for row in records)
    report = counterfactual.summarize(prepared, records, diagnostics)
    method = counterfactual.method_name("jev", "jev", counterfactual.VARIANTS[1])
    counts = report["mechanism_funnels"][method]
    assert counts["edge_query_occurrences"]["dependent"] == 2
    assert counts["distinct_document_edges"]["dependent"] == 1
    assert counts["questions_with_selected_set_change"] == 2
    assert report["comparisons_to_I"][method]["minus_same_support_I"]["official_evidence_f1"]["positive"] == 2
    assert report["general_relation_minus_jev_relation"]["support_jev_uncapped_chunk_diagnostic"]["official_evidence_f1"]["negative"] == 2


def test_support_swap_can_stop_funnel_even_when_relation_is_accepted():
    prepared, documents, annotations, labels = synthetic()
    for task in prepared["support_tasks"]:
        if task["unit_id"] == "u1":
            labels["general"][task["id"]] = "no"
    _, diagnostics = counterfactual.replay_counterfactuals(prepared, documents, annotations, labels, CharacterTokenizer())
    method = counterfactual.method_name("general", "jev", counterfactual.VARIANTS[1])
    row = next(row for row in diagnostics if row["method"] == method)
    assert row["funnel"]["counts"] == {"dependent": 1, "accepted_merge": 1, "both_support_eligible": 0,
                                       "bonus_used": 0, "priority_changed": 0}
    assert row["selected_ids"] == row["independent_selected_ids"]


def test_selected_ids_do_not_depend_on_scoring_references():
    prepared, documents, annotations, labels = synthetic()
    original, traces = counterfactual.replay_counterfactuals(prepared, documents, annotations, labels, CharacterTokenizer())
    altered = {key: [annotation([])] for key in annotations}
    records, changed_traces = counterfactual.replay_counterfactuals(prepared, documents, altered, labels, CharacterTokenizer())
    assert traces == changed_traces
    assert [row["selected_ids"] for row in records] == [row["selected_ids"] for row in original]
    assert [row["official_evidence_f1"] for row in records] != [row["official_evidence_f1"] for row in original]


def test_uncapped_bound_handles_nonmonotonic_tokenizer():
    class NonmonotonicTokenizer:
        def encode(self, text, **kwargs):
            return list(range(900 if "u1" in text and "u2" not in text else 20))
    units = document(["a", "b", "c"])
    ids = [u.unit_id for u in units]
    assert counterfactual.uncapped_chunk_budget(units, ids, NonmonotonicTokenizer()) == 900


def test_rejected_selection_priority_does_not_count_as_accepted_mechanism():
    units = document(["s", "n" * 200, "far", "end"])
    relations = [AdjacentRelation("u0", "u1", "dependent")]
    relevance = {u.unit_id: "yes" for u in units}
    trace = replay_policy(units, list(relevance), relevance, relations, ["u0", "u2", "u3", "u1"],
                          mode="S", budget=50, chunk_budget=1000, max_units=3, tokenizer=CharacterTokenizer())
    assert any(step["relation_changed_priority"] and not step["accepted"] for step in trace["selection_trace"])
    result = counterfactual.funnel(trace, relations, relevance)
    assert result["counts"]["both_support_eligible"] == 1
    assert result["counts"]["bonus_used"] == result["counts"]["priority_changed"] == 0


def test_completed_execution_verified_and_local_details_separated(completed):
    report = counterfactual.analyze(completed)
    assert report["status"] == "completed" and report["method_count"] == 10
    assert report["api_calls"] == 0 and report["thresholds_selected_using_gold"] is False
    assert "post-hoc" in report["analysis_type"]
    assert len(analysis.read_rows(completed.output / "per_question.jsonl")) == 20
    assert len(analysis.read_rows(completed.output / "diagnostics.jsonl")) == 16
    content = (completed.output / "analysis.json").read_text(encoding="utf-8")
    for forbidden in ('"q0"', '"q1"', '"u0"', "FORBIDDEN_QA_ANSWER", "What is 0?", "Evidence 0", str(completed.run)):
        assert forbidden not in content
    for name, expected in report["output_files_sha256"].items():
        assert counterfactual.digest(completed.output / name) == expected


@pytest.mark.parametrize("target", ["record", "trace", "labels", "response", "summary", "ledger", "extra_call"])
def test_counterfactual_refuses_tampered_or_incomplete_execution(completed, target):
    if target in {"record", "trace"}:
        path = completed.run / ("per_question.jsonl" if target == "record" else "traces.jsonl")
        rows = analysis.read_rows(path)
        rows[0]["selected_ids"] = ["changed"]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    elif target == "labels":
        path = completed.run / "labels.json"
        value = counterfactual.pilot.read_json(path)
        value["jev"].pop(next(iter(value["jev"])))
        path.write_text(json.dumps(value), encoding="utf-8")
    elif target == "response":
        path = completed.run / "provider_calls" / "response_001.json"
        value = counterfactual.pilot.read_json(path)
        value["id"] = "changed"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif target == "summary":
        path = completed.run / "summary.json"
        value = counterfactual.pilot.read_json(path)
        value["status"] = "halted"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif target == "ledger":
        path = completed.run / "provider_calls" / "ledger.json"
        value = counterfactual.pilot.read_json(path)
        value["attempts"][0]["status"] = "halted"
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        counterfactual.pilot.write_json(completed.run / "provider_calls" / "response_999.json", {})
    with pytest.raises(ValueError):
        counterfactual.analyze(completed)
    assert not completed.output.exists()


def test_invalid_output_coverage_refused():
    prepared, documents, annotations, labels = synthetic()
    records, diagnostics = counterfactual.replay_counterfactuals(prepared, documents, annotations, labels, CharacterTokenizer())
    with pytest.raises(ValueError, match="coverage"):
        counterfactual.summarize(prepared, records[:-1], diagnostics)
    broken = deepcopy(diagnostics)
    broken.append(broken[0])
    with pytest.raises(ValueError, match="duplicate"):
        counterfactual.summarize(prepared, records, broken)
