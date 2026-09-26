from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import analyze_qasper_relation_cardinality as cardinality
from test_qasper_relation_counterfactuals import synthetic
from test_qasper_relation_pilot import annotation
from test_qasper_relation_replay import CharacterTokenizer


def test_complete_uniform_grid_keeps_caps_and_dense_comparator():
    prepared, documents, annotations, labels = synthetic()
    records, traces = cardinality.replay(prepared, documents, annotations, labels, CharacterTokenizer())
    assert len(records) == 30 and len(traces) == 24
    assert len({(r["method"], r["question_id"]) for r in records}) == 30
    for row in records:
        k = int(row["method"][-1])
        assert row["selected_units"] <= k and row["actual_evidence_tokens"] <= 1024
        assert len(row["selected_ids"]) == len(set(row["selected_ids"]))
    assert {r["method"] for r in records if r["method"].startswith("dense")} == {"dense_k1", "dense_k2", "dense_k3"}


def test_scoring_gold_cannot_choose_k_or_change_selected_evidence():
    prepared, documents, annotations, labels = synthetic()
    annotations = {key: [annotation([documents["doc"][0].native_text])] for key in annotations}
    original, traces = cardinality.replay(prepared, documents, annotations, labels, CharacterTokenizer())
    changed, changed_traces = cardinality.replay(prepared, documents,
        {key: [annotation([])] for key in annotations}, labels, CharacterTokenizer())
    assert traces == changed_traces
    assert [(r["method"], r["selected_ids"]) for r in original] == [(r["method"], r["selected_ids"]) for r in changed]
    assert [r["official_evidence_f1"] for r in original] != [r["official_evidence_f1"] for r in changed]


def test_relation_cannot_bypass_support_no_in_any_k():
    prepared, documents, annotations, labels = synthetic()
    for backend in labels:
        for task in prepared["support_tasks"]:
            if task["unit_id"] == "u1":
                labels[backend][task["id"]] = "no"
    records, _ = cardinality.replay(prepared, documents, annotations, labels, CharacterTokenizer())
    assert all("u1" not in r["selected_ids"] for r in records if not r["method"].startswith("dense"))
