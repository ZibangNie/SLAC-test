"""Hand-derived evidence scoring cases; no real references or model access."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import evaluate_cached_budget_selection as evaluation


def annotation(evidence, *, unanswerable=False):
    return {"native_answer": {"unanswerable": unanswerable, "extractive_spans": [],
                              "free_form_answer": "synthetic answer", "yes_no": None,
                              "evidence": evidence}}


def fixture():
    units = [{"unit_id": f"u{i}", "order": i, "text": f"canonical {i}", "native_text": text}
             for i, text in enumerate(("native A", "native B", "FLOAT SELECTED: synthetic figure"))]
    documents = {"synthetic-doc": units}
    queries = [{"family_id": "A" if i < 2 else "B", "doc_id": "synthetic-doc", "question_id": f"q{i}",
                "candidate_ids": ["u0", "u1", "u2"]} for i in range(3)]
    choices = {"score_greedy": [[0], [], [1]], "score_exact": [[1], [0], [2]],
               "density_greedy": [[0, 1], [], [0]], "matched_resource_exact": [[0], [], [2]]}
    selections = []
    for method, selections_for_method in choices.items():
        for query, selected in zip(queries, selections_for_method):
            selections.append({**{k: query[k] for k in evaluation.IDENTITY}, "method": method,
                               "selected_indexes": selected, "tokens": 10 * len(selected),
                               "unit_count": len(selected),
                               "pack_sha256": evaluation.sha(evaluation.render(units, selected).encode())})
    references = {
        evaluation.identity(queries[0]): [annotation(["native A", "native A"])],
        evaluation.identity(queries[1]): [annotation([], unanswerable=True)],
        evaluation.identity(queries[2]): [annotation(["FLOAT SELECTED: synthetic figure"])],
    }
    old_values = {evaluation.identity(q): value for q, value in zip(queries, [2 / 3, 1., 0.])}
    return documents, queries, selections, references, old_values


def test_native_exact_evidence_duplicates_unanswerable_and_figures_are_retained():
    records = evaluation.score_selections(*fixture())
    observed = {(row["method"], row["question_id"]): row["official_evidence_f1"] for row in records}
    assert len(records) == 12
    # Reference duplicates retain list-length denominator; canonical text would score zero.
    assert observed["score_greedy", "q0"] == 2 / 3
    assert observed["density_greedy", "q0"] == .5
    assert observed["score_greedy", "q1"] == 1
    assert observed["score_exact", "q1"] == 0
    assert observed["score_exact", "q2"] == 1


def test_complete_weightings_and_all_three_paired_contrasts():
    data = fixture()
    summary = evaluation.summarize(evaluation.score_selections(*data), data[1])
    assert summary["question_count"] == 3 and summary["family_count"] == 2
    assert summary["record_count"] == 12
    baseline = summary["official_evidence_f1"]["score_greedy"]
    assert baseline["question_weighted"] == pytest.approx(5 / 9)
    assert baseline["family_balanced"] == pytest.approx(5 / 12)
    exact = summary["paired_comparisons"][0]
    assert len(summary["paired_comparisons"]) == 3
    assert exact["plus"] == "score_exact" and exact["minus"] == "score_greedy"
    assert exact["mean_delta"]["question_weighted"] == pytest.approx(-2 / 9)
    assert exact["mean_delta"]["family_balanced"] == pytest.approx(1 / 12)
    assert exact["question_win_tie_loss"] == {"wins": 1, "ties": 0, "losses": 2}
    assert exact["family_win_tie_loss"] == {"wins": 1, "ties": 0, "losses": 1}
    assert "synthetic-doc" not in json.dumps(summary)


def test_multiple_annotators_use_maximum_not_annotation_average():
    documents, queries, selections, references, old = fixture()
    references[evaluation.identity(queries[0])].append(annotation(["native B"]))
    rows = evaluation.score_selections(documents, queries, selections, references, old)
    assert next(r for r in rows if r["method"] == "score_exact" and r["question_id"] == "q0")["official_evidence_f1"] == 1


@pytest.mark.parametrize("change", ["missing_reference", "empty_annotations", "changed_baseline", "missing_method", "duplicate_selection"])
def test_incomplete_or_invalid_inputs_stop_without_partial_score_result(change):
    documents, queries, selections, references, old = fixture()
    first = evaluation.identity(queries[0])
    if change == "missing_reference":
        del references[first]
    elif change == "empty_annotations":
        references[first] = []
    elif change == "changed_baseline":
        old[first] = .5
    elif change == "missing_method":
        selections.pop()
    else:
        selections.append(deepcopy(selections[0]))
    with pytest.raises(ValueError):
        evaluation.score_selections(documents, queries, selections, references, old)


def test_non_unanswerable_empty_evidence_is_not_excluded():
    documents, queries, selections, references, old = fixture()
    key = evaluation.identity(queries[1])
    references[key] = [annotation([])]
    rows = evaluation.score_selections(documents, queries, selections, references, old)
    assert len(rows) == 12
    assert next(r for r in rows if r["method"] == "score_greedy" and r["question_id"] == "q1")["official_evidence_f1"] == 1


def test_reference_projection_preserves_full_annotations_and_checks_identity():
    _, queries, _, references, _ = fixture()
    rows = [{**dict(zip(evaluation.IDENTITY, key)), "answer_annotations": value, "ignored_field": "not projected"}
            for key, value in references.items()]
    rows.append({"family_id": "unused", "doc_id": "unused", "question_id": "unused", "answer_annotations": []})
    raw = "\n".join(json.dumps(row) for row in rows).encode()
    assert evaluation.project_references(raw, queries) == references
    rows[0]["family_id"] = "different"
    with pytest.raises(ValueError, match="reference identity"):
        evaluation.project_references("\n".join(json.dumps(row) for row in rows).encode(), queries)


def test_historical_baseline_bridge_requires_same_selection_and_complete_coverage():
    documents, queries, selections, _, old_values = fixture()
    query_map, indexed = evaluation.index_selections(documents, queries, selections)
    old = []
    for key, value in old_values.items():
        selected = indexed["score_greedy", key]
        old.append({**dict(zip(evaluation.IDENTITY, key)), "method": "p_yes_only_k3",
                    "selected_ids": [documents[key[1]][i]["unit_id"] for i in selected["selected_indexes"]],
                    "pack_sha256": selected["pack_sha256"], "actual_evidence_tokens": selected["tokens"],
                    "official_evidence_f1": value})
    assert evaluation.project_baseline(old, query_map, indexed, documents) == old_values
    old[0]["actual_evidence_tokens"] += 1
    with pytest.raises(ValueError, match="baseline selection"):
        evaluation.project_baseline(old, query_map, indexed, documents)


def test_selected_native_duplicate_or_forged_pack_hash_is_rejected():
    documents, queries, selections, _, _ = fixture()
    selections[0]["pack_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pack content"):
        evaluation.index_selections(documents, queries, selections)


def test_malformed_json_duplicate_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        evaluation.parse(b'{"field": 1, "field": 2}')


def test_wrong_selection_pin_stops_before_any_reference_read(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ARTIFACTS", tmp_path)
    source = tmp_path / "bad-selection.json"
    source.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(evaluation, "PINS", {"selections": (source.name, "0" * 64)})
    with pytest.raises(ValueError, match="input binding"):
        evaluation.evaluate(tmp_path / "output")
    assert not (tmp_path / "output/summary.json").exists()
