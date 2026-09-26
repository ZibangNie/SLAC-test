"""Synthetic regression fixtures for the pinned official Qasper metric contract."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
from qasper_metrics import (
    evidence_metrics,
    evaluate_qa,
    normalize_answer,
    paragraph_f1_score,
    paragraph_recall_diagnostic,
    references_from_annotations,
    token_f1_score,
)


def annotation(*, evidence=(), extractive=(), freeform="", yes_no=None, unanswerable=False):
    return {"native_answer": {
        "unanswerable": unanswerable,
        "extractive_spans": list(extractive),
        "free_form_answer": freeform,
        "yes_no": yes_no,
        "evidence": list(evidence),
    }}


@pytest.mark.parametrize("predicted,reference,expected", [
    ([], [], 1.0),
    ([], ["A"], 0.0),
    (["A"], [], 0.0),
    (["A"], ["A"], 1.0),
    (["A"], ["A", "B"], 2 / 3),
    (["A", "B"], ["A"], 2 / 3),
    (["A", "A"], ["A"], 2 / 3),
    (["A"], ["A", "A"], 2 / 3),
    (["A", "A"], ["A", "A"], 0.5),
    (["A", "B"], ["B", "A"], 1.0),
    (["A "], ["A"], 0.0),
    (["a"], ["A"], 0.0),
    (["e\u0301"], ["\u00e9"], 0.0),
    (["A\nB"], ["A B"], 0.0),
])
def test_exact_evidence_and_duplicate_denominators(predicted, reference, expected):
    assert paragraph_f1_score(predicted, reference) == pytest.approx(expected)


@pytest.mark.parametrize("kwargs,answer,kind", [
    ({"unanswerable": True, "evidence": ["irrelevant"]}, "Unanswerable", "none"),
    ({"extractive": ["red", "blue"]}, "red, blue", "extractive"),
    ({"freeform": "Because of calibration."}, "Because of calibration.", "abstractive"),
    ({"yes_no": True}, "Yes", "boolean"),
    ({"yes_no": False}, "No", "boolean"),
    ({"extractive": ["span"], "freeform": "ignored", "yes_no": True}, "span", "extractive"),
    ({"freeform": "explanation", "yes_no": False}, "explanation", "abstractive"),
])
def test_all_answer_types_and_priority(kwargs, answer, kind):
    ref = references_from_annotations([annotation(**kwargs)])[0]
    assert ref["answer"] == answer
    assert ref["type"] == kind
    if kind == "none":
        assert ref["evidence"] == []


def test_native_official_and_sidecar_annotation_shapes():
    sidecar = annotation(yes_no=False, evidence=["P"])
    native = sidecar["native_answer"]
    assert references_from_annotations([sidecar]) == references_from_annotations([native])
    assert references_from_annotations([sidecar]) == references_from_annotations([{"answer": native}])


def test_multiple_evidence_references_are_not_unioned():
    refs = [annotation(yes_no=True, evidence=["A"]), annotation(yes_no=True, evidence=["B", "C"])]
    result = evidence_metrics(["A", "B"], refs)
    assert result["evidence_f1"] == pytest.approx(2 / 3)
    assert result["evidence_recall"] == 1.0
    assert result["reference_count"] == 2
    assert result["best_f1_reference_index"] == 0


def test_recall_and_f1_maxima_are_independent():
    refs = [annotation(yes_no=True, evidence=["A"]), annotation(yes_no=True, evidence=["A", "B", "C"])]
    result = evidence_metrics(["A", "B"], refs)
    assert result["evidence_f1"] == pytest.approx(0.8)
    assert result["evidence_recall"] == 1.0
    assert result["best_f1_reference_index"] == 1
    assert result["best_recall_reference_index"] == 0


def test_text_only_filter_applies_only_to_gold_exact_case_substring():
    refs = [annotation(yes_no=True, evidence=["P", "x FLOAT SELECTED y", "float selected lower"])]
    assert references_from_annotations(refs, text_evidence_only=True)[0]["evidence"] == ["P", "float selected lower"]
    result = evidence_metrics(["P", "x FLOAT SELECTED y"], refs, text_evidence_only=True)
    assert result["evidence_f1"] == pytest.approx(0.5)
    assert evidence_metrics(["P", "x FLOAT SELECTED y"], refs)["evidence_f1"] == pytest.approx(0.8)


def test_empty_reference_recall_convention_is_explicit():
    assert paragraph_recall_diagnostic([], []) == 1.0
    assert paragraph_recall_diagnostic(["P"], []) == 0.0
    assert paragraph_recall_diagnostic(["P"], ["P", "P"]) == 0.5


@pytest.mark.parametrize("predicted,reference,expected", [
    ("The, CAT!", "a cat", 1.0),
    ("red blue", "red red blue", 0.8),
    ("", "", 0.0),
    ("the", "a", 0.0),
    ("No", "no", 1.0),
    ("Unanswerable", "Unanswerable", 1.0),
])
def test_official_answer_token_f1(predicted, reference, expected):
    assert token_f1_score(predicted, reference) == pytest.approx(expected)


def test_answer_normalization_only_removes_ascii_punctuation():
    assert normalize_answer("The dog's  red-blue") == "dogs redblue"
    assert normalize_answer("The cat—dog") == "cat—dog"


def synthetic_qa_fixture():
    gold = {
        "extractive": [annotation(extractive=["red", "blue"], evidence=["P", "Q"])],
        "freeform": [annotation(freeform="the old bridge", evidence=["R"])],
        "yes": [annotation(yes_no=True, evidence=["S"])],
        "no": [annotation(yes_no=False, evidence=["T"])],
        "none": [annotation(unanswerable=True, evidence=["ignored"])],
        "multiple": [annotation(extractive=["alpha"], evidence=["A"]), annotation(freeform="beta", evidence=["B", "C"])],
        "missing": [annotation(yes_no=True, evidence=["M"])],
    }
    predictions = {
        "extractive": {"predicted_answer": "red", "predicted_evidence": ["P"]},
        "freeform": {"predicted_answer": "old bridge", "predicted_evidence": ["R"]},
        "yes": {"predicted_answer": "Yes", "predicted_evidence": ["S"]},
        "no": {"predicted_answer": "No", "predicted_evidence": ["T"]},
        "none": {"predicted_answer": "Unanswerable", "predicted_evidence": []},
        "multiple": {"predicted_answer": "alpha", "predicted_evidence": ["B", "C"]},
        "extra": {"predicted_answer": "ignored", "predicted_evidence": ["ignored"]},
    }
    return gold, predictions


def test_macro_question_average_missing_and_independent_answer_evidence_maxima():
    gold, predictions = synthetic_qa_fixture()
    result = evaluate_qa(gold, predictions)
    assert result["Answer F1"] == pytest.approx(17 / 21)
    assert result["Evidence F1"] == pytest.approx(17 / 21)
    assert result["Missing predictions"] == 1
    assert result["Answer F1 by type"] == pytest.approx({
        "extractive": 5 / 6, "abstractive": 1.0, "boolean": 1.0, "none": 1.0,
    })


def test_answer_type_tie_uses_first_reference_and_empty_gold_is_zero():
    gold = {"q": [annotation(freeform="same"), annotation(extractive=["same"])]}
    result = evaluate_qa(gold, {"q": {"predicted_answer": "same", "predicted_evidence": []}})
    assert result["Answer F1 by type"] == {"extractive": 0.0, "abstractive": 1.0, "boolean": 0.0, "none": 0.0}
    assert result["Evidence F1"] == 1.0
    assert evaluate_qa({}, {})["Evidence F1"] == 0.0


def test_metric_adapter_does_not_mutate_annotations_or_predictions():
    gold, predicted = synthetic_qa_fixture()
    old_gold, old_predicted = deepcopy(gold), deepcopy(predicted)
    evaluate_qa(gold, predicted, text_evidence_only=True)
    assert gold == old_gold and predicted == old_predicted


def test_invalid_or_empty_annotations_fail_instead_of_silently_scoring():
    with pytest.raises(ValueError):
        evidence_metrics([], [])
    with pytest.raises(ValueError):
        references_from_annotations([annotation()])
    with pytest.raises(TypeError):
        paragraph_f1_score("P", ["P"])
    with pytest.raises(TypeError):
        paragraph_f1_score([1], ["P"])
