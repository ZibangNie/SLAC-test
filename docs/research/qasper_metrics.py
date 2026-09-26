"""Qasper official metric semantics with an adapter for native QA sidecars.

Adapted from Allen Institute for AI's qasper-led-baseline/scripts/evaluator.py,
commit afd0fb96bf78ce8cd8157639c6f6a6995e4f9089 (Apache-2.0).
Changes: typed, validated sidecar inputs; per-question evidence-only interface;
recall diagnostics; no file loading, network access, or command-line interface.
The Answer F1 normalization retains upstream attribution to SQuAD v1.1.
See QASPER_METRICS.md and licenses/QASPER_APACHE_2_0.txt.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import re
import string
from typing import Any


OFFICIAL_COMMIT = "afd0fb96bf78ce8cd8157639c6f6a6995e4f9089"
OFFICIAL_SOURCE = (
    "https://github.com/allenai/qasper-led-baseline/blob/"
    f"{OFFICIAL_COMMIT}/scripts/evaluator.py"
)
ANSWER_TYPES = ("extractive", "abstractive", "boolean", "none")


def _strings(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise TypeError(f"{name} must be a list of strings")
    return value


def normalize_answer(text: str) -> str:
    """Upstream SQuAD v1.1 normalization; never use it for evidence."""
    if not isinstance(text, str):
        raise TypeError("answer must be a string")
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1_score(prediction: str, ground_truth: str) -> float:
    """Official answer token F1, including zero for two empty token lists."""
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(ground_truth).split()
    common_count = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if not common_count:
        return 0.0
    precision = common_count / len(predicted_tokens)
    recall = common_count / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def paragraph_f1_score(prediction: list[str], ground_truth: list[str]) -> float:
    """Official exact evidence-string F1, retaining list-length denominators."""
    _strings(prediction, "predicted_evidence")
    _strings(ground_truth, "reference evidence")
    if not prediction and not ground_truth:
        return 1.0
    common_count = len(set(prediction).intersection(ground_truth))
    if not common_count:
        return 0.0
    precision = common_count / len(prediction)
    recall = common_count / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def paragraph_recall_diagnostic(prediction: list[str], ground_truth: list[str]) -> float:
    """Non-official recall diagnostic; empty-empty=1, other empty-gold=0."""
    _strings(prediction, "predicted_evidence")
    _strings(ground_truth, "reference evidence")
    if not ground_truth:
        return float(not prediction)
    return len(set(prediction).intersection(ground_truth)) / len(ground_truth)


def references_from_annotations(
    answer_annotations: list[dict], *, text_evidence_only: bool = False
) -> list[dict]:
    """Convert sidecar/native annotations into the official references format.

    Accepts {native_answer: ...}, official {answer: ...}, or the native answer
    dictionary itself. Annotation order and duplicate evidence are preserved.
    """
    if not isinstance(answer_annotations, list) or not answer_annotations:
        raise ValueError("answer_annotations must be a nonempty list")
    references = []
    for annotation in answer_annotations:
        if not isinstance(annotation, Mapping):
            raise TypeError("annotation must be a dictionary")
        answer = annotation.get("native_answer", annotation.get("answer", annotation))
        if not isinstance(answer, Mapping):
            raise TypeError("native answer must be a dictionary")
        if not isinstance(answer.get("unanswerable"), bool):
            raise TypeError("unanswerable must be boolean")
        if answer["unanswerable"]:
            references.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
            continue
        extractive = _strings(answer["extractive_spans"], "extractive_spans")
        freeform = answer["free_form_answer"]
        yes_no = answer["yes_no"]
        if not isinstance(freeform, str):
            raise TypeError("free_form_answer must be a string")
        if yes_no is not None and not isinstance(yes_no, bool):
            raise TypeError("yes_no must be boolean or null")
        if extractive:
            answer_text, answer_type = ", ".join(extractive), "extractive"
        elif freeform:
            answer_text, answer_type = freeform, "abstractive"
        elif yes_no is True:
            answer_text, answer_type = "Yes", "boolean"
        elif yes_no is False:
            answer_text, answer_type = "No", "boolean"
        else:
            raise ValueError("answer annotation has no answer")
        evidence = _strings(answer["evidence"], "reference evidence")
        if text_evidence_only:
            evidence = [text for text in evidence if "FLOAT SELECTED" not in text]
        references.append({"answer": answer_text, "evidence": list(evidence), "type": answer_type})
    return references


def evidence_metrics(
    predicted_evidence: list[str], answer_annotations: list[dict], *,
    text_evidence_only: bool = False,
) -> dict:
    """Official per-question Evidence F1 plus separately maximized recall.

    Average evidence_f1 across questions (not documents or annotations) for the
    official macro metric. Recall is diagnostic, not a Qasper reported metric.
    """
    _strings(predicted_evidence, "predicted_evidence")
    references = references_from_annotations(answer_annotations, text_evidence_only=text_evidence_only)
    f1s = [paragraph_f1_score(predicted_evidence, ref["evidence"]) for ref in references]
    recalls = [paragraph_recall_diagnostic(predicted_evidence, ref["evidence"]) for ref in references]
    best_f1 = max(range(len(f1s)), key=f1s.__getitem__)
    best_recall = max(range(len(recalls)), key=recalls.__getitem__)
    return {
        "evidence_f1": f1s[best_f1],
        "evidence_recall": recalls[best_recall],
        "best_f1_reference_index": best_f1,
        "best_recall_reference_index": best_recall,
        "reference_count": len(references),
    }


def evaluate_qa(
    gold_annotations: Mapping[str, list[dict]],
    predictions: Mapping[str, dict], *, text_evidence_only: bool = False,
) -> dict:
    """Official aggregate QA metrics, with sidecar annotation inputs.

    predictions maps question ID to {predicted_answer: str,
    predicted_evidence: list[str]}. Missing predictions contribute zero to both
    overall means and are excluded from type means, following the official code.
    Extra predicted IDs are ignored. Answer and evidence maxima are independent.
    """
    answer_scores, evidence_scores = [], []
    by_type = {kind: [] for kind in ANSWER_TYPES}
    missing = 0
    for question_id, annotations in gold_annotations.items():
        references = references_from_annotations(annotations, text_evidence_only=text_evidence_only)
        if question_id not in predictions:
            missing += 1
            answer_scores.append(0.0)
            evidence_scores.append(0.0)
            continue
        prediction = predictions[question_id]
        answer = prediction["predicted_answer"]
        evidence = _strings(prediction["predicted_evidence"], "predicted_evidence")
        scores = [token_f1_score(answer, ref["answer"]) for ref in references]
        best = max(range(len(scores)), key=scores.__getitem__)
        answer_scores.append(scores[best])
        by_type[references[best]["type"]].append(scores[best])
        evidence_scores.append(max(paragraph_f1_score(evidence, ref["evidence"]) for ref in references))
    mean = lambda items: sum(items) / len(items) if items else 0.0
    return {
        "Answer F1": mean(answer_scores),
        "Answer F1 by type": {kind: mean(scores) for kind, scores in by_type.items()},
        "Evidence F1": mean(evidence_scores),
        "Missing predictions": missing,
    }
