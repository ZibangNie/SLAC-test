"""Synthetic two-arm projection, quota expectations and end-to-end isolation."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import run_cached_routing as stage
from SLAC.retrieval.routing.offline_router import RoutingInput, assign_family_folds, oof_routes


def encoded(rows):
    return b"\n".join(json.dumps(row).encode() for row in rows)


def arm_fixture():
    key = ("synthetic-family", "synthetic-doc", "synthetic-question")
    mapping = dict(zip(stage.IDENTITY, key)) | {"method": "arm", "selected_ids": ["unit"],
        "pack_sha256": "digest", "cache_key": "cache", "actual_evidence_tokens": 12}
    quality = mapping | {"official_answer_f1": 0.75, "predicted_answer": "PRIVATE SYNTHETIC SENTINEL"}
    return key, mapping, quality


def test_exact_arm_projection_does_not_retain_prediction():
    key, mapping, quality = arm_fixture()
    result = stage.project_arm(encoded([quality]), encoded([mapping]), "arm", {key})
    assert result[key]["answer_f1"] == 0.75
    assert "PRIVATE SYNTHETIC SENTINEL" not in repr(result)
    assert "predicted_answer" not in result[key]


@pytest.mark.parametrize("field,value", [("cache_key", "different"), ("selected_ids", ["another"]),
                                         ("pack_sha256", "changed"), ("actual_evidence_tokens", 13)])
def test_target_cannot_attach_to_a_different_original_pack(field, value):
    key, mapping, quality = arm_fixture()
    quality[field] = value
    with pytest.raises(ValueError, match="original pack"):
        stage.project_arm(encoded([quality]), encoded([mapping]), "arm", {key})


def test_missing_and_duplicate_answers_refuse_partial_denominator():
    key, mapping, quality = arm_fixture()
    with pytest.raises(ValueError, match="coverage"):
        stage.project_arm(b"", encoded([mapping]), "arm", {key})
    with pytest.raises(ValueError, match="duplicate"):
        stage.project_arm(encoded([quality, quality]), encoded([mapping]), "arm", {key})


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan")])
def test_invalid_target_metric_is_rejected(value):
    key, mapping, quality = arm_fixture()
    quality["official_answer_f1"] = value
    with pytest.raises(ValueError, match="Answer F1"):
        stage.project_arm(encoded([quality]), encoded([mapping]), "arm", {key})


def toy():
    rows, features, arms = [], [], {"bge": {}, "jev": {}}
    for family in range(10):
        for question in range(1 + family % 3):
            key = (f"family{family}", f"doc{family}", f"q{question}")
            values = ((family + 1) / 11, question / 3, 0.1, 0.0)
            rows.append(RoutingInput(key, values))
            features.append(dict(zip(stage.IDENTITY, key)) | {"support_task_count": 2 + question})
            arms["bge"][key] = {"answer_f1": 0.5, "actual_evidence_tokens": 100}
            arms["jev"][key] = {"answer_f1": 0.8 if family % 2 else 0.2, "actual_evidence_tokens": 150}
    targets = {row.key: arms["jev"][row.key]["answer_f1"] - 0.5 for row in rows}
    folds = assign_family_folds(tuple(row.key for row in rows))
    return tuple(rows), features, arms, targets, folds


def test_complete_toy_replay_preserves_all_six_methods_and_fold_quotas():
    rows, features, arms, targets, folds = toy()
    results = oof_routes(rows, targets, folds)
    decisions = stage.make_decisions(results)
    records = stage.score_decisions(decisions, features, arms)
    summary = stage.summarize(records)
    assert len(records) == 6 * len(rows)
    assert len(summary["paired_comparisons"]) == 15
    assert summary["family_count"] == 10
    expected = sum(f.quota for f in folds)
    for method in stage.METHODS[2:]:
        assert summary["methods"][method]["jev_queries"] == pytest.approx(expected)
    for decision in decisions:
        fold = folds[decision["fold_index"]]
        assert decision["jev_probabilities"]["random_expectation"] == fold.quota / len(fold.test_keys)
    assert summary["methods"]["random_expectation"]["expectation_only"]
    assert summary["methods"]["always_bge"]["answer_f1"]["question_weighted"] == 0.5
    assert summary["methods"]["always_jev"]["jev_queries"] == len(rows)
    assert sum(r["selected_support_judgments"] + r["skipped_support_judgments"] for r in records) == pytest.approx(
        6 * sum(r["support_task_count"] for r in features))


def test_held_out_target_probe_always_changes_every_test_value():
    rows, _, _, targets, folds = toy()
    targets = {key: 0.5 if i % 2 else 0.0 for i, key in enumerate(targets)}
    results = oof_routes(rows, targets, folds)
    report = stage.verify_target_isolation(rows, targets, folds, results)
    assert [r["changed_targets"] for r in report] == [len(f.test_keys) for f in folds]
    assert all(r["prediction_and_routes_unchanged"] for r in report)


def test_leaky_prediction_is_detected(monkeypatch):
    rows, _, _, targets, folds = toy()
    results = oof_routes(rows, targets, folds)

    def leaky(*args):
        out = list(results)
        out[0] = replace(out[0], gap_predictions=tuple(v + 0.1 for v in out[0].gap_predictions))
        return tuple(out)

    monkeypatch.setattr(stage, "oof_routes", leaky)
    with pytest.raises(ValueError, match="held-out targets"):
        stage.verify_target_isolation(rows, targets, folds, results)


def test_family_balancing_and_gate_do_not_select_question_weighted_winner():
    identities = [("large", "doc", f"q{i}") for i in range(3)] + [("small", "other", "q")]
    records = []
    for method in stage.METHODS:
        for key in identities:
            value = 0.5
            if method == "four_feature_ridge":
                value = 0.6 if key[0] == "large" else 0.3
            records.append(dict(zip(stage.IDENTITY, key)) | {"method": method, "answer_f1": value,
                "evidence_tokens": 100, "fold_index": 0, "jev_probability": 0.5,
                "selected_support_judgments": 1, "skipped_support_judgments": 1})
    result = stage.summarize(records)
    delta = result["primary_comparison"]["answer_f1_delta"]
    assert delta["question_weighted"] > 0
    assert delta["family_balanced"] < 0
    assert result["descriptive_structural_feature_gate_passed"] is False


def test_nonrandom_fractional_route_is_rejected():
    rows, features, arms, targets, folds = toy()
    decisions = stage.make_decisions(oof_routes(rows, targets, folds))
    decisions[0]["jev_probabilities"]["four_feature_ridge"] = 0.5
    with pytest.raises(ValueError, match="select one arm"):
        stage.score_decisions(decisions, features, arms)


def test_random_expectation_preserves_identical_arm_values_exactly():
    rows, features, arms, targets, folds = toy()
    decisions = stage.make_decisions(oof_routes(rows, targets, folds))
    for key in arms["bge"]:
        for arm in arms.values():
            arm[key]["answer_f1"] = 9 / 19
            arm[key]["actual_evidence_tokens"] = 101
    for row in decisions:
        row["jev_probabilities"]["random_expectation"] = 7 / 15
    records = stage.score_decisions(decisions, features, arms)
    assert all(row["answer_f1"] == 9 / 19 and row["evidence_tokens"] == 101 for row in records)
    assert all(pair["question_win_tie_loss"] == {"wins": 0, "ties": len(rows), "losses": 0}
               for pair in stage.summarize(records)["paired_comparisons"])


def test_cli_denies_network_and_redacts_errors(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["runner"])
    for owner, name in ((stage.socket, "create_connection"), (stage.socket.socket, "connect"),
                        (stage.socket.socket, "connect_ex")):
        monkeypatch.setattr(owner, name, getattr(owner, name))

    def fail(*args):
        with pytest.raises(RuntimeError, match="network disabled"):
            stage.socket.create_connection(("synthetic.invalid", 443))
        raise ValueError("PRIVATE SYNTHETIC SENTINEL")

    monkeypatch.setattr(stage, "run", fail)
    assert stage.main() == 1
    output = capsys.readouterr().out
    assert "PRIVATE SYNTHETIC SENTINEL" not in output
    assert json.loads(output) == {"status": "failed", "error_class": "ValueError", "api_calls": 0}
