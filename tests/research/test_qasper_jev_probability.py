"""Synthetic score validation and complete-run replay; no credentials or network."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import analyze_qasper_jev_probability as probability
import openrouter_decision_client as api
import run_qasper_relation_pilot as pilot
from test_qasper_relation_counterfactuals import synthetic
from test_qasper_relation_pilot import make_prepared, responses, annotation
from test_qasper_relation_replay import CharacterTokenizer, document


@pytest.mark.parametrize("value", [None, {}, {"yes": 1}, {"yes": .5, "no": .3, "unknown": .2, "extra": 0},
    {"yes": float("nan"), "no": 0, "unknown": 0}, {"yes": float("inf"), "no": 0, "unknown": 0},
    {"yes": True, "no": 0, "unknown": 0}, {"yes": "1", "no": 0, "unknown": 0},
    {"yes": 1.1, "no": 0, "unknown": -.1}])
def test_both_contracts_refuse_missing_keys_or_invalid_values(value):
    assert all(probability.validate_probabilities(value, contract) for contract in probability.CONTRACTS)


def test_distribution_and_raw_score_contracts_are_distinct_without_normalization():
    exact = {"yes": .22, "no": .32, "unknown": .45999999999999996}
    rounded = {"yes": .55, "no": .34, "unknown": .1}
    original = deepcopy(rounded)
    assert probability.validate_probabilities(exact) is None
    assert probability.validate_probabilities(rounded) == "probabilities_do_not_sum_to_one"
    assert probability.validate_probabilities(rounded, "reported-scores") is None
    assert rounded == original
    assert probability.validate_probabilities({"yes": .54, "no": .34, "unknown": .1}, "reported-scores")
    assert probability.validate_probabilities({"yes": .57, "no": .35, "unknown": .1}, "reported-scores")
    with pytest.raises(ValueError, match="contract"):
        probability.validate_probabilities(exact, "anything")


def test_two_rules_preserve_no_exclusion_and_distinguish_ordinal_tiers():
    units = document(["a", "b", "c", "d"])
    ids = [u.unit_id for u in units]
    labels = dict(zip(ids, ["yes", "unknown", "yes", "no"]))
    scores = {uid: {"yes": p, "no": 1 - p, "unknown": 0} for uid, p in zip(ids, [.4, .8, .6, .9])}
    assert probability.probability_ranking(units, ids, labels, scores, ids, "ordinal_then_p_yes") == [2, 0, 1]
    assert probability.probability_ranking(units, ids, labels, scores, ids, "p_yes_only") == [1, 2, 0]
    scores["u0"]["yes"] = .6
    scores["u0"]["no"] = .4
    assert probability.probability_ranking(units, ids, labels, scores, ["u2", "u0", "u3", "u1"], "ordinal_then_p_yes") == [2, 0, 1]


def test_frozen_choices_not_recomputed_from_score_argmax():
    units = document(["a", "b"])
    scores = {"u0": {"yes": .9, "no": .1, "unknown": 0}, "u1": {"yes": .2, "no": .8, "unknown": 0}}
    assert probability.probability_ranking(units, ["u0", "u1"], {"u0": "no", "u1": "yes"}, scores,
        ["u0", "u1"], "p_yes_only") == [1]


def test_replay_reports_all_six_rules_and_selection_is_gold_independent():
    prepared, documents, annotations, labels = synthetic()
    scores = {task["id"]: {"yes": .9 if task["unit_id"] == "u1" else .5,
                           "no": .1 if task["unit_id"] == "u1" else .5, "unknown": 0}
              for task in prepared["support_tasks"]}
    rows, traces = probability.replay(prepared, documents, annotations, labels, scores, CharacterTokenizer())
    changed, changed_traces = probability.replay(prepared, documents,
        {key: [annotation([])] for key in annotations}, labels, scores, CharacterTokenizer())
    assert len(rows) == 24 and len(traces) == 12
    assert traces == changed_traces
    assert all(row["actual_evidence_tokens"] <= 1024 and row["selected_units"] <= int(row["method"][-1]) for row in rows)
    assert [row["selected_ids"] for row in rows] == [row["selected_ids"] for row in changed]
    assert [row["official_evidence_f1"] for row in rows] != [row["official_evidence_f1"] for row in changed]
    report = probability.summarize(prepared, rows)
    assert report["new_method_count"] == 6 and len(report["paired_comparisons"]) == 12


@pytest.fixture
def completed_scores(tmp_path, monkeypatch):
    api.select_general_profile("gpt41mini")
    monkeypatch.setattr(api, "read_key", lambda *_: pytest.fail("diagnostic must not read credentials"))
    args = make_prepared(tmp_path, monkeypatch)
    pilot.plan(args)
    run_dir = tmp_path / "run"

    def with_scores(backend, payload):
        response = responses(backend, payload)
        if backend == "jev":
            for key, answer in response["answers"].items():
                if "yes" in payload["questions"][key]["criteria"]:
                    answer["probabilities"] = {"yes": .55, "no": .34, "unknown": .1}
                    answer["confidence"] = {"deliberately": "not used"}
        return response

    pilot.run(SimpleNamespace(plan=args.output, output=run_dir, key_file="unused", proxy=None),
              client_factory=lambda path, **kwargs: api.BoundedClient(path, transport=with_scores, **kwargs))
    yield SimpleNamespace(plan=args.output, run=run_dir, prepared=args.prepared, output=tmp_path / "score-analysis")
    api.select_general_profile("gpt41mini")


def test_strict_default_refuses_all_ranking_results_but_preserves_coverage(completed_scores):
    report = probability.analyze(completed_scores)
    assert report["status"] == "refused_score_contract"
    coverage = report["probability_coverage"]
    assert coverage["invalid_task_count"] == coverage["support_task_count"]
    assert coverage["strict_distribution_invalid_but_support_eligible_count"] == coverage["support_task_count"]
    assert not (completed_scores.output / "per_question.jsonl").exists()
    assert (completed_scores.output / "probability_sources.jsonl").exists()
    assert "metrics" not in report


def test_explicit_reported_scores_retains_every_item_and_is_separate_from_distribution(completed_scores):
    completed_scores.score_contract = "reported-scores"
    report = probability.analyze(completed_scores)
    assert report["status"] == "completed" and report["new_method_count"] == 6
    coverage = report["probability_coverage"]
    assert coverage["invalid_task_count"] == 0
    assert coverage["strict_distribution_invalid_task_count"] == coverage["support_task_count"]
    assert report["confidence_used"] is report["calibration_claimed"] is False
    assert coverage["renormalized"] is False
    content = (completed_scores.output / "summary.json").read_text(encoding="utf-8")
    for forbidden in ('"q0"', '"q1"', '"u0"', "FORBIDDEN_QA_ANSWER", "What is 0?", "Evidence 0", str(completed_scores.run)):
        assert forbidden not in content


def test_reused_complete_run_extracts_scores_from_original_responses(completed_scores):
    new_plan, new_run = completed_scores.plan.parent / "reuse-plan", completed_scores.run.parent / "reuse-run"
    config = pilot.read_json(completed_scores.plan / "experiment_config.json")
    pilot.plan(SimpleNamespace(prepared=completed_scores.prepared, sidecar=config["sidecar"], tokenizer=config["tokenizer"],
                               output=new_plan, reuse_run=completed_scores.run))
    pilot.run(SimpleNamespace(plan=new_plan, output=new_run, key_file="unused", proxy=None),
              client_factory=lambda *_args, **_kwargs: pytest.fail("reused run must not create client"))
    completed_scores.plan, completed_scores.run = new_plan, new_run
    completed_scores.score_contract = "reported-scores"
    report = probability.analyze(completed_scores)
    assert report["status"] == "completed"
    assert all(row["source_reused"] for row in probability.audit.read_rows(completed_scores.output / "probability_sources.jsonl"))


def test_score_changed_after_verification_is_refused(completed_scores, monkeypatch):
    ledger = pilot.read_json(completed_scores.run / "provider_calls" / "ledger.json")
    item = next(a for a in ledger["attempts"] if a["backend"] == "jev" and a["kind"] == "support")
    path = completed_scores.run / "provider_calls" / f"response_{item['attempt']:03d}.json"
    real_analyze = probability.audit.analyze

    def mutate_after_verification(args):
        verified = real_analyze(args)
        response = pilot.read_json(path)
        next(iter(response["answers"].values()))["probabilities"]["yes"] = .56
        path.write_text(json.dumps(response), encoding="utf-8")
        return verified

    monkeypatch.setattr(probability.audit, "analyze", mutate_after_verification)
    with pytest.raises(ValueError):
        probability.analyze(completed_scores)
    assert not (completed_scores.output / "summary.json").exists()
