"""Synthetic post-hoc analysis and tamper refusal; no provider/key access."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import analyze_qasper_relation_pilot as analysis
import openrouter_decision_client as api
import run_qasper_relation_pilot as pilot
from test_qasper_relation_pilot import make_prepared, responses, annotation


@pytest.fixture
def completed(tmp_path, monkeypatch):
    api.select_general_profile("gpt41mini")
    monkeypatch.setattr(api, "read_key", lambda *_: pytest.fail("analysis must not read credentials"))
    args = make_prepared(tmp_path, monkeypatch)
    pilot.plan(args)
    run_dir = tmp_path / "run"
    pilot.run(SimpleNamespace(plan=args.output, output=run_dir, key_file="unused", proxy=None),
              client_factory=lambda path, **kwargs: api.BoundedClient(path, transport=responses, **kwargs))
    result = SimpleNamespace(plan=args.output, run=run_dir, prepared=args.prepared, output=tmp_path / "analysis")
    yield result
    api.select_general_profile("gpt41mini")


def loaded(args):
    config = pilot.read_json(args.plan / "experiment_config.json")
    prepared = pilot.read_json(args.prepared / "prepared.json")
    return (prepared, pilot.read_json(args.run / "labels.json"), analysis.read_rows(args.run / "per_question.jsonl"),
            analysis.read_rows(args.run / "traces.jsonl"), analysis.read_rows(args.plan / "baseline_per_question.jsonl"),
            pilot.selected_gold(config["sidecar"], prepared))


def test_completed_analysis_verifies_replay_and_emits_aggregates_only(completed):
    result = analysis.analyze(completed)
    assert result["status"] == "completed" and result["question_count"] == 2
    assert result["api_calls"] == 0 and result["significance_claimed"] is False
    assert "post-hoc" in result["analysis_type"]
    assert result["input_hashes_unchanged"] is True
    assert len(result["input_binding_sha256"]) == 64
    assert all(set(row) == {"path_sha256", "content_sha256"} for row in result["input_sha256"])
    assert all(row["all_equal"] for row in result["I_C_equivalence"].values())
    assert result["label_diagnostics"]["static"]["agreement_rate"] == 1
    assert result["label_diagnostics"]["support"]["agreement_rate"] == 1
    assert result["reference_strata"]["all_nonempty_references"]["questions"] == 2
    assert result["reference_strata"]["any_empty_reference"]["questions"] == 0
    content = (completed.output / "analysis.json").read_text(encoding="utf-8")
    for forbidden in ('"q0"', '"q1"', '"u0"', "FORBIDDEN_QA_ANSWER", "What is 0?", "Evidence 0", str(completed.run)):
        assert forbidden not in content
    assert len(result["paired_method_minus_dense_top3_capped"]) == 6


@pytest.mark.parametrize("target", ["record", "trace", "labels", "response", "summary", "ledger", "baseline", "extra_call"])
def test_changed_or_incomplete_artifacts_are_refused(completed, target):
    if target == "record":
        path = completed.run / "per_question.jsonl"
        rows = analysis.read_rows(path)
        rows[0]["official_evidence_f1"] = 0.123
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    elif target == "trace":
        path = completed.run / "traces.jsonl"
        rows = analysis.read_rows(path)
        rows[0]["selected_ids"] = []
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    elif target == "labels":
        path = completed.run / "labels.json"
        value = pilot.read_json(path)
        value["jev"].pop(next(iter(value["jev"])))
        path.write_text(json.dumps(value), encoding="utf-8")
    elif target == "response":
        path = completed.run / "provider_calls" / "response_001.json"
        value = pilot.read_json(path)
        value["id"] = "different-response"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif target == "summary":
        path = completed.run / "summary.json"
        value = pilot.read_json(path)
        value["status"] = "halted"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif target == "ledger":
        path = completed.run / "provider_calls" / "ledger.json"
        value = pilot.read_json(path)
        value["attempts"][0]["status"] = "halted"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif target == "baseline":
        path = completed.plan / "baseline_per_question.jsonl"
        path.write_text("{}", encoding="utf-8")
    else:
        pilot.write_json(completed.run / "provider_calls" / "response_999.json", {})
    with pytest.raises(ValueError):
        analysis.analyze(completed)
    assert not completed.output.exists()


def test_summary_reference_strata_use_any_official_empty_reference(completed):
    prepared, labels, records, traces, baseline, annotations = loaded(completed)
    annotations[("doc", "q0")].append(annotation([]))
    result = analysis.summarize(prepared, labels, records, traces, baseline, annotations)
    assert result["reference_strata"]["any_empty_reference"]["questions"] == 1
    assert result["reference_strata"]["all_nonempty_references"]["questions"] == 1
    # Official handling discards evidence for an unanswerable annotation.
    annotations[("doc", "q1")][0]["native_answer"]["unanswerable"] = True
    result = analysis.summarize(prepared, labels, records, traces, baseline, annotations)
    assert result["reference_strata"]["any_empty_reference"]["questions"] == 2
    assert result["reference_strata"]["all_nonempty_references"]["questions"] == 0


def test_label_confusion_counts_unique_tasks_and_not_replayed_queries(completed):
    prepared, labels, *_ = loaded(completed)
    task_id = prepared["support_tasks"][0]["id"]
    labels["jev"][task_id] = "no"
    result = analysis.label_diagnostics(prepared, labels)
    count = len(prepared["support_tasks"])
    assert result["support"]["task_count"] == count
    assert result["support"]["confusion_rows_jev_columns_general"]["no"]["yes"] == 1
    assert result["support"]["agreement_count"] == count - 1
    assert result["support"]["label_counts"]["jev"]["no"] == 1


@pytest.mark.parametrize("change", ["duplicate", "missing", "ic_pack", "label"])
def test_summary_refuses_incomplete_or_inconsistent_inputs(completed, change):
    prepared, labels, records, traces, baseline, annotations = loaded(completed)
    if change == "duplicate":
        records.append(deepcopy(records[0]))
    elif change == "missing":
        traces.pop()
    elif change == "ic_pack":
        next(row for row in records if row["method"] == "C_jev")["pack_sha256"] = "different"
    else:
        labels["jev"].pop(next(iter(labels["jev"])))
    with pytest.raises(ValueError):
        analysis.summarize(prepared, labels, records, traces, baseline, annotations)


def test_delta_statistics_preserve_family_macro_and_win_tie_loss():
    questions = [("f1", "d1", "q1"), ("f1", "d1", "q2"), ("f2", "d2", "q3")]
    plus = {key: {"f1": value} for key, value in zip(questions, [1, 0, 0])}
    minus = {key: {"f1": value} for key, value in zip(questions, [0, 0, 1])}
    result = analysis.delta_stats(plus, minus, questions, "f1")
    assert result["question_macro_delta"] == 0
    assert result["family_macro_delta"] == -0.25
    assert result["positive"] == result["equal"] == result["negative"] == 1


def test_fully_reused_completed_run_can_be_verified_without_new_provider_calls(completed):
    new_plan, new_run = completed.plan.parent / "reuse-plan", completed.run.parent / "reuse-run"
    config = pilot.read_json(completed.plan / "experiment_config.json")
    pilot.plan(SimpleNamespace(prepared=completed.prepared, sidecar=config["sidecar"], tokenizer=config["tokenizer"],
                               output=new_plan, reuse_run=completed.run))
    pilot.run(SimpleNamespace(plan=new_plan, output=new_run, key_file="unused", proxy=None),
              client_factory=lambda *_args, **_kwargs: pytest.fail("fully reused run must not create a provider client"))
    completed.plan, completed.run = new_plan, new_run
    result = analysis.analyze(completed)
    assert result["status"] == "completed"
    assert not (new_run / "provider_calls").exists()
