"""Synthetic publication prerequisites and privacy checks; no real run reads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import publish_qasper_local_answers as publication


def complete_fixture():
    e = publication.experiment
    config = {"answer_reservation_usd": e.RESERVATION, "prior_night_accounting": e.PRIOR.copy()}
    ledger = {"reservation_total_usd": e.RESERVATION, "actual_reported_cost_usd": "0.123456789",
        "attempts": [{"status": "completed", "actual_cost_usd": "synthetic"} for _ in range(367)]}
    accounting = e.accounting(config, ledger, True)
    summary = {**accounting, "question_count": 77, "family_count": 24, "record_count": 462,
        "shared_resamples_sha256": "1" * 64, "plan_sha256": "2" * 64, "input_binding_sha256": "3" * 64}
    qw = (.4, .38, .5, .45, .455, .2)
    fb = (.44, .40, .43, .43, .425, .22)
    summary["metrics"] = [{"method": method, "questions": 77, "families": 24,
        "official_answer_f1_question_weighted": qw[i], "answer_f1_family_balanced": fb[i],
        "actual_evidence_tokens_question_weighted": 0 if method == "empty" else 410.1 + i,
        "actual_evidence_tokens_family_balanced": 0 if method == "empty" else 430.2 + i,
        "unanswerable_predictions": 10 + i,
        "official_metrics": {"Answer F1": qw[i], "Answer F1 by type": {kind: qw[i] for kind in publication.ANSWER_TYPES},
            "Evidence F1": .3, "Missing predictions": 0}}
        for i, method in enumerate(e.METHODS)]
    summary["paired_comparisons"] = []
    for plus, minus in e.PAIRS:
        a, b = e.METHODS.index(plus), e.METHODS.index(minus)
        pair = {"plus": plus, "minus": minus, "questions": 77, "families": 24,
            "question_positive": 20, "question_ties": 37, "question_negative": 20,
            "question_wins": 20, "question_losses": 20}
        for name, values in (("question_weighted", qw), ("family_balanced", fb)):
            delta = values[a] - values[b]
            pair[name] = {"delta": delta, "bootstrap_percentile_95": [delta - .05, delta + .05]}
        summary["paired_comparisons"].append(pair)
    audit = {"status": "verified_complete", "accounting": deepcopy(accounting),
        "all_bound_inputs_outputs_unchanged": True, "api_calls_by_audit": 0,
        "key_read": False, "partial_quality_metrics_computed": False}
    return summary, audit


def driver_fixture():
    return {"status": "completed_and_audited_pending_report", "paid_run_exit_code": 0,
        "audit_exit_code": 0, "child_pid": None, "readiness_sha256": "9" * 64,
        "stages": [{"name": name, "exit_code": 0} for name in ("local_answer_run", "local_answer_audit")]}


def test_complete_projection_preserves_all_six_methods_and_signed_intervals():
    summary, audit = complete_fixture()
    result = publication.project_summary(summary, audit)
    assert result["metrics"] == summary["metrics"]
    assert result["paired_comparisons"] == summary["paired_comparisons"]
    assert len(result["metrics"]) == len(result["paired_comparisons"]) == 6
    assert result["paired_comparisons"][0]["question_weighted"]["delta"] < 0
    assert result["paired_comparisons"][-1]["family_balanced"]["delta"] < 0
    md = publication.markdown(result)
    assert "-0.020000" in md and "-0.005000" in md
    assert "20 / 37 / 20" in md and "95%" in md


def test_historical_unknown_is_not_lost_when_generation_is_fully_known():
    result = publication.project_summary(*complete_fixture())
    cost = result["cost_accounting"]
    assert cost["generation_unknown_cost_attempts"] == 0
    assert cost["night_unknown_cost_attempts"] == 1 and cost["night_actual_total_known"] is False
    assert cost["night_known_reported_cost_subtotal_usd"] == "0.336259771"
    assert cost["night_attempted_reservation_usd"] == "2.8067622925"
    assert cost["night_attempts"] == 532
    assert "已知小计" in publication.markdown(result)


def test_unlisted_private_fields_are_never_copied():
    summary, audit = complete_fixture()
    summary["private_question_id"] = "SENTINEL-PRIVATE-ID"
    summary["per_question"] = [{"predicted_answer": "SENTINEL-PRIVATE-ANSWER"}]
    summary["metrics"][0]["selected_ids"] = ["SENTINEL-UNIT-ID"]
    summary["paired_comparisons"][0]["debug_question"] = "SENTINEL-QUESTION-TEXT"
    audit["private_path"] = "SENTINEL-LOCAL-PATH"
    result = publication.project_summary(summary, audit)
    serialized = json.dumps(result)
    assert "SENTINEL" not in serialized and "SENTINEL" not in publication.markdown(result)
    assert "input_sha256" not in result and "per_question" not in result


@pytest.mark.parametrize("mutation", ["partial_status", "attempts", "records", "missing_method", "missing_pair",
    "method_denominator", "pair_denominator", "missing_predictions", "prior_unknown", "audit_accounting",
    "delta_disagrees", "interval_reversed", "nan", "empty_tokens", "known_cost_exceeds", "boolean_count"])
def test_incomplete_or_inconsistent_aggregate_cannot_publish(mutation):
    summary, audit = complete_fixture()
    if mutation == "partial_status": audit["status"] = "verified_incomplete"
    elif mutation == "attempts": summary["new_api_calls"] = 366
    elif mutation == "records": summary["record_count"] = 461
    elif mutation == "missing_method": summary["metrics"].pop()
    elif mutation == "missing_pair": summary["paired_comparisons"].pop()
    elif mutation == "method_denominator": summary["metrics"][0]["questions"] = 76
    elif mutation == "pair_denominator": summary["paired_comparisons"][0]["question_ties"] = 38
    elif mutation == "missing_predictions": summary["metrics"][0]["official_metrics"]["Missing predictions"] = 1
    elif mutation == "prior_unknown": summary["prior_night_accounting"]["unknown_cost_attempts"] = 0
    elif mutation == "audit_accounting": audit["accounting"]["known_generation_cost_usd"] = "0"
    elif mutation == "delta_disagrees": summary["paired_comparisons"][0]["question_weighted"]["delta"] = .123
    elif mutation == "interval_reversed": summary["paired_comparisons"][0]["family_balanced"]["bootstrap_percentile_95"] = [.3, -.3]
    elif mutation == "nan": summary["metrics"][0]["answer_f1_family_balanced"] = float("nan")
    elif mutation == "empty_tokens": summary["metrics"][-1]["actual_evidence_tokens_question_weighted"] = 2
    elif mutation == "known_cost_exceeds":
        summary["known_generation_cost_usd"] = audit["accounting"]["known_generation_cost_usd"] = "2"
    elif mutation == "boolean_count": summary["metrics"][0]["unanswerable_predictions"] = True
    with pytest.raises(ValueError): publication.project_summary(summary, audit)


@pytest.mark.parametrize("mutation", ["running", "paid_failure", "audit_failure", "child_alive", "child_field_missing", "stage_failure", "stage_missing"])
def test_driver_must_finish_both_stages_before_publication(mutation):
    driver = driver_fixture()
    if mutation == "running": driver["status"] = "running"
    elif mutation == "paid_failure": driver["paid_run_exit_code"] = 1
    elif mutation == "audit_failure": driver["audit_exit_code"] = 1
    elif mutation == "child_alive": driver["child_pid"] = 123
    elif mutation == "child_field_missing": driver.pop("child_pid")
    elif mutation == "stage_failure": driver["stages"][0]["exit_code"] = 1
    elif mutation == "stage_missing": driver["stages"].pop()
    with pytest.raises(ValueError): publication.validate_driver(driver)


def release_fixture(tmp_path):
    e = publication.experiment
    plan = tmp_path / "plan"; plan.mkdir()
    for name in e.PLAN_FILES: e.write(plan / name, {"synthetic": True})
    summary, audit = complete_fixture()
    summary["plan_sha256"] = e.legacy.digest(plan / "experiment_config.json")
    summary_path = tmp_path / "summary.json"; e.write(summary_path, summary)
    audit_dir = tmp_path / "audit"; audit_dir.mkdir()
    e.write(audit_dir / "audit.json", audit)
    driver_path = tmp_path / "driver.json"; e.write(driver_path, driver_fixture())
    bindings = e.hashes([summary_path, audit_dir / "audit.json", driver_path,
        *(plan / name for name in e.PLAN_FILES), Path(e.__file__)])
    e.write(audit_dir / "source_binding.json", {"schema": "synthetic-root-release",
        "root_release": publication.ROOT_RELEASE, "input_sha256": bindings})
    return SimpleNamespace(plan=plan, summary=summary_path, audit_directory=audit_dir, driver_state=driver_path,
        report=tmp_path / "public/report.md", aggregate=tmp_path / "public/result.json")


def test_file_publication_requires_release_and_source_bindings_without_private_reads(tmp_path, monkeypatch):
    args = release_fixture(tmp_path)
    monkeypatch.setattr(publication.experiment, "load_plan", lambda *a: pytest.fail("publisher must not audit model/data again"))
    monkeypatch.setattr(publication.experiment, "audit", lambda *a: pytest.fail("publisher must not start audit"))
    monkeypatch.setattr(publication.experiment, "score_all", lambda *a: pytest.fail("publisher must not recompute any question scores"))
    result = publication.publish(args)
    assert result["status"] == "published_complete_aggregates"
    public = publication.read(args.aggregate)
    assert "publication_provenance" in public
    assert str(tmp_path) not in json.dumps(public)
    assert args.report.exists()
    with pytest.raises(FileExistsError): publication.publish(args)


@pytest.mark.parametrize("mutation", ["no_release", "changed_summary", "changed_driver", "missing_plan_binding", "wrong_plan_identity"])
def test_binding_or_release_changes_block_before_output(tmp_path, mutation):
    args = release_fixture(tmp_path)
    binding_path = args.audit_directory / "source_binding.json"
    binding = publication.read(binding_path)
    if mutation == "no_release": binding["root_release"] = "not_released"
    elif mutation == "changed_summary": args.summary.write_text("{}", encoding="utf-8")
    elif mutation == "changed_driver": args.driver_state.write_text("{}", encoding="utf-8")
    elif mutation == "missing_plan_binding": binding["input_sha256"].pop(str((args.plan / "jobs.json").resolve()))
    elif mutation == "wrong_plan_identity":
        summary = publication.read(args.summary); summary["plan_sha256"] = "0" * 64
        args.summary.write_text(json.dumps(summary), encoding="utf-8")
        binding["input_sha256"][str(args.summary.resolve())] = publication.experiment.legacy.digest(args.summary)
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises((ValueError, KeyError)): publication.publish(args)
    assert not args.report.exists() and not args.aggregate.exists()
