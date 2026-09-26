"""New-stage planning, bounded segments, explicit prefix reuse and full audit."""
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import run_qasper_extended_development as extended
import openrouter_decision_client as client
import run_qasper_relation_pilot as pilot
from test_qasper_relation_pilot import make_prepared, responses


@pytest.fixture
def planned(tmp_path, monkeypatch):
    args = make_prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(extended, "utc_now", lambda: datetime(2026, 9, 26, 17, tzinfo=timezone.utc))
    monkeypatch.setattr(extended.preparation, "load_prepared", pilot.load_prepared)
    monkeypatch.setattr(client, "read_key", lambda *_: pytest.fail("no credential reads in offline tests"))
    history = tmp_path / "historical.json"
    history.write_text(json.dumps({"status": "completed", "all_results_available": True,
        "cumulative_accounting": {**pilot.accounting(), "attempts": 4,
                                  "unknown_cost_attempts": 2, "known_cost_usd": ".02", "reservation_usd": ".1"}}), encoding="utf-8")
    args.historical_summary = history
    config = extended.plan(args)
    yield args, config
    client.select_general_profile("gpt41mini")


def scored_response(backend, payload):
    response = responses(backend, payload)
    if backend == "jev":
        for answer in response["answers"].values():
            answer["probabilities"] = {"yes": .55, "no": .34, "unknown": .1}
    return response


def factory(path, **kwargs):
    return client.BoundedClient(path, transport=scored_response, **kwargs)


def run_args(args, name="run", **kwargs):
    return SimpleNamespace(plan=args.output, output=args.output.parent / name, key_file="unused", proxy=None, **kwargs)


def test_support_only_plan_binds_common_payloads_and_leaves_generation_budget(planned):
    args, config = planned
    assert config["api_calls"] == 0 and config["static_api_calls_planned"] == 0
    assert config["general_profile"] == "qwen36plus-json"
    assert config["predicted_reservations"]["requests"] == 4
    assert config["predicted_reservations"]["questions"] == 24
    assert Decimal(config["generation_reservation_remaining_usd"]) >= 1
    assert config["historical_pilot"]["accounting"]["unknown_cost_attempts"] == 2
    batches = pilot.read_json(args.output / "batches.json")
    assert all(batch["kind"] == "support" for batch in batches)
    assert "FORBIDDEN_QA_ANSWER" not in json.dumps(batches)
    for batch in batches:
        jev = client.make_payload(batch["tasks"], "support", "jev")
        qwen = client.make_payload(batch["tasks"], "support", "general")
        visible = json.loads(qwen["messages"][1]["content"])
        assert jev["state"] == visible["state"] and jev["questions"] == visible["questions"]
    assert extended.load_plan(args.output)[0] == config


def test_segments_obey_individual_caps_and_do_not_reset_total(monkeypatch):
    rows = [{"schedule_index": i, "task_ids": [str(i)] * 8, "reserved_usd": ".6",
             "input_allowance": 3000000, "output_allowance": 1024} for i in range(5)]
    segments = extended.segment_schedule(rows)
    assert [x["requests"] for x in segments] == [2, 2, 1]
    assert sum(Decimal(x["reservation_usd"]) for x in segments) == Decimal("3")
    too_large = [{**rows[0], "reserved_usd": "2.01"}]
    with pytest.raises(ValueError, match="one frozen request"):
        extended.segment_schedule(too_large)


def test_plan_rejects_stage_budget_excess_before_creating_outputs(planned, monkeypatch):
    args, _ = planned
    args.output = args.output.parent / "over-budget"
    original = client.reservation
    monkeypatch.setattr(client, "reservation", lambda payload, backend: (Decimal("1.01"), *original(payload, backend)[1:]))
    with pytest.raises(ValueError, match="stage cap"):
        extended.plan(args)
    assert not args.output.exists()


def test_complete_run_and_pure_generator_verifier(planned):
    args, config = planned
    execute = run_args(args)
    summary = extended.run(execute, client_factory=factory)
    assert summary["record_count"] == 30 and summary["new_api_calls"] == 4
    assert summary["raw_score_coverage"]["strict_distribution_invalid"] == 12
    assert summary["stage_accounting"]["attempts"] == 4
    assert summary["historical_pilot"]["accounting"]["attempts"] == 4
    assert Decimal(summary["stage_accounting"]["reservation_usd"]) == Decimal(config["predicted_reservations"]["reservation_usd"])
    restored, prepared, documents, rows, verified = extended.verify_completed_run(args.output, execute.output)
    assert restored == config and verified == summary and len(rows) == 30
    assert all(row["actual_evidence_tokens"] <= 1024 and row["selected_units"] <= int(row["method"][-1]) for row in rows)
    assert set(summary["output_files_sha256"]) >= {"run_manifest.json", "labels.json", "raw_scores.json", "per_question.jsonl", "traces.jsonl"}
    assert "FORBIDDEN_QA_ANSWER" not in json.dumps(summary)
    with pytest.raises(ValueError, match="already started"):
        extended.run(run_args(args, "duplicate-fresh-output"), client_factory=lambda *_a, **_k: pytest.fail("must refuse before client"))
    assert not (args.output / "execution.lock").exists()


def test_explicit_verified_completed_prefix_resume_retains_cost_once(planned):
    args, config = planned
    first = run_args(args, "interrupted")

    class StopAfterOne:
        def __init__(self, actual):
            self.actual, self.calls = actual, 0

        def __getattr__(self, key):
            return getattr(self.actual, key)

        def submit(self, *args):
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError("offline stop between provider calls")
            return self.actual.submit(*args)

    with pytest.raises(TimeoutError):
        extended.run(first, client_factory=lambda path, **kw: StopAfterOne(factory(path, **kw)))
    failure = pilot.read_json(first.output / "failure.json")
    assert failure["stage_accounting"]["attempts"] == 1
    assert not (first.output / "summary.json").exists()
    second = run_args(args, "resumed", resume_from=first.output)
    summary = extended.run(second, client_factory=factory)
    assert summary["new_api_calls"] == 3 and summary["reused_prefix_requests"] == 1
    assert summary["stage_accounting"]["attempts"] == 4
    assert Decimal(summary["stage_accounting"]["reservation_usd"]) == Decimal(config["predicted_reservations"]["reservation_usd"])
    extended.verify_completed_run(args.output, second.output)


def test_halted_provider_attempt_is_costed_and_refuses_automatic_retry(planned):
    args, _ = planned
    first = run_args(args, "halted")

    def unavailable(*_):
        raise TimeoutError("simulated ambiguous transport")

    with pytest.raises(RuntimeError, match="halted"):
        extended.run(first, client_factory=lambda path, **kw: client.BoundedClient(path, transport=unavailable, **kw))
    failure = pilot.read_json(first.output / "failure.json")
    assert failure["stage_accounting"]["attempts"] == 1
    assert failure["stage_accounting"]["unknown_cost_attempts"] == 1
    with pytest.raises(ValueError, match="failed or uncertain"):
        extended.run(run_args(args, "invalid-resume", resume_from=first.output),
                     client_factory=lambda *_a, **_k: pytest.fail("must not retry refused provider"))


def test_raw_score_contract_failure_stops_without_partial_main_results(planned):
    args, _ = planned
    execute = run_args(args)

    def invalid(backend, payload):
        response = scored_response(backend, payload)
        if backend == "jev":
            next(iter(response["answers"].values()))["probabilities"] = {"yes": .9}
        return response

    with pytest.raises(ValueError, match="reported-scores"):
        extended.run(execute, client_factory=lambda path, **kw: client.BoundedClient(path, transport=invalid, **kw))
    assert not (execute.output / "summary.json").exists()
    failure = pilot.read_json(execute.output / "failure.json")
    assert failure["stage_accounting"]["attempts"] == 1
    assert failure["stage_accounting"]["known_cost_usd"] == "0.001"


def test_provider_cost_overshoot_remains_in_failure_accounting(planned):
    args, _ = planned
    execute = run_args(args)

    def expensive(backend, payload):
        response = scored_response(backend, payload)
        response["usage"]["cost"] = ".1"
        return response

    with pytest.raises(RuntimeError, match="halted"):
        extended.run(execute, client_factory=lambda path, **kw: client.BoundedClient(path, transport=expensive, **kw))
    failure = pilot.read_json(execute.output / "failure.json")
    assert failure["stage_accounting"]["known_cost_usd"] == "0.1"
    assert failure["stage_accounting"]["cost_over_reservation_attempts"] == 1


@pytest.mark.parametrize("target", ["labels", "raw_scores", "records", "response", "summary", "generation_budget"])
def test_completed_output_tamper_refused(planned, target):
    args, _ = planned
    execute = run_args(args)
    extended.run(execute, client_factory=factory)
    if target in {"labels", "raw_scores"}:
        path = execute.output / (target + ".json")
        path.write_text("{}", encoding="utf-8")
    elif target == "records":
        path = execute.output / "per_question.jsonl"
        rows = pilot_audit_rows(path)
        rows[0]["official_evidence_f1"] = .123
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    elif target == "response":
        path = execute.output / "provider_calls/segment_001/response_001.json"
        response = pilot.read_json(path)
        response["model"] = "different-model"
        path.write_text(json.dumps(response), encoding="utf-8")
    else:
        path = execute.output / "summary.json"
        summary = pilot.read_json(path)
        if target == "generation_budget":
            summary["generation_reservation_remaining_usd"] = "5"
        else:
            summary["stage_accounting"]["reservation_usd"] = "0"
        path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        extended.verify_completed_run(args.output, execute.output)


def pilot_audit_rows(path):
    return extended.pilot_audit.read_rows(path)


def test_cross_segment_model_identity_drift_stops(planned, monkeypatch):
    args, _ = planned
    # Recreate a pre-execution plan with one request per accounting segment.
    monkeypatch.setitem(extended.SEGMENT_CAPS, "request_cap", 1)
    args.output = args.output.parent / "segmented-plan"
    extended.plan(args)
    calls = 0

    def drifting(backend, payload):
        nonlocal calls
        calls += 1
        response = scored_response(backend, payload)
        if backend == "general":
            response["model"] = sorted(client.RESPONSE_MODELS[backend])[0 if calls == 2 else -1]
        return response

    with pytest.raises(ValueError, match="changed across"):
        extended.run(run_args(args), client_factory=lambda path, **kw: client.BoundedClient(path, transport=drifting, **kw))


def test_fixed_overnight_deadline_refuses_before_loading_a_provider_client(planned, monkeypatch):
    args, _ = planned
    monkeypatch.setattr(extended, "utc_now", lambda: datetime(2026, 9, 27, 0, 59, tzinfo=timezone.utc))
    with pytest.raises(TimeoutError, match="overnight deadline"):
        extended.run(run_args(args), client_factory=lambda *_a, **_k: pytest.fail("no client after admission deadline"))
    failure = pilot.read_json(args.output.parent / "run" / "failure.json")
    assert failure["stage_accounting"]["attempts"] == 0
