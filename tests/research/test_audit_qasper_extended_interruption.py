"""An uncertain paid request is preserved, never turned into a reusable prefix."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import audit_qasper_extended_interruption as audit
from test_qasper_extended_development import planned, run_args, scored_response


@pytest.fixture
def interrupted(planned):
    args, config = planned
    execute = run_args(args, "timeout")
    calls = []
    def transport(backend, payload):
        calls.append(backend)
        if len(calls) == 3:
            raise TimeoutError("synthetic uncertain completion")
        return scored_response(backend, payload)
    with pytest.raises(RuntimeError, match="halted"):
        audit.stage.run(execute, client_factory=lambda path, **kw:
                        audit.stage.client.BoundedClient(path, transport=transport, **kw))
    return SimpleNamespace(plan=args.output, run=execute.output, output=args.output.parent / "audit"), calls


def test_prefix_replays_successes_and_preserves_uncertain_cost_without_calls(interrupted):
    args, calls = interrupted
    result = audit.audit(args)
    assert len(calls) == 3
    assert result["completed_requests"] == 2 and result["attempted_requests"] == 3
    assert result["planned_requests"] == 4 and result["unattempted_requests"] == 1
    assert result["new_accounting"]["unknown_cost_attempts"] == 1
    assert result["terminal_attempt"]["response_received"] is False
    assert result["partial_quality_metrics_computed"] is result["main_results_available"] is False
    assert result["api_calls_by_audit"] == 0 and result["automatic_retries"] == 0
    assert len(calls) == 3
    raw = (args.output / "audit.json").read_text(encoding="utf-8")
    assert "support:" not in raw and str(args.run) not in raw
    with pytest.raises(FileExistsError): audit.audit(args)


@pytest.mark.parametrize("mutation", ["cost", "count", "invent_response", "status", "extra_file",
                                     "ledger_total", "labels", "unknown_zero", "missing_request"])
def test_tampered_or_invented_timeout_outcomes_refused(interrupted, mutation):
    args, _ = interrupted
    ledger_path = args.run / "provider_calls" / "segment_001" / "ledger.json"
    ledger = audit.pilot.read_json(ledger_path)
    failure_path = args.run / "failure.json"
    failure = audit.pilot.read_json(failure_path)
    if mutation == "cost": ledger["attempts"][-1]["actual_cost_usd"] = "0"
    elif mutation == "count": failure["completed_requests"] += 1
    elif mutation == "invent_response": ledger["attempts"][-1]["response_id"] = "made-up"
    elif mutation == "status": ledger["attempts"][-1]["status"] = "in_flight"
    elif mutation == "extra_file": (ledger_path.parent / "response_003.json").write_text("{}", encoding="utf-8")
    elif mutation == "ledger_total": ledger["reservation_total_usd"] = "0"
    elif mutation == "labels": ledger["attempts"][-1]["labels"] = {}
    elif mutation == "unknown_zero": failure["stage_accounting"]["unknown_cost_attempts"] = 0
    else: (ledger_path.parent / "request_003.json").unlink()
    ledger_path.write_text(__import__("json").dumps(ledger), encoding="utf-8")
    failure_path.write_text(__import__("json").dumps(failure), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)): audit.audit(args)
    assert not args.output.exists()


def test_completed_response_reparsed_and_not_trusted_from_ledger(interrupted):
    args, _ = interrupted
    response = args.run / "provider_calls" / "segment_001" / "response_001.json"
    row = audit.pilot.read_json(response)
    row["usage"]["cost"] = 1
    response.write_text(__import__("json").dumps(row), encoding="utf-8")
    with pytest.raises(ValueError): audit.audit(args)
    assert not args.output.exists()


def test_mutation_during_audit_is_not_rebound(interrupted, monkeypatch):
    args, _ = interrupted
    original = audit.validate_timeout_prefix
    def change(*values):
        result = original(*values)
        (args.run / "unexpected.txt").write_text("changed", encoding="utf-8")
        return result
    monkeypatch.setattr(audit, "validate_timeout_prefix", change)
    with pytest.raises(ValueError, match="changed during"): audit.audit(args)
    assert not args.output.exists()
