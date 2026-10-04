"""Synthetic tests only: no credential access or live request is exercised."""

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import socket
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import run_conditional_probe as runner


@pytest.fixture
def fixture():
    return json.loads((runner.ROOT / runner.FIXTURE_NAME).read_bytes())


@pytest.fixture
def plan(fixture):
    entries, _ = runner.freeze_requests(fixture)
    return {"requests": entries, "planned_reserved_usd": "0.030",
            "source_sha256": {}, "schema": runner.PLAN_SCHEMA}


def completed_attempt(entry, *, added="yes", cost="0.00001"):
    labels = {name: "unknown" for name in entry["expected_ids"]}
    if "conditional_added_information" in labels:
        labels["conditional_added_information"] = added
        labels["conflict"] = "no"
    return {"cache_key": entry["wire_cache_key"], "core_cache_key": entry["core_cache_key"],
            "request_sha256": entry["wire_body_sha256"], "reserved_usd": entry["reserved_usd"],
            "status": "completed", "cost_status": "provider_reported", "actual_cost_usd": cost,
            "labels": labels}


def test_plan_has_fixed_physical_order_and_separate_supervision(fixture):
    entries, wires = runner.freeze_requests(fixture)
    assert [(e["pair_id"], e["case_id"], e["arm"]) for e in entries] == [
        ("P01", "C01", "standalone"), ("P01", "C01", "plain_conditional"),
        ("P01", "C02", "plain_conditional"), ("P02", "C03", "standalone"),
        ("P02", "C03", "plain_conditional"), ("P02", "C04", "plain_conditional")]
    assert len({wire.cache_key for wire in wires}) == 6
    assert sum(len(wire.expected_ids) for wire in wires) == 10
    assert sum(Decimal(wire.reserved_usd) for wire in wires) <= Decimal("0.03")
    assert [entry["expected_added_information"] for entry in entries] == [None, "yes", "no", None, "yes", "no"]
    for wire in wires:
        assert len(wire.payload_bytes) <= 24000
        assert not any(value in wire.payload_bytes for value in (b"pair_id", b"case_id", b"supervision", b"expected_labels"))
        assert "relations" not in wire.payload()["state"]


def test_design_metadata_mutation_does_not_change_wire_bytes(fixture):
    _, before = runner.freeze_requests(fixture)
    changed = deepcopy(fixture)
    for pair in changed["pairs"][:2]:
        pair["supervision"]["rationale"] = "HIDDEN_REASON"
        pair["supervision"]["fixture_group"] = "HIDDEN_GROUP"
        for case in pair["states"]:
            case["expected"] = "HIDDEN_EXPECTED"
    _, after = runner.freeze_requests(changed)
    assert [wire.payload_bytes for wire in before] == [wire.payload_bytes for wire in after]
    assert all(b"HIDDEN" not in wire.payload_bytes for wire in after)


def test_fixed_pair_selection_cannot_be_reordered(fixture):
    fixture["pairs"][:2] = list(reversed(fixture["pairs"][:2]))
    with pytest.raises(ValueError, match="pair ordering"):
        runner.freeze_requests(fixture)


def test_plan_is_immutable_and_source_or_body_drift_rejected(tmp_path, monkeypatch):
    directory = tmp_path / "plan"
    runner.prepare_plan(directory)
    path = directory / "plan.json"
    original = path.read_bytes()
    assert runner.verify_plan(path)[2] == runner.sha(original)
    with pytest.raises(FileExistsError):
        runner.prepare_plan(directory)
    snapshots = runner._snapshots()
    snapshots[runner.SOURCE_NAMES[0]] += b"\n# source drift\n"
    monkeypatch.setattr(runner, "_snapshots", lambda: snapshots)
    with pytest.raises(ValueError, match="source commitments"):
        runner.verify_plan(path)
    monkeypatch.undo()
    changed = json.loads(original)
    changed["requests"][0]["wire_payload"]["state"]["query"] = "mutated query"
    path.write_bytes(runner.canonical(changed) + b"\n")
    with pytest.raises(ValueError, match="plan or wire bytes"):
        runner.verify_plan(path)


def test_complete_readout_only_scores_four_added_observations(plan):
    attempts = [completed_attempt(entry, added=choice) for entry, choice in zip(
        plan["requests"], ("yes", "yes", "no", "yes", "yes", "no"))]
    summary = runner.analyze_ledger(plan, {"status": "completed", "halt_reason": None, "attempts": attempts})
    assert summary["planned_requests"] == summary["completed_requests"] == 6
    assert summary["planned_typed_questions"] == summary["typed_labels_observed"] == 10
    assert summary["added_observations_planned"] == summary["added_matches_planned_expectation"] == 4
    assert summary["paired_reversals_planned"] == summary["paired_reversals_observed"] == 2
    assert Decimal(summary["provider_reported_cost_usd"]) == Decimal("0.00006")
    assert summary["cost_unknown_attempts"] == 0
    assert summary["total_actual_cost_known"]
    assert all(not any("accuracy" in key for key in row) for row in summary["requests"])


def test_unknown_mismatch_and_unobserved_are_distinct_and_denominators_fixed(plan):
    attempts = [completed_attempt(plan["requests"][0]),
                completed_attempt(plan["requests"][1], added="unknown"),
                completed_attempt(plan["requests"][2], added="yes")]
    summary = runner.analyze_ledger(plan, {"status": "halted", "halt_reason": "deadline", "attempts": attempts})
    assert len(summary["requests"]) == 6
    added = [row for row in summary["requests"] if "added_observation_status" in row]
    assert [row["added_observation_status"] for row in added] == ["unknown", "mismatch", "unobserved", "unobserved"]
    assert summary["added_observations_completed"] == 2
    assert summary["added_observations_planned"] == 4
    assert summary["added_matches_planned_expectation"] == 0
    assert summary["paired_reversals_planned"] == 2


def test_inflight_unknown_cost_keeps_reservation_and_all_planned_rows(plan):
    first = completed_attempt(plan["requests"][0])
    second = completed_attempt(plan["requests"][1])
    second.update(status="in_flight", cost_status="cost_unknown")
    del second["actual_cost_usd"], second["labels"]
    summary = runner.analyze_ledger(plan, {"status": "running", "halt_reason": None, "attempts": [first, second]})
    assert summary["completed_requests"] == 1
    assert summary["attempted_requests"] == 2
    assert summary["unattempted_requests"] == 4
    assert summary["in_flight_attempts"] == summary["cost_unknown_attempts"] == 1
    assert not summary["total_actual_cost_known"]
    assert Decimal(summary["attempted_reserved_usd"]) == Decimal("0.010")
    assert Decimal(summary["unresolved_reserved_usd"]) == Decimal("0.005")
    assert summary["added_matches_planned_expectation"] == 0


def test_zero_attempt_failure_has_full_denominators(plan):
    summary = runner.analyze_ledger(plan, {"status": "halted", "halt_reason": "preflight", "attempts": []})
    assert summary["attempted_requests"] == summary["completed_requests"] == 0
    assert summary["unattempted_requests"] == len(summary["requests"]) == 6
    assert summary["added_observations_planned"] == 4
    assert summary["paired_reversals_planned"] == 2


@pytest.mark.parametrize("change", ["reorder", "duplicate", "body_sha", "reservation"])
def test_readout_rejects_ledger_identity_or_accounting_drift(plan, change):
    attempts = [completed_attempt(entry) for entry in plan["requests"][:2]]
    if change == "reorder":
        attempts.reverse()
    elif change == "duplicate":
        attempts.append(attempts[0])
    elif change == "body_sha":
        attempts[0]["request_sha256"] = "0" * 64
    else:
        attempts[0]["reserved_usd"] = "0"
    with pytest.raises(ValueError):
        runner.analyze_ledger(plan, {"status": "completed", "attempts": attempts})


def test_public_summary_omits_raw_responses_provider_ids_and_private_paths(plan):
    attempt = completed_attempt(plan["requests"][0])
    attempt.update(raw_response="PRIVATE_RAW", provider_id="PRIVATE_ID", key_file="PRIVATE_PATH")
    summary = runner.analyze_ledger(plan, {"status": "halted", "attempts": [attempt]})
    assert b"PRIVATE" not in runner.canonical(summary)


def test_fake_end_to_end_never_opens_network_or_reads_key(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access is forbidden in runner tests")
    monkeypatch.setattr(socket, "socket", forbidden)
    plan_dir = tmp_path / "plan"
    runner.prepare_plan(plan_dir)
    calls = []
    choices = ("unknown", "yes", "no", "unknown", "yes", "no")
    def fake(wire, timeout_seconds):
        position = len(calls)
        calls.append(wire.cache_key)
        assert 0 < timeout_seconds <= 30
        return {"model": runner.projector.RESPONSE_MODEL, "provider": "TypeSafe", "answers": {
            name: {"type": "choice", "choice": choices[position] if name != "conflict" else "no"}
            for name in wire.expected_ids},
            "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100,
                      "input_tokens": 100, "output_tokens": 0, "cost": 0.00001}}
    output = tmp_path / "run"
    ledger = runner.execute_plan(plan_dir / "plan.json", output, key_file=tmp_path / "must-not-be-read",
                                 transport=fake, live=False)
    assert ledger["status"] == "completed"
    assert len(calls) == 6
    summary = json.loads((output / "summary.json").read_bytes())
    assert summary["paired_reversals_observed"] == 2
    assert runner.write_analysis(plan_dir / "plan.json", output) == summary
    with pytest.raises(FileExistsError):
        runner.execute_plan(plan_dir / "plan.json", output, transport=fake)
    assert len(calls) == 6
