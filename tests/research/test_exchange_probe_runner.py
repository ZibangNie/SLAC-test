"""Fake-only exchange runner tests; no keys, network or real research dataset."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import socket
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import run_exchange_probe as runner


@pytest.fixture
def inputs():
    return json.loads((runner.ROOT / runner.INPUT_NAME).read_bytes())


@pytest.fixture
def plan():
    return runner._assemble(runner._snapshots())[0]


def attempt(entry, labels=("yes", "no", "no"), *, cost="0.0001"):
    return {"cache_key": entry["wire_cache_key"], "core_cache_key": entry["core_cache_key"],
            "request_sha256": entry["wire_body_sha256"], "reserved_usd": entry["reserved_usd"],
            "status": "completed", "cost_status": "provider_reported", "actual_cost_usd": cost,
            "labels": dict(zip(runner.DIMENSIONS, labels)), "response_model": "typesafe/jev-1.13-20260917",
            "provider_response_status": "reported", "elapsed_seconds": 0.1}


def test_fixed_six_input_only_requests_and_native_byte_counter(inputs):
    entries, exchanges, wires = runner.freeze_requests(inputs)
    assert [entry["case_id"] for entry in entries] == list(runner.CASE_IDS)
    assert len(wires) == len({wire.cache_key for wire in wires}) == 6
    assert sum(len(wire.expected_ids) for wire in wires) == 18
    assert sum(Decimal(wire.reserved_usd) for wire in wires) == Decimal("0.030")
    for wire, exchange in zip(wires, exchanges):
        assert len(wire.payload_bytes) <= 24000
        assert exchange.max_tokens == 2048 and exchange.max_judge_tokens == 4096 and exchange.max_units == 3
        assert exchange.request.evidence_tokens == len(exchange.request.rendered_evidence.encode("utf-8"))
        assert json.loads(exchange.request.binding_bytes)["counter_version"] == runner.COUNTER_VERSION
        assert set(wire.payload()["questions"]) == set(runner.DIMENSIONS)
        assert not any(term in wire.payload_bytes for term in (b"case_id", b"expected_labels", b"choices_readout", b"expectations"))
    assert runner.artificial_counter("证据") == 6


def test_input_supervision_fields_never_enter_payload(inputs):
    _, _, before = runner.freeze_requests(inputs)
    changed = deepcopy(inputs)
    for case in changed["cases"]:
        case["expected"] = "HIDDEN_EXPECTATION"
        case["fixture_group"] = "HIDDEN_GROUP"
        case["candidate"]["metadata"] = "HIDDEN_METADATA"
    _, _, after = runner.freeze_requests(changed)
    assert [wire.payload_bytes for wire in before] == [wire.payload_bytes for wire in after]
    assert all(b"HIDDEN" not in wire.payload_bytes for wire in after)


def test_expectation_mutation_changes_readout_commitment_not_requests():
    sources = runner._snapshots()
    first, wires = runner._assemble(sources)
    expected = json.loads(sources[runner.EXPECTED_NAME])
    expected["cases"][0]["choices"] = ["no", "no", "no"]
    expected["cases"][0]["status"] = "rejected"
    expected["cases"][0]["reason"] = "SYNTHETIC_EXPECTATION_MUTATION"
    sources[runner.EXPECTED_NAME] = runner.canonical(expected)
    second, changed_wires = runner._assemble(sources)
    assert first["source_sha256"][runner.EXPECTED_NAME] != second["source_sha256"][runner.EXPECTED_NAME]
    assert first["expectations_readout_only"] != second["expectations_readout_only"]
    assert [wire.payload_bytes for wire in wires] == [wire.payload_bytes for wire in changed_wires]


def test_order_and_removal_identity_are_frozen(inputs):
    reordered = deepcopy(inputs)
    reordered["cases"].reverse()
    with pytest.raises(ValueError, match="six-case order"):
        runner.freeze_requests(reordered)
    entries, _, _ = runner.freeze_requests(inputs)
    a, b = entries[:2]
    left, right = deepcopy(a["core_payload"]["state"]), deepcopy(b["core_payload"]["state"])
    assert left.pop("removed_id") != right.pop("removed_id")
    assert left.pop("proposed_pack_ids") != right.pop("proposed_pack_ids")
    assert left == right
    assert a["wire_cache_key"] != b["wire_cache_key"]


def test_plan_and_source_drift_abort_before_client_or_key(tmp_path, monkeypatch):
    folder = tmp_path / "plan"
    runner.prepare_plan(folder)
    path = folder / "plan.json"
    original = path.read_bytes()
    assert runner.verify_plan(path)[2] == runner.sha(original)
    with pytest.raises(FileExistsError):
        runner.prepare_plan(folder)
    def forbidden(*args, **kwargs):
        raise AssertionError("client must not be constructed after drift")
    monkeypatch.setattr(runner.client, "ExchangeProbeClient", forbidden)
    sources = runner._snapshots()
    sources[runner.SOURCE_NAMES[0]] += b"\n# drift\n"
    monkeypatch.setattr(runner, "_snapshots", lambda: sources)
    with pytest.raises(ValueError, match="source commitments"):
        runner.execute_plan(path, tmp_path / "run", key_file=tmp_path / "never-read", live=True)
    assert not (tmp_path / "run").exists()
    monkeypatch.undo()
    changed = json.loads(original)
    changed["requests"][0]["wire_body_sha256"] = "0" * 64
    path.write_bytes(runner.canonical(changed) + b"\n")
    with pytest.raises(ValueError, match="plan or request"):
        runner.verify_plan(path)


def test_full_gate_and_gain_only_use_same_observed_labels(plan):
    # Independently fixed fake values for mechanical analysis, never model predictions.
    matrix = (("yes", "no", "no"), ("yes", "yes", "no"), ("yes", "no", "no"),
              ("no", "no", "no"), ("yes", "no", "yes"), ("no", "yes", "no"))
    attempts = [attempt(entry, values) for entry, values in zip(plan["requests"], matrix)]
    summary = runner.analyze_ledger(plan, {"status": "completed", "mode": "fake", "attempts": attempts})
    assert summary["completed_requests"] == summary["planned_requests"] == 6
    assert summary["typed_expectation_matches"] == summary["typed_dimensions_observed"] == 18
    assert summary["full_gate_matches_expected"] == 6
    assert summary["gain_only_matches_expected_full_gate"] == 4
    assert summary["full_gate_vs_gain_only_disagreements"] == 2
    assert summary["acceptance_by_authored_expectation"] == {
        "expected_accept_planned": 2, "expected_reject_planned": 4,
        "full_gate": {"accepted_expected_accept": 2, "accepted_expected_reject": 0},
        "gain_only": {"accepted_expected_accept": 2, "accepted_expected_reject": 2}}
    assert summary["contrast_patterns_matched"] == summary["contrast_pairs_planned"] == 2
    assert summary["requests"][1]["full_gate_status"] == "rejected"
    assert summary["requests"][1]["gain_only_status"] == "accepted"
    assert summary["execution_mode"] == "fake"
    assert "SAME exchange gain" in summary["interpretation"]


def test_unknown_failure_and_unattempted_are_kept_distinct(plan):
    first = attempt(plan["requests"][0], ("yes", "unknown", "no"))
    second = attempt(plan["requests"][1])
    second.update(status="halted", cost_status="cost_unknown")
    del second["actual_cost_usd"], second["labels"]
    summary = runner.analyze_ledger(plan, {"status": "halted", "halt_reason": "transport_timeout_or_error", "attempts": [first, second]})
    assert len(summary["requests"]) == 6
    assert summary["requests"][0]["full_gate_status"] == "abstained"
    assert summary["requests"][0]["gain_only_status"] == "accepted"
    assert summary["requests"][0]["dimension_observations"]["original_information_lost"] == "unknown"
    assert summary["requests"][1]["status"] == "halted"
    assert summary["requests"][1]["full_gate_status"] == "unobserved"
    assert summary["requests"][2]["status"] == "unattempted"
    assert summary["planned_typed_dimensions"] == 18 and summary["typed_dimensions_observed"] == 3
    assert summary["full_gate_decisions_planned"] == summary["gain_only_decisions_planned"] == 6
    assert summary["contrast_pairs_planned"] == 2
    assert summary["cost_unknown_attempts"] == 1 and not summary["total_actual_cost_known"]
    assert Decimal(summary["unresolved_reserved_usd"]) == Decimal("0.005")


def test_inflight_and_zero_attempt_readouts_retain_six_rows(plan):
    pending = attempt(plan["requests"][0])
    pending.update(status="in_flight", cost_status="cost_unknown")
    del pending["actual_cost_usd"], pending["labels"]
    result = runner.analyze_ledger(plan, {"status": "running", "attempts": [pending]})
    assert result["in_flight_attempts"] == result["cost_unknown_attempts"] == 1
    assert result["unattempted_requests"] == 5
    assert result["typed_dimensions_observed"] == 0
    empty = runner.analyze_ledger(plan, {"status": "halted", "attempts": []})
    assert empty["unattempted_requests"] == len(empty["requests"]) == 6
    assert empty["planned_typed_dimensions"] == 18


@pytest.mark.parametrize("mutation", ["order", "body", "reservation", "missing_dimension"])
def test_ledger_drift_fails_closed(plan, mutation):
    attempts = [attempt(entry) for entry in plan["requests"][:2]]
    if mutation == "order":
        attempts.reverse()
    elif mutation == "body":
        attempts[0]["request_sha256"] = "0" * 64
    elif mutation == "reservation":
        attempts[0]["reserved_usd"] = "0"
    else:
        del attempts[0]["labels"]["proposed_conflict"]
    with pytest.raises(ValueError):
        runner.analyze_ledger(plan, {"status": "completed", "attempts": attempts})


def test_public_summary_excludes_private_ledger_fields(plan):
    row = attempt(plan["requests"][0])
    row.update(raw_response="PRIVATE_RESPONSE", provider_id="PRIVATE_IDENTIFIER", key_file="PRIVATE_PATH")
    summary = runner.analyze_ledger(plan, {"status": "halted", "attempts": [row]})
    assert b"PRIVATE" not in runner.canonical(summary)


@pytest.mark.parametrize("fail_at", [None, 2])
def test_fake_end_to_end_complete_and_partial_without_key_or_network(tmp_path, monkeypatch, fail_at):
    def forbidden(*args, **kwargs):
        raise AssertionError("network is forbidden")
    monkeypatch.setattr(socket, "socket", forbidden)
    folder = tmp_path / "plan"
    runner.prepare_plan(folder)
    calls = []
    def fake(wire, timeout_seconds):
        calls.append(wire.cache_key)
        assert 0 < timeout_seconds <= 30
        if len(calls) == fail_at:
            raise TimeoutError("unsafe exception text must not appear in summary")
        return {"model": "typesafe/jev-1.13-20260917", "provider": "typesafe",
                "usage": {"cost": "0.0001", "input_tokens": 25, "output_tokens": 3},
                "answers": {name: {"type": "choice", "choice": "unknown"} for name in wire.expected_ids}}
    output = tmp_path / "run"
    ledger = runner.execute_plan(folder / "plan.json", output, transport=fake, key_file=tmp_path / "must-not-read")
    summary = runner.write_analysis(folder / "plan.json", output)
    assert len(calls) == (6 if fail_at is None else 2)
    assert ledger["status"] == ("completed" if fail_at is None else "halted")
    assert len(summary["requests"]) == 6 and summary["planned_typed_dimensions"] == 18
    assert summary["execution_mode"] == "fake"
    assert b"unsafe exception" not in runner.canonical(summary)
    with pytest.raises(FileExistsError):
        runner.execute_plan(folder / "plan.json", output, transport=fake)
    assert len(calls) == (6 if fail_at is None else 2)


def test_cli_stdout_is_sanitized_and_halted_run_has_nonzero_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_exchange_probe.py", "plan", "--output-dir", str(tmp_path / "plan")])
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out) == {"command": "plan", "status": "frozen_before_model_observations"}
    monkeypatch.setattr(runner, "execute_plan", lambda *args, **kwargs: {"status": "halted", "private": "DO_NOT_PRINT"})
    monkeypatch.setattr(sys, "argv", ["run_exchange_probe.py", "run", "--plan", "synthetic-plan", "--output-dir", "synthetic-run", "--key-file", "never-read"])
    assert runner.main() == 1
    assert json.loads(capsys.readouterr().out) == {"command": "run", "status": "halted"}
