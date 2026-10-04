"""Actual child-process deadline checks; no credentials or network activity."""
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import run_conditional_probe_bounded as watchdog


def worker(run_dir, *, delay=0, reported=True):
    ledger = {"attempts": [{"status": "completed" if reported else "in_flight",
                            "cost_status": "provider_reported" if reported else "cost_unknown",
                            "reserved_usd": "0.005"}]}
    code = ("import json,pathlib,time; p=pathlib.Path(" + repr(str(run_dir)) + "); "
            "p.mkdir(); (p/'ledger.json').write_text(" + repr(json.dumps(ledger)) + "); "
            "time.sleep(" + str(delay) + ")")
    return [sys.executable, "-c", code]


def test_successful_child_and_duplicate_controller_are_one_shot(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "OUTPUT_ROOT", tmp_path)
    run_dir, receipt = tmp_path / "run", tmp_path / "controller.json"
    result = watchdog.supervise(worker(run_dir), run_dir, receipt, timeout_seconds=5, plan_sha256="test")
    assert result["status"] == "worker_exited" and result["returncode"] == 0
    assert result["worker_terminal_verified"] and result["attempt_count"] == 1
    assert result["unresolved_cost_attempts"] == 0
    with pytest.raises(FileExistsError):
        watchdog.supervise(worker(run_dir), run_dir, receipt, timeout_seconds=5, plan_sha256="test")
    with pytest.raises(FileExistsError):
        watchdog.supervise(worker(tmp_path / "second"), tmp_path / "second", receipt,
                            timeout_seconds=5, plan_sha256="test")
    assert not (tmp_path / "second").exists()


def test_hard_deadline_kills_worker_preserves_unresolved_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "OUTPUT_ROOT", tmp_path)
    run_dir, receipt = tmp_path / "run", tmp_path / "controller.json"
    result = watchdog.supervise(worker(run_dir, delay=20, reported=False), run_dir, receipt,
                                 timeout_seconds=1.5, plan_sha256="test")
    assert result["status"] == "hard_deadline_exceeded"
    assert result["worker_terminal_verified"] and result["automatic_restarts"] == 0
    assert result["in_flight_attempts"] == result["unresolved_cost_attempts"] == 1
    assert not result["upstream_cancellation_claimed"]
    ledger = json.loads((run_dir / "ledger.json").read_bytes())
    assert ledger["attempts"][0] == {"status": "in_flight", "cost_status": "cost_unknown",
                                      "reserved_usd": "0.005"}


def test_output_boundary_and_receipt_placement_reject_before_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(ValueError):
        watchdog.supervise([], tmp_path.parent / "outside", tmp_path / "receipt.json", plan_sha256="test")
    with pytest.raises(ValueError):
        watchdog.supervise([], tmp_path / "run", tmp_path / "run/receipt.json", plan_sha256="test")
    assert not (tmp_path / "run").exists()
    with pytest.raises(ValueError):
        watchdog.supervise([], tmp_path / "run", tmp_path / "receipt.json", timeout_seconds=181, plan_sha256="test")


def test_startup_failure_does_not_invent_zero_attempts_or_serialize_error(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "OUTPUT_ROOT", tmp_path)
    def fail(*args, **kwargs):
        raise OSError("private diagnostic must not be written")
    monkeypatch.setattr(watchdog.subprocess, "Popen", fail)
    receipt = tmp_path / "receipt.json"
    result = watchdog.supervise(["unused"], tmp_path / "run", receipt, plan_sha256="test")
    assert result["status"] == "controller_aborted" and result["error_class"] == "OSError"
    assert result["worker_terminal_verified"]
    assert not result["ledger_observed"] and result["attempt_count"] is None
    assert "private diagnostic" not in receipt.read_text()
