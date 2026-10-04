"""Check new CLI dispatch while reusing the existing real-process watchdog tests."""
import hashlib
import sys

import pytest

from docs.research import run_exchange_probe_bounded as wrapper
from docs.research import run_conditional_probe_bounded as watchdog


def test_dispatches_only_new_runner_with_fixed_proxy_deadline_and_plan_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "OUTPUT_ROOT", tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_bytes(b'{"test":true}\n')
    run = tmp_path / "run"
    receipt = tmp_path / "controller.json"
    key_path = tmp_path / "absent-key.txt"
    observed = {}

    def supervise(command, output, destination, **kwargs):
        observed.update(command=command, output=output, destination=destination, **kwargs)
        return {"status": "worker_exited", "returncode": 0, "worker_terminal_verified": True}

    monkeypatch.setattr(wrapper, "supervise", supervise)
    monkeypatch.setattr(sys, "argv", ["wrapper", "--plan", str(plan), "--output-dir", str(run),
                                    "--receipt", str(receipt), "--key-file", str(key_path)])
    assert wrapper.main() == 0
    assert observed["command"] == [sys.executable, str(wrapper.ROOT / "docs/research/run_exchange_probe.py"),
                                   "run", "--plan", str(plan.resolve()), "--output-dir", str(run.resolve()),
                                   "--key-file", str(key_path.resolve()), "--proxy", "http://127.0.0.1:7897"]
    assert observed["timeout_seconds"] == 180
    assert observed["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert not key_path.exists() and not run.exists()


def test_outside_artifact_output_rejected_before_any_worker(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(watchdog, "OUTPUT_ROOT", artifacts)
    plan = artifacts / "plan.json"
    plan.write_bytes(b'{}')
    monkeypatch.setattr(sys, "argv", ["wrapper", "--plan", str(plan), "--output-dir", str(tmp_path / "bad"),
                                    "--receipt", str(artifacts / "controller.json"),
                                    "--key-file", str(tmp_path / "no-key")])
    monkeypatch.setattr(wrapper, "supervise", lambda *a, **k: pytest.fail("must not dispatch"))
    with pytest.raises(ValueError, match="dedicated artifact"):
        wrapper.main()
