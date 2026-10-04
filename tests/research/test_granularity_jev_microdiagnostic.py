"""Synthetic 117-pair fixtures only; real inputs, keys, transport and gold forbidden."""
from datetime import timedelta
from decimal import Decimal
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "granularity_jev_test", ROOT / "docs/research/run_granularity_jev_microdiagnostic.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Guard:
    def __init__(self, *args):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def prohibit_real_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real key/network/reference/old runner forbidden")
    monkeypatch.setattr(m.client, "read_key", forbidden)
    monkeypatch.setattr(m.client.urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.setattr(m.support, "prepare", forbidden)
    monkeypatch.setattr(m.support, "load_plan", forbidden)
    monkeypatch.setattr(m.support, "run", forbidden)
    monkeypatch.setattr(m.support.pilot, "selected_gold", forbidden)
    monkeypatch.setattr(m.support.os, "_exit", forbidden)


def replace(path, value):
    path.write_bytes(m.canonical(value) + b"\n")


def fake_response(backend, payload):
    assert backend == "jev"
    return {"id": "synthetic-response", "model": m.MODEL, "provider": "TypeSafe",
            "usage": {"cost": 0.00001, "input_tokens": 10, "output_tokens": 10},
            "answers": {task_id: {"type": "choice", "choice": "unknown",
                "probabilities": {"yes": .8, "no": .1, "unknown": .1}}
                for task_id in payload["questions"]}}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    prep, live = tmp_path / "FAKE_ONLY_input", tmp_path / "FAKE_ONLY_live"
    prep.mkdir(); live.mkdir()
    monkeypatch.setattr(m, "INPUT", prep / "selected_inputs.json")
    monkeypatch.setattr(m, "WIRE_REPORT", prep / "wire.json")
    monkeypatch.setattr(m, "LIVE_ROOT", live)
    monkeypatch.setattr(m, "EXTRA_SOURCES", ())
    # Artificial ASCII bodies hit the fixed numeric byte/reservation bounds;
    # no real query, passage, identifier, or wire payload is read.
    target_bytes = [10181, 9487, 9832, 9403, 10042, 10296, 10153, 9598,
                    10996, 10520, 9643, 11187, 9477, 9949, 5917]
    pairs, cases, rows = [], [], []
    for ordinal, count in ((1, 64), (2, 53)):
        doc, qid, query = f"fake-doc-{ordinal}", f"fake-query-{ordinal}", f"Synthetic question {ordinal}?"
        cases.append({"ordinal": ordinal, "doc_id": doc, "question_id": qid, "query": query})
        local = [{"task_id": m.client.object_hash(["fake", ordinal, i]), "doc_id": doc,
                  "question_id": qid, "unit_id": f"fake-unit-{i}", "query": query,
                  "passage": f"Synthetic evidence {i}."} for i in range(count)]
        for batch_index, start in enumerate(range(0, count, 8)):
            subset = local[start:start+8]
            def tasks():
                return [{"id": row["task_id"], "item": {"query": row["query"],
                         "unit": {"id": row["task_id"], "text": row["passage"]}}} for row in subset]
            payload = m.client.make_payload(tasks(), "support", "jev")
            padding = target_bytes[len(rows)] - len(m.canonical(payload))
            assert padding >= 0
            subset[0]["passage"] += "x" * padding
            payload = m.client.make_payload(tasks(), "support", "jev")
            reserve, inp, out = m.client.reservation(payload, "jev")
            rows.append({"ordinal": ordinal, "batch_index": batch_index, "decisions": len(subset),
                         "wire_bytes": len(m.canonical(payload)), "payload_sha256": m.client.object_hash(payload),
                         "historical_reservation_usd": str(reserve),
                         "input_allowance": inp, "output_allowance": out})
        pairs.extend(local)
    replace(m.INPUT, {"schema": "slac-candidate-granularity-reranker-input-v1",
                      "cases": cases, "pairs": pairs, "input_bindings": {}})
    monkeypatch.setattr(m, "INPUT_SHA", m.digest(m.INPUT))
    wire = {"schema": "slac-granularity-jev-wire-inspection-v1", "input_sha256": m.INPUT_SHA,
            "source_sha256": {name: m.digest(ROOT / "docs/research" / name) for name in
                              ("inspect_granularity_jev_wire.py", "openrouter_decision_client.py")},
            "batches": rows, "summary": {"questions": 2, "decisions": 117, "requests_if_executed": 15,
            "wire_bytes_total": sum(row["wire_bytes"] for row in rows),
            "wire_bytes_maximum": max(row["wire_bytes"] for row in rows),
            "historical_reservation_usd": str(sum((Decimal(row["historical_reservation_usd"])
                                                  for row in rows), Decimal(0)))}}
    replace(m.WIRE_REPORT, wire)
    monkeypatch.setattr(m, "WIRE_SHA", m.digest(m.WIRE_REPORT))
    now = m.utc_now()
    endpoint = {"name": "TypeSafe | " + m.MODEL, "model_id": "typesafe/jev-1.13",
                "provider_name": "TypeSafe", "tag": "typesafe", "status": 0, "context_length": 32000,
                "pricing": {"prompt": "0.000000042", "completion": "0"}}
    replace(live / "provider_endpoints.json", {"data": {"id": "typesafe/jev-1.13",
        "architecture": {"modality": "text->decisions", "input_modalities": ["text"],
                         "output_modalities": ["decisions"]}, "endpoints": [endpoint]}})
    provider = {"schema": "slac-score-transfer-provider-check-v1", "status": "verified_current_public_metadata",
        "models": [{"role": "jev", "request_endpoint": m.ENDPOINT, "request_model_id": "typesafe/jev-1.13",
        "http_status": 200, "endpoint_count": 1, "endpoint_metadata": endpoint,
        "received_at_utc": (now-timedelta(minutes=1)).isoformat(), "raw_response_file": "provider_endpoints.json",
        "raw_response_sha256": m.digest(live / "provider_endpoints.json"),
        "raw_response_bytes": (live / "provider_endpoints.json").stat().st_size,
        "base_prices_usd_per_million": {"prompt": "0.042", "completion": "0"}}]}
    replace(live / "provider_check.json", provider)
    plan, run = live / "plan", live / "run"
    m.prepare(plan, run, live / "provider_check.json", (now-timedelta(seconds=1)).isoformat(),
              (now+timedelta(minutes=30)).isoformat())
    return {"prep": prep, "live": live, "plan": plan, "run": run, "rows": rows}


def execute(f, transport=fake_response):
    return m.run(f["plan"], transport=transport, guard_factory=Guard)


def test_complete_15_requests_117_scores_tail_and_verified_bindings(fixture):
    calls = []
    def transport(backend, payload):
        claim = m.read(fixture["live"] / "consumed.json")
        assert claim["run_dir"] == str(fixture["run"])
        ledger = m.read(fixture["run"] / "ledger.json")
        assert ledger["attempts"][-1]["status"] == "in_flight"
        assert ledger["attempts"][-1]["cost_status"] == "cost_unknown"
        assert Decimal(ledger["reservation_total_usd"]) > 0
        assert all(key == item["unit"]["id"] and len(key) == 64
                   for key, item in payload["state"]["items"].items())
        calls.append(payload)
        return fake_response(backend, payload)
    result = execute(fixture, transport)
    assert [len(p["questions"]) for p in calls] == [8]*14 + [5]
    assert len(result["labels"]) == len(result["reported_scores"]) == 117
    assert set(result["labels"].values()) == {"unknown"}  # Preserve choice, not score argmax.
    assert all(s == {"yes": .8, "no": .1, "unknown": .1} for s in result["reported_scores"].values())
    verified = m.verify_run(fixture["plan"])
    assert verified["accounting"]["unknown_cost_attempts"] == 0
    assert verified["accounting"]["attempts"] == 15
    assert Decimal(verified["accounting"]["reserved_usd"]) == Decimal(".0903957440")
    assert verified["bindings"][str(m.INPUT.resolve())] == m.INPUT_SHA
    assert str(Path(m.__file__).resolve()) in verified["bindings"]
    assert m.read(fixture["run"] / "judgments.json") == {
        k: result[k] for k in ("complete", "labels", "reported_scores")}


@pytest.mark.parametrize("case", ["input", "wire", "source", "expiry", "plan", "caps"])
def test_admission_drift_precedes_key_and_consumption(fixture, monkeypatch, case):
    if case in ("input", "wire"):
        (m.INPUT if case == "input" else m.WIRE_REPORT).write_bytes(b"{}")
    elif case == "source":
        original = m._sources
        monkeypatch.setattr(m, "_sources", lambda: original() | {"fake-drift": "0"*64})
    elif case == "expiry":
        end = m.timestamp(m.read(fixture["plan"] / "plan.json")["valid_until_utc"])
        monkeypatch.setattr(m, "utc_now", lambda: end)
    else:
        plan = m.read(fixture["plan"] / "plan.json")
        if case == "plan": plan["jobs"][0]["payload"]["model"] = "changed"
        else: plan["limits"]["requests"] = 16
        replace(fixture["plan"] / "plan.json", plan)
        replace(fixture["plan"] / "seal.json", {"plan_sha256": m.digest(fixture["plan"] / "plan.json")})
    with pytest.raises(ValueError):
        m.run(fixture["plan"], key_file="FAKE_ONLY_KEY", live=True, guard_factory=Guard)
    assert not (fixture["live"] / "consumed.json").exists()


@pytest.mark.parametrize("case", ["timeout", "unknown_cost", "invalid_scores", "model", "cost_overrun"])
def test_first_failure_stops_preserves_accounting_and_cannot_restart(fixture, case):
    calls = []
    def transport(backend, payload):
        calls.append(1)
        if case == "timeout": raise TimeoutError()
        response = fake_response(backend, payload)
        if case == "unknown_cost": del response["usage"]["cost"]
        if case == "invalid_scores": next(iter(response["answers"].values()))["probabilities"]["yes"] = .5
        if case == "model": response["model"] = "wrong-version"
        if case == "cost_overrun": response["usage"]["cost"] = .2
        return response
    with pytest.raises(RuntimeError): execute(fixture, transport)
    assert len(calls) == 1 and not (fixture["run"] / "judgments.json").exists()
    ledger = m.read(fixture["run"] / "ledger.json")
    assert ledger["attempts"][0]["status"] == "halted"
    assert Decimal(ledger["reservation_total_usd"]) == Decimal(fixture["rows"][0]["historical_reservation_usd"])
    unknown = case in ("timeout", "unknown_cost")
    assert ledger["attempts"][0]["cost_status"] == ("cost_unknown" if unknown else "provider_reported")
    if case == "cost_overrun": assert Decimal(ledger["actual_reported_cost_usd"]) == Decimal('.2')
    with pytest.raises((ValueError, FileExistsError)): execute(fixture)
    with pytest.raises(ValueError): m.verify_run(fixture["plan"])


def test_key_access_follows_claim_and_claim_blocks_another_plan(fixture, monkeypatch):
    old = m.read(fixture["plan"] / "plan.json")
    second = fixture["live"] / "plan-second"
    m.prepare(second, fixture["live"] / "run-second", old["provider_snapshot"],
              old["valid_from_utc"], old["valid_until_utc"])
    calls = []
    def fake_key(path):
        assert (fixture["live"] / "consumed.json").exists() and fixture["run"].is_dir()
        calls.append(path)
        raise ValueError("synthetic credential failure")
    monkeypatch.setattr(m.client, "read_key", fake_key)
    with pytest.raises(RuntimeError):
        m.run(fixture["plan"], key_file="FAKE_ONLY_KEY", live=True, guard_factory=Guard)
    assert calls == ["FAKE_ONLY_KEY"]
    assert (fixture["run"] / "failure.json").exists()
    with pytest.raises(FileExistsError):
        m.run(second, transport=fake_response, guard_factory=Guard)
    assert not (fixture["live"] / "run-second").exists()


def test_midrun_input_drift_stops_before_second_dispatch(fixture):
    calls = []
    def transport(backend, payload):
        calls.append(1)
        m.INPUT.write_bytes(b"{}")
        return fake_response(backend, payload)
    with pytest.raises(RuntimeError): execute(fixture, transport)
    assert len(calls) == 1
    assert m.read(fixture["run"] / "failure.json")["accounting"]["completed_requests"] == 1


@pytest.mark.parametrize("artifact", ["response", "mirror"])
def test_saved_completion_tampering_is_rejected(fixture, artifact):
    execute(fixture)
    if artifact == "response":
        path = fixture["run"] / "provider_calls/response_015.json"
        value = m.read(path)
        next(iter(value["answers"].values()))["choice"] = "yes"
    else:
        path = fixture["run"] / "ledger.json"
        value = m.read(path); value["actual_reported_cost_usd"] = "0"
    replace(path, value)
    with pytest.raises(ValueError): m.verify_run(fixture["plan"])


@pytest.mark.parametrize("case", ["stale", "raw_divergence"])
def test_provider_freshness_and_raw_envelope_consistency(fixture, case):
    path = fixture["live"] / "provider_check.json"
    provider = m.read(path)
    if case == "stale":
        provider["models"][0]["received_at_utc"] = (m.utc_now()-timedelta(days=2)).isoformat()
    else:
        raw_path = fixture["live"] / "provider_endpoints.json"
        raw = m.read(raw_path); raw["data"]["architecture"]["modality"] = "text->text"
        replace(raw_path, raw)
        provider["models"][0].update(raw_response_sha256=m.digest(raw_path), raw_response_bytes=raw_path.stat().st_size)
    replace(path, provider)
    with pytest.raises(ValueError): m._provider(path, m.utc_now(), fresh=True)
    assert not (fixture["live"] / "consumed.json").exists()


@pytest.mark.parametrize("status,code,exit_code", [("worker_exited", 0, 0),
                                                ("worker_exited", 1, 1),
                                                ("hard_deadline_exceeded", 0, 1)])
def test_supervised_cli_reports_failure_and_preserves_180_second_cap(fixture, monkeypatch, capsys, status, code, exit_code):
    calls = []
    def supervise(command, run_dir, receipt, **kwargs):
        assert command[2] == "run" and kwargs["timeout_seconds"] == 180
        assert Path(command[1]).name == "run_granularity_jev_microdiagnostic.py"
        calls.append(command)
        return {"status": status, "returncode": code}
    monkeypatch.setattr(m, "supervise", supervise)
    monkeypatch.setattr(m.sys, "argv", ["runner", "bounded", "--plan", str(fixture["plan"]),
                                      "--key-file", "FAKE_ONLY_KEY"])
    assert m.main() == exit_code and len(calls) == 1
    assert json.loads(capsys.readouterr().out)["status"] == status
    assert not (fixture["live"] / "consumed.json").exists()


@pytest.mark.parametrize("sleep_seconds,expected", [(0, "worker_exited"), (30, "hard_deadline_exceeded")])
def test_new_stage_supervisor_starts_and_terminates_real_fake_worker(fixture, sleep_seconds, expected):
    # This child only writes a tiny synthetic ledger and sleeps; no runner import.
    code = (
        "import json,sys,time; from pathlib import Path; "
        "p=Path(sys.argv[1]); p.mkdir(); "
        "(p/'ledger.json').write_text(json.dumps({'attempts':[{'status':'in_flight',"
        "'cost_status':'cost_unknown'}]})); "
        "print('FAKE_SENSITIVE_OUTPUT',flush=True); "
        "print('FAKE_SENSITIVE_ERROR',file=sys.stderr,flush=True); "
        "time.sleep(float(sys.argv[2]))"
    )
    receipt = fixture["live"] / "controller-test.json"
    result = m.supervise([m.sys.executable, "-c", code, str(fixture["run"]), str(sleep_seconds)],
        fixture["run"], receipt, timeout_seconds=.8, plan_sha256="0"*64)
    assert result["status"] == expected and result["worker_terminal_verified"]
    assert result["ledger_observed"] and result["attempt_count"] == 1
    assert result["in_flight_attempts"] == result["unresolved_cost_attempts"] == 1
    assert result["automatic_restarts"] == 0 and result["upstream_cancellation_claimed"] is False
    if sleep_seconds == 0: assert result["returncode"] == 0
    assert "FAKE_SENSITIVE" not in receipt.read_text(encoding="utf-8")
    assert m.read(receipt) == result
    with pytest.raises(ValueError):
        m.supervise(["must-not-start"], fixture["run"], receipt,
                    timeout_seconds=.8, plan_sha256="0"*64)
