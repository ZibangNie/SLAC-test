"""Synthetic packets and transports only; never open real prep, credentials or references."""
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("score_transfer_test", ROOT / "docs/research/run_score_transfer_microdiagnostic.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Guard:
    def __init__(self, *args):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def no_real_execution(monkeypatch):
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


def response(backend, payload):
    assert backend == "jev"
    return {"id": "fake-only-response", "model": m.MODEL, "provider": "TypeSafe",
            "usage": {"cost": 0.00001, "input_tokens": 10, "output_tokens": 10},
            "answers": {i: {"type": "choice", "choice": "unknown",
                            "probabilities": {"yes": .8, "no": .1, "unknown": .1}}
                        for i in payload["questions"]}}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    prep, live = tmp_path / "FAKE_ONLY_prep", tmp_path / "FAKE_ONLY_live"
    prep.mkdir(); live.mkdir()
    monkeypatch.setattr(m, "PREP_DIR", prep)
    monkeypatch.setattr(m, "LIVE_ROOT", live)
    monkeypatch.setattr(m, "EXTRA_SOURCES", ())
    packets = []
    for ordinal in range(1, 9):
        group = [f"fake-doc-{(ordinal-1)//2}", f"fake-query-{(ordinal-1)//2}"]
        tasks = []
        for j in range(8):
            item = {"query": "Synthetic fixed question?", "unit": {"id": f"fake-{ordinal}-{j}", "text": "Synthetic evidence."}}
            task_id = "support:" + m.client.object_hash({"kind": "support", "doc_id": group[0], "question_id": group[1], "item": item})
            tasks.append({"id": task_id, "item": item})
        payload = m.client.make_payload(tasks, "support", "jev")
        reserve, inp, out = m.client.reservation(payload, "jev")
        packets.append({"ordinal": ordinal, "batch": {"kind": "support", "group": group, "tasks": tasks},
                        "payload": payload, "payload_sha256": m.client.object_hash(payload),
                        "canonical_payload_bytes": len(m.canonical(payload)), "reserved_usd": str(reserve),
                        "input_allowance": inp, "output_allowance": out})
    replace(prep / "request_packets.json", packets)
    replace(prep / "handoff.json", {"sealed_inputs": {"request_packets.json": m.digest(prep/"request_packets.json")}})
    replace(prep / "transport_feasibility.json", {"source_sha256": {}})
    for name, file in [("PACKETS_SHA", "request_packets.json"), ("HANDOFF_SHA", "handoff.json"),
                       ("HISTORICAL_SHA", "transport_feasibility.json")]:
        monkeypatch.setattr(m, name, m.digest(prep/file))
    now = m.utc_now()
    endpoint = {"name": "TypeSafe | " + m.MODEL, "model_id": "typesafe/jev-1.13",
                "provider_name": "TypeSafe", "tag": "typesafe", "status": 0, "context_length": 32000,
                "pricing": {"prompt": "0.000000042", "completion": "0"}}
    replace(live/"provider_endpoints.json", {"data": {"id": "typesafe/jev-1.13",
            "architecture": {"modality": "text->decisions", "input_modalities": ["text"], "output_modalities": ["decisions"]},
            "endpoints": [endpoint]}})
    provider = {"schema": "slac-score-transfer-provider-check-v1", "status": "verified_current_public_metadata",
                "models": [{"role": "jev", "request_endpoint": m.ENDPOINT, "request_model_id": "typesafe/jev-1.13",
                            "http_status": 200, "endpoint_count": 1, "endpoint_metadata": endpoint,
                            "received_at_utc": (now-timedelta(minutes=1)).isoformat(),
                            "raw_response_file": "provider_endpoints.json",
                            "raw_response_sha256": m.digest(live/"provider_endpoints.json"),
                            "raw_response_bytes": (live/"provider_endpoints.json").stat().st_size,
                            "base_prices_usd_per_million": {"prompt": "0.042", "completion": "0"}}]}
    replace(live/"provider_check.json", provider)
    plan, run = live/"plan", live/"run"
    m.prepare(plan, run, live/"provider_check.json", (now-timedelta(seconds=1)).isoformat(),
              (now+timedelta(minutes=30)).isoformat())
    return {"prep": prep, "live": live, "plan": plan, "run": run, "packets": packets}


def execute(f, transport=response):
    return m.run(f["plan"], transport=transport, guard_factory=Guard)


def reseal(f, mutate):
    plan = m.read(f["plan"]/"plan.json")
    mutate(plan)
    replace(f["plan"]/"plan.json", plan)
    replace(f["plan"]/"seal.json", {"plan_sha256": m.digest(f["plan"]/"plan.json")})


def test_success_eight_requests_64_raw_scores_and_independent_verify(fixture):
    calls = []
    def transport(backend, payload):
        claim = m.read(fixture["live"]/"consumed.json")
        assert claim["run_dir"] == str(fixture["run"])
        ledger = m.read(fixture["run"]/"ledger.json")
        assert ledger["attempts"][-1]["status"] == "in_flight"
        assert ledger["attempts"][-1]["cost_status"] == "cost_unknown"
        assert Decimal(ledger["reservation_total_usd"]) > 0
        calls.append(payload)
        return response(backend, payload)
    result = execute(fixture, transport)
    assert len(calls) == 8 and len(result["labels"]) == len(result["reported_scores"]) == 64
    assert set(result["labels"].values()) == {"unknown"}  # Do not replace choice with score argmax.
    assert all(s == {"yes": .8, "no": .1, "unknown": .1} for s in result["reported_scores"].values())
    verified = m.verify_run(fixture["plan"])
    assert verified["accounting"]["unknown_cost_attempts"] == 0
    assert verified["accounting"]["attempts"] == 8
    assert m.read(fixture["run"]/"judgments.json") == {k: result[k] for k in ("complete", "labels", "reported_scores")}


@pytest.mark.parametrize("case", ["plan", "packet", "source", "expiry", "budget", "count", "job_payload"])
def test_admission_rejects_drift_before_key_or_consumption(fixture, monkeypatch, case):
    if case == "plan":
        value = m.read(fixture["plan"]/"plan.json"); value["jobs"][0]["ordinal"] = 99
        replace(fixture["plan"]/"plan.json", value)
    elif case == "packet":
        value = deepcopy(fixture["packets"]); value[0]["payload"]["model"] = "changed"
        replace(fixture["prep"]/"request_packets.json", value)
    elif case == "source":
        original = m._sources
        monkeypatch.setattr(m, "_sources", lambda: original() | {"fake-drift": "0"*64})
    elif case == "expiry":
        monkeypatch.setattr(m, "utc_now", lambda: m.timestamp(m.read(fixture["plan"]/"plan.json")["valid_until_utc"]))
    elif case == "budget":
        reseal(fixture, lambda p: p["limits"].update(budget_usd="2"))
    elif case == "count":
        reseal(fixture, lambda p: p["jobs"].pop())
    else:
        reseal(fixture, lambda p: p["jobs"][0]["payload"].update(model="changed"))
    with pytest.raises(ValueError):
        m.run(fixture["plan"], key_file="not-a-real-key", live=True, guard_factory=Guard)
    assert not (fixture["live"]/"consumed.json").exists()


@pytest.mark.parametrize("case", ["missing_cost", "transport", "scores", "nan_scores", "model", "cost_overrun"])
def test_failure_stops_without_retry_and_preserves_known_or_unknown_cost(fixture, case):
    calls = []
    def transport(backend, payload):
        calls.append(1)
        if case == "transport":
            raise TimeoutError()
        value = response(backend, payload)
        if case == "missing_cost": del value["usage"]["cost"]
        if case == "scores": next(iter(value["answers"].values()))["probabilities"]["yes"] = .5
        if case == "nan_scores": next(iter(value["answers"].values()))["probabilities"]["yes"] = float("nan")
        if case == "model": value["model"] = "unexpected-model"
        if case == "cost_overrun": value["usage"]["cost"] = .2
        return value
    with pytest.raises(RuntimeError): execute(fixture, transport)
    assert len(calls) == 1 and not (fixture["run"]/"judgments.json").exists()
    ledger = m.read(fixture["run"]/"ledger.json")
    assert ledger["attempts"][0]["status"] == "halted"
    assert Decimal(ledger["reservation_total_usd"]) == Decimal(fixture["packets"][0]["reserved_usd"])
    unknown = case in ("missing_cost", "transport")
    assert ledger["attempts"][0]["cost_status"] == ("cost_unknown" if unknown else "provider_reported")
    if case == "cost_overrun": assert Decimal(ledger["actual_reported_cost_usd"]) == Decimal('.2')
    with pytest.raises((ValueError, FileExistsError)): execute(fixture)
    with pytest.raises(ValueError): m.verify_run(fixture["plan"])


def test_stage_claim_blocks_a_second_prepared_plan(fixture):
    plan2 = fixture["live"]/"second-plan"
    old = m.read(fixture["plan"]/"plan.json")
    m.prepare(plan2, fixture["live"]/"second-run", old["provider_snapshot"], old["valid_from_utc"], old["valid_until_utc"])
    execute(fixture)
    with pytest.raises(FileExistsError):
        m.run(plan2, transport=response, guard_factory=Guard)
    assert not (fixture["live"]/"second-run").exists()


def test_packet_drift_midrun_stops_before_second_dispatch(fixture):
    calls = []
    def transport(backend, payload):
        calls.append(1)
        (fixture["prep"]/"request_packets.json").write_bytes(b"[]")
        return response(backend, payload)
    with pytest.raises(RuntimeError): execute(fixture, transport)
    assert len(calls) == 1
    assert m.read(fixture["run"]/"failure.json")["accounting"]["completed_requests"] == 1


def test_no_mode_does_not_consume_or_read_key(fixture):
    with pytest.raises(ValueError): m.run(fixture["plan"], guard_factory=Guard)
    assert not (fixture["live"]/"consumed.json").exists()


@pytest.mark.parametrize("artifact", ["mirror", "response"])
def test_completion_verification_rejects_response_and_mirror_tampering(fixture, artifact):
    execute(fixture)
    if artifact == "mirror":
        mirror = m.read(fixture["run"]/"ledger.json")
        mirror["actual_reported_cost_usd"] = "0"
        replace(fixture["run"]/"ledger.json", mirror)
    else:
        path = fixture["run"]/"provider_calls/response_001.json"
        value = m.read(path); next(iter(value["answers"].values()))["choice"] = "yes"
        replace(path, value)
    with pytest.raises(ValueError): m.verify_run(fixture["plan"])


@pytest.mark.parametrize("change", ["stale", "model", "modality", "envelope", "override"])
def test_provider_review_cannot_override_raw_or_freshness(fixture, change):
    path = fixture["live"]/"provider_check.json"
    raw_path = fixture["live"]/"provider_endpoints.json"
    value, raw = m.read(path), m.read(raw_path)
    model = value["models"][0]
    if change == "stale": model["received_at_utc"] = (m.utc_now()-timedelta(days=2)).isoformat()
    if change == "model": raw["data"]["id"] = "different"
    if change == "modality": raw["data"]["architecture"]["modality"] = "text->text"
    if change == "envelope": raw["data"]["endpoints"][0]["status"] = -1
    if change == "override": raw["data"]["pricing_overrides"] = {"prompt": "1"}
    replace(raw_path, raw)
    model.update(raw_response_sha256=m.digest(raw_path), raw_response_bytes=raw_path.stat().st_size)
    replace(path, value)
    with pytest.raises(ValueError): m._provider(path, m.utc_now(), fresh=True)
    assert not (fixture["live"]/"consumed.json").exists()


@pytest.mark.parametrize("field", ["budget_usd", "request_cap", "prompt_version", "automatic_retries",
                                  "cache_key", "question_count", "attempt", "started_at"])
def test_complete_verifier_checks_semantics_beyond_a_valid_event_chain(fixture, field):
    execute(fixture)
    # Rehash every event to demonstrate that the semantic checks do more than hash validation.
    previous = None
    final = None
    for path in sorted((fixture["run"]/"provider_calls_events").glob("event_*.json")):
        event = m.read(path)
        ledger = event["ledger"]
        if field in {"budget_usd", "request_cap", "prompt_version", "automatic_retries"}:
            ledger[field] = {"budget_usd": "1", "request_cap": 9, "prompt_version": "changed", "automatic_retries": 1}[field]
        elif ledger["attempts"]:
            ledger["attempts"][0][field] = {"cache_key": "0"*64, "question_count": 7, "attempt": 9,
                                          "started_at": "2000-01-01T00:00:00+00:00"}[field]
        event["previous_sha256"] = previous
        replace(path, event)
        previous, final = m.digest(path), ledger
    replace(fixture["run"]/"provider_calls/ledger.json", final)
    replace(fixture["run"]/"ledger.json", m._mirror(final))
    m.support.event_ledger(fixture["run"]/"provider_calls")  # Integrity is valid; semantics are wrong.
    with pytest.raises(ValueError): m.verify_run(fixture["plan"])


@pytest.mark.parametrize("status,code,expected", [("worker_exited", 0, 0), ("worker_exited", 1, 1),
                                                 ("hard_deadline_exceeded", 0, 1), ("controller_aborted", None, 1)])
def test_bounded_cli_preserves_terminal_failure(fixture, monkeypatch, capsys, status, code, expected):
    seen = []
    def supervisor(command, run_dir, receipt, **kwargs):
        seen.append(command)
        assert command[2] == "run" and kwargs["timeout_seconds"] == 180
        return {"status": status, "returncode": code}
    monkeypatch.setattr(m, "supervise", supervisor)
    monkeypatch.setattr(m.sys, "argv", ["runner", "bounded", "--plan", str(fixture["plan"]), "--key-file", "FAKE_ONLY_KEY"])
    assert m.main() == expected and len(seen) == 1
    assert json.loads(capsys.readouterr().out)["status"] == status
    assert not (fixture["live"]/"consumed.json").exists()


def test_key_read_occurs_after_consumption_and_full_admission(fixture, monkeypatch):
    calls = []
    def key_reader(path):
        assert (fixture["live"]/"consumed.json").exists()
        assert fixture["run"].is_dir()
        calls.append(path)
        raise ValueError("synthetic credential failure")
    monkeypatch.setattr(m.client, "read_key", key_reader)
    with pytest.raises(RuntimeError):
        m.run(fixture["plan"], key_file="fake-key-path", live=True, guard_factory=Guard)
    assert calls == ["fake-key-path"] and (fixture["run"]/"failure.json").exists()
