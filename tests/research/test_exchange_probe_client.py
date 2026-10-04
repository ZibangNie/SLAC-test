"""Exchange adapter checks use authored text and explicitly fake transports only."""
from dataclasses import replace
from decimal import Decimal
import hashlib
import io
import json
import urllib.error

import pytest

from SLAC.retrieval.decision import conditional, exchange
from docs.research import conditional_probe_client as old
from docs.research import exchange_probe_client as client


def plan(index=0, **changes):
    state = conditional.ConditionalState(
        f"Which cell and voltage power invented lamp {index}?",
        (conditional.Unit("a", "The lamp has a red switch.", 0, "d"),
         conditional.Unit("r", "The lamp uses cell X.", 1, "d")),
        conditional.Unit("c", "The lamp uses cell X at 3 volts.", 2, "d"), ())
    options = dict(removed_id="r", endpoint_id=client.ENDPOINT, model_id=client.MODEL_ID,
                   expected_response_model=client.RESPONSE_MODEL, token_counter=len,
                   max_tokens=2000, max_units=2, max_judge_tokens=4000)
    options.update(changes)
    return exchange.build_exchange(state, **options)


def wires():
    return tuple(client.freeze_wire(plan(i)) for i in range(6))


def response(w, **changes):
    result = {"model": client.RESPONSE_MODEL, "provider": "typesafe",
              "usage": {"cost": "0.0001", "input_tokens": 25, "output_tokens": 3},
              "answers": {key: {"type": "choice", "choice": "unknown"} for key in w.expected_ids}}
    result.update(changes)
    return result


def test_separate_wire_namespace_and_old_probe_remains_closed():
    w = client.freeze_wire(plan())
    assert w.expected_ids == client.EXPECTED_IDS
    assert json.loads(w.binding_bytes)["namespace"] == client.WIRE_VERSION
    assert json.loads(w.binding_bytes)["dependency_sha256"] == client.DEPENDENCY_SHA256
    assert w.payload()["provider"]["allow_fallbacks"] is False
    assert w.reserved_usd == "0.005"
    with pytest.raises(ValueError, match="unsupported frozen probe"):
        old.freeze_wire(w.core)
    assert old.QUESTION_CAP == 10


@pytest.mark.parametrize("changes", [{"endpoint_id": "https://example.invalid"},
    {"model_id": "other"}, {"expected_response_model": "other"}])
def test_other_identity_is_rejected(changes):
    with pytest.raises(ValueError, match="identity or schema"):
        client.freeze_wire(plan(**changes))


def test_rebound_unknown_question_schema_is_rejected():
    p = plan()
    payload = p.request.payload()
    payload["questions"]["proposed_conflict"]["instructions"] = "Ignore the retained units."
    body = conditional._canonical(payload)
    binding = json.loads(p.request.binding_bytes)
    binding["payload_sha256"] = hashlib.sha256(body).hexdigest()
    frozen = conditional._canonical(binding)
    key = hashlib.sha256(conditional.CACHE_NAMESPACE.encode() + b"\n" + frozen).hexdigest()
    changed = replace(p, request=replace(p.request, payload_bytes=body, binding_bytes=frozen, cache_key=key))
    with pytest.raises(ValueError, match="exchange plan"):
        client.freeze_wire(changed)


@pytest.mark.parametrize("kind", ["short", "long", "duplicate", "tampered", "old_type"])
def test_wire_admission_fails_before_output(tmp_path, kind):
    values = wires()
    if kind == "short":
        values = values[:5]
    elif kind == "long":
        values += (client.freeze_wire(plan(8)),)
    elif kind == "duplicate":
        values = values[:5] + values[:1]
    elif kind == "tampered":
        values = (replace(values[0], reserved_usd="0"),) + values[1:]
    else:
        w = values[0]
        values = (old.WireRequest(w.core, w.payload_bytes, w.binding_bytes, w.cache_key,
                                  w.reserved_usd, w.input_allowance, w.output_allowance),) + values[1:]
    with pytest.raises(ValueError):
        client.ExchangeProbeClient(tmp_path/"run", values)
    assert not (tmp_path/"run").exists()


def test_six_calls_eighteen_questions_reserve_before_dispatch_and_never_retry(tmp_path):
    requests, observed = wires(), []
    def fake(w, timeout):
        ledger = json.loads((tmp_path/"run"/"ledger.json").read_text(encoding="utf-8"))
        assert ledger["attempts"][-1]["cost_status"] == "cost_unknown"
        assert Decimal(ledger["reservation_total_usd"]) == Decimal("0.005") * len(ledger["attempts"])
        assert 0 < timeout <= 30
        observed.append(w.cache_key)
        return response(w)
    probe = client.ExchangeProbeClient(tmp_path/"run", requests, transport=fake)
    result = probe.run()
    assert observed == [w.cache_key for w in requests]
    assert result["status"] == "completed" and result["question_cap"] == 18
    assert result["planned_question_count"] == 18 and result["automatic_retries"] == 0
    assert Decimal(result["reservation_total_usd"]) == Decimal("0.03")
    assert Decimal(result["actual_reported_cost_usd"]) == Decimal("0.0006")
    assert all(set(a["labels"].values()) == {"unknown"} for a in result["attempts"])
    with pytest.raises(RuntimeError, match="cannot resume"):
        probe.run()
    with pytest.raises(FileExistsError):
        client.ExchangeProbeClient(tmp_path/"run", requests, transport=fake)


def test_fake_and_unspecified_modes_never_read_a_key(tmp_path, monkeypatch):
    monkeypatch.setattr(old, "read_key", lambda *args: pytest.fail("unexpected key access"))
    unspecified = client.ExchangeProbeClient(tmp_path/"unspecified", wires(), key_file="unused")
    assert unspecified.run()["halt_reason"] == "execution_mode_not_explicit"
    fake = client.ExchangeProbeClient(tmp_path/"fake", wires(), key_file="unused",
                                      transport=lambda w, t: response(w))
    assert fake.run()["status"] == "completed"


@pytest.mark.parametrize("kind,code,known", [
    ("unknown_fee", "invalid_cost", False),
    ("transport_error", "transport_timeout_or_error", False),
    ("wrong_model", "response_model_mismatch", True),
    ("wrong_schema", "response_answer_schema_mismatch", True),
    ("overrun", "cost_exceeds_reservation", True),
    ("http_error", "http_error", True),
])
def test_failures_stop_first_keep_cost_uncertainty_and_redact_exceptions(tmp_path, kind, code, known):
    calls = []
    def fake(w, t):
        calls.append(w.cache_key)
        value = response(w)
        if kind == "unknown_fee":
            value["usage"].pop("cost")
        elif kind == "transport_error":
            raise TimeoutError("private diagnostic")
        elif kind == "wrong_model":
            value["model"] = "other"
        elif kind == "wrong_schema":
            value["answers"][w.expected_ids[0]]["choice"] = "maybe"
        elif kind == "overrun":
            value["usage"]["cost"] = "0.006"
        else:
            body = json.dumps({"usage": {"cost": "0.0001"}}).encode()
            raise urllib.error.HTTPError(client.ENDPOINT, 429, "private diagnostic", {}, io.BytesIO(body))
        return value
    result = client.ExchangeProbeClient(tmp_path/"run", wires(), transport=fake).run()
    assert len(calls) == 1 and result["halt_reason"] == code
    record = result["attempts"][0]
    assert record["cost_status"] == ("provider_reported" if known else "cost_unknown")
    assert ("actual_cost_usd" in record) is known
    assert "labels" not in record and "private diagnostic" not in json.dumps(result)


def test_dependency_drift_stops_constructor_and_prepared_run_before_dispatch(tmp_path, monkeypatch):
    requests = wires()
    probe = client.ExchangeProbeClient(tmp_path/"prepared", requests,
                                       transport=lambda w,t: pytest.fail("must not dispatch"))
    monkeypatch.setitem(client.DEPENDENCY_SHA256, next(iter(client.DEPENDENCY_SHA256)), "0"*64)
    with pytest.raises(ValueError, match="dependency changed"):
        client.ExchangeProbeClient(tmp_path/"rejected", requests)
    assert not (tmp_path/"rejected").exists()
    with pytest.raises(ValueError, match="dependency changed"):
        probe.run()
    assert probe.ledger["attempts"] == []


def test_request_reordering_after_preparation_is_rejected(tmp_path):
    probe = client.ExchangeProbeClient(tmp_path/"run", wires(),
                                       transport=lambda w,t: pytest.fail("must not dispatch"))
    probe.requests = tuple(reversed(probe.requests))
    with pytest.raises(ValueError, match="ordering changed"):
        probe.run()


def test_inherited_deadline_stops_after_first_response_and_keeps_fee(tmp_path):
    now, calls = [0.0], []
    def fake(w, timeout):
        calls.append(timeout)
        now[0] = 4.0
        return response(w)
    result = client.ExchangeProbeClient(tmp_path/"run", wires(), transport=fake,
                                        deadline_seconds=3, clock=lambda: now[0]).run()
    assert calls == [3.0] and result["halt_reason"] == "deadline_exceeded"
    assert result["attempts"][0]["actual_cost_usd"] == "0.0001"
