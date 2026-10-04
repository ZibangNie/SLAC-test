"""Bounded transport tests use only invented state and explicitly fake responses."""
from dataclasses import replace
from decimal import Decimal
import io
import json
import urllib.error

import pytest

from SLAC.retrieval.decision.conditional import ConditionalState, Unit, build_request
from docs.research import conditional_probe_client as client


def wire(index=0, arm="plain_conditional", **kwargs):
    state = ConditionalState(f"Which cell powers invented lamp {index}?",
                             (Unit("u001", f"Invented lamp {index} has a red switch.", 0, "d001"),),
                             Unit("u003", f"Invented lamp {index} uses cell X.", 2, "d001"), ())
    options = dict(arm=arm, endpoint_id=client.ENDPOINT, model_id=client.MODEL_ID,
                   expected_response_model=client.RESPONSE_MODEL,
                   token_counter=len, max_tokens=10000)
    options.update(kwargs)
    return client.freeze_wire(build_request(state, **options))


def response(w, **changes):
    value = {"model": client.RESPONSE_MODEL, "provider": "typesafe",
             "usage": {"cost": "0.0001", "input_tokens": 25, "output_tokens": 1},
             "answers": {key: {"type": "choice", "choice": "unknown"} for key in w.expected_ids}}
    value.update(changes)
    return value


def run_one(tmp_path, value, w=None):
    w = w or wire()
    calls = []
    def fake(request, timeout):
        calls.append((request.cache_key, timeout))
        if isinstance(value, Exception):
            raise value
        return value
    result = client.ProbeClient(tmp_path/"run", (w, wire(2)), transport=fake).run()
    return result, calls


def test_wire_binds_complete_provider_core_and_accounting_and_returns_copy():
    w = wire()
    payload, binding = w.payload(), json.loads(w.binding_bytes)
    assert payload["provider"] == {"only": ["typesafe"], "allow_fallbacks": False,
                                   "max_price": {"prompt": "0.042", "completion": "0", "request": "0"}}
    assert binding["core_binding"] == json.loads(w.core.binding_bytes)
    assert binding["core_cache_key"] == w.core.cache_key
    assert w.cache_key != w.core.cache_key
    assert Decimal(w.reserved_usd) == Decimal("0.005")
    payload["provider"]["only"].append("other")
    assert w.payload()["provider"]["only"] == ["typesafe"]
    assert wire(prompt_version="new-contract-version").cache_key != w.cache_key


@pytest.mark.parametrize("change", [dict(endpoint_id="https://example.invalid"),
    dict(model_id="other"), dict(expected_response_model="other"), dict(arm="relation_conditioned")])
def test_rejects_unfrozen_model_endpoint_or_arm(change):
    with pytest.raises(ValueError):
        wire(**change)


def test_rejects_full_wire_byte_overflow():
    state = ConditionalState("Q?", (), Unit("u001", "a"*24000, 0, "d001"), ())
    req = build_request(state, arm="standalone", endpoint_id=client.ENDPOINT,
                        model_id=client.MODEL_ID, expected_response_model=client.RESPONSE_MODEL,
                        token_counter=len, max_tokens=30000)
    with pytest.raises(ValueError, match="byte cap"):
        client.freeze_wire(req)


@pytest.mark.parametrize("requests", [(), tuple(wire(i, "standalone") for i in range(7)),
    (wire(), wire()), tuple(wire(i) for i in range(6)),
    (replace(wire(), payload_bytes=b"{}"),), (replace(wire(), reserved_usd="0"),)])
def test_caps_duplicates_and_tampering_rejected_before_output(tmp_path, requests):
    with pytest.raises(ValueError):
        client.ProbeClient(tmp_path/"run", requests, transport=lambda w,t: response(w))
    assert not (tmp_path/"run").exists()


@pytest.mark.parametrize("value", [True, 0, -1, 181, float("nan"), float("inf")])
def test_invalid_deadline_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        client.ProbeClient(tmp_path/"run", (wire(),), deadline_seconds=value)


def test_six_calls_ten_questions_reserve_before_dispatch_unknown_is_valid(tmp_path):
    wires = tuple(wire(i, "standalone" if i in (0,3) else "plain_conditional") for i in range(6))
    observed = []
    def fake(w, timeout):
        saved = json.loads((tmp_path/"run"/"ledger.json").read_text(encoding="utf-8"))
        current = saved["attempts"][-1]
        assert current["status"] == "in_flight" and current["cost_status"] == "cost_unknown"
        assert Decimal(saved["reservation_total_usd"]) == Decimal("0.005") * len(saved["attempts"])
        assert timeout <= 30
        observed.append(w.cache_key)
        return response(w)
    c = client.ProbeClient(tmp_path/"run", wires, transport=fake)
    result = c.run()
    assert observed == [w.cache_key for w in wires]
    assert result["status"] == "completed" and result["planned_question_count"] == 10
    assert Decimal(result["reservation_total_usd"]) == Decimal("0.03")
    assert Decimal(result["actual_reported_cost_usd"]) == Decimal("0.0006")
    assert all(set(a["labels"].values()) == {"unknown"} for a in result["attempts"])
    with pytest.raises(RuntimeError, match="cannot resume"):
        c.run()
    with pytest.raises(FileExistsError):
        client.ProbeClient(tmp_path/"run", wires, transport=fake)


def test_keys_lazy_and_fake_never_reads_keys(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("credential reader must not run")
    monkeypatch.setattr(client, "read_key", forbidden)
    c = client.ProbeClient(tmp_path/"disabled", (wire(),), key_file="not-a-real-key-file")
    assert c.run()["halt_reason"] == "execution_mode_not_explicit"
    fake = client.ProbeClient(tmp_path/"fake", (wire(),), key_file="not-a-real-key-file",
                              transport=lambda w,t: response(w))
    assert fake.run()["status"] == "completed"


@pytest.mark.parametrize("change,code", [
    (dict(model="typesafe/jev-new"), "response_model_mismatch"),
    (dict(provider="other"), "response_provider_mismatch"),
    (dict(error={"message":"do not retain this exception text"}), "response_error_object"),
    (dict(answers={}), "response_answer_ids_mismatch"),
    (dict(answers={"extra":{"type":"choice","choice":"yes"}}), "response_answer_ids_mismatch"),
])
def test_identity_schema_halt_first_and_preserve_reported_cost(tmp_path, change, code):
    w = wire()
    result,calls = run_one(tmp_path, response(w, **change), w)
    assert len(calls) == 1 and result["halt_reason"] == code
    a = result["attempts"][0]
    assert a["cost_status"] == "provider_reported" and a["actual_cost_usd"] == "0.0001"
    assert Decimal(result["reservation_total_usd"]) == Decimal("0.005")
    assert "labels" not in a


@pytest.mark.parametrize("bad", [{"type":"text","choice":"yes"}, {"type":"choice","choice":"maybe"}, None])
def test_bad_dimension_halts_without_labels(tmp_path, bad):
    w=wire(); value=response(w)
    value["answers"][w.expected_ids[0]]=bad
    result,_=run_one(tmp_path,value,w)
    assert result["halt_reason"] == "response_answer_schema_mismatch"


@pytest.mark.parametrize("bad", [None, True, -1, "NaN", "Infinity", {}, "x"*129])
def test_unknown_or_invalid_cost_never_means_zero_cost(tmp_path, bad):
    w=wire(); value=response(w); value["usage"]["cost"]=bad
    result,calls=run_one(tmp_path,value,w)
    assert len(calls)==1 and result["halt_reason"] == "invalid_cost"
    a=result["attempts"][0]
    assert a["cost_status"] == "cost_unknown" and "actual_cost_usd" not in a
    assert Decimal(result["reservation_total_usd"]) == Decimal("0.005")


@pytest.mark.parametrize("usage", [
    {"cost":"0.0001","input_tokens":True,"output_tokens":1},
    {"cost":"0.0001","input_tokens":2.0,"output_tokens":1},
    {"cost":"0.0001","output_tokens":1},
    {"cost":"0.0001","input_tokens":2,"prompt_tokens":3,"output_tokens":1},
])
def test_bad_usage_keeps_known_cost(tmp_path, usage):
    result,_=run_one(tmp_path,response(wire(),usage=usage))
    assert result["halt_reason"]=="invalid_usage_tokens"
    assert result["attempts"][0]["cost_status"]=="provider_reported"


@pytest.mark.parametrize("usage,code", [
    ({"cost":"0.006","input_tokens":2,"output_tokens":1},"cost_exceeds_reservation"),
    ({"cost":"0.0001","input_tokens":1000000,"output_tokens":1},"usage_exceeds_reservation")])
def test_usage_and_cost_overruns_are_recorded_then_halt(tmp_path, usage, code):
    result,calls=run_one(tmp_path,response(wire(),usage=usage))
    assert len(calls)==1 and result["halt_reason"]==code
    assert result["attempts"][0]["actual_cost_usd"]==usage["cost"]


def test_http_error_retains_only_status_and_known_cost_without_body(tmp_path):
    body=json.dumps({"error":{"message":"private upstream message"},"usage":{"cost":"0.0002"}}).encode()
    error=urllib.error.HTTPError(client.ENDPOINT,429,"private exception",{},io.BytesIO(body))
    result,calls=run_one(tmp_path,error)
    assert len(calls)==1 and result["halt_reason"]=="http_error"
    assert result["attempts"][0]["http_status"]==429
    assert result["attempts"][0]["actual_cost_usd"]=="0.0002"
    stored="".join(p.read_text(encoding="utf-8") for p in (tmp_path/"run").iterdir())
    assert "private upstream" not in stored and "private exception" not in stored


@pytest.mark.parametrize("error", [TimeoutError("private transport text"),
    RuntimeError("private transport text"), client.ProbeFailure("private transport text")])
def test_transport_errors_unknown_cost_no_retry_and_no_exception_text(tmp_path,error):
    result,calls=run_one(tmp_path,error)
    assert len(calls)==1 and result["status"]=="halted"
    assert result["attempts"][0]["cost_status"]=="cost_unknown"
    assert "private transport text" not in json.dumps(result)


@pytest.mark.parametrize("raw", [b'not JSON', b'{"model":"a","model":"b"}', b'a'*65537, b'[]'],
                         ids=["not_json", "duplicate_field", "oversized", "wrong_root_type"])
def test_bad_or_oversized_raw_response_is_never_saved(tmp_path,raw):
    result,calls=run_one(tmp_path,raw)
    assert result["status"]=="halted" and len(calls)==1
    assert not list((tmp_path/"run").glob("response_*.json"))


def test_parsed_response_redacted_and_absent_provider_explicit(tmp_path):
    w=wire(); value=response(w); del value["provider"]
    fake_secret="sk-or-v1-"+"A"*40
    value["untrusted_optional"]=fake_secret
    result,_=run_one(tmp_path,value,w)
    a=result["attempts"][0]
    assert a["provider_response_status"]=="not_reported_route_pinned_in_request"
    stored=(tmp_path/"run"/"response_001.json").read_text(encoding="utf-8")
    assert fake_secret not in stored and "[REDACTED]" in stored


def test_deadline_after_response_keeps_known_cost_and_stops(tmp_path):
    now=[0.0]; calls=[]
    def fake(w,timeout):
        calls.append(timeout); now[0]=4.0; return response(w)
    result=client.ProbeClient(tmp_path/"run",(wire(),wire(2)),transport=fake,
                              deadline_seconds=3,clock=lambda:now[0]).run()
    assert calls==[3.0] and result["halt_reason"]=="deadline_exceeded"
    assert result["attempts"][0]["actual_cost_usd"]=="0.0001"


def test_deadline_before_dispatch_has_no_attempt(tmp_path):
    ticks=iter((0.0,181.0,181.0))
    result=client.ProbeClient(tmp_path/"run",(wire(),),transport=lambda w,t:pytest.fail("no call"),
                              clock=lambda:next(ticks)).run()
    assert result["halt_reason"]=="deadline_exceeded" and result["attempts"]==[]
    assert Decimal(result["reservation_total_usd"])==0


@pytest.mark.parametrize("proxy",["http://remote.example:8080", "http://user:secret@localhost:8080",
                                 "http://127.0.0.1:8080/?secret=x", "http://localhost"])
def test_proxy_must_be_explicit_credential_free_loopback(tmp_path,proxy):
    with pytest.raises(ValueError):
        client.ProbeClient(tmp_path/"run",(wire(),),proxy=proxy)
