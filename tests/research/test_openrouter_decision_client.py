"""Offline contracts for paid-call admission, provenance and secret-safe logs."""
from copy import deepcopy
from decimal import Decimal
import io
import json
from pathlib import Path
import sys
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
from openrouter_decision_client import (
    BoundedClient, ERROR_CODE_CHAR_CAP, ERROR_MESSAGE_CHAR_CAP, ERROR_RAW_BYTE_CAP, RESPONSE_BYTE_CAP,
    make_payload, parse_labels, task_batches,
)


def task(identifier="q0", text="A method combines two estimates."):
    return {"id": identifier, "item": {
        "query": "What method is used?", "unit": {"id": "p0", "text": text}}}


def response_for(backend, payload):
    result = {"id": "offline-request", "provider": "TypeSafe" if backend == "jev" else "OpenAI",
              "model": "typesafe/jev-1.13-20260917" if backend == "jev" else "openai/gpt-4.1-mini-2025-04-14",
              "usage": {"cost": 0.00001}}
    if backend == "jev":
        result["usage"].update(input_tokens=100, output_tokens=10)
        result["answers"] = {key: {"type": "choice", "choice": "yes"} for key in payload["questions"]}
    else:
        result["usage"].update(prompt_tokens=100, completion_tokens=10)
        ids = payload["response_format"]["json_schema"]["schema"]["required"]
        result["choices"] = [{"finish_reason": "stop", "message": {"content": json.dumps(dict.fromkeys(ids, "yes"))}}]
    return result


@pytest.mark.parametrize("backend", ["jev", "general"])
def test_valid_response_is_charged_and_reservation_persisted_before_transport(tmp_path, backend):
    output = tmp_path / backend
    calls = []

    def transport(kind, payload):
        persisted = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
        assert persisted["attempts"][-1]["status"] == "in_flight"
        assert Decimal(persisted["reservation_total_usd"]) >= Decimal("0.005")
        calls.append(kind)
        return response_for(kind, payload)

    client = BoundedClient(output, transport=transport)
    assert client.submit([task()], "support", backend) == {"q0": "yes"}
    assert calls == [backend]
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["attempts"][0]["status"] == "completed"
    assert Decimal(ledger["actual_reported_cost_usd"]) == Decimal("0.00001")


@pytest.mark.parametrize("backend", ["jev", "general"])
@pytest.mark.parametrize("field,bad_value", [("model", "another/model"), ("provider", "UnexpectedProvider")])
def test_unrequested_model_or_provider_halts(tmp_path, backend, field, bad_value):
    def transport(kind, payload):
        result = response_for(kind, payload)
        result[field] = bad_value
        return result

    client = BoundedClient(tmp_path / "run", transport=transport)
    with pytest.raises(RuntimeError):
        client.submit([task()], "support", backend)
    assert client.ledger["halt_reason"]


def test_echoed_secret_cannot_reach_any_persisted_response_or_ledger_field(tmp_path):
    secret = "test-credential"

    def transport(kind, payload):
        result = response_for(kind, payload)
        result["id"] = "provider-echo:" + secret
        result["usage"]["provider_metadata"] = {"echo": secret}
        return result

    output = tmp_path / "run"
    client = BoundedClient(output, transport=transport)
    try:
        client.submit([task()], "support", "jev")
    except RuntimeError:
        pass  # Rejecting a suspicious response is also safe.
    for path in output.glob("*.json"):
        assert secret not in path.read_text(encoding="utf-8"), path.name


def test_transport_failure_keeps_reservation_and_prevents_a_second_charge(tmp_path):
    calls = []

    def transport(backend, payload):
        calls.append(backend)
        raise TimeoutError("Potential authentication material must not be logged")

    client = BoundedClient(tmp_path / "run", transport=transport)
    with pytest.raises(RuntimeError):
        client.submit([task()], "support", "jev")
    reserved = client.ledger["reservation_total_usd"]
    with pytest.raises(RuntimeError):
        client.submit([task("q1")], "support", "jev")
    assert calls == ["jev"]
    assert Decimal(reserved) >= Decimal("0.005")
    assert client.ledger["reservation_total_usd"] == reserved
    assert "Potential authentication" not in (tmp_path / "run" / "ledger.json").read_text()


@pytest.mark.parametrize("change", ["missing_cost", "nan_cost", "too_expensive", "too_many_tokens", "bool_tokens"])
def test_bad_or_unaccounted_usage_halts_without_retry(tmp_path, change):
    calls = []

    def transport(backend, payload):
        calls.append(backend)
        result = response_for(backend, payload)
        if change == "missing_cost":
            del result["usage"]["cost"]
        elif change == "nan_cost":
            result["usage"]["cost"] = "NaN"
        elif change == "too_expensive":
            result["usage"]["cost"] = 1
        elif change == "too_many_tokens":
            result["usage"]["input_tokens"] = 10**9
        else:
            result["usage"]["input_tokens"] = True
        return result

    client = BoundedClient(tmp_path / "run", transport=transport)
    with pytest.raises(RuntimeError):
        client.submit([task()], "support", "jev")
    with pytest.raises(RuntimeError):
        client.submit([task("q1")], "support", "jev")
    assert len(calls) == 1


def test_admission_stops_before_any_transport(tmp_path):
    def transport(*args):
        pytest.fail("No request is allowed below its conservative reservation")

    client = BoundedClient(tmp_path / "run", transport=transport, budget_usd="0.001")
    with pytest.raises(RuntimeError):
        client.submit([task()], "support", "general")
    assert client.ledger["attempts"] == []


def test_task_grouping_preserves_same_visible_state_and_questions_for_both_backends():
    tasks = [task(f"q{i}", "Unicode 段落 " * 180) for i in range(11)]
    grouped = task_batches(tasks, "support")
    assert [entry["id"] for batch in grouped for entry in batch] == [entry["id"] for entry in tasks]
    assert all(len(batch) <= 8 for batch in grouped)
    for batch in grouped:
        jev = make_payload(batch, "support", "jev")
        general = make_payload(batch, "support", "general")
        visible = json.loads(general["messages"][1]["content"])
        assert visible == {"state": jev["state"], "questions": jev["questions"]}
        for entry in batch:
            assert jev["state"]["items"][entry["id"]] == entry["item"]


def test_no_gold_or_extra_context_can_be_added_to_visible_item():
    bad_task = deepcopy(task())
    bad_task["item"]["answer_annotations"] = ["gold label"]
    with pytest.raises(ValueError):
        make_payload([bad_task], "support", "jev")


@pytest.mark.parametrize("backend", ["jev", "general"])
def test_missing_or_extra_decision_ids_and_unknown_labels_rejected(backend):
    payload = make_payload([task()], "support", backend)
    for labels in ({}, {"q0": "yes", "unexpected": "no"}, {"q0": "made-up"}):
        result = response_for(backend, payload)
        if backend == "jev":
            result["answers"] = {key: {"type": "choice", "choice": value} for key, value in labels.items()}
        else:
            result["choices"][0]["message"]["content"] = json.dumps(labels)
        with pytest.raises(ValueError):
            parse_labels(result, payload, backend, "support")


def test_ambiguous_duplicate_json_key_is_rejected():
    payload = make_payload([task()], "support", "general")
    result = response_for("general", payload)
    result["choices"][0]["message"]["content"] = '{"q0":"no","q0":"yes"}'
    with pytest.raises(ValueError):
        parse_labels(result, payload, "general", "support")


def test_general_truncated_response_is_not_a_valid_decision():
    payload = make_payload([task()], "support", "general")
    result = response_for("general", payload)
    result["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError):
        parse_labels(result, payload, "general", "support")


@pytest.mark.parametrize("backend", ["jev", "general"])
def test_completed_request_cannot_be_paid_again(tmp_path, backend):
    calls = []

    def transport(kind, payload):
        calls.append(kind)
        return response_for(kind, payload)

    client = BoundedClient(tmp_path / "run", transport=transport)
    client.submit([task()], "support", backend)
    before = client.ledger["reservation_total_usd"]
    with pytest.raises(RuntimeError):
        client.submit([task()], "support", backend)
    assert calls == [backend]
    assert client.ledger["reservation_total_usd"] == before


@pytest.mark.parametrize("backend", ["jev", "general"])
def test_optional_provider_is_explicitly_recorded_as_unreported(tmp_path, backend):
    def transport(kind, payload):
        result = response_for(kind, payload)
        del result["provider"]
        return result

    client = BoundedClient(tmp_path / "run", transport=transport)
    client.submit([task()], "support", backend)
    record = client.ledger["attempts"][0]
    assert record["provider"] is None
    assert record["provider_response_status"] == "not_reported_route_pinned_in_request"


def test_large_single_unit_is_rejected_instead_of_truncated():
    with pytest.raises(ValueError):
        task_batches([task(text="完整文本" * 10000)], "support")


def test_colon_bearing_id_has_an_explicit_bracket_reference():
    payload = make_payload([task("paper:query:unit")], "support", "jev")
    instructions = payload["questions"]["paper:query:unit"]["instructions"]
    assert 'state.items["paper:query:unit"]' in instructions


def test_http_error_saves_redacted_limited_diagnostics_without_retry(tmp_path):
    other_secret = "sk-or-v1-" + "Z" * 64
    calls = []
    body = {"error": {"code": 403, "message": (
        "test-credential " + other_secret + " " + "x" * ERROR_MESSAGE_CHAR_CAP),
        "metadata": {"headers": {"Authorization": "test-credential"}}}}

    def transport(backend, payload):
        calls.append(backend)
        raise urllib.error.HTTPError("https://example.invalid", 403, "test-credential", {},
                                     io.BytesIO(json.dumps(body).encode("utf-8")))

    output = tmp_path / "run"
    client = BoundedClient(output, transport=transport)
    with pytest.raises(RuntimeError, match="HTTP_403"):
        client.submit([task()], "support", "general")
    reserved = client.ledger["reservation_total_usd"]
    with pytest.raises(RuntimeError, match="client halted"):
        client.submit([task("q1")], "support", "general")
    assert calls == ["general"]
    assert client.ledger["reservation_total_usd"] == reserved
    assert Decimal(reserved) >= Decimal("0.005")
    record = client.ledger["attempts"][0]
    assert record["status"] == "halted"
    assert record["error_class"] == "HTTP_403"
    assert record["cost_status"] == "cost_unknown"
    assert "actual_cost_usd" not in record
    diagnostic = json.loads((output / "error_response_001.json").read_text(encoding="utf-8"))
    assert diagnostic["body_status"] == "json"
    assert set(diagnostic["error"]) == {"code", "message"}
    assert diagnostic["error"]["code"] == 403
    assert "[REDACTED]" in diagnostic["error"]["message"]
    assert len(diagnostic["error"]["message"]) == ERROR_MESSAGE_CHAR_CAP
    for path in output.glob("*.json"):
        saved = path.read_text(encoding="utf-8")
        assert "test-credential" not in saved
        assert other_secret not in saved
        assert "Authorization" not in saved


@pytest.mark.parametrize("body", [b'not JSON test-credential', b'\xfftest-credential',
                                   b'{"error": "a", "error": "test-credential"}', b'[]'])
def test_invalid_http_error_body_is_not_persisted_or_treated_as_zero_cost(tmp_path, body):
    def transport(*args):
        raise urllib.error.HTTPError("https://example.invalid", 403, "forbidden", {}, io.BytesIO(body))

    client = BoundedClient(tmp_path / "run", transport=transport)
    with pytest.raises(RuntimeError, match="HTTP_403"):
        client.submit([task()], "support", "general")
    diagnostic = json.loads((client.output / "error_response_001.json").read_text(encoding="utf-8"))
    assert diagnostic == {"http_status": 403, "body_status": "invalid_json", "cost_status": "cost_unknown"}
    assert client.ledger["halt_reason"] == "HTTP_403"
    assert "actual_cost_usd" not in client.ledger["attempts"][0]
    assert "test-credential" not in (client.output / "ledger.json").read_text(encoding="utf-8")


def test_http_error_read_has_strict_two_mib_cap(tmp_path):
    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            assert size == RESPONSE_BYTE_CAP
            raw = super().read(size)
            self.bytes_read = len(raw)
            return raw

    stream = BoundedReader(b'{"error":{"message":"test-credential ' + b"x" * RESPONSE_BYTE_CAP + b'"}}')

    def transport(*args):
        raise urllib.error.HTTPError("https://example.invalid", 403, "forbidden", {}, stream)

    client = BoundedClient(tmp_path / "run", transport=transport)
    with pytest.raises(RuntimeError, match="HTTP_403"):
        client.submit([task()], "support", "general")
    assert stream.bytes_read == RESPONSE_BYTE_CAP
    diagnostic = json.loads((client.output / "error_response_001.json").read_text(encoding="utf-8"))
    assert diagnostic == {"http_status": 403, "body_status": "size_limit", "cost_status": "cost_unknown"}
    assert "actual_cost_usd" not in client.ledger["attempts"][0]


def test_http_error_body_read_failure_preserves_http_status(tmp_path):
    class BrokenReader(io.BytesIO):
        def read(self, size=-1):
            raise OSError("test-credential must never enter diagnostics")

    def transport(*args):
        raise urllib.error.HTTPError("https://example.invalid", 403, "forbidden", {}, BrokenReader())

    client = BoundedClient(tmp_path / "run", transport=transport)
    with pytest.raises(RuntimeError, match="HTTP_403"):
        client.submit([task()], "support", "general")
    diagnostic = json.loads((client.output / "error_response_001.json").read_text(encoding="utf-8"))
    assert diagnostic == {"http_status": 403, "body_status": "read_failed", "cost_status": "cost_unknown"}
    assert client.ledger["halt_reason"] == "HTTP_403"
    assert "test-credential" not in (client.output / "ledger.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("cost", [None, True, "NaN", "-0.1", "1e9999999999999"])
def test_http_error_invalid_cost_stays_unknown_but_valid_tokens_are_recorded(tmp_path, cost):
    body = {"error": {"code": 403, "message": "forbidden"},
            "usage": {"cost": cost, "prompt_tokens": 12, "completion_tokens": 0,
                      "metadata": "test-credential", "total_tokens": True}}

    def transport(*args):
        raise urllib.error.HTTPError("https://example.invalid", 403, "forbidden", {},
                                     io.BytesIO(json.dumps(body).encode("utf-8")))

    client = BoundedClient(tmp_path / "run", transport=transport)
    with pytest.raises(RuntimeError, match="HTTP_403"):
        client.submit([task()], "support", "general")
    record = client.ledger["attempts"][0]
    assert record["cost_status"] == "cost_unknown"
    assert "actual_cost_usd" not in record
    assert record["usage"] == {"prompt_tokens": 12, "completion_tokens": 0}
    assert record["input_tokens"] == 12
    assert record["output_tokens"] == 0
    assert client.ledger["actual_reported_cost_usd"] == "0"


@pytest.mark.parametrize("cost", ["0", "0.002"])
def test_http_error_reported_cost_is_added_once_to_completed_costs(tmp_path, cost):
    def transport(backend, payload):
        if backend == "jev":
            return response_for(backend, payload)
        body = {"error": {"code": 403, "message": "forbidden"}, "usage": {"cost": cost}}
        raise urllib.error.HTTPError("https://example.invalid", 403, "forbidden", {},
                                     io.BytesIO(json.dumps(body).encode("utf-8")))

    client = BoundedClient(tmp_path / "run", transport=transport)
    client.submit([task()], "support", "jev")
    with pytest.raises(RuntimeError, match="HTTP_403"):
        client.submit([task()], "support", "general")
    record = client.ledger["attempts"][1]
    assert record["actual_cost_usd"] == cost
    assert record["cost_status"] == "provider_reported"
    assert record["usage"] == {"cost": cost}
    assert Decimal(client.ledger["actual_reported_cost_usd"]) == Decimal("0.00001") + Decimal(cost)
    assert "input_tokens" not in record
    assert "output_tokens" not in record
    assert Decimal(client.ledger["reservation_total_usd"]) >= Decimal("0.01")
    assert (client.output / "error_response_002.json").exists()


@pytest.mark.parametrize("raw_format", ["object", "json_string", "escaped_json_string", "flat_object"])
def test_upstream_provider_error_retains_only_bounded_redacted_fields(tmp_path, raw_format):
    other_secret = "sk-or-v1-" + "Z" * 64
    upstream = {"code": "InvalidParameter", "message": "Unsupported parameter test-credential " + other_secret,
                "headers": {"Authorization": "test-credential"}, "unrequested": "do-not-save"}
    raw = upstream if raw_format == "flat_object" else {"error": upstream, "unrequested": "do-not-save"}
    if raw_format in {"json_string", "escaped_json_string"}:
        raw = json.dumps(raw)
    if raw_format == "escaped_json_string":
        raw = raw.replace("test-credential", r"\u0074est-credential").replace("sk-or-v1-", r"\u0073k-or-v1-")
    body = {"error": {"code": 400, "message": "Provider returned error", "metadata": {
        "provider_name": "Alibaba test-credential", "raw": raw, "unrequested": "do-not-save"}}}

    def transport(*args):
        raise urllib.error.HTTPError("https://example.invalid", 400, "bad request", {},
                                     io.BytesIO(json.dumps(body).encode("utf-8")))

    client = BoundedClient(tmp_path / "run", transport=transport)
    with pytest.raises(RuntimeError, match="HTTP_400"):
        client.submit([task()], "support", "general")
    diagnostic = json.loads((client.output / "error_response_001.json").read_text(encoding="utf-8"))
    error = diagnostic["error"]
    assert error == {
        "code": 400, "message": "Provider returned error", "provider_name": "Alibaba [REDACTED]",
        "upstream_body_status": "json", "upstream_error": {
            "code": "InvalidParameter", "message": "Unsupported parameter [REDACTED] [REDACTED]"}}
    assert client.ledger["attempts"][0]["provider_error"] == error
    assert client.ledger["attempts"][0]["cost_status"] == "cost_unknown"
    for path in client.output.glob("*.json"):
        content = path.read_text(encoding="utf-8")
        for unwanted in ("test-credential", other_secret, "do-not-save", "Authorization", '"raw"'):
            assert unwanted not in content


@pytest.mark.parametrize("raw,status", [
    ("not JSON test-credential", "invalid_json"),
    ('{"code": 400, "code": "test-credential"}', "invalid_json"),
    ([], "invalid_json"),
    ('"test-credential"', "invalid_json"),
    ({"error": "test-credential"}, "invalid_json"),
    ('{"message":"' + "x" * ERROR_RAW_BYTE_CAP + '"}', "size_limit"),
    ({"message": "x" * ERROR_RAW_BYTE_CAP}, "size_limit"),
], ids=["plain-text", "duplicate-key", "list", "json-scalar", "non-object-error", "large-json", "large-object"])
def test_unusable_upstream_raw_is_not_saved_and_does_not_change_halt(tmp_path, raw, status):
    body = {"error": {"code": 400, "message": "Provider returned error", "metadata": {
        "provider_name": "Alibaba", "raw": raw}}}

    def transport(*args):
        raise urllib.error.HTTPError("https://example.invalid", 400, "bad request", {},
                                     io.BytesIO(json.dumps(body).encode("utf-8")))

    client = BoundedClient(tmp_path / "run", transport=transport)
    with pytest.raises(RuntimeError, match="HTTP_400"):
        client.submit([task()], "support", "general")
    record = client.ledger["attempts"][0]
    assert record["provider_error"] == {
        "code": 400, "message": "Provider returned error", "provider_name": "Alibaba", "upstream_body_status": status}
    assert record["cost_status"] == "cost_unknown"
    assert record["error_class"] == "HTTP_400"
    assert client.ledger["halt_reason"] == "HTTP_400"
    assert "actual_cost_usd" not in record
    assert Decimal(client.ledger["reservation_total_usd"]) >= Decimal("0.005")


def test_upstream_diagnostic_fields_are_redacted_before_truncation(tmp_path):
    client = BoundedClient(tmp_path / "run", transport=lambda *_: None)
    metadata = {"provider_name": "P" * (ERROR_CODE_CHAR_CAP - 2) + "test-credential", "raw": {
        "error": {"code": "C" * (ERROR_CODE_CHAR_CAP - 2) + "test-credential",
                  "message": "M" * (ERROR_MESSAGE_CHAR_CAP - 2) + "test-credential"}}}
    result = client.provider_error_metadata(metadata)
    assert result["provider_name"] == "P" * (ERROR_CODE_CHAR_CAP - 2) + "[R"
    assert result["upstream_error"]["code"] == "C" * (ERROR_CODE_CHAR_CAP - 2) + "[R"
    assert result["upstream_error"]["message"] == "M" * (ERROR_MESSAGE_CHAR_CAP - 2) + "[R"
