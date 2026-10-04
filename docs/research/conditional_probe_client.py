"""Six-request conditional probe transport. Importing never reads keys or calls APIs.

The caller must add a process-level watchdog: socket deadlines cannot terminate
every blocked operation. Reservations are conservative admission checks, not an
account-level spending cap. This client never resumes or retries an attempt.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import ssl
import time
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request

from SLAC.retrieval.decision.conditional import FrozenRequest, decode_response
from docs.research.openrouter_decision_client import (
    NoRedirect, canonical_bytes, object_hash, read_key, reservation, unique_object,
    MODELS,
)

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL_ID = "typesafe/jev-1.13"
RESPONSE_MODEL = "typesafe/jev-1.13-20260917"
WIRE_VERSION = "slac-conditional-probe-wire-v1"
LEDGER_VERSION = "slac-conditional-probe-ledger-v1"
WIRE_BYTE_CAP = 24000
RESPONSE_BYTE_CAP = 65536
REQUEST_CAP = 6
QUESTION_CAP = 10
BUDGET_USD = Decimal("0.03")
ERROR_CODES = frozenset({
    "invalid_response_json", "response_size_limit", "invalid_response_object",
    "invalid_usage", "invalid_cost", "cost_exceeds_reservation", "invalid_usage_tokens",
    "usage_exceeds_reservation", "response_error_object", "response_model_mismatch",
    "response_provider_mismatch", "response_answer_ids_mismatch", "response_answer_schema_mismatch",
    "execution_mode_not_explicit", "missing_key_file", "credential_load_failed", "deadline_exceeded",
    "http_error", "transport_timeout_or_error", "transport_or_local_error", "local_execution_error",
})


def _provider() -> dict:
    return {"only": ["typesafe"], "allow_fallbacks": False,
            "max_price": {"prompt": "0.042", "completion": "0", "request": "0"}}


def _parse(value: bytes | str | dict) -> dict:
    if type(value) is dict:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    elif type(value) is str:
        raw = value.encode("utf-8")
    elif type(value) is bytes:
        raw = value
    else:
        raise ProbeFailure("invalid_response_json")
    if len(raw) > RESPONSE_BYTE_CAP:
        raise ProbeFailure("response_size_limit")
    def invalid_constant(_):
        raise ValueError("nonfinite JSON number")
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                            parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ProbeFailure("invalid_response_json") from None
    if type(parsed) is not dict:
        raise ProbeFailure("invalid_response_object")
    return parsed


@dataclass(frozen=True)
class WireRequest:
    core: FrozenRequest
    payload_bytes: bytes
    binding_bytes: bytes
    cache_key: str
    reserved_usd: str
    input_allowance: int
    output_allowance: int

    @property
    def endpoint(self) -> str:
        return ENDPOINT

    @property
    def expected_ids(self) -> tuple[str, ...]:
        return self.core.expected_ids

    @property
    def expected_response_model(self) -> str:
        return RESPONSE_MODEL

    def payload(self) -> dict:
        return json.loads(self.payload_bytes)


def freeze_wire(core: FrozenRequest) -> WireRequest:
    """Bind the complete core contract and provider-augmented, immutable wire bytes."""
    if not isinstance(core, FrozenRequest):
        raise TypeError("core must be a FrozenRequest")
    payload = core.payload()  # Also verifies the core's immutable binding.
    binding = json.loads(core.binding_bytes)
    if (binding["endpoint_id"] != ENDPOINT or binding["model_id"] != MODEL_ID
            or core.expected_response_model != RESPONSE_MODEL
            or binding["arm"] not in ("standalone", "plain_conditional")):
        raise ValueError("unsupported frozen probe identity or arm")
    expected_ids = (("standalone_support",) if binding["arm"] == "standalone"
                    else ("conditional_added_information", "conflict"))
    if core.expected_ids != expected_ids or set(payload) != {"model", "state", "questions"}:
        raise ValueError("unsupported frozen probe schema")
    # The shared accounting helper reads this registry; reject mutable price drift.
    expected_model = {"id": MODEL_ID, "endpoint": ENDPOINT, "provider": "typesafe",
                      "prompt_per_million": "0.042", "completion_per_million": "0"}
    if any(MODELS["jev"].get(k) != v for k, v in expected_model.items()):
        raise ValueError("shared accounting identity differs from frozen probe")
    payload["provider"] = _provider()
    body = canonical_bytes(payload)
    if len(body) > WIRE_BYTE_CAP:
        raise ValueError("complete wire exceeds byte cap")
    reserved, input_allowance, output_allowance = reservation(payload, "jev")
    if reserved != Decimal("0.005"):
        raise ValueError("request reservation differs from frozen five-mill floor")
    wire_binding = {"namespace": WIRE_VERSION, "core_binding": binding,
                    "core_cache_key": core.cache_key, "endpoint": ENDPOINT,
                    "expected_response_model": RESPONSE_MODEL,
                    "provider": _provider(), "wire_sha256": hashlib.sha256(body).hexdigest(),
                    "reserved_usd": str(reserved), "input_allowance": input_allowance,
                    "output_allowance": output_allowance}
    frozen_binding = canonical_bytes(wire_binding)
    return WireRequest(core, body, frozen_binding, object_hash(wire_binding),
                       str(reserved), input_allowance, output_allowance)


class ProbeFailure(Exception):
    """Only constructed with a fixed local diagnostic code; no provider text."""


def _proxy(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError("proxy must be an explicit loopback URL")
    parts = urllib.parse.urlsplit(value)
    if (parts.scheme not in ("http", "https") or parts.hostname not in ("127.0.0.1", "localhost", "::1")
            or parts.username is not None or parts.password is not None
            or parts.path not in ("", "/") or parts.query or parts.fragment or parts.port is None):
        raise ValueError("proxy must be an explicit loopback URL without credentials")
    return value


class ProbeClient:
    """Exclusive new ledger; explicit fake transport or run(live=True) only.

    Fake transport signature: (WireRequest, timeout_seconds) -> bytes | str | dict.
    A supplied fake transport never receives a credential. Live credentials are
    loaded only inside run(), are never represented in the ledger, and are cleared
    afterward. run() returns a copy of the completed or halted ledger.
    """

    def __init__(self, output, wire_requests, *, key_file=None, proxy=None,
                 transport: Callable | None = None, deadline_seconds=180,
                 clock: Callable[[], float] = time.monotonic):
        requests = tuple(wire_requests)
        if not 1 <= len(requests) <= REQUEST_CAP:
            raise ValueError("request cap exceeded")
        if any(not isinstance(w, WireRequest) or freeze_wire(w.core) != w for w in requests):
            raise ValueError("wire request binding mismatch")
        if (len({w.cache_key for w in requests}) != len(requests)
                or len({w.payload_bytes for w in requests}) != len(requests)):
            raise ValueError("duplicate physical request")
        if sum(len(w.expected_ids) for w in requests) > QUESTION_CAP:
            raise ValueError("question cap exceeded")
        planned = sum((Decimal(w.reserved_usd) for w in requests), Decimal(0))
        if planned > BUDGET_USD:
            raise ValueError("reservation cap exceeded")
        if (type(deadline_seconds) not in (int, float) or not math.isfinite(deadline_seconds)
                or not 0 < deadline_seconds <= 180 or not callable(clock)):
            raise ValueError("invalid deadline")
        if transport is not None and not callable(transport):
            raise ValueError("fake transport must be callable")
        self._proxy = _proxy(proxy)
        self.output = Path(output)
        self.requests = requests
        self._key_file, self._key = key_file, ""
        self._transport, self._clock = transport, clock
        self._deadline_seconds = float(deadline_seconds)
        self._ran = False
        self.output.mkdir(parents=True, exist_ok=False)
        self.ledger = {"schema": LEDGER_VERSION, "wire_version": WIRE_VERSION,
                       "status": "prepared", "halt_reason": None,
                       "endpoint": ENDPOINT, "requested_model": MODEL_ID,
                       "expected_response_model": RESPONSE_MODEL, "provider_policy": _provider(),
                       "proxy": self._proxy, "deadline_seconds": self._deadline_seconds,
                       "hard_watchdog_required": True, "request_cap": REQUEST_CAP,
                       "question_cap": QUESTION_CAP, "budget_usd": str(BUDGET_USD),
                       "planned_request_count": len(requests),
                       "planned_question_count": sum(len(w.expected_ids) for w in requests),
                       "planned_reservation_usd": str(planned), "reservation_total_usd": "0",
                       "actual_reported_cost_usd": "0", "automatic_retries": 0,
                       "wire_keys": [w.cache_key for w in requests], "attempts": []}
        self._save()

    def _redacted(self, value):
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if self._key:
            text = text.replace(self._key, "[REDACTED]")
        return json.loads(re.sub(r"sk-or-v1-[A-Za-z0-9]{32,}", "[REDACTED]", text))

    def _save(self):
        temporary = self.output / "ledger.tmp"
        temporary.write_text(json.dumps(self._redacted(self.ledger), ensure_ascii=False,
                                         indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.output / "ledger.json")

    def _response_file(self, index, response, *, error=False):
        name = ("error_response_" if error else "response_") + f"{index:03d}.json"
        (self.output / name).write_text(json.dumps(self._redacted(response), ensure_ascii=False,
                                                  allow_nan=False), encoding="utf-8")
        return name

    def _account(self, response, record, wire, *, require_tokens=True):
        usage = response.get("usage")
        if type(usage) is not dict:
            raise ProbeFailure("invalid_usage")
        value = usage.get("cost")
        if type(value) not in (str, int, float) or len(str(value)) > 128:
            raise ProbeFailure("invalid_cost")
        try:
            cost = Decimal(str(value))
        except InvalidOperation:
            raise ProbeFailure("invalid_cost") from None
        if not cost.is_finite() or cost < 0:
            raise ProbeFailure("invalid_cost")
        record.update(cost_status="provider_reported", actual_cost_usd=str(cost))
        total = Decimal(self.ledger["actual_reported_cost_usd"]) + cost
        self.ledger["actual_reported_cost_usd"] = str(total)
        self._save()  # Known cost survives a later schema, usage, or identity failure.
        if cost > Decimal(wire.reserved_usd) or total > BUDGET_USD:
            raise ProbeFailure("cost_exceeds_reservation")
        if not require_tokens:
            return
        tokens = {}
        for target, aliases in (("input_tokens", ("input_tokens", "prompt_tokens")),
                                ("output_tokens", ("output_tokens", "completion_tokens"))):
            values = [usage[k] for k in aliases if k in usage]
            if not values or any(type(v) is not int or v < 0 for v in values) or len(set(values)) != 1:
                raise ProbeFailure("invalid_usage_tokens")
            tokens[target] = values[0]
        record["usage"] = {**tokens, "cost": str(cost)}
        if tokens["input_tokens"] > wire.input_allowance or tokens["output_tokens"] > wire.output_allowance:
            raise ProbeFailure("usage_exceeds_reservation")

    def _validate(self, response, wire, record):
        if "error" in response:
            raise ProbeFailure("response_error_object")
        if response.get("model") != RESPONSE_MODEL:
            raise ProbeFailure("response_model_mismatch")
        provider = response.get("provider")
        if provider is not None and (type(provider) is not str or provider.casefold() != "typesafe"):
            raise ProbeFailure("response_provider_mismatch")
        record["provider_response_status"] = ("reported" if provider is not None
                                                else "not_reported_route_pinned_in_request")
        record["response_model"] = response["model"]
        if provider is not None:
            record["provider"] = provider
        answers = response.get("answers")
        if type(answers) is not dict or set(answers) != set(wire.expected_ids):
            raise ProbeFailure("response_answer_ids_mismatch")
        decoded = decode_response(wire.core, {"request_key": wire.core.cache_key, "response": response},
                                  max_response_bytes=RESPONSE_BYTE_CAP + 1024)
        if not decoded.envelope_valid or any(d.reason not in (None, "model_unknown") for d in decoded.decisions):
            raise ProbeFailure("response_answer_schema_mismatch")
        return {d.id: d.choice for d in decoded.decisions}

    def _failure(self, code, record=None):
        code = code if code in ERROR_CODES else "transport_or_local_error"
        self.ledger.update(status="halted", halt_reason=code)
        if record is not None:
            record.update(status="halted", error_code=code)

    def run(self, *, live=False) -> dict:
        if self._ran:
            raise RuntimeError("probe run cannot resume or retry")
        self._ran = True
        if type(live) is not bool or (self._transport is None and not live) or (self._transport is not None and live):
            self._failure("execution_mode_not_explicit")
            self._save()
            return self._redacted(self.ledger)
        started = self._clock()
        deadline = started + self._deadline_seconds
        self.ledger.update(status="running", mode="live" if live else "fake",
                           started_at=datetime.now(timezone.utc).isoformat())
        self._save()
        try:
            if live:
                if self._key_file is None:
                    raise ProbeFailure("missing_key_file")
                try:
                    self._key = read_key(self._key_file)
                except Exception:
                    raise ProbeFailure("credential_load_failed") from None
                opener = urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler(
                    {"https": self._proxy} if self._proxy else {}),
                    urllib.request.HTTPSHandler(context=ssl.create_default_context()))
            for index, wire in enumerate(self.requests, 1):
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise ProbeFailure("deadline_exceeded")
                body = wire.payload_bytes
                (self.output / f"request_{index:03d}.json").write_bytes(body)
                record = {"attempt": index, "cache_key": wire.cache_key,
                          "core_cache_key": wire.core.cache_key,
                          "request_sha256": hashlib.sha256(body).hexdigest(),
                          "status": "in_flight", "cost_status": "cost_unknown",
                          "reserved_usd": wire.reserved_usd, "input_allowance": wire.input_allowance,
                          "output_allowance": wire.output_allowance, "question_count": len(wire.expected_ids),
                          "started_at": datetime.now(timezone.utc).isoformat()}
                self.ledger["attempts"].append(record)
                self.ledger["reservation_total_usd"] = str(
                    Decimal(self.ledger["reservation_total_usd"]) + Decimal(wire.reserved_usd))
                self._save()
                attempt_start = self._clock()
                try:
                    remaining = deadline - attempt_start
                    if remaining <= 0:
                        raise ProbeFailure("deadline_exceeded")
                    timeout = min(30.0, remaining)
                    if self._transport is not None:
                        raw = self._transport(wire, timeout)
                    else:
                        request = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
                            "Authorization": "Bearer " + self._key, "Content-Type": "application/json",
                            "User-Agent": "SLAC-conditional-probe/1"})
                        with opener.open(request, timeout=timeout) as response_stream:
                            raw = response_stream.read(RESPONSE_BYTE_CAP + 1)
                    response = _parse(raw)
                    record["response_file"] = self._response_file(index, response)
                    self._account(response, record, wire)
                    if self._clock() >= deadline:
                        raise ProbeFailure("deadline_exceeded")
                    record.update(labels=self._validate(response, wire, record), status="completed")
                except urllib.error.HTTPError as exc:
                    record["http_status"] = exc.code if type(exc.code) is int and 100 <= exc.code <= 599 else None
                    try:
                        error_response = _parse(exc.read(RESPONSE_BYTE_CAP + 1))
                        self._account(error_response, record, wire, require_tokens=False)
                    except Exception:
                        pass
                    self._failure("http_error", record)
                except ProbeFailure as exc:
                    self._failure(str(exc), record)
                except (TimeoutError, urllib.error.URLError):
                    self._failure("transport_timeout_or_error", record)
                except Exception:
                    self._failure("transport_or_local_error", record)
                finally:
                    record["elapsed_seconds"] = max(0.0, self._clock() - attempt_start)
                    self._save()
                if self.ledger["status"] == "halted":
                    break
            else:
                self.ledger["status"] = "completed"
        except ProbeFailure as exc:
            self._failure(str(exc))
        except Exception:
            self._failure("local_execution_error")
        finally:
            self.ledger["elapsed_seconds"] = max(0.0, self._clock() - started)
            self.ledger["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._save()
            self._key = ""
        return self._redacted(self.ledger)
