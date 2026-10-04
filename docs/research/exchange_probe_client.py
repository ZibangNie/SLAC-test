"""Six-request exchange probe adapter; importing never reads keys or calls APIs.

Only initialization and wire admission differ from the sealed conditional probe.
Its transport, accounting, redaction and no-retry loop are reused unchanged after
checking their source and runtime constants. A separate process watchdog remains
mandatory. This module does not reopen the old probe or widen its admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

from SLAC.retrieval.decision import conditional, exchange
from docs.research import conditional_probe_client as base
from docs.research import openrouter_decision_client as accounting

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL_ID = "typesafe/jev-1.13"
RESPONSE_MODEL = "typesafe/jev-1.13-20260917"
WIRE_VERSION = "slac-exchange-probe-wire-v1"
LEDGER_VERSION = "slac-exchange-probe-ledger-v1"
REQUEST_CAP = 6
QUESTION_CAP = 18
BUDGET_USD = Decimal("0.03")
WIRE_BYTE_CAP = 24000
RESPONSE_BYTE_CAP = 65536
EXPECTED_IDS = ("original_information_lost", "proposed_adds_information", "proposed_conflict")
DEPENDENCY_SHA256 = {
    "docs/research/conditional_probe_client.py": "2c0e46384b4b731742271a7854b6d78712c9df65b7c1b0b9b585976ea350d26d",
    "docs/research/openrouter_decision_client.py": "1400d8fd16d65da1640cd0bb3ff3bb4e7ea4e3a80cb75356f0be2327bf1c97ba",
    "SLAC/retrieval/decision/conditional.py": "ce2bc425dceb13eee9706b46951b4a02e51eff01491382a4bf2de5d6598c39de",
    "SLAC/retrieval/decision/exchange.py": "0986c93bbfb7ecb30e8837fb0b3b22f67a57276848d5bba4f9b21be59f637e72",
}


def _verify_dependencies() -> None:
    modules = (base, accounting, conditional, exchange)
    for (name, expected), module in zip(DEPENDENCY_SHA256.items(), modules, strict=True):
        path = Path(module.__file__).resolve()
        if path != (ROOT / name).resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("sealed exchange probe dependency changed")
    expected_constants = {"ENDPOINT": ENDPOINT, "MODEL_ID": MODEL_ID,
                          "RESPONSE_MODEL": RESPONSE_MODEL, "BUDGET_USD": BUDGET_USD,
                          "REQUEST_CAP": REQUEST_CAP, "WIRE_BYTE_CAP": WIRE_BYTE_CAP,
                          "RESPONSE_BYTE_CAP": RESPONSE_BYTE_CAP}
    if any(getattr(base, name) != value for name, value in expected_constants.items()):
        raise ValueError("inherited exchange probe transport constants changed")
    expected_model = {"id": MODEL_ID, "endpoint": ENDPOINT, "provider": "typesafe",
                      "prompt_per_million": "0.042", "completion_per_million": "0"}
    if any(accounting.MODELS["jev"].get(k) != v for k, v in expected_model.items()):
        raise ValueError("shared accounting identity differs from frozen exchange probe")


@dataclass(frozen=True)
class ExchangeWireRequest(base.WireRequest):
    plan: exchange.ExchangePlan


def freeze_wire(plan: exchange.ExchangePlan) -> ExchangeWireRequest:
    """Admit the exact exchange contract without passing through old freeze_wire."""
    _verify_dependencies()
    exchange._verify_plan(plan)
    core = plan.request
    payload, binding = core.payload(), json.loads(core.binding_bytes)
    if (binding["endpoint_id"] != ENDPOINT or binding["model_id"] != MODEL_ID
            or core.expected_response_model != RESPONSE_MODEL
            or core.expected_ids != EXPECTED_IDS
            or set(payload) != {"model", "state", "questions"}):
        raise ValueError("unsupported frozen exchange identity or schema")
    payload["provider"] = base._provider()
    body = accounting.canonical_bytes(payload)
    if len(body) > WIRE_BYTE_CAP:
        raise ValueError("complete exchange wire exceeds byte cap")
    reserved, input_allowance, output_allowance = accounting.reservation(payload, "jev")
    if reserved != Decimal("0.005"):
        raise ValueError("exchange reservation differs from frozen five-mill floor")
    wire_binding = {"namespace": WIRE_VERSION, "core_binding": binding,
                    "core_cache_key": core.cache_key, "endpoint": ENDPOINT,
                    "expected_response_model": RESPONSE_MODEL,
                    "provider": base._provider(), "wire_sha256": hashlib.sha256(body).hexdigest(),
                    "reserved_usd": str(reserved), "input_allowance": input_allowance,
                    "output_allowance": output_allowance,
                    "dependency_sha256": DEPENDENCY_SHA256.copy()}
    return ExchangeWireRequest(core, body, accounting.canonical_bytes(wire_binding),
                               accounting.object_hash(wire_binding), str(reserved),
                               input_allowance, output_allowance, plan)


class ExchangeProbeClient(base.ProbeClient):
    """Exactly six unique three-dimension requests, exclusive output, one run.

    The enclosing runner must bind the six-request ordering to its frozen plan.
    Only this constructor's admission is new; no inherited module is modified.
    Inherited run() loads credentials lazily, disables redirects/fallbacks, writes
    reservations before dispatch, records known costs and halts without retry.
    """

    def __init__(self, output, wire_requests, *, key_file=None, proxy=None,
                 transport: Callable | None = None, deadline_seconds=180,
                 clock: Callable[[], float] = time.monotonic):
        _verify_dependencies()
        requests = tuple(wire_requests)
        if len(requests) != REQUEST_CAP:
            raise ValueError("exactly six exchange requests required")
        if any(type(w) is not ExchangeWireRequest or freeze_wire(w.plan) != w for w in requests):
            raise ValueError("exchange wire request binding mismatch")
        if (len({w.cache_key for w in requests}) != REQUEST_CAP
                or len({w.payload_bytes for w in requests}) != REQUEST_CAP):
            raise ValueError("duplicate physical exchange request")
        if sum(len(w.expected_ids) for w in requests) != QUESTION_CAP:
            raise ValueError("exactly eighteen exchange dimensions required")
        planned = sum((Decimal(w.reserved_usd) for w in requests), Decimal(0))
        if planned != BUDGET_USD:
            raise ValueError("exchange reservation differs from frozen total")
        if (type(deadline_seconds) not in (int, float) or not math.isfinite(deadline_seconds)
                or not 0 < deadline_seconds <= 180 or not callable(clock)):
            raise ValueError("invalid deadline")
        if transport is not None and not callable(transport):
            raise ValueError("fake transport must be callable")
        self._proxy = base._proxy(proxy)
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
                       "expected_response_model": RESPONSE_MODEL, "provider_policy": base._provider(),
                       "proxy": self._proxy, "deadline_seconds": self._deadline_seconds,
                       "hard_watchdog_required": True, "request_cap": REQUEST_CAP,
                       "question_cap": QUESTION_CAP, "budget_usd": str(BUDGET_USD),
                       "planned_request_count": REQUEST_CAP, "planned_question_count": QUESTION_CAP,
                       "planned_reservation_usd": str(planned), "reservation_total_usd": "0",
                       "actual_reported_cost_usd": "0", "automatic_retries": 0,
                       "wire_keys": [w.cache_key for w in requests], "attempts": [],
                       "dependency_sha256": DEPENDENCY_SHA256.copy(),
                       "inherited_transport": "conditional_probe_client.ProbeClient.run"}
        self._save()

    def run(self, *, live=False) -> dict:
        _verify_dependencies()
        if tuple(w.cache_key for w in self.requests) != tuple(self.ledger["wire_keys"]):
            raise ValueError("exchange request ordering changed after preparation")
        if any(type(w) is not ExchangeWireRequest or freeze_wire(w.plan) != w for w in self.requests):
            raise ValueError("exchange wire changed after preparation")
        return super().run(live=live)
