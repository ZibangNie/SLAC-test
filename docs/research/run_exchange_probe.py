"""Offline planning and complete readout for one bounded six-case exchange probe.

Only the explicit run subcommand permits live execution through the bounded
client. Expectations never enter requests. Gain-only readout is an ablation of
the same observed gain label, not another prompt, inference or quality baseline.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for directory in (ROOT, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import exchange_probe_client as client
from SLAC.retrieval.decision.conditional import ConditionalState, DecodedResult, TypedDecision, Unit
from SLAC.retrieval.decision.exchange import build_exchange, decide

INPUT_NAME = "docs/research/fixtures/exchange_probe_inputs_v1.json"
EXPECTED_NAME = "docs/research/fixtures/exchange_probe_expectations_v1.json"
SOURCE_NAMES = (
    "docs/research/run_exchange_probe.py", "docs/research/exchange_probe_client.py",
    "docs/research/conditional_probe_client.py", "docs/research/openrouter_decision_client.py",
    "SLAC/retrieval/decision/conditional.py", "SLAC/retrieval/decision/exchange.py",
    INPUT_NAME, EXPECTED_NAME, "docs/research/EXCHANGE_JEV_PROBE_PROTOCOL_20261004.md",
    "docs/research/run_exchange_probe_bounded.py", "docs/research/run_conditional_probe_bounded.py",
    "tests/research/test_exchange_probe_runner.py", "tests/research/test_exchange_probe_client.py",
    "tests/research/test_exchange_probe_watchdog.py", "tests/research/test_exchange_decision.py",
    "tests/research/test_conditional_decision.py",
    "tests/research/test_conditional_probe_client.py", "tests/research/test_conditional_probe_watchdog.py",
)
SCHEMA = "slac-six-case-exchange-probe-plan-v1"
COUNTER_VERSION = "slac-exchange-synthetic-utf8-byte-counter-v1"
INPUT_VERSION = "slac-authored-exchange-input-state-v1"
DIMENSIONS = ("proposed_adds_information", "original_information_lost", "proposed_conflict")
CASE_IDS = tuple(f"X{i:02d}" for i in range(1, 7))
LIMITS = {"max_requests": 6, "typed_questions": 18, "max_reserved_usd": "0.03",
          "deadline_seconds": 180, "generation_max_artificial_units": 2048,
          "judge_max_artificial_units": 4096, "generation_max_units": 3,
          "max_wire_payload_bytes": 24000, "max_response_bytes": 65536,
          "max_request_timeout_seconds": 30}


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def artificial_counter(text: str) -> int:
    """UTF-8 byte units for contract testing; not actual tokenizer/billed tokens."""
    return len(text.encode("utf-8"))


def _make_exchange(state: ConditionalState, removed_id: str):
    return build_exchange(
        state, removed_id=removed_id, endpoint_id="https://openrouter.ai/api/alpha/decisions",
        model_id="typesafe/jev-1.13", expected_response_model="typesafe/jev-1.13-20260917",
        token_counter=artificial_counter, max_tokens=2048, max_units=3,
        max_judge_tokens=4096, max_payload_bytes=65536, counter_version=COUNTER_VERSION,
    )


def project_case(case: dict):
    """Project source fields explicitly, excluding case IDs and any supervision."""
    def unit(raw):
        return Unit(id=raw["id"], text=raw["text"], order=raw["order"], doc_id="synthetic")
    state = ConditionalState(case["query"], tuple(unit(raw) for raw in case["current_pack"]),
                             unit(case["candidate"]), version=INPUT_VERSION)
    return _make_exchange(state, case["removed_id"])


def freeze_requests(inputs: dict) -> tuple[list[dict], tuple, tuple]:
    """Input-only request preparation; this function never accepts expectations."""
    require(inputs.get("schema") == "slac-authored-exchange-inputs-v1", "input schema mismatch")
    require(tuple(case["case_id"] for case in inputs["cases"]) == CASE_IDS, "fixed six-case order changed")
    entries, exchanges, wires = [], [], []
    for case in inputs["cases"]:
        exchange = project_case(case)
        wire = client.freeze_wire(exchange)
        request = exchange.request
        require(set(wire.expected_ids) == set(DIMENSIONS), "exchange dimensions changed")
        entries.append({"ordinal": len(entries) + 1, "case_id": case["case_id"],
                        "core_cache_key": request.cache_key, "wire_cache_key": wire.cache_key,
                        "core_payload": request.payload(), "core_binding": json.loads(request.binding_bytes),
                        "wire_payload": wire.payload(), "wire_binding": json.loads(wire.binding_bytes),
                        "wire_body_sha256": sha(wire.payload_bytes), "wire_payload_bytes": len(wire.payload_bytes),
                        "expected_ids": list(wire.expected_ids), "reserved_usd": wire.reserved_usd,
                        "input_allowance": wire.input_allowance, "output_allowance": wire.output_allowance})
        exchanges.append(exchange)
        wires.append(wire)
    require(len({wire.cache_key for wire in wires}) == 6, "six unique requests required")
    require(all(len(wire.payload_bytes) <= 24000 for wire in wires), "wire payload too large")
    require(sum((Decimal(w.reserved_usd) for w in wires), Decimal(0)) <= Decimal("0.03"), "reservation cap exceeded")
    return entries, tuple(exchanges), tuple(wires)


def _snapshots() -> dict[str, bytes]:
    return {name: (ROOT / name).read_bytes() for name in SOURCE_NAMES}


def _assemble(sources: dict[str, bytes]) -> tuple[dict, tuple]:
    entries, exchanges, wires = freeze_requests(json.loads(sources[INPUT_NAME]))
    expected = json.loads(sources[EXPECTED_NAME])
    require(expected.get("schema") == "slac-authored-exchange-expectations-v1", "expectation schema mismatch")
    require(expected["label_order"] == list(DIMENSIONS), "expectation dimensions changed")
    require(tuple(case["case_id"] for case in expected["cases"]) == CASE_IDS, "expectation cases changed")
    for case, exchange in zip(expected["cases"], exchanges):
        require(len(case["choices"]) == 3 and all(v in ("yes", "no", "unknown") for v in case["choices"]),
                "invalid frozen expected choices")
        labels = dict(zip(DIMENSIONS, case["choices"]))
        require(_decision(exchange, labels).status == case["status"], "expected decision disagrees with fixed gate")
    for pair in expected["pairs"]:
        require(len(pair["case_ids"]) == len(pair["expected_pattern"]) == 2
                and set(pair["case_ids"]) <= set(CASE_IDS) and pair["expected_dimension"] in DIMENSIONS,
                "invalid expectation pair")
    return {"schema": SCHEMA, "status": "frozen_before_model_observations",
            "source_sha256": {name: sha(data) for name, data in sources.items()}, "limits": LIMITS.copy(),
            "requests": entries, "expectations_readout_only": expected,
            "planned_reserved_usd": str(sum((Decimal(w.reserved_usd) for w in wires), Decimal(0))),
            "counter": {"version": COUNTER_VERSION, "units": "UTF-8 bytes of rendered evidence",
                        "actual_tokenizer": False, "billed_tokens": False},
            "planned_denominators": {"requests": 6, "typed_dimensions": 18, "full_gate_decisions": 6,
                                      "gain_only_decisions": 6, "contrast_pairs": len(expected["pairs"])},
            "ablation_boundary": "Gain-only uses the SAME observed exchange gain; not independent inference, an add-only prompt baseline, or answer quality."}, wires


def prepare_plan(output_dir: Path) -> dict:
    sources = _snapshots()
    plan, _ = _assemble(sources)
    require({k: sha(v) for k, v in _snapshots().items()} == plan["source_sha256"], "sources changed during planning")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "plan.json").open("xb") as stream:
        stream.write(canonical(plan) + b"\n")
    return plan


def verify_plan(path: Path) -> tuple[dict, tuple, str]:
    body = path.read_bytes()
    frozen = json.loads(body)
    sources = _snapshots()
    require(frozen.get("source_sha256") == {k: sha(v) for k, v in sources.items()}, "frozen source commitments changed")
    rebuilt, wires = _assemble(sources)
    require(body == canonical(rebuilt) + b"\n", "frozen plan or request bytes changed")
    return frozen, wires, sha(body)


def _exchange_from_entry(entry):
    state = entry["core_payload"]["state"]
    exchange = _make_exchange(ConditionalState(
        state["query"], tuple(Unit(**unit) for unit in state["current_pack"]),
        Unit(**state["candidate"]), version=INPUT_VERSION), state["removed_id"])
    require(exchange.request.cache_key == entry["core_cache_key"]
            and exchange.request.payload() == entry["core_payload"], "analysis core binding changed")
    return exchange


def _decision(exchange, labels):
    decoded = DecodedResult(exchange.request.cache_key, tuple(
        TypedDecision(name, labels[name], "model_unknown" if labels[name] == "unknown" else None)
        for name in exchange.request.expected_ids), True)
    return decide(exchange, decoded)


def analyze_ledger(plan: dict, ledger: dict) -> dict:
    """Keep failed/unattempted observations distinct from semantic unknown."""
    entries, attempts = plan["requests"], ledger.get("attempts", [])
    require(isinstance(attempts, list) and len(attempts) <= 6, "invalid attempt count")
    index = {}
    for entry, attempt in zip(entries, attempts):
        require(attempt.get("cache_key") == entry["wire_cache_key"]
                and attempt.get("core_cache_key") == entry["core_cache_key"]
                and attempt.get("request_sha256") == entry["wire_body_sha256"]
                and Decimal(attempt["reserved_usd"]) == Decimal(entry["reserved_usd"]), "attempt identity/order/reservation drift")
        require(attempt.get("status") in ("in_flight", "halted", "completed")
                and attempt.get("cost_status") in ("cost_unknown", "provider_reported"), "invalid attempt state")
        index[entry["case_id"]] = attempt
    expectations = {case["case_id"]: case for case in plan["expectations_readout_only"]["cases"]}
    rows, known_cost, reserved, unresolved = [], Decimal(0), Decimal(0), Decimal(0)
    known_count = 0
    for entry in entries:
        attempt = index.get(entry["case_id"])
        status = "unattempted" if attempt is None else attempt["status"]
        labels = attempt.get("labels", {}) if status == "completed" else {}
        require(not labels or (set(labels) == set(DIMENSIONS) and all(v in ("yes", "no", "unknown") for v in labels.values())),
                "invalid completed labels")
        require(status != "completed" or len(labels) == 3, "completed request missing dimensions")
        expected = expectations[entry["case_id"]]
        expected_labels = dict(zip(DIMENSIONS, expected["choices"]))
        full_status, full_reason, gain_status = "unobserved", "no_completed_observation", "unobserved"
        if labels:
            decision = _decision(_exchange_from_entry(entry), labels)
            full_status, full_reason = decision.status, decision.reason
            gain_status = {"yes": "accepted", "no": "rejected", "unknown": "abstained"}[labels[DIMENSIONS[0]]]
        if attempt is not None:
            reservation = Decimal(attempt["reserved_usd"])
            reserved += reservation
            if attempt["cost_status"] == "provider_reported":
                amount = Decimal(attempt["actual_cost_usd"])
                require(amount.is_finite() and amount >= 0, "invalid reported cost")
                known_cost += amount
                known_count += 1
            else:
                unresolved += reservation
        rows.append({"case_id": entry["case_id"], "ordinal": entry["ordinal"], "status": status,
                     "typed_labels": labels, "expected_labels": expected_labels,
                     "dimension_observations": {name: "unobserved" if name not in labels else "unknown" if labels[name] == "unknown"
                                                else "match" if labels[name] == expected_labels[name] else "mismatch" for name in DIMENSIONS},
                     "typed_expectation_matches": sum(labels.get(name) == expected_labels[name] for name in DIMENSIONS),
                     "full_gate_status": full_status, "full_gate_reason": full_reason,
                     "expected_full_gate_status": expected["status"], "full_gate_matches_expected": full_status == expected["status"],
                     "gain_only_status": gain_status, "gain_only_matches_expected_full_gate": gain_status == expected["status"],
                     "cost_status": "unattempted" if attempt is None else attempt["cost_status"],
                     "response_model": None if attempt is None else attempt.get("response_model"),
                     "provider_response_status": "unattempted" if attempt is None else attempt.get("provider_response_status", "not_available"),
                     "elapsed_seconds": None if attempt is None else attempt.get("elapsed_seconds")})
    by_case = {row["case_id"]: row for row in rows}
    pairs = []
    for pair in plan["expectations_readout_only"]["pairs"]:
        observed = [by_case[name]["typed_labels"].get(pair["expected_dimension"]) for name in pair["case_ids"]]
        pairs.append({"pair_id": pair["pair_id"], "case_ids": pair["case_ids"], "dimension": pair["expected_dimension"],
                      "expected_pattern": pair["expected_pattern"], "observed_pattern": observed,
                      "both_observed": all(value is not None for value in observed), "matches_expected_pattern": observed == pair["expected_pattern"]})
    return {"schema": "slac-exchange-probe-summary-v1", "status": ledger.get("status", "unknown"),
            "execution_mode": ledger.get("mode", "unreported"), "halt_reason": ledger.get("halt_reason"),
            "elapsed_seconds": ledger.get("elapsed_seconds"), "planned_requests": 6, "planned_typed_dimensions": 18,
            "attempted_requests": len(attempts), "completed_requests": sum(row["status"] == "completed" for row in rows),
            "unattempted_requests": 6 - len(attempts), "in_flight_attempts": sum(a["status"] == "in_flight" for a in attempts),
            "typed_dimensions_observed": sum(len(row["typed_labels"]) for row in rows),
            "typed_expectation_matches": sum(row["typed_expectation_matches"] for row in rows),
            "full_gate_decisions_planned": 6, "full_gate_status_counts": dict(Counter(row["full_gate_status"] for row in rows)),
            "full_gate_matches_expected": sum(row["full_gate_matches_expected"] for row in rows),
            "gain_only_decisions_planned": 6, "gain_only_status_counts": dict(Counter(row["gain_only_status"] for row in rows)),
            "gain_only_matches_expected_full_gate": sum(row["gain_only_matches_expected_full_gate"] for row in rows),
            "full_gate_vs_gain_only_disagreements": sum(row["full_gate_status"] != row["gain_only_status"] for row in rows),
            "acceptance_by_authored_expectation": {
                "expected_accept_planned": sum(row["expected_full_gate_status"] == "accepted" for row in rows),
                "expected_reject_planned": sum(row["expected_full_gate_status"] == "rejected" for row in rows),
                **{name: {"accepted_expected_accept": sum(row[field] == "accepted" and row["expected_full_gate_status"] == "accepted" for row in rows),
                           "accepted_expected_reject": sum(row[field] == "accepted" and row["expected_full_gate_status"] == "rejected" for row in rows)}
                   for name, field in (("full_gate", "full_gate_status"), ("gain_only", "gain_only_status"))}},
            "contrast_pairs_planned": len(pairs), "contrast_patterns_matched": sum(pair["matches_expected_pattern"] for pair in pairs),
            "provider_reported_cost_usd": str(known_cost), "cost_known_attempts": known_count,
            "cost_unknown_attempts": len(attempts) - known_count, "total_actual_cost_known": known_count == len(attempts),
            "planned_reserved_usd": plan["planned_reserved_usd"], "attempted_reserved_usd": str(reserved),
            "unresolved_reserved_usd": str(unresolved),
            "returned_model_identities": sorted({row["response_model"] for row in rows if row["response_model"] is not None}),
            "returned_model_identity_unavailable_attempts": sum(a.get("response_model") is None for a in attempts),
            "requests": rows, "pairs": pairs,
            "interpretation": "Six exposed authored cases only. Gain-only reuses the SAME exchange gain label, not independent or add-only prompt inference. No answer quality, natural-data retrieval benefit or population accuracy is measured."}


def write_analysis(plan_path: Path, output_dir: Path) -> dict:
    body = plan_path.read_bytes()
    plan = json.loads(body)
    receipt = json.loads((output_dir / "plan_receipt.json").read_bytes())
    require(receipt["plan_sha256"] == sha(body) and receipt["source_sha256"] == plan["source_sha256"], "frozen run receipt mismatch")
    summary = analyze_ledger(plan, json.loads((output_dir / "ledger.json").read_bytes()))
    summary.update(plan_sha256=sha(body), source_sha256=plan["source_sha256"])
    result = canonical(summary) + b"\n"
    target = output_dir / "summary.json"
    if target.exists():
        require(target.read_bytes() == result, "existing summary changed; overwriting is forbidden")
    else:
        with target.open("xb") as stream:
            stream.write(result)
    return summary


def execute_plan(plan_path: Path, output_dir: Path, *, key_file=None, proxy=None, transport=None, live=False):
    plan, wires, digest = verify_plan(plan_path)
    probe = client.ExchangeProbeClient(output_dir, wires, key_file=key_file, proxy=proxy,
                                       transport=transport, deadline_seconds=180)
    with (output_dir / "plan_receipt.json").open("xb") as stream:
        stream.write(canonical({"plan_sha256": digest, "source_sha256": plan["source_sha256"]}) + b"\n")
    ledger = probe.run(live=live)
    write_analysis(plan_path, output_dir)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("plan")
    prepare.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--key-file", type=Path, required=True)
    run.add_argument("--proxy")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--plan", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = prepare_plan(args.output_dir)
    elif args.command == "run":
        result = execute_plan(args.plan, args.output_dir, key_file=args.key_file, proxy=args.proxy, live=True)
    else:
        result = write_analysis(args.plan, args.output_dir)
    print(json.dumps({"command": args.command, "status": result.get("status")}, sort_keys=True))
    return 1 if args.command == "run" and result.get("status") != "completed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
