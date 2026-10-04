"""Freeze and read out one six-request synthetic semantic probe.

The plan command is offline. The run command is an explicit live entry point;
only the separately bounded client can read credentials or dispatch requests.
All planned observations remain in the readout after any partial failure.
"""

from __future__ import annotations

import argparse
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

import conditional_probe_client as client
import run_conditional_contract as projector
from SLAC.retrieval.decision.conditional import build_request


SOURCE_NAMES = (
    "docs/research/run_conditional_probe.py",
    "docs/research/conditional_probe_client.py",
    "docs/research/openrouter_decision_client.py",
    "SLAC/retrieval/decision/conditional.py",
    "docs/research/fixtures/conditional_jev_contract_v1.json",
    "docs/research/run_conditional_contract.py",
    "docs/research/CONDITIONAL_JEV_CONTRACT_PROTOCOL_20261004.md",
    "docs/research/CONDITIONAL_JEV_PROBE_PROTOCOL_20261004.md",
    "docs/research/run_conditional_probe_bounded.py",
    "tests/research/test_conditional_probe_runner.py",
    "tests/research/test_conditional_probe_client.py",
    "tests/research/test_conditional_probe_watchdog.py",
    "tests/research/test_conditional_decision.py",
    "tests/research/test_conditional_contract_runner.py",
)
LIMITS = {"max_requests": 6, "typed_questions": 10, "max_reserved_usd": "0.03",
          "deadline_seconds": 180, "max_wire_payload_bytes": 24000,
          "max_response_bytes": 65536, "max_request_timeout_seconds": 30}
FIXTURE_NAME = "docs/research/fixtures/conditional_jev_contract_v1.json"
PLAN_SCHEMA = "slac-six-request-conditional-probe-plan-v1"


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _snapshots() -> dict[str, bytes]:
    return {name: (ROOT / name).read_bytes() for name in SOURCE_NAMES}


def freeze_requests(fixture: dict) -> tuple[list[dict], tuple]:
    """Use exactly the first published two pairs without outcome-dependent choice."""
    pairs = fixture["pairs"][:2]
    _require([pair["pair_id"] for pair in pairs] == ["P01", "P02"], "fixed pair ordering changed")
    _require([[case["case_id"] for case in pair["states"]] for pair in pairs]
             == [["C01", "C02"], ["C03", "C04"]], "fixed case ordering changed")
    entries, wires = [], []
    for pair in pairs:
        first, second = pair["states"]
        _require(pair["supervision"]["expected_changed_dimension"] == "added", "supervised dimension changed")
        for arm, case in (("standalone", first), ("plain_conditional", first), ("plain_conditional", second)):
            _require(not case["relations"] and not case["control_relations"], "first two pairs must have no relations")
            state = projector.project_state(pair, case)
            core = build_request(state, arm=arm, endpoint_id="https://openrouter.ai/api/alpha/decisions",
                                 model_id=projector.MODEL, expected_response_model=projector.RESPONSE_MODEL,
                                 token_counter=projector.synthetic_counter, max_tokens=1024,
                                 max_payload_bytes=65536, counter_version=projector.COUNTER_VERSION)
            wire = client.freeze_wire(core)
            expected = None if arm == "standalone" else pair["supervision"]["expected_labels_by_case"][case["case_id"]]
            if expected is not None:
                _require(expected in ("yes", "no"), "fixed added supervision changed")
            entries.append({"ordinal": len(entries) + 1, "pair_id": pair["pair_id"],
                            "case_id": case["case_id"], "arm": arm,
                            "expected_added_information": expected,
                            "core_cache_key": core.cache_key, "wire_cache_key": wire.cache_key,
                            "core_payload": core.payload(), "core_binding": json.loads(core.binding_bytes),
                            "wire_payload": wire.payload(), "wire_binding": json.loads(wire.binding_bytes),
                            "wire_body_sha256": sha(wire.payload_bytes), "wire_payload_bytes": len(wire.payload_bytes),
                            "expected_ids": list(wire.expected_ids), "reserved_usd": wire.reserved_usd,
                            "input_allowance": wire.input_allowance, "output_allowance": wire.output_allowance})
            wires.append(wire)
    _require(len(wires) == 6 and len({wire.cache_key for wire in wires}) == 6, "exact six unique requests required")
    _require(sum(len(wire.expected_ids) for wire in wires) == 10, "ten typed questions required")
    _require(all(len(wire.payload_bytes) <= 24000 for wire in wires), "wire byte limit exceeded")
    _require(sum((Decimal(wire.reserved_usd) for wire in wires), Decimal(0)) <= Decimal("0.03"),
             "total reservation exceeds probe cap")
    _require([entry["expected_added_information"] for entry in entries] == [None, "yes", "no", None, "yes", "no"],
             "predeclared added expectations changed")
    return entries, tuple(wires)


def _plan_from_snapshots(sources: dict[str, bytes]) -> tuple[dict, tuple]:
    fixture = json.loads(sources[FIXTURE_NAME])
    entries, wires = freeze_requests(fixture)
    return {"schema": PLAN_SCHEMA, "status": "frozen_before_model_observations",
            "source_sha256": {name: sha(data) for name, data in sources.items()},
            "limits": LIMITS.copy(), "requests": entries,
            "planned_reserved_usd": str(sum((Decimal(wire.reserved_usd) for wire in wires), Decimal(0))),
            "supervision_boundary": "Expected added labels are readout-only, outside all model payloads.",
            "planned_denominators": {"requests": 6, "typed_questions": 10, "added_observations": 4, "paired_reversals": 2},
            "interpretation": "Artificial sanity probe only; no relation treatment, retrieval benefit or novelty claim."}, wires


def prepare_plan(output_dir: Path) -> dict:
    sources = _snapshots()
    plan, _ = _plan_from_snapshots(sources)
    _require({name: sha(data) for name, data in _snapshots().items()} == plan["source_sha256"],
             "source changed while preparing plan")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "plan.json").open("xb") as stream:
        stream.write(canonical(plan) + b"\n")
    return plan


def verify_plan(plan_path: Path) -> tuple[dict, tuple, str]:
    frozen_bytes = plan_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    sources = _snapshots()
    _require(frozen.get("source_sha256") == {name: sha(data) for name, data in sources.items()},
             "frozen source commitments changed")
    rebuilt, wires = _plan_from_snapshots(sources)
    _require(frozen_bytes == canonical(rebuilt) + b"\n", "frozen request plan or wire bytes changed")
    return frozen, wires, sha(frozen_bytes)


def execute_plan(plan_path: Path, output_dir: Path, *, key_file: Path | None = None,
                 proxy: str | None = None, transport=None, live: bool = False) -> dict:
    plan, wires, plan_hash = verify_plan(plan_path)
    # ProbeClient alone creates the new output directory and loads a credential
    # only for an explicit live run. No resume or existing-ledger reuse exists.
    probe = client.ProbeClient(output_dir, wires, key_file=key_file, proxy=proxy,
                               transport=transport, deadline_seconds=180)
    receipt = {"schema": "slac-conditional-probe-plan-receipt-v1", "plan_sha256": plan_hash,
               "source_sha256": plan["source_sha256"]}
    with (output_dir / "plan_receipt.json").open("xb") as stream:
        stream.write(canonical(receipt) + b"\n")
    ledger = probe.run(live=live)
    write_analysis(plan_path, output_dir)
    return ledger


def analyze_ledger(plan: dict, ledger: dict) -> dict:
    """Retain every frozen denominator, including unattempted and failed calls."""
    planned = plan["requests"]
    by_key = {entry["wire_cache_key"]: entry for entry in planned}
    attempts = ledger.get("attempts", [])
    _require(isinstance(attempts, list) and len(attempts) <= 6, "invalid attempt ledger")
    index = {}
    for position, attempt in enumerate(attempts):
        key = attempt["cache_key"]
        _require(key in by_key and key not in index, "attempt is unknown or duplicated")
        _require(key == planned[position]["wire_cache_key"], "attempt order differs from frozen plan")
        expected = by_key[key]
        _require(attempt.get("status") in ("in_flight", "completed", "halted")
                 and attempt.get("cost_status") in ("cost_unknown", "provider_reported"),
                 "invalid attempt status or cost evidence state")
        _require(attempt.get("core_cache_key") == expected["core_cache_key"]
                 and attempt.get("request_sha256") == expected["wire_body_sha256"]
                 and Decimal(attempt["reserved_usd"]) == Decimal(expected["reserved_usd"]),
                 "attempt identity or reservation changed")
        index[key] = attempt
    rows, added = [], []
    known_cost = Decimal(0)
    known_count = completed = 0
    reserved = Decimal(0)
    unresolved_reserved = Decimal(0)
    for entry in planned:
        attempt = index.get(entry["wire_cache_key"])
        status = "unattempted" if attempt is None else attempt["status"]
        labels = {} if attempt is None or status != "completed" else attempt.get("labels", {})
        _require(set(labels) <= set(entry["expected_ids"]), "unexpected typed label")
        _require(all(value in ("yes", "no", "unknown") for value in labels.values()), "invalid typed label")
        if status == "completed":
            _require(set(labels) == set(entry["expected_ids"]), "completed request missing typed labels")
            completed += 1
        if attempt is not None:
            reserved += Decimal(attempt["reserved_usd"])
            if attempt.get("cost_status") == "provider_reported":
                amount = Decimal(attempt["actual_cost_usd"])
                _require(amount.is_finite() and amount >= 0, "invalid reported cost")
                known_cost += amount
                known_count += 1
            else:
                unresolved_reserved += Decimal(attempt["reserved_usd"])
        row = {"ordinal": entry["ordinal"], "pair_id": entry["pair_id"], "case_id": entry["case_id"],
               "arm": entry["arm"], "status": status, "typed_labels": labels,
               "cost_status": "unattempted" if attempt is None else attempt.get("cost_status", "cost_unknown"),
               "response_model": None if attempt is None else attempt.get("response_model"),
               "provider_response_status": "unattempted" if attempt is None else attempt.get("provider_response_status", "not_available"),
               "elapsed_seconds": None if attempt is None else attempt.get("elapsed_seconds")}
        if entry["expected_added_information"] is not None:
            row["expected_added_information"] = entry["expected_added_information"]
            observed = labels.get("conditional_added_information")
            row["added_matches_expected"] = observed == entry["expected_added_information"]
            row["added_observation_status"] = ("unobserved" if observed is None else "unknown" if observed == "unknown"
                                                else "match" if row["added_matches_expected"] else "mismatch")
            added.append(row)
        rows.append(row)
    pairs = []
    for pair_id in ("P01", "P02"):
        pair_rows = [row for row in added if row["pair_id"] == pair_id]
        observations = [row["typed_labels"].get("conditional_added_information") for row in pair_rows]
        pairs.append({"pair_id": pair_id, "added_labels": observations,
                      "both_observed": all(value is not None for value in observations),
                      "expected_yes_to_no_observed": observations == ["yes", "no"]})
    return {"schema": "slac-conditional-probe-summary-v1", "status": ledger.get("status", "unknown"),
            "execution_mode": ledger.get("mode", "unreported"),
            "elapsed_seconds": ledger.get("elapsed_seconds"),
            "returned_model_identities": sorted({row["response_model"] for row in rows if row["response_model"] is not None}),
            "returned_model_identity_unavailable_attempts": sum(attempt.get("response_model") is None for attempt in attempts),
            "halt_reason": ledger.get("halt_reason"), "planned_requests": 6, "planned_typed_questions": 10,
            "attempted_requests": len(attempts), "completed_requests": completed,
            "unattempted_requests": 6 - len(attempts),
            "typed_labels_observed": sum(len(row["typed_labels"]) for row in rows),
            "added_observations_planned": 4,
            "added_observations_completed": sum("conditional_added_information" in row["typed_labels"] for row in added),
            "added_matches_planned_expectation": sum(row["added_matches_expected"] for row in added),
            "paired_reversals_planned": 2,
            "paired_reversals_observed": sum(pair["expected_yes_to_no_observed"] for pair in pairs),
            "provider_reported_cost_usd": str(known_cost), "cost_known_attempts": known_count,
            "cost_unknown_attempts": len(attempts) - known_count,
            "in_flight_attempts": sum(attempt["status"] == "in_flight" for attempt in attempts),
            "unresolved_reserved_usd": str(unresolved_reserved),
            "total_actual_cost_known": len(attempts) == known_count,
            "planned_reserved_usd": plan["planned_reserved_usd"], "attempted_reserved_usd": str(reserved),
            "requests": rows, "pairs": pairs,
            "interpretation": "Fixed artificial sanity probe. Standalone and conflict are descriptive, not scored. No relation or retrieval benefit is measured."}


def write_analysis(plan_path: Path, output_dir: Path) -> dict:
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    receipt = json.loads((output_dir / "plan_receipt.json").read_bytes())
    _require(receipt["plan_sha256"] == sha(plan_bytes), "run receipt differs from frozen plan")
    _require(receipt["source_sha256"] == plan["source_sha256"], "run source receipt differs from frozen plan")
    ledger = json.loads((output_dir / "ledger.json").read_bytes())
    summary = analyze_ledger(plan, ledger)
    summary["plan_sha256"] = sha(plan_bytes)
    summary["source_sha256"] = plan["source_sha256"]
    output = output_dir / "summary.json"
    body = canonical(summary) + b"\n"
    if output.exists():
        _require(output.read_bytes() == body, "existing summary differs; immutable output cannot be overwritten")
    else:
        with output.open("xb") as stream:
            stream.write(body)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("plan")
    prepare.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--key-file", type=Path, required=True)
    run.add_argument("--proxy")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--plan", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = prepare_plan(args.output_dir)
    elif args.command == "run":
        result = execute_plan(args.plan, args.output_dir, key_file=args.key_file, proxy=args.proxy, live=True)
    else:
        result = write_analysis(args.plan, args.output_dir)
    print(json.dumps({"status": result.get("status"), "command": args.command}, sort_keys=True))


if __name__ == "__main__":
    main()
