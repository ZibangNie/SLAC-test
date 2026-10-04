"""Four fixed authored query contrasts over two unchanged natural evidence pairs.

Planning is offline. The explicit bounded command runs a single-consumption
worker through the existing supervisor. Expectations remain readout-only.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.research import exchange_probe_client as client
from docs.research import run_natural_exchange_probe as common
from docs.research.run_conditional_probe_bounded import _contained, supervise
from SLAC.retrieval.decision.conditional import ConditionalState, Unit, render_evidence
from SLAC.retrieval.decision.exchange import build_exchange

canonical, sha, require = common.canonical, common.sha, common.require
PACKET, SAMPLE_PLAN = common.PACKET, common.SAMPLE_PLAN
REVIEW = "artifacts/research-foundation/offline-20261004/natural-loss-witness-01/review_a.json"
INDEPENDENT_REVIEW = "artifacts/research-foundation/offline-20261004/natural-loss-witness-01/independent_contrast_review.json"
SEMANTIC_PREFLIGHT = "artifacts/research-foundation/offline-20261004/loss-scope-probe-admission-01/semantic_preflight.json"
PINNED = {
    REVIEW: "f5e9c4abb01f5948fa611ee14b5015bd9d1b050204685230a3b4dd210404d7ec",
    INDEPENDENT_REVIEW: "4a86c535cc268cddc5f1a0c7e1c23e63ee55c954d02798a7b62981f850a3101e",
    PACKET: "bd4e903a2de589d0f823288a9274428e434b5f57dab86840db00f72bec7984f1",
    SAMPLE_PLAN: "ca453244fb94d6dd91f119a92731425adb4684f983b6e11f57b7e845dfd16bba",
    SEMANTIC_PREFLIGHT: "2af2cdde4da9cdf1f19d521ba42d7ae019af0b0e28365dcdf8a588ad6cdcb601",
}
SOURCE_NAMES = (
    "docs/research/run_loss_scope_probe.py", "tests/research/test_loss_scope_probe.py",
    "docs/research/LOSS_SCOPE_JEV_PROTOCOL_20261004.md", "docs/research/run_natural_exchange_probe.py",
    "docs/research/exchange_probe_client.py", "docs/research/conditional_probe_client.py",
    "docs/research/openrouter_decision_client.py", "docs/research/run_conditional_probe_bounded.py",
    "SLAC/retrieval/decision/conditional.py", "SLAC/retrieval/decision/exchange.py",
    "tests/research/test_exchange_probe_client.py", "tests/research/test_conditional_probe_watchdog.py",
    "artifacts/research-foundation/offline-20261004/read_loss_scope_probe.py",
    *PINNED,
)
CONTRAST_IDS = ("Q1", "Q2", "Q3", "Q4")
LIMITS = {"max_requests": 4, "typed_questions": 12, "max_reserved_usd": "0.020",
          "deadline_seconds": 120, "generation_max_tokens": 1024, "generation_max_units": 3,
          "judge_max_tokens": 2048, "max_wire_payload_bytes": 24000,
          "max_response_bytes": 65536, "max_request_timeout_seconds": 30}


def _admit(wires):
    require(len(wires) == 4, "exactly four scope requests required")
    require(all(type(w) is client.ExchangeWireRequest and client.freeze_wire(w.plan) == w for w in wires),
            "scope wire binding changed")
    require(len({w.cache_key for w in wires}) == len({w.payload_bytes for w in wires}) == 4,
            "four unique physical requests required")
    require(sum(len(w.expected_ids) for w in wires) == 12
            and all(w.expected_ids == client.EXPECTED_IDS for w in wires), "twelve fixed dimensions required")
    require(all(Decimal(w.reserved_usd) == Decimal("0.005") for w in wires)
            and sum((Decimal(w.reserved_usd) for w in wires), Decimal(0)) == Decimal("0.020"),
            "fixed scope reservation changed")


class ScopeProbeClient(client.ExchangeProbeClient):
    """Only four-request admission/ledger differ; sealed transport is inherited.

    Four reservations of .005 give the .020 total. Inherited accounting saves
    known fees before halting any per-request overrun; unknown fees halt too.
    The six-request constructor is not called and no old global is modified.
    """
    def __init__(self, output, wires, *, key_file=None, proxy=None, transport=None, clock=time.monotonic):
        self.requests = tuple(wires)
        _admit(self.requests)
        require(callable(clock) and (transport is None or callable(transport)), "invalid execution helpers")
        self._proxy = client.base._proxy(proxy)
        self.output = Path(output)
        self._key_file, self._key = key_file, ""
        self._transport, self._clock, self._deadline_seconds, self._ran = transport, clock, 120.0, False
        self.output.mkdir(parents=True, exist_ok=False)
        self.ledger = {"schema": "slac-loss-scope-probe-ledger-v1", "wire_version": client.WIRE_VERSION,
                       "status": "prepared", "halt_reason": None, "endpoint": client.ENDPOINT,
                       "requested_model": client.MODEL_ID, "expected_response_model": client.RESPONSE_MODEL,
                       "provider_policy": client.base._provider(), "proxy": self._proxy,
                       "deadline_seconds": 120, "hard_watchdog_required": True,
                       "request_cap": 4, "question_cap": 12, "budget_usd": "0.020",
                       "planned_request_count": 4, "planned_question_count": 12,
                       "planned_reservation_usd": "0.020", "reservation_total_usd": "0",
                       "actual_reported_cost_usd": "0", "automatic_retries": 0,
                       "wire_keys": [w.cache_key for w in self.requests], "attempts": [],
                       "dependency_sha256": client.DEPENDENCY_SHA256.copy(),
                       "inherited_transport": "conditional_probe_client.ProbeClient.run"}
        self._save()

    def run(self, *, live=False):
        _admit(self.requests)
        return super().run(live=live)


def freeze_requests(review, packet, counter):
    """Read only the fixed Q1-Q4 prefix; held Q5/Q6 never produce requests."""
    selected = review["authored_contrasts"][:4]
    require(tuple(c["contrast_id"] for c in selected) == CONTRAST_IDS, "fixed Q1-Q4 order changed")
    bases = {c["ordinal"]: c for c in packet["cases"][:2]}
    require(set(bases) == {1, 2}, "fixed source cases changed")
    entries, wires = [], []
    for ordinal, contrast in enumerate(selected, 1):
        base = bases[(ordinal + 1) // 2]
        inventory = {k: base[k] for k in ("doc_id", "original_pack", "proposed_pack", "candidate", "removed")}
        require(contrast["base_ordinal"] == base["ordinal"] and contrast["evidence_inventory"] == inventory
                and contrast["evidence_inventory_sha256"] == sha(canonical(inventory)), "same-evidence source binding changed")
        def unit(raw):
            digest = sha(raw["text"].encode("utf-8"))
            require(digest == raw["retrieval_text_sha256"] == raw["native_text_sha256"], "source unit hash changed")
            return Unit(raw["unit_id"], raw["text"], raw["source_order"], inventory["doc_id"])
        original, candidate = tuple(unit(u) for u in inventory["original_pack"]), unit(inventory["candidate"])
        removed = unit(inventory["removed"])
        require(removed in original, "removed unit identity changed")
        plan = build_exchange(ConditionalState(contrast["query"], original, candidate,
                              version="slac-natural-exchange-input-state-v1"), removed_id=removed.id,
                              endpoint_id=client.ENDPOINT, model_id=client.MODEL_ID,
                              expected_response_model=client.RESPONSE_MODEL, token_counter=counter,
                              max_tokens=1024, max_units=3, max_judge_tokens=2048,
                              max_payload_bytes=65536, counter_version=common.COUNTER_VERSION)
        require(plan.proposed_pack == tuple(unit(u) for u in inventory["proposed_pack"]), "proposed pack identity changed")
        wire = client.freeze_wire(plan)
        legacy = lambda units: "\n\n".join(f"[{u.id}]\n{u.text}" for u in units)
        surfaces = {"original": render_evidence(plan.original_pack), "proposed": render_evidence(plan.proposed_pack),
                    "judge_union": plan.request.rendered_evidence, "legacy_original": legacy(plan.original_pack),
                    "legacy_proposed": legacy(plan.proposed_pack)}
        require(max(counter(surfaces[k]) for k in ("legacy_original", "legacy_proposed")) <= 1024,
                "legacy pack exceeds frozen budget")
        entries.append({"ordinal": ordinal, "contrast_id": contrast["contrast_id"], "base_ordinal": base["ordinal"],
                        "core_cache_key": plan.request.cache_key, "wire_cache_key": wire.cache_key,
                        "core_payload": plan.request.payload(), "core_binding": json.loads(plan.request.binding_bytes),
                        "wire_payload": wire.payload(), "wire_binding": json.loads(wire.binding_bytes),
                        "wire_body_sha256": sha(wire.payload_bytes), "wire_payload_bytes": len(wire.payload_bytes),
                        "expected_ids": list(wire.expected_ids), "reserved_usd": wire.reserved_usd,
                        "input_allowance": wire.input_allowance, "output_allowance": wire.output_allowance,
                        "surfaces": {k: {"tokens": counter(v), "render_sha256": sha(v.encode("utf-8"))}
                                     for k, v in surfaces.items()}})
        wires.append(wire)
    _admit(wires)
    return entries, tuple(wires)


def _snapshots():
    return {name: (ROOT / name).read_bytes() for name in SOURCE_NAMES}


def _assemble(sources):
    require(all(sha(sources[p]) == h for p, h in PINNED.items()), "pinned input changed")
    review, packet = json.loads(sources[REVIEW]), json.loads(sources[PACKET])
    require(review["packet_sha256"] == sha(sources[PACKET]), "review source packet changed")
    independent = json.loads(sources[INDEPENDENT_REVIEW])
    require(independent["input_sha256"] == sha(sources[REVIEW]), "independent review source changed")
    counter, metadata = common._load_counter(json.loads(sources[SAMPLE_PLAN]))
    entries, wires = freeze_requests(review, packet, counter)
    readout = [{k: c[k] for k in ("contrast_id", "base_ordinal", "target_claim_id", "authored_expected_loss")}
               for c in review["authored_contrasts"][:4]]
    require([r["authored_expected_loss"] for r in readout] == ["yes", "no", "yes", "no"]
            and [r["target_claim_id"] for r in readout] == ["C1-L", "C1-R", "C2-L", "C2-R"],
            "fixed authored readout changed")
    return {"schema": "slac-loss-scope-probe-plan-v1", "status": "frozen_before_model_observations",
            "source_sha256": {p: sha(v) for p, v in sources.items()}, "limits": LIMITS.copy(),
            "counter": metadata, "requests": entries, "planned_reserved_usd": "0.020",
            "readout_only": {"contrasts": readout, "loss_patterns": [["yes", "no"], ["yes", "no"]],
                             "boundary": "Authored posthoc scope contrasts, not natural gold, independent accuracy, or RAG effectiveness."},
            "projection_boundary": "Unchanged natural input/state/counter contracts; only query changes within each pair. No held contrast is submitted."}, wires


def prepare_plan(output_dir):
    sources = _snapshots()
    plan, _ = _assemble(sources)
    require({p: sha(v) for p, v in _snapshots().items()} == plan["source_sha256"], "sources changed during planning")
    Path(output_dir).mkdir(parents=True, exist_ok=False)
    (Path(output_dir) / "plan.json").write_bytes(canonical(plan) + b"\n")
    return plan


def verify_plan(path):
    body, sources = Path(path).read_bytes(), _snapshots()
    frozen = json.loads(body)
    require(frozen.get("source_sha256") == {p: sha(v) for p, v in sources.items()}, "frozen source commitments changed")
    rebuilt, wires = _assemble(sources)
    require(body == canonical(rebuilt) + b"\n", "frozen plan or wire changed")
    return frozen, wires, sha(body)


def execute_plan(path, output, *, key_file=None, proxy=None, transport=None, live=False):
    path, output = Path(path), Path(output)
    plan, wires, digest = verify_plan(path)
    require(not output.exists(), "run output already exists")
    receipt = {"plan_sha256": digest, "source_sha256": plan["source_sha256"],
               "output_dir": str(output.resolve()), "mode": "live" if live else "fake_or_unspecified"}
    with (path.parent / "execution_claim.json").open("xb") as stream:
        stream.write(canonical(receipt) + b"\n")
    probe = ScopeProbeClient(output, wires, key_file=key_file, proxy=proxy, transport=transport)
    (output / "plan_receipt.json").write_bytes(canonical(receipt) + b"\n")
    return probe.run(live=live)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("plan")
    prepare.add_argument("--output-dir", type=Path, required=True)
    for name in ("run", "bounded"):
        command = sub.add_parser(name)
        for flag in ("plan", "output-dir", "key-file"):
            command.add_argument("--" + flag, type=Path, required=True)
        if name == "bounded":
            command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = prepare_plan(args.output_dir)
    elif args.command == "run":
        result = execute_plan(args.plan, args.output_dir, key_file=args.key_file,
                              proxy="http://127.0.0.1:7897", live=True)
    else:
        path, output, receipt = _contained(args.plan), _contained(args.output_dir), _contained(args.receipt)
        command = [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(path),
                   "--output-dir", str(output), "--key-file", str(args.key_file.resolve())]
        result = supervise(command, output, receipt, timeout_seconds=120, plan_sha256=sha(path.read_bytes()))
    print(json.dumps({"command": args.command, "status": result.get("status")}, sort_keys=True))
    if args.command == "bounded":
        return 0 if result.get("status") == "worker_exited" and result.get("returncode") == 0 else 1
    return 1 if args.command == "run" and result.get("status") != "completed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
