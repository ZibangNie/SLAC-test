"""One fixed natural six-case exchange probe; plan is offline, run is explicit.

Reviewer annotations are readout-only and never projected into a request. The
existing sealed client supplies transport/accounting; its existing supervisor
supplies the process deadline. No resume, substitution or automatic retry exists.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.research import exchange_probe_client as client
from docs.research.run_conditional_probe_bounded import _contained, supervise
from SLAC.retrieval.decision.conditional import ConditionalState, Unit, render_evidence
from SLAC.retrieval.decision.exchange import build_exchange

BASE = "artifacts/research-foundation/offline-20261004/natural-exchange-sample-01/"
PACKET, REVIEW_A, REVIEW_B, SAMPLE_PLAN = (BASE + n for n in
    ("review_packet.json", "semantic_review_a.json", "semantic_review_b.json", "plan.json"))
PINNED_INPUT_SHA256 = {
    PACKET: "bd4e903a2de589d0f823288a9274428e434b5f57dab86840db00f72bec7984f1",
    REVIEW_A: "f6ad210b3f767d6912a09f6b591b4058e878e3cdd255c736dac61ff2da4b430f",
    REVIEW_B: "a414d6d9e10aafa14304926ecd2e1588b53a87e5a4942b48cee92c3d96a94c9b",
    SAMPLE_PLAN: "ca453244fb94d6dd91f119a92731425adb4684f983b6e11f57b7e845dfd16bba",
}
SOURCE_NAMES = (
    "docs/research/run_natural_exchange_probe.py", "tests/research/test_natural_exchange_probe.py",
    "docs/research/NATURAL_EXCHANGE_JEV_PROTOCOL_20261004.md",
    "docs/research/exchange_probe_client.py", "docs/research/conditional_probe_client.py",
    "docs/research/openrouter_decision_client.py", "SLAC/retrieval/decision/conditional.py",
    "SLAC/retrieval/decision/exchange.py", "docs/research/run_conditional_probe_bounded.py",
    "tests/research/test_exchange_probe_client.py", "tests/research/test_conditional_probe_watchdog.py",
    *PINNED_INPUT_SHA256,
)
DIMENSIONS = ("proposed_adds_information", "original_information_lost", "proposed_conflict")
COUNTER_VERSION = "slac-natural-exchange-bge-m3-whole-render-v1"
SCHEMA = "slac-natural-exchange-probe-plan-v1"
LIMITS = {"max_requests": 6, "typed_questions": 18, "max_reserved_usd": "0.03",
          "deadline_seconds": 180, "generation_max_tokens": 1024, "generation_max_units": 3,
          "judge_max_tokens": 2048, "max_wire_payload_bytes": 24000,
          "max_response_bytes": 65536, "max_request_timeout_seconds": 30}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _snapshots():
    return {name: (ROOT / name).read_bytes() for name in SOURCE_NAMES}


def _load_counter(sample_plan):
    """Load the same local fast tokenizer, without weights or remote access."""
    pins = sample_plan["tokenizer_sha256"]
    required = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"}
    require(len(pins) == 4 and {Path(p).name for p in pins} == required, "tokenizer pin schema changed")
    parents = {Path(p).resolve().parent for p in pins}
    require(len(parents) == 1, "tokenizer files must share one local directory")
    for name, digest in pins.items():
        require(sha(Path(name).read_bytes()) == digest, "pinned tokenizer bytes changed")
    directory = parents.pop()
    # AutoTokenizer may inspect model config; freeze it too. No weight is loaded.
    extra = {str(directory / "config.json"): sha((directory / "config.json").read_bytes())}
    require(not (directory / "added_tokens.json").exists(), "unbound additional tokenizer file")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    import tokenizers
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False, use_fast=True)
    require(tokenizer.is_fast, "fast tokenizer required")
    def count(text):
        return 0 if not text else len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    metadata = {"version": COUNTER_VERSION, "tokenizer_sha256": pins,
                "additional_config_sha256": extra, "transformers_version": transformers.__version__,
                "tokenizers_version": tokenizers.__version__, "add_special_tokens": True,
                "truncation": False, "billed_tokens": False,
                "core_render": "[doc_id/unit_id] newline text, source order, blank-line separator",
                "legacy_render_comparison": "[unit_id] newline text; a separate measurement"}
    require(all(sha(Path(p).read_bytes()) == v for p, v in (pins | extra).items()), "tokenizer changed while loading")
    return count, metadata


def freeze_requests(packet, counter):
    """Project input fields only; no reviewer data or ranking enters this path."""
    require(packet.get("schema") == "slac-natural-exchange-review-inputs-v1", "packet schema changed")
    require([c["ordinal"] for c in packet["cases"]] == list(range(1, 7)), "fixed six-case order changed")
    entries, wires = [], []
    for case in packet["cases"]:
        def unit(raw):
            require(sha(raw["text"].encode("utf-8")) == raw["retrieval_text_sha256"], "retrieval text identity changed")
            require(raw["native_text_sha256"] == raw["retrieval_text_sha256"], "frozen native/retrieval identity changed")
            return Unit(raw["unit_id"], raw["text"], raw["source_order"], case["doc_id"])
        original = tuple(unit(u) for u in case["original_pack"])
        candidate, removed = unit(case["candidate"]), unit(case["removed"])
        require(removed in original, "removed source unit changed")
        plan = build_exchange(ConditionalState(case["query"], original, candidate,
                              version="slac-natural-exchange-input-state-v1"), removed_id=removed.id,
                              endpoint_id=client.ENDPOINT, model_id=client.MODEL_ID,
                              expected_response_model=client.RESPONSE_MODEL, token_counter=counter,
                              max_tokens=1024, max_units=3, max_judge_tokens=2048,
                              max_payload_bytes=65536, counter_version=COUNTER_VERSION)
        require(plan.proposed_pack == tuple(unit(u) for u in case["proposed_pack"]), "proposed source pack changed")
        wire = client.freeze_wire(plan)
        legacy = lambda units: "\n\n".join(f"[{u.id}]\n{u.text}" for u in units)
        surfaces = {"original": render_evidence(plan.original_pack),
                    "proposed": render_evidence(plan.proposed_pack),
                    "judge_union": plan.request.rendered_evidence,
                    "legacy_original": legacy(plan.original_pack), "legacy_proposed": legacy(plan.proposed_pack)}
        require(max(counter(surfaces[k]) for k in ("legacy_original", "legacy_proposed")) <= 1024,
                "legacy generation render exceeds frozen budget")
        entries.append({"ordinal": case["ordinal"], "doc_id": case["doc_id"], "question_id": case["question_id"],
                        "core_cache_key": plan.request.cache_key, "wire_cache_key": wire.cache_key,
                        "core_payload": plan.request.payload(), "core_binding": json.loads(plan.request.binding_bytes),
                        "wire_payload": wire.payload(), "wire_binding": json.loads(wire.binding_bytes),
                        "wire_body_sha256": sha(wire.payload_bytes), "wire_payload_bytes": len(wire.payload_bytes),
                        "expected_ids": list(wire.expected_ids), "reserved_usd": wire.reserved_usd,
                        "input_allowance": wire.input_allowance, "output_allowance": wire.output_allowance,
                        "surfaces": {k: {"tokens": counter(v), "render_sha256": sha(v.encode("utf-8"))}
                                     for k, v in surfaces.items()}})
        wires.append(wire)
    require(len({w.cache_key for w in wires}) == 6 and sum(len(w.expected_ids) for w in wires) == 18,
            "six unique requests and eighteen dimensions required")
    require(sum((Decimal(w.reserved_usd) for w in wires), Decimal(0)) == Decimal("0.03"), "reservation cap changed")
    return entries, tuple(wires)


def review_readout(sources):
    matrices = {}
    for name, path in (("A", REVIEW_A), ("B", REVIEW_B)):
        review = json.loads(sources[path])
        require(review["packet_sha256"] == sha(sources[PACKET]), "review packet commitment changed")
        require([r["ordinal"] for r in review["reviews"]] == list(range(1, 7)), "review case order changed")
        matrix = [[r["judgments"][d]["label"] for d in DIMENSIONS] for r in review["reviews"]]
        require(all(v in ("yes", "no", "unknown") for row in matrix for v in row), "invalid reviewer label")
        matrices[name] = matrix
    categories = {"agreed_determinate": [], "agreed_unknown": [], "disagreed": []}
    for i, (a, b) in enumerate(zip(matrices["A"], matrices["B"]), 1):
        for dimension, left, right in zip(DIMENSIONS, a, b):
            group = "disagreed" if left != right else "agreed_unknown" if left == "unknown" else "agreed_determinate"
            categories[group].append({"ordinal": i, "dimension": dimension, "A": left, "B": right})
    return {"label_order": list(DIMENSIONS), "reviewer_matrices": matrices, "categories": categories,
            "boundary": "Two qualitative AI references, not gold; disagreements and agreed unknown are separate. No fabricated consensus label."}


def _assemble(sources):
    require(all(sha(sources[p]) == h for p, h in PINNED_INPUT_SHA256.items()), "pinned input commitment changed")
    counter, metadata = _load_counter(json.loads(sources[SAMPLE_PLAN]))
    entries, wires = freeze_requests(json.loads(sources[PACKET]), counter)
    readout = review_readout(sources)
    require([len(readout["categories"][k]) for k in ("agreed_determinate", "agreed_unknown", "disagreed")]
            == [15, 1, 2], "fixed reference partition changed")
    return {"schema": SCHEMA, "status": "frozen_before_model_observations", "limits": LIMITS.copy(),
            "source_sha256": {p: sha(v) for p, v in sources.items()}, "counter": metadata,
            "requests": entries, "planned_reserved_usd": "0.030", "readout_only": readout,
            "planned_denominators": {"requests": 6, "typed_dimensions": 18, "agreed_determinate": 15,
                                     "agreed_unknown": 1, "disagreed": 2},
            "interpretation": "Fixed exposed natural sample; source-support diagnostic only, not gold accuracy, answer quality, or an independent effectiveness result."}, wires


def prepare_plan(output_dir):
    sources = _snapshots()
    plan, _ = _assemble(sources)
    require({p: sha(v) for p, v in _snapshots().items()} == plan["source_sha256"], "sources changed during planning")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "plan.json").write_bytes(canonical(plan) + b"\n")
    return plan


def verify_plan(path):
    body = Path(path).read_bytes()
    frozen, sources = json.loads(body), _snapshots()
    require(frozen.get("source_sha256") == {p: sha(v) for p, v in sources.items()}, "frozen source commitments changed")
    rebuilt, wires = _assemble(sources)
    require(body == canonical(rebuilt) + b"\n", "frozen plan or request bytes changed")
    return frozen, wires, sha(body)


def execute_plan(plan_path, output_dir, *, key_file=None, proxy=None, transport=None, live=False):
    plan_path, output_dir = Path(plan_path), Path(output_dir)
    plan, wires, digest = verify_plan(plan_path)
    require(not output_dir.exists(), "run output already exists")
    # A failed attempt is terminal for this plan, including uncertain billing.
    receipt = {"plan_sha256": digest, "source_sha256": plan["source_sha256"],
               "output_dir": str(output_dir.resolve()), "mode": "live" if live else "fake_or_unspecified"}
    with (plan_path.parent / "execution_claim.json").open("xb") as stream:
        stream.write(canonical(receipt) + b"\n")
    probe = client.ExchangeProbeClient(output_dir, wires, key_file=key_file, proxy=proxy,
                                       transport=transport, deadline_seconds=180)
    (output_dir / "plan_receipt.json").write_bytes(canonical(receipt) + b"\n")
    return probe.run(live=live)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("plan")
    prepare.add_argument("--output-dir", type=Path, required=True)
    for name in ("run", "bounded"):
        command = sub.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--key-file", type=Path, required=True)
        if name == "bounded":
            command.add_argument("--receipt", type=Path, required=True)
        else:
            command.add_argument("--proxy", default="http://127.0.0.1:7897", choices=["http://127.0.0.1:7897"])
    args = parser.parse_args()
    if args.command == "plan":
        result = prepare_plan(args.output_dir)
    elif args.command == "run":
        result = execute_plan(args.plan, args.output_dir, key_file=args.key_file, proxy=args.proxy, live=True)
    else:
        path, output, receipt = _contained(args.plan), _contained(args.output_dir), _contained(args.receipt)
        command = [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(path),
                   "--output-dir", str(output), "--key-file", str(args.key_file.resolve()),
                   "--proxy", "http://127.0.0.1:7897"]
        result = supervise(command, output, receipt, timeout_seconds=180, plan_sha256=sha(path.read_bytes()))
    print(json.dumps({"command": args.command, "status": result.get("status")}, sort_keys=True))
    if args.command == "bounded":
        return 0 if result.get("status") == "worker_exited" and result.get("returncode") == 0 else 1
    return 1 if args.command == "run" and result.get("status") != "completed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
