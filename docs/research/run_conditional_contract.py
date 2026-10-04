"""Run artificial JEV request/cache contracts with fixed fake responses only.

No model predictions or semantic quality metrics are produced. The word-count
counter is a synthetic resource-accounting probe, not a model tokenizer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SLAC.retrieval.decision.conditional import (  # noqa: E402
    CachedResponse, ConditionalState, SourceRelation, Unit, build_request,
    decode_cached, decode_response, freeze_response, render_evidence,
)

FIXTURE_PATH = ROOT / "docs/research/fixtures/conditional_jev_contract_v1.json"
PROTOCOL_PATH = ROOT / "docs/research/CONDITIONAL_JEV_CONTRACT_PROTOCOL_20261004.md"
MODEL = "typesafe/jev-1.13"
RESPONSE_MODEL = "typesafe/jev-1.13-20260917"
ENDPOINT_ID = "openrouter-decisions-contract-only"
COUNTER_VERSION = "synthetic-whitespace-counter-v1"
FIXED_CHOICES = {
    "standalone_support": "unknown",
    "conditional_added_information": "yes",
    "conflict": "no",
}


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def synthetic_counter(text: str) -> int:
    return len(text.split())


def project_state(pair: dict, case: dict, *, control: bool = False) -> ConditionalState:
    """Whitelist inputs; fixture scope maps to the fixed core qualifier kind.

    Supervision, fixture groups, pair IDs and case IDs never enter model state.
    """
    texts = case["current_texts"]
    if not isinstance(texts, list) or len(texts) != 2:
        raise ValueError("each synthetic case must have two current units")
    current = tuple(Unit(f"u{i + 1:03d}", text, i, "s001") for i, text in enumerate(texts))
    candidate = Unit("u003", pair["candidate_text"], 2, "s001")
    relations = []
    for raw in case["control_relations" if control else "relations"]:
        if set(raw) != {"kind", "from", "to", "anchor_text"}:
            raise ValueError("relation fixture must use the fixed input schema")
        if raw["to"] != candidate.id:
            raise ValueError("synthetic relation must anchor in the candidate")
        anchor = raw["anchor_text"]
        if not isinstance(anchor, str) or not anchor or candidate.text.count(anchor) != 1:
            raise ValueError("fixture anchor must occur exactly once in candidate")
        start = candidate.text.index(anchor)
        relations.append(SourceRelation(raw["to"], raw["from"],
                                        "qualifier" if raw["kind"] == "scope" else raw["kind"],
                                        start, start + len(anchor)))
    return ConditionalState(pair["question"], current, candidate, tuple(relations))


def request_for(state: ConditionalState, arm: str, *, max_tokens: int = 1024,
                max_payload_bytes: int = 65536):
    return build_request(state, arm=arm, endpoint_id=ENDPOINT_ID, model_id=MODEL,
                         expected_response_model=RESPONSE_MODEL,
                         token_counter=synthetic_counter, max_tokens=max_tokens,
                         max_payload_bytes=max_payload_bytes, counter_version=COUNTER_VERSION)


def variants(pair: dict, case: dict) -> dict:
    state = project_state(pair, case)
    result = {"standalone": request_for(state, "standalone"),
              "plain_conditional": request_for(state, "plain_conditional")}
    if case["relations"]:
        result["relation_conditioned"] = request_for(state, "relation_conditioned")
        result["permuted_relation_control"] = request_for(
            project_state(pair, case, control=True), "relation_conditioned")
    elif case["control_relations"]:
        raise ValueError("no relation control may be invented for a relation-free case")
    return result


def fixed_fake_response(transport_bytes: bytes) -> dict:
    """Mechanical constants keyed only by requested dimension, never supervision."""
    envelope = json.loads(transport_bytes)
    questions = envelope["payload"]["questions"]
    if set(questions) - set(FIXED_CHOICES):
        raise ValueError("unknown synthetic question dimension")
    return {"request_key": envelope["request_key"], "response": {
        "model": RESPONSE_MODEL,
        "answers": {name: {"type": "choice", "choice": FIXED_CHOICES[name]}
                    for name in questions},
    }}


def cached_roundtrip(request, cache: dict[str, CachedResponse], transport=fixed_fake_response):
    if request.cache_key in cache:
        return decode_cached(request, cache[request.cache_key]), True
    envelope = transport(request.transport_bytes())
    cache[request.cache_key] = freeze_response(request, envelope)
    return decode_cached(request, cache[request.cache_key]), False


def malformed_probes(request) -> dict[str, dict]:
    """Separately authored structural failures, without semantic supervision."""
    if set(request.expected_ids) != {"conditional_added_information", "conflict"}:
        raise ValueError("malformed probes require a conditional request")
    valid = fixed_fake_response(request.transport_bytes())
    probes = {}
    probes["missing_added"] = copy.deepcopy(valid)
    del probes["missing_added"]["response"]["answers"]["conditional_added_information"]
    probes["invalid_added_choice"] = copy.deepcopy(valid)
    probes["invalid_added_choice"]["response"]["answers"]["conditional_added_information"]["choice"] = "maybe"
    probes["invalid_conflict_type"] = copy.deepcopy(valid)
    probes["invalid_conflict_type"]["response"]["answers"]["conflict"]["type"] = "number"
    probes["wrong_request_key"] = copy.deepcopy(valid)
    probes["wrong_request_key"]["request_key"] = "0" * 64
    probes["wrong_returned_model"] = copy.deepcopy(valid)
    probes["wrong_returned_model"]["response"]["model"] = "different-synthetic-model"
    probes["extra_answer_id"] = copy.deepcopy(valid)
    probes["extra_answer_id"]["response"]["answers"]["unexpected"] = {"type": "choice", "choice": "yes"}
    return probes


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_relation_control(authored, control) -> None:
    left = authored.payload()["state"]
    right = control.payload()["state"]
    left_relations, right_relations = left.pop("relations"), right.pop("relations")
    _require(left == right, "relation controls changed visible text or order")
    _require(len(left_relations) == len(right_relations) > 0, "relation count differs")
    _require(len(canonical(left_relations)) == len(canonical(right_relations)),
             "encoded relation metadata byte sizes differ")
    _require(left_relations != right_relations, "control did not permute a relation")
    _require(authored.cache_key != control.cache_key, "control reused authored relation cache")
    _require(authored.rendered_evidence == control.rendered_evidence,
             "control changed evidence-only rendering")


def run_contract(fixture: dict) -> tuple[dict, list[dict]]:
    """Check all authored cases. Expected labels are deliberately never scored."""
    _require(fixture.get("schema") == "slac-human-authored-conditional-fixtures-v1", "fixture schema mismatch")
    pairs = fixture["pairs"]
    _require(len(pairs) == 8, "contract requires all eight pairs")
    cache, records, pair_ids, case_ids = {}, [], set(), set()
    arm_counts, checks = Counter(), Counter()
    first_conditional = None
    for pair in pairs:
        _require(pair["pair_id"] not in pair_ids, "duplicate pair ID")
        pair_ids.add(pair["pair_id"])
        _require(len(pair["states"]) == 2, "each pair must have exactly two cases")
        pair_requests = []
        for case in pair["states"]:
            _require(case["case_id"] not in case_ids, "duplicate case ID")
            case_ids.add(case["case_id"])
            built = variants(pair, case)
            pair_requests.append(built)
            poisoned = copy.deepcopy(pair)
            poisoned["supervision"] = {"fixture_group": "HIDDEN_GROUP_SENTINEL",
                                        "expected_labels_by_case": "HIDDEN_EXPECTED_SENTINEL"}
            alternate = variants(poisoned, case)
            _require(all(built[name].payload_bytes == alternate[name].payload_bytes
                         and built[name].cache_key == alternate[name].cache_key for name in built),
                     "supervision leaked into request")
            checks["supervision_isolation_cases"] += 1
            _require(len({request.cache_key for request in built.values()}) == len(built),
                     "arm cache namespaces collided")
            checks["arm_isolation_cases"] += 1
            if "relation_conditioned" in built:
                check_relation_control(built["relation_conditioned"], built["permuted_relation_control"])
                checks["relation_controls"] += 1
            for name, request in built.items():
                state = project_state(pair, case, control=name == "permuted_relation_control")
                arm = "relation_conditioned" if name == "permuted_relation_control" else name
                expected_render = render_evidence((state.candidate,) if arm == "standalone"
                                                  else (*state.current_pack, state.candidate))
                _require(request.rendered_evidence == expected_render, "wrong evidence budget surface")
                count = synthetic_counter(expected_render)
                exact = request_for(state, arm, max_tokens=count)
                _require(exact.evidence_tokens == count, "exact budget rejected")
                try:
                    request_for(state, arm, max_tokens=count - 1)
                except ValueError:
                    checks["over_budget_rejections"] += 1
                else:
                    raise ValueError("budget plus one was accepted")
                checks["exact_budget_acceptances"] += 1
                decoded, hit = cached_roundtrip(request, cache)
                _require(decoded.envelope_valid, "fixed fake response failed roundtrip")
                checks["cache_hits" if hit else "fake_transport_calls"] += 1
                arm_counts[name] += 1
                records.append({"pair_id": pair["pair_id"], "case_id": case["case_id"],
                                "variant": name, "request_key": request.cache_key,
                                "payload": request.payload(), "binding": json.loads(request.binding_bytes),
                                "rendered_evidence": request.rendered_evidence,
                                "simulated_response": json.loads(cache[request.cache_key].response_bytes),
                                "decoded_simulated": asdict(decoded), "cache_hit": hit})
            first_conditional = first_conditional or built["plain_conditional"]
        left, right = pair_requests
        _require(left["standalone"].cache_key == right["standalone"].cache_key,
                 "hidden current pack changed standalone key")
        _require(left["plain_conditional"].cache_key != right["plain_conditional"].cache_key,
                 "current pack mutation failed to invalidate conditional key")
        checks["counterfactual_pairs"] += 1
    _require(len(case_ids) == 16, "contract requires all sixteen cases")
    expected_probe_results = {
        "missing_added": (True, "unknown", "no"),
        "invalid_added_choice": (True, "unknown", "no"),
        "invalid_conflict_type": (True, "yes", "unknown"),
        "wrong_request_key": (False, "unknown", "unknown"),
        "wrong_returned_model": (False, "unknown", "unknown"),
        "extra_answer_id": (False, "unknown", "unknown"),
    }
    for name, envelope in malformed_probes(first_conditional).items():
        decoded = decode_response(first_conditional, envelope)
        choices = {item.id: item.choice for item in decoded.decisions}
        actual = (decoded.envelope_valid, choices["conditional_added_information"], choices["conflict"])
        _require(actual == expected_probe_results[name], f"malformed probe failed: {name}")
        checks["malformed_response_probes"] += 1
    stale = CachedResponse("0" * 64, cache[first_conditional.cache_key].response_bytes)
    _require(not decode_cached(first_conditional, stale).envelope_valid, "stale receipt accepted")
    checks["stale_cache_rejections"] += 1

    def forbidden_transport(_: bytes):
        raise AssertionError("a valid cache hit called transport")

    _, hit = cached_roundtrip(first_conditional, cache, forbidden_transport)
    _require(hit, "cache hit was lost")
    checks["cache_hit_transport_suppression"] += 1
    summary = {
        "schema": "slac-conditional-contract-result-v1",
        "status": "synthetic_contract_checks_passed",
        "interpretation": "Fixed fake responses only. NOT model predictions. No semantic accuracy computed.",
        "real_dataset_read": False, "api_calls": 0, "model_inference_calls": 0,
        "pair_count": len(pair_ids), "case_count": len(case_ids),
        "request_records": len(records), "unique_request_keys": len(cache),
        "variant_counts": dict(arm_counts), "contract_checks": dict(checks),
        "counter": {"version": COUNTER_VERSION, "measurement": "synthetic whitespace units over exact evidence rendering",
                    "model_tokenizer": False, "total_billed_tokens_measured": False},
        "relation_control": "Equal canonical metadata bytes, visible text/order and relation counts; not token-matched inference.",
        "fixed_fake_choices": FIXED_CHOICES.copy(),
    }
    return summary, records


def run_to_directory(output_dir: Path) -> dict:
    sources = [FIXTURE_PATH, PROTOCOL_PATH, Path(__file__).resolve(),
               ROOT / "SLAC/retrieval/decision/conditional.py",
               ROOT / "tests/research/test_conditional_contract_runner.py",
               ROOT / "tests/research/test_conditional_decision.py"]
    # One byte snapshot per input is both hashed and parsed; no target data exists.
    source_bytes = {str(path.resolve()): path.read_bytes() for path in sources}
    fixture = json.loads(source_bytes[str(FIXTURE_PATH.resolve())])
    summary, records = run_contract(fixture)
    hashes = {path: sha(data) for path, data in source_bytes.items()}
    for path, expected in hashes.items():
        _require(sha(Path(path).read_bytes()) == expected, "bound source changed during contract run")
    summary["source_sha256"] = hashes
    summary["protocol_sha256"] = hashes[str(PROTOCOL_PATH.resolve())]
    record_bytes = canonical(records) + b"\n"
    summary["synthetic_requests_sha256"] = sha(record_bytes)
    summary_bytes = canonical(summary) + b"\n"
    plan = {"schema": "slac-conditional-contract-plan-v1", "source_sha256": hashes,
            "protocol_sha256": summary["protocol_sha256"], "api_calls": 0,
            "artifact_sha256": {"summary.json": sha(summary_bytes), "synthetic_requests.json": sha(record_bytes)}}
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in (("summary.json", summary_bytes), ("synthetic_requests.json", record_bytes),
                       ("plan.json", canonical(plan) + b"\n")):
        with (output_dir / name).open("xb") as stream:
            stream.write(data)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New immutable output directory. Execution always uses fixed fake responses.")
    args = parser.parse_args()
    summary = run_to_directory(args.output_dir)
    print(json.dumps({key: summary[key] for key in ("status", "pair_count", "case_count",
                                                   "request_records", "unique_request_keys", "api_calls")}, indent=2))


if __name__ == "__main__":
    main()
