"""Two fixed documents: exact source coverage and offline JEV request identity."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from SLAC.refiner.pipeline.assemble.source_document_view import DocumentSourceView
from SLAC.refiner.pipeline.assemble.source_coverage import build_native_coverage_index
from SLAC.retrieval.decision import conditional as core
from SLAC.retrieval.decision.refiner_bridge import capture_refiner_source_chunk, build_refiner_source_request

BASE = ROOT / "artifacts/research-foundation/offline-20261004"
STAGE = BASE / "refiner-jev-text-contract-01"
INPUTS = {
    "bounded_inputs": (BASE / "refiner-granularity-probe-01/bounded_inputs.json", "e860ca0d755ce4f1ce24b73f64e89b4ad6e9e1eb1da0659efa333eb8fc7d603c"),
    "document_exports": (BASE / "refiner-document-source-export-01/run-01/document_exports.json", "e28538bf86ed10b452257c887067c21d5fdd1588baef3d964ac24229de87df87"),
}
SOURCES = {
    "docs/research/probe_refiner_jev_source_contract.py",
    "SLAC/refiner/pipeline/assemble/source_document_view.py",
    "SLAC/refiner/pipeline/assemble/source_coverage.py",
    "SLAC/retrieval/decision/refiner_bridge.py",
    "SLAC/retrieval/decision/conditional.py",
    "tests/research/test_source_coverage.py",
    "tests/research/test_refiner_decision_bridge.py",
}
TOKENIZER = Path("D:/code/Github/SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
TOKEN_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"}
QUERY = "Offline text-identity fixture; no model judgment is requested."
REQUEST_CONFIG = {"arm": "standalone", "endpoint_id": "offline-only",
                  "model_id": "offline-identity-model", "expected_response_model": "offline-identity-response",
                  "counter_version": "local-bge-m3-specials-no-truncation-v1", "max_tokens": 1024}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def write(path, value):
    with Path(path).open("xb") as output:
        output.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def classify(coverage):
    full, partial = len(coverage.full_native_unit_ids), len(coverage.partial_native_unit_ids)
    if full and partial:
        return "mixed_full_partial"
    if full:
        return "full_only_single" if full == 1 else "full_only_multiple"
    if partial:
        return "partial_only_single" if partial == 1 else "partial_only_multiple"
    return "gap_only"


def inspect_partition(index, rows, atom_span):
    details, covered, previous, hits = [], Counter(), 0, Counter()
    full_hits = partial_hits = 0
    for ordinal, row in enumerate(rows, 1):
        start, end = atom_span(row)
        coverage = index.cover_atoms(start, end)
        a, b = coverage.source_char_span
        require(a == previous and row["text"] == index.view.source_text[a:b], "source partition differs")
        previous = b
        category = classify(coverage)
        hits[category] += 1
        full_hits += len(coverage.full_native_unit_ids)
        partial_hits += len(coverage.partial_native_unit_ids)
        for hit in coverage.hits:
            x, y = hit.overlap_source_span
            covered[hit.native_unit_id] += y - x
        details.append({"ordinal": ordinal, "atom_span": [start, end], "category": category,
                        "coverage": asdict(coverage), "exact_native_unit_id": coverage.exact_native_unit_id})
    require(previous == len(index.view.source_text), "source partition does not reach document end")
    summary = {"records": len(rows), "categories": dict(hits),
               "full_native_hits": full_hits,
               "partial_native_hits": partial_hits,
               "exact_native_text_records": sum(row["exact_native_unit_id"] is not None for row in details),
               "gap_chars_total": sum(row["coverage"]["gap_chars"] for row in details),
               "unit_char_coverage": dict(covered)}
    return summary, details


def run(protocol_sha, output):
    protocol_path = STAGE / "protocol.json"
    require(sha(protocol_path.read_bytes()) == protocol_sha, "frozen protocol changed")
    protocol = read(protocol_path)
    require(protocol["schema"] == "slac-refiner-jev-source-contract-protocol-v1"
            and set(protocol["inputs"]) == set(INPUTS) and set(protocol["source_sha256"]) == SOURCES,
            "protocol inventory differs")
    require(protocol["synthetic_query"] == QUERY and protocol["request_config"] == REQUEST_CONFIG,
            "fixed request contract differs")
    bindings, loaded = {str(protocol_path): protocol_sha}, {}
    for role, (path, expected) in INPUTS.items():
        item = protocol["inputs"][role]
        require(Path(item["path"]).resolve() == path.resolve() and item["sha256"] == expected
                and sha(path.read_bytes()) == expected, "fixed input binding differs")
        loaded[role] = read(path)
        bindings[str(path)] = expected
    for name, expected in protocol["source_sha256"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and sha(path.read_bytes()) == expected, "source changed")
        bindings[str(path)] = expected
    tok = protocol["tokenizer"]
    require(Path(tok["path"]).resolve() == TOKENIZER.resolve() and set(tok["verified_file_sha256"]) == TOKEN_FILES,
            "tokenizer inventory differs")
    for name, expected in tok["verified_file_sha256"].items():
        path = TOKENIZER / name
        require(sha(path.read_bytes()) == expected, "tokenizer changed")
        bindings[str(path)] = expected
    data, exports = loaded["bounded_inputs"], loaded["document_exports"]
    ids = [row["doc_id"] for row in data["queries"]]
    require(len(ids) == len(set(ids)) == 2 and [len(data["documents"][d]) for d in ids] == [60, 23]
            and [e["doc_id"] for e in exports] == ids, "two-document scope differs")
    output = Path(output).resolve()
    require(output.is_relative_to(STAGE.resolve()) and output != STAGE.resolve() and not output.exists(),
            "new stage output subdirectory required")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "started.json", {"bindings": bindings})
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    def denied(*args, **kwargs):
        raise RuntimeError("network forbidden in source-contract probe")
    socket.create_connection = denied
    socket.socket.connect = denied
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, trust_remote_code=False)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    docs, artifacts = [], []
    for ordinal, item in enumerate(exports, 1):
        doc_id = item["doc_id"]
        view = DocumentSourceView(**item["view"])
        native = next(row for row in data["native_documents"] if row["doc_id"] == doc_id)
        require(view.source_text == native["native_document_text"]
                and sha(view.source_text.encode()) == native["native_document_text_sha256"], "view source differs")
        mappings = {row["native_unit_id"]: row for row in data["unit_mapping"] if row["doc_id"] == doc_id}
        units = [{"native_unit_id": unit["unit_id"], "source_span": mappings[unit["unit_id"]]["raw_native_char_span"],
                  "source_text": unit["native_text"]} for unit in data["documents"][doc_id]]
        require(len(mappings) == len(units), "native unit identity differs")
        index = build_native_coverage_index(view, units)
        group_rows = {"raw_b0_chunks": item["exports"]["raw_b0_source"]["refined_chunks"],
                      "rule_projected_chunks": item["exports"]["rule_projected_source"]["refined_chunks"],
                      "source_leaves": item["exports"]["raw_b0_source"]["leaf_records"]}
        group_stats, coverage_rows = {}, {}
        unit_lengths = {u["native_unit_id"]: len(u["source_text"]) for u in units}
        for name, rows in group_rows.items():
            span = (lambda r: (r["atom_index"], r["atom_index"] + 1)) if name == "source_leaves" else (
                lambda r: (r["atom_start"], r["atom_end"]))
            stats, details = inspect_partition(index, rows, span)
            require(stats.pop("unit_char_coverage") == unit_lengths, "partition lost or duplicated raw unit characters")
            stats["complete_document_partition"] = True
            group_stats[name], coverage_rows[name] = stats, details
        other_leaves = item["exports"]["rule_projected_source"]["leaf_records"]
        require([(r["atom_index"], r["text"], r["source_char_span"]) for r in group_rows["source_leaves"]]
                == [(r["atom_index"], r["text"], r["source_char_span"]) for r in other_leaves],
                "leaf coverage changed under boundary-only projection")
        requests, request_stats = {}, {}
        for group, source_mode, legacy_mode in (("raw_b0_chunks", "raw_b0_source", "raw_b0_legacy"),
                                               ("rule_projected_chunks", "rule_projected_source", "rule_projected_legacy")):
            pair_rows = []
            for position, (chunk, legacy) in enumerate(zip(group_rows[group],
                    item["exports"][legacy_mode]["refined_chunks"], strict=True), 1):
                require((chunk["chunk_id"], chunk["atom_start"], chunk["atom_end"])
                        == (legacy["chunk_id"], legacy["atom_start"], legacy["atom_end"]),
                        "counterfactual legacy/source chunk identity differs")
                snapshot = capture_refiner_source_chunk(chunk, index)
                wrapped = build_refiner_source_request(QUERY, (), snapshot, token_counter=count, **REQUEST_CONFIG)
                source_request = wrapped.request
                binding = json.loads(source_request.binding_bytes)
                legacy_unit = core.Unit(snapshot.unit.id, legacy["text"], snapshot.unit.order, snapshot.unit.doc_id)
                legacy_request = core.build_request(core.ConditionalState(QUERY, (), legacy_unit,
                    version=binding["state_version"]), token_counter=count, **REQUEST_CONFIG)
                source_payload, legacy_payload = source_request.payload(), legacy_request.payload()
                legacy_payload["state"]["candidate"]["text"] = source_payload["state"]["candidate"]["text"]
                require(source_payload == legacy_payload, "paired request payload differs beyond candidate text")
                legacy_binding = json.loads(legacy_request.binding_bytes)
                derived = {"payload_sha256", "evidence_tokens"}
                require({k: v for k, v in binding.items() if k not in derived}
                        == {k: v for k, v in legacy_binding.items() if k not in derived},
                        "paired request binding differs beyond text-derived fields")
                same_text = chunk["text"] == legacy["text"]
                require((source_request.cache_key == legacy_request.cache_key) == same_text,
                        "exact request identity did not track text-only change")
                require(source_request.payload()["state"]["candidate"]["text"] == chunk["text"],
                        "request candidate did not preserve actual exported bytes")
                require(source_request.evidence_tokens == count(source_request.rendered_evidence), "request evidence count differs")
                pair_rows.append({"chunk_ordinal": position, "same_text": same_text,
                    "same_counterfactual_request_key": source_request.cache_key == legacy_request.cache_key,
                    "source_request": {"payload": source_request.payload(), "binding": binding,
                        "cache_key": source_request.cache_key, "rendered_evidence": source_request.rendered_evidence,
                        "provenance": wrapped.provenance_receipt()},
                    "legacy_counterfactual_request": {"payload": legacy_request.payload(),
                        "binding": json.loads(legacy_request.binding_bytes), "cache_key": legacy_request.cache_key,
                        "rendered_evidence": legacy_request.rendered_evidence}})
            requests[group] = pair_rows
            request_stats[group] = {"pairs": len(pair_rows), "same_text": sum(r["same_text"] for r in pair_rows),
                "same_counterfactual_request_key": sum(r["same_counterfactual_request_key"] for r in pair_rows),
                "changed_text_changed_key": sum(not r["same_text"] and not r["same_counterfactual_request_key"] for r in pair_rows),
                "source_request_max_evidence_tokens": max(r["source_request"]["binding"]["evidence_tokens"] for r in pair_rows),
                "transport_calls": 0}
        docs.append({"document_ordinal": ordinal, "native_units": len(units), "model_atoms": len(view.model_atoms),
                     "coverage": group_stats, "request_identity": request_stats})
        artifacts.append({"document_ordinal": ordinal, "doc_id": doc_id, "coverage": coverage_rows, "requests": requests})
    totals = {"documents": len(docs), "native_units": sum(d["native_units"] for d in docs),
              "model_atoms": sum(d["model_atoms"] for d in docs),
              "coverage_records": sum(g["records"] for d in docs for g in d["coverage"].values()),
              "request_pairs": sum(g["pairs"] for d in docs for g in d["request_identity"].values()),
              "transport_calls": 0}
    for path, expected in bindings.items():
        require(sha(Path(path).read_bytes()) == expected, "bound file drift")
    write(output / "coverage_requests.json", artifacts)
    report = {"schema": "slac-refiner-jev-source-contract-result-v1", "status": "completed", "scope": protocol["scope"],
              "bindings": bindings, "documents": docs, "totals": totals,
              "artifact_sha256": sha((output / "coverage_requests.json").read_bytes()), "limits": protocol["interpretation"]}
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--output", type=Path, default=STAGE / "run-01")
    args = parser.parse_args()
    result = run(args.protocol_sha, args.output)
    print(json.dumps({"status": result["status"], "totals": result["totals"]}, sort_keys=True))
