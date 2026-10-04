"""Frozen two-document projector/exporter text and budget comparison, no model."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from SLAC.refiner.slac_refiner.decoding import projector
from SLAC.refiner.pipeline.assemble import export_refined_chunks as exporter

STAGE = ROOT / "artifacts/research-foundation/offline-20261004/refiner-source-budget-probe-01"
PREVIOUS = ROOT / "artifacts/research-foundation/offline-20261004/refiner-granularity-probe-01"
INPUTS = {
    "bounded_inputs": (PREVIOUS / "bounded_inputs.json", "e860ca0d755ce4f1ce24b73f64e89b4ad6e9e1eb1da0659efa333eb8fc7d603c"),
    "fixture_exports": (PREVIOUS / "run-02/fixture_exports.json", "ba8bfbcc4b12587323195a9395c1c3b32048d5657d354a3e09c0d50f05636024"),
}
TOKENIZER = Path("D:/code/Github/SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
TOKENIZER_FILES = {
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "sentencepiece.bpe.model",
}
REQUIRED_SOURCES = {
    "docs/research/probe_refiner_projection_export.py",
    "SLAC/refiner/slac_refiner/decoding/projector.py",
    "SLAC/refiner/pipeline/assemble/export_refined_chunks.py",
}
ALLOWED_SOURCES = REQUIRED_SOURCES | {
    "docs/research/probe_refiner_source_alignment.py",
    "tests/research/test_refiner_source_alignment.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def stats(text, atoms, count):
    tokens = count(text)
    require(type(tokens) is int and tokens >= 0, "invalid whole-span token count")
    return {"atoms": atoms, "chars": len(text), "tokens": tokens,
            "utf8_bytes": len(text.encode()), "text_sha256": sha(text.encode())}


def violations(measured, config, prefix):
    return [key for key in ("atoms", "chars", "tokens")
            if (measured[key] > getattr(config, prefix + key) if prefix == "max_chunk_"
                else measured[key] < getattr(config, prefix + key))]


def compare_units(units, chunks, count, config):
    require(len(units) == len(chunks), "projector/exporter chunk count differs")
    rows = []
    for index, (unit, chunk) in enumerate(zip(units, chunks, strict=True)):
        start, end = unit["start_atom"], unit["end_atom"]
        require((start, end) == (chunk["atom_start"], chunk["atom_end"]), "comparison spans differ")
        left, right = stats(unit["text"], end-start, count), stats(chunk["text"], end-start, count)
        rows.append({"chunk_index": index, "atom_span": [start, end],
            "projector": left, "exporter": right,
            "byte_equal": unit["text"].encode() == chunk["text"].encode(),
            "whitespace_normalized_equal": " ".join(unit["text"].split()) == " ".join(chunk["text"].split()),
            "token_equal": left["tokens"] == right["tokens"],
            "exporter_minus_projector": {key: right[key]-left[key] for key in ("chars", "tokens", "utf8_bytes")},
            "projector_hard_violations": violations(left, config, "max_chunk_"),
            "exporter_hard_violations": violations(right, config, "max_chunk_"),
            "projector_soft_unmet": violations(left, config, "min_chunk_"),
            "exporter_soft_unmet": violations(right, config, "min_chunk_")})
    return rows


def aggregate(rows):
    return {"chunks": len(rows), "byte_equal": sum(r["byte_equal"] for r in rows),
        "whitespace_normalized_equal": sum(r["whitespace_normalized_equal"] for r in rows),
        "token_equal": sum(r["token_equal"] for r in rows),
        "exporter_minus_projector_token_sum": sum(r["exporter_minus_projector"]["tokens"] for r in rows),
        "projector_over_hard_caps": sum(bool(r["projector_hard_violations"]) for r in rows),
        "exporter_over_hard_caps": sum(bool(r["exporter_hard_violations"]) for r in rows),
        "exporter_violation_dimensions": dict(Counter(d for r in rows for d in r["exporter_hard_violations"]))}


def boundary_changes(left, right):
    require(len(left) == len(right), "boundary lengths differ")
    return {"added_gaps": [i for i, (a, b) in enumerate(zip(left, right)) if a == 0 and b == 1],
            "removed_gaps": [i for i, (a, b) in enumerate(zip(left, right)) if a == 1 and b == 0],
            "changed_gaps": sum(a != b for a, b in zip(left, right))}


def probe_document(fixture, count):
    record = fixture["refiner_input"]
    atoms, b0 = record["atoms"], record["b0"]
    config = projector.ProjectorConfig()
    spans = projector.boundary_vector_to_spans(len(atoms), b0)
    legacy_units = projector.spans_to_units(atoms, spans)
    old_chunks = fixture["raw_export"]["refined_chunks"]
    fixed_rows = compare_units(legacy_units, old_chunks, count, config)
    report = {"doc_id": record["doc_id"], "atoms": len(atoms), "default_projector_config": asdict(config),
        "fixed_b0": {"spans": spans, "rows": fixed_rows, "summary": aggregate(fixed_rows)}}
    artifact = {"doc_id": record["doc_id"], "mode": "rule_projection_fixture", "model_inference": False,
                "fixed_b0_projector_units": legacy_units}
    try:
        projection = projector.project_boundary_vector(atoms, b0, config,
            gap_scores=None, token_counter=count, strict=True)
    except projector.ProjectorBudgetError as exc:
        failure = {"status": "strict_projection_infeasible", "overlong_spans": exc.overlong_spans,
                   "export_executed": False, "relaxed_retry": False}
        report["rule_projection"] = failure
        artifact["rule_projection"] = failure
        return report, artifact
    projected_b = projection["projected_b"]
    candidate = {"candidate_id": "rule-projection-fixture", "candidate_type": "rule_projection_fixture",
        "teacher_ckpt": None, "prediction": {
            "b0_sparse": [i for i, value in enumerate(b0) if value],
            "b_pred_sparse": [i for i, value in enumerate(projected_b) if value]}}
    exported = exporter.export_refined_chunks_from_candidate(record, candidate)
    rows = compare_units(projection["projected_units"], exported["refined_chunks"], count, config)
    require(projection["hard_max_satisfied"] and not any(r["projector_hard_violations"] for r in rows),
            "strict projector returned over-budget text")
    split_b = projector.spans_to_boundary_vector(len(atoms), projection["spans_after_split"])
    split_units = projector.spans_to_units(atoms, projection["spans_after_split"])
    short_after_split = [{"atom_span": [u["start_atom"], u["end_atom"]],
                         "unmet_dimensions": violations(stats(u["text"], u["end_atom"]-u["start_atom"], count), config, "min_chunk_")}
                        for u in split_units]
    report["rule_projection"] = {"status": "completed", "strict": True, "gap_scores": None,
        "source_text": None, "atom_char_spans": None,
        "text_mode": projection["text_mode"], "token_count_mode": projection["token_count_mode"],
        "spans_before": projection["spans_before"], "spans_after_split": projection["spans_after_split"],
        "spans_after_merge": projection["spans_after_merge"], "projected_b": projected_b,
        "hard_split_boundary_changes": boundary_changes(b0, split_b),
        "soft_min_boundary_changes": boundary_changes(split_b, projected_b),
        "net_boundary_changes": boundary_changes(b0, projected_b),
        "soft_unmet_after_split": [r for r in short_after_split if r["unmet_dimensions"]],
        "short_spans_after_projection": projection["short_spans"],
        "projector_hard_max_satisfied": projection["hard_max_satisfied"],
        "exporter_hard_max_satisfied": not any(r["exporter_hard_violations"] for r in rows),
        "rows": rows, "summary": aggregate(rows)}
    artifact.update(projection=projection, authored_rule_candidate=candidate, raw_export=exported)
    return report, artifact


def verify_inputs(data, fixtures):
    ids = [q["doc_id"] for q in data["queries"]]
    require(len(ids) == len(set(ids)) == 2, "two distinct documents required")
    require([len(data["documents"][doc]) for doc in ids] == [60, 23], "83 native-unit scope differs")
    require(len(fixtures) == 2 and [f["refiner_input"]["doc_id"] for f in fixtures] == ids,
            "fixed fixtures differ")
    require(sum(len(f["refiner_input"]["atoms"]) for f in fixtures) == 251, "251-atom scope differs")
    for fixture in fixtures:
        require(fixture["mode"] == "b0_identity_fixture" and fixture["model_inference"] is False,
                "input is not the frozen identity fixture")
        record = fixture["refiner_input"]
        units = data["documents"][record["doc_id"]]
        require([s["source_unit_id"] for s in record["chunk0_units"]] == [u["unit_id"] for u in units]
                and [s["text"] for s in record["chunk0_units"]] == [u["text"] for u in units], "seed input projection differs")
        gaps = [i for i, value in enumerate(record["b0"]) if value]
        require(fixture["authored_identity_candidate"]["prediction"]["b_pred_sparse"] == gaps,
                "input boundaries are not b0 identity")


def run(protocol_sha, output):
    protocol_path = STAGE / "protocol.json"
    require(sha(protocol_path.read_bytes()) == protocol_sha, "frozen protocol changed")
    protocol = read(protocol_path)
    require(protocol["schema"] == "slac-refiner-source-budget-protocol-v1", "protocol schema differs")
    require(set(protocol["inputs"]) == set(INPUTS), "unexpected input roles")
    bindings = {str(protocol_path): protocol_sha}
    loaded = {}
    for name, (path, expected) in INPUTS.items():
        item = protocol["inputs"][name]
        require(Path(item["path"]).resolve() == path.resolve() and item["sha256"] == expected, "input path or commitment differs")
        require(sha(path.read_bytes()) == expected, "fixed input changed")
        bindings[str(path)] = expected
        loaded[name] = read(path)
    source_hashes = protocol["source_sha256"]
    require(REQUIRED_SOURCES <= set(source_hashes) <= ALLOWED_SOURCES, "source closure differs")
    for name, expected in source_hashes.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.suffix == ".py", "invalid source path")
        require(sha(path.read_bytes()) == expected, "source changed")
        bindings[str(path)] = expected
    tokenizer = protocol["tokenizer"]
    require(Path(tokenizer["path"]).resolve() == TOKENIZER.resolve()
            and set(tokenizer["verified_file_sha256"]) == TOKENIZER_FILES, "tokenizer path or inventory differs")
    for name, expected in tokenizer["verified_file_sha256"].items():
        path = TOKENIZER / name
        require(sha(path.read_bytes()) == expected, "tokenizer changed")
        bindings[str(path)] = expected
    verify_inputs(loaded["bounded_inputs"], loaded["fixture_exports"])
    output = Path(output).resolve()
    require(output.is_relative_to(STAGE.resolve()) and output != STAGE.resolve() and not output.exists(),
            "new stage output subdirectory required")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "started.json", {"status": "started", "bindings": bindings, "model_inference": False})
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    def denied(*args, **kwargs):
        raise RuntimeError("network forbidden in offline projection fixture")
    socket.create_connection = denied
    socket.socket.connect = denied
    from transformers import AutoTokenizer
    local_tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, trust_remote_code=False)
    def count(text):
        return len(local_tokenizer.encode(text, add_special_tokens=True, truncation=False))
    documents, artifacts = [], []
    for fixture in loaded["fixture_exports"]:
        report, artifact = probe_document(fixture, count)
        documents.append(report)
        artifacts.append(artifact)
    for path, expected in bindings.items():
        require(sha(Path(path).read_bytes()) == expected, "bound input/source/tokenizer drift")
    write(output / "projection_exports.json", artifacts)
    successful = [d for d in documents if d["rule_projection"]["status"] == "completed"]
    result = {"schema": "slac-refiner-projection-export-result-v1",
        "status": "completed" if len(successful) == 2 else "completed_with_infeasible_projection",
        "mode": "rule_projection_fixture", "scope": {"documents": 2, "source_units": 83, "atoms": 251,
            "model_inference": False, "api_calls": 0, "key_reads": 0, "weight_reads": 0, "gold_labels_scores_read": False},
        "bindings": bindings, "projection_exports_sha256": sha((output / "projection_exports.json").read_bytes()),
        "documents": documents,
        "totals": {"fixed_b0": aggregate([row for d in documents for row in d["fixed_b0"]["rows"]]),
                   "rule_projection": aggregate([row for d in successful for row in d["rule_projection"]["rows"]]),
                   "strict_projection_infeasible_documents": len(documents)-len(successful),
                   "net_changed_boundaries": sum(d["rule_projection"]["net_boundary_changes"]["changed_gaps"] for d in successful),
                   "soft_min_removed_boundaries": sum(len(d["rule_projection"]["soft_min_boundary_changes"]["removed_gaps"]) for d in successful)},
        "limits": ["All changes are measured on the fixed exposed sample; no model or quality inference.",
            "Fixed-b0 text comparison and rule projection are separate layers, not separate predictors.",
            "No atom source character offsets; legacy newline rendering is used explicitly.",
            "Exporter hardcaps are checked on actual exported text, not inherited from projector success.",
            "Whole-span tokens include specials; token sums summarize per-chunk counts, not a combined evidence-pack budget.",
            "Soft minima are goals, not hard guarantees; default merge decisions may remove native boundaries.",
            "Exporter refiner_epoch8 source literals do not prove learned prediction provenance."]}
    write(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--output", type=Path, default=STAGE / "projection-export-run-01")
    args = parser.parse_args()
    result = run(args.protocol_sha, args.output)
    print(json.dumps({"status": result["status"], "totals": result["totals"]}, sort_keys=True))


if __name__ == "__main__":
    main()
