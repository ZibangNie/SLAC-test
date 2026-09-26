"""Supplement the frozen reranker scientific audit with explicit metadata checks.

No model load, GPU/process probe, provider call or credential access. Historical
resource measurements are validated as recorded data, not independently rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import run_qasper_reranker_baseline as runner


SCHEMA = "slac-qasper-reranker-metadata-audit-v1"
PLAN_FILES = ("experiment_config.json", "plan_manifest.json", "pair_audit.jsonl", "length_audit.json")
RUN_FILES = (*runner.OUTPUT_FILES, "summary.json", "run_manifest.json")
FIXED_FIELDS = {
    "model_id": runner.MODEL_ID, "revision": runner.REVISION,
    "api_calls": 0, "test_payload_read": False, "training_performed": False,
    "answer_generation_performed": False, "all_packs_within_budget": True,
    "input_hashes_unchanged": True, "paid_api_cost_usd": "0",
    "raw_text_or_question_ids_in_summary": False, "limits": runner.LIMITS,
}


def exact(value, expected, name):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"reranker metadata field differs: {name}")


def nonnegative(value, name, *, integer=False):
    if ((type(value) is not int if integer else type(value) not in (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"invalid recorded nonnegative measurement: {name}")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def expected_padding(pair_audit):
    lengths = sorted((row["pair_tokens"] for row in pair_audit), reverse=True)
    if not lengths or any(type(length) is not int or not 0 < length <= runner.CONTRACT["max_pair_tokens"] for length in lengths):
        raise ValueError("invalid frozen pair lengths")
    microbatch = runner.CONTRACT["microbatch"]
    padded = [lengths[start] * len(lengths[start:start + microbatch]) for start in range(0, len(lengths), microbatch)]
    return {"padded_input_tokens": sum(padded), "max_padded_tokens_per_batch": max(padded),
            "max_microbatch": microbatch}


def validate_metadata(config, summary, records, pair_scores, pair_audit, prepared):
    """Pure supplement; the caller must first pass audit_saved_run."""
    for name, value in FIXED_FIELDS.items():
        exact(summary.get(name), value, name)
    counts = {
        "question_count": len({(row["doc_id"], row["question_id"]) for row in records}),
        "family_count": len({row["family_id"] for row in records}),
        "pair_count": len(pair_scores), "record_count": len(records),
    }
    for name, value in counts.items():
        exact(summary.get(name), value, name)
    if (counts["question_count"] != config["question_count"]
            or counts["pair_count"] != config["pair_count"]
            or counts["pair_count"] != len(pair_audit)
            or counts["record_count"] != counts["question_count"] * len(runner.CONTRACT["selection_k"])):
        raise ValueError("metadata denominator differs from frozen plan or pair audit")
    for row in records:
        exact(row.get("budget"), 1024, "record.budget")
        tokens, selected = row["actual_evidence_tokens"], row["selected_units"]
        nonnegative(tokens, "record.actual_evidence_tokens", integer=True)
        nonnegative(selected, "record.selected_units", integer=True)
        k = int(row["method"].rsplit("_k", 1)[1])
        if k not in runner.CONTRACT["selection_k"] or tokens > 1024 or selected > k:
            raise ValueError("derived whole-pack token or item budget violated")
    exact(summary["contract"], runner.CONTRACT, "contract")
    exact(summary["length_audit"], runner.length_summary(pair_audit), "length_audit")

    execution = summary.get("execution")
    expected_runtime = {"torch": str(runner.torch.__version__), "transformers": runner.transformers.__version__,
        "cuda_runtime": runner.torch.version.cuda, "device": runner.CONTRACT["device"], "dtype": runner.CONTRACT["dtype"]}
    if not isinstance(execution, dict) or set(execution) != set(expected_runtime) | {"gpu_name", "gpu_capability"}:
        raise ValueError("execution field inventory differs")
    for name, expected in expected_runtime.items():
        exact(execution.get(name), expected, "execution." + name)
    capability = execution["gpu_capability"]
    if (not isinstance(execution["gpu_name"], str) or not execution["gpu_name"].strip()
            or not isinstance(capability, list) or len(capability) != 2
            or any(type(value) is not int or value < 0 for value in capability)):
        raise ValueError("invalid recorded GPU identity")

    gate = runner.CONTRACT["gpu_admission"]
    samples = summary.get("gpu_admission_samples")
    if not isinstance(samples, list) or len(samples) != gate["samples"]:
        raise ValueError("GPU admission sample count differs")
    for sample in samples:
        if set(sample) != {"name", "free_mib", "utilization_percent"}:
            raise ValueError("GPU admission sample fields differ")
        if not isinstance(sample["name"], str) or sample["name"].strip() != execution["gpu_name"]:
            raise ValueError("GPU admission device differs from execution")
        nonnegative(sample["free_mib"], "gpu_admission.free_mib", integer=True)
        nonnegative(sample["utilization_percent"], "gpu_admission.utilization_percent", integer=True)
        if (sample["free_mib"] < gate["minimum_free_mib"]
                or sample["utilization_percent"] > gate["maximum_utilization_percent"]):
            raise ValueError("recorded GPU admission threshold violated")
    exact(gate["defer_if_process_running"], "FIFA18.exe", "FIFA process veto contract")
    # The original code checks this process before each sample, but does not save
    # the process query result. Never substitute a present-day process query.
    process_claim = {"frozen_gate_requires_no_running_fifa": True,
        "historical_process_snapshot_recorded": False,
        "historical_process_absence_independently_verified": False,
        "interpretation": "No-FIFA admission is inferred from the bound runner control flow; only memory/utilization samples were persisted."}

    compute = summary.get("compute")
    padding = expected_padding(pair_audit)
    if not isinstance(compute, dict) or set(compute) != set(padding) | {"peak_allocated_bytes", "peak_reserved_bytes"}:
        raise ValueError("compute measurement inventory differs")
    for name, expected in padding.items():
        exact(compute[name], expected, "compute." + name)
    for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        nonnegative(compute[name], "compute." + name, integer=True)
    if compute["peak_reserved_bytes"] < compute["peak_allocated_bytes"]:
        raise ValueError("recorded reserved GPU peak is smaller than allocated peak")
    nonnegative(summary.get("elapsed_seconds"), "elapsed_seconds")

    private = {str(query[name]) for query in prepared["queries"]
               for name in ("doc_id", "family_id", "question_id", "query")}
    if private.intersection(strings(summary)):
        raise ValueError("raw query identity or query text appears in public reranker summary")
    return {"fixed_fields_verified": sorted(FIXED_FIELDS), "derived_counts": counts,
        "derived_all_packs_within_budget": True, "derived_padding": padding,
        "execution_package_dtype_device_metadata_verified": True,
        "gpu_admission_samples": len(samples),
        "gpu_admission_minimum_free_mib": min(row["free_mib"] for row in samples),
        "gpu_admission_maximum_utilization_percent": max(row["utilization_percent"] for row in samples),
        "fifa_veto": process_claim, "public_summary_query_identity_scan_passed": True,
        "resource_measurements": "Recorded values have valid types/ranges and consistent padding; GPU peaks and elapsed time are not independently remeasured."}


def buffers_for(directory, expected):
    directory = Path(directory).resolve()
    if {path.name for path in directory.iterdir()} != set(expected):
        raise ValueError("frozen plan or completed run inventory differs")
    buffers = {name: (directory / name).read_bytes() for name in expected}
    return buffers, {str(directory / name): hashlib.sha256(value).hexdigest() for name, value in buffers.items()}


def audit(args):
    started = time.monotonic()
    plan, run, output = (Path(getattr(args, name)).resolve() for name in ("plan", "run", "output"))
    if output.exists():
        raise FileExistsError("metadata audit output already exists")
    if output.is_relative_to(plan) or output.is_relative_to(run):
        raise ValueError("metadata audit output must be outside immutable plan/run directories")
    plan_buffers, bindings = buffers_for(plan, PLAN_FILES)
    run_buffers, run_bindings = buffers_for(run, RUN_FILES)
    bindings.update(run_bindings)
    for path in (Path(__file__).resolve(), Path(runner.__file__).resolve()):
        bindings[str(path)] = runner.digest(path)
    verified = runner.audit_saved_run(plan, run, deadline=started + 600)
    expected_audit = {"status": "verified", "queries": runner.EXPECTED_QUERIES,
        "pairs": runner.EXPECTED_PAIRS, "records": runner.EXPECTED_QUERIES * len(runner.CONTRACT["selection_k"]),
        "all_pair_identities_and_encoding_hashes_match": True, "all_rankings_and_metrics_reproduced": True,
        "all_source_plan_output_hashes_unchanged": True, "model_inference_performed": False, "api_calls": 0}
    for name, expected in expected_audit.items():
        exact(verified.get(name), expected, "scientific_audit." + name)
    config = json.loads(plan_buffers["experiment_config.json"])
    summary = json.loads(run_buffers["summary.json"])
    for path, value in config["input_sha256"].items():
        resolved = str(Path(path).resolve())
        if resolved in bindings and bindings[resolved] != value:
            raise ValueError("conflicting frozen source binding")
        bindings[resolved] = value
    runner.verify_hashes(bindings)
    parse_rows = lambda value: [json.loads(line) for line in value.decode("utf-8").splitlines() if line.strip()]
    records = parse_rows(run_buffers["per_question.jsonl"])
    pair_scores = parse_rows(run_buffers["pair_scores.jsonl"])
    pair_audit = parse_rows(plan_buffers["pair_audit.jsonl"])
    prepared_path = Path(config["prepared_dir"]) / "prepared.json"
    raw_prepared = prepared_path.read_bytes()
    if hashlib.sha256(raw_prepared).hexdigest() != bindings[str(prepared_path.resolve())]:
        raise ValueError("prepared query identity source changed")
    details = validate_metadata(config, summary, records, pair_scores, pair_audit, json.loads(raw_prepared))
    runner.verify_hashes(bindings)
    # File hashes alone do not detect a newly inserted extra file.
    if buffers_for(plan, PLAN_FILES)[1] != {str(plan / name): bindings[str(plan / name)] for name in PLAN_FILES}:
        raise ValueError("plan changed during metadata audit")
    if buffers_for(run, RUN_FILES)[1] != run_bindings:
        raise ValueError("run changed during metadata audit")
    runner.deadline_check(started + 600)
    report = {"schema": SCHEMA, "status": "verified", "scientific_audit": verified,
        "metadata_audit": details, "plan_sha256": bindings[str(plan / "experiment_config.json")],
        "run_manifest_sha256": bindings[str(run / "run_manifest.json")],
        "input_binding_sha256": runner.object_hash(bindings), "all_source_plan_run_hashes_unchanged": True,
        "api_calls": 0, "key_read": False, "model_loaded": False, "gpu_or_process_probe_performed": False,
        "test_payload_read": False, "raw_text_or_question_ids_in_public_summary": False,
        "elapsed_seconds": time.monotonic() - started}
    output.mkdir(parents=True, exist_ok=False)
    runner.write(output / "source_binding.json", {"input_sha256": bindings})
    report["source_binding_file_sha256"] = runner.digest(output / "source_binding.json")
    runner.write(output / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "output"):
        parser.add_argument("--" + name, required=True)
    result = audit(parser.parse_args())
    print(json.dumps({"status": result["status"], "queries": result["scientific_audit"]["queries"],
                      "records": result["scientific_audit"]["records"], "api_calls": 0}))
