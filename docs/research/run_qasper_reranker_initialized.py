"""Explicitly initialize CUDA before the unchanged, sealed reranker runner.

This is a separately authorized correction of a diagnosed pre-model failure,
not an automatic retry. prepare never initializes CUDA; run is a single attempt.
PyTorch 2.9 documents cuda.init as initializing PyTorch CUDA state, whereas
is_available only reports availability. The 2.9.1 reset_peak_memory_stats Python
binding forwards an explicit device index without calling _lazy_init:
https://docs.pytorch.org/docs/2.9/generated/torch.cuda.init.html
https://github.com/pytorch/pytorch/blob/v2.9.1/torch/cuda/memory.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch.cuda._utils as cuda_utils
import run_qasper_reranker_baseline as runner


SCHEMA = "slac-qasper-reranker-initialized-launch-v1"
BASE_FILES = ("experiment_config.json", "plan_manifest.json", "pair_audit.jsonl", "length_audit.json")
PLAN_FILES = ("launch_config.json", "launch_manifest.json")
CONTRACT = {
    "correction": "explicit torch.cuda.init before the unchanged frozen runner.run",
    "authorization": "root explicitly authorized a correction after diagnosing the initial pre-model failure",
    "automatic_retries": 0, "maximum_runner_invocations": 1, "cpu_fallback": False,
    "outer_gpu_guard": "unchanged runner.ensure_gpu_ready before CUDA initialization",
    "inner_gpu_guard": "unchanged runner.run repeats its own guard before loading the model",
    "runner_max_seconds": 600,
    "stop_at_utc": "2026-09-27T01:00:00+00:00",
    "timing": "Initialization and launcher overhead are additional recorded time; the original runner keeps its own 600-second limit.",
    "deadline": "Check the absolute cutoff before admission, initialization, runner invocation and completion; the queue supervises process termination.",
}


def now():
    return datetime.now(timezone.utc)


def check_stop():
    if now() >= datetime.fromisoformat(CONTRACT["stop_at_utc"]):
        raise TimeoutError("initialized launcher reached the frozen overnight cutoff")


def snapshot(directory, names):
    directory = Path(directory).resolve()
    if {path.name for path in directory.iterdir()} != set(names):
        raise ValueError("immutable launch input inventory differs")
    buffers = {name: (directory / name).read_bytes() for name in names}
    return buffers, {str(directory / name): hashlib.sha256(raw).hexdigest() for name, raw in buffers.items()}


def sources():
    return [Path(path).resolve() for path in (
        __file__, runner.__file__, runner.torch.__file__, runner.torch.cuda.__file__,
        runner.torch.cuda.memory.__file__, cuda_utils.__file__)]


def merge(*mappings):
    result = {}
    for mapping in mappings:
        for key, value in mapping.items():
            path = str(Path(key).resolve())
            if path in result and result[path] != value:
                raise ValueError("conflicting launch source hashes")
            result[path] = value
    return result


def separate_paths(*paths):
    resolved = [Path(path).resolve() for path in paths]
    if any(left.is_relative_to(right) or right.is_relative_to(left)
           for i, left in enumerate(resolved) for right in resolved[i + 1:]):
        raise ValueError("base plan, failure, launch plan, run and receipt must be separate directories")


def diagnosed_failure(directory, stderr):
    buffers, bindings = snapshot(directory, ("failure.json",))
    failure = json.loads(buffers["failure.json"])
    expected = {"status": "failed", "stage": "run", "error_class": "RuntimeError",
        "api_calls": 0, "automatic_retries": 0, "cpu_fallback": False,
        "experimental_scores_available": False}
    for name, value in expected.items():
        if type(failure.get(name)) is not type(value) or failure[name] != value:
            raise ValueError("prior failure does not match the diagnosed pre-model attempt")
    elapsed = failure.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid prior failure timing")
    stderr = Path(stderr).resolve()
    raw = stderr.read_bytes()
    text = raw.decode("utf-8")
    if "torch.cuda.reset_peak_memory_stats(0)" not in text or "RuntimeError: Invalid device argument" not in text:
        raise ValueError("prior traceback does not identify the CUDA statistics initialization failure")
    bindings[str(stderr)] = hashlib.sha256(raw).hexdigest()
    return failure, bindings


def prepare(args):
    base, failed, output, run_output, receipt = [Path(getattr(args, name)).resolve()
        for name in ("base_plan", "failed_run", "output", "run_output", "receipt_output")]
    separate_paths(base, failed, output, run_output, receipt)
    if any(path.exists() for path in (output, run_output, receipt)):
        raise FileExistsError("new launch plan, run and receipt paths must be unused")
    _, base_hashes = snapshot(base, BASE_FILES)
    config = runner.load_plan(base)
    if (config["question_count"] != runner.EXPECTED_QUERIES or config["pair_count"] != runner.EXPECTED_PAIRS
            or runner.CONTRACT["max_seconds"] != CONTRACT["runner_max_seconds"]):
        raise ValueError("original frozen scientific scope differs")
    failure, failure_hashes = diagnosed_failure(failed, args.failed_stderr)
    inputs = merge(config["input_sha256"], base_hashes, failure_hashes,
                   {str(path): runner.digest(path) for path in sources()})
    runner.verify_hashes(inputs)
    launch = {"schema": SCHEMA, "status": "prepared_no_cuda_initialization",
        "created_at_utc": now().isoformat(), "contract": CONTRACT,
        "base_plan_dir": str(base), "base_plan_sha256": base_hashes[str(base / "experiment_config.json")],
        "failed_run_dir": str(failed), "failed_stderr": str(Path(args.failed_stderr).resolve()),
        "prior_failure_sha256": failure_hashes[str(failed / "failure.json")],
        "prior_failure_binding_sha256": runner.object_hash(failure_hashes),
        "prior_failed_attempt_seconds": failure["elapsed_seconds"],
        "run_output": str(run_output), "receipt_output": str(receipt), "input_sha256": inputs,
        "launcher_source_sha256": inputs[str(Path(__file__).resolve())],
        "runtime": {"torch": str(runner.torch.__version__), "transformers": runner.transformers.__version__,
                    "cuda_runtime": runner.torch.version.cuda},
        "api_calls": 0, "key_read": False, "model_loaded": False, "cuda_initialization_performed": False,
        "test_payload_read": False, "automatic_retries": 0}
    output.mkdir(parents=True, exist_ok=False)
    runner.write(output / "launch_config.json", launch)
    runner.write(output / "launch_manifest.json", {"schema": SCHEMA, "status": launch["status"],
        "launch_config_sha256": runner.digest(output / "launch_config.json")})
    return launch


def load_plan(directory):
    directory = Path(directory).resolve()
    raw, own = snapshot(directory, PLAN_FILES)
    config, seal = (json.loads(raw[name]) for name in PLAN_FILES)
    if (seal.get("schema") != SCHEMA or config.get("schema") != SCHEMA
            or seal.get("status") != "prepared_no_cuda_initialization"
            or config.get("status") != seal["status"]
            or config.get("contract") != CONTRACT
            or seal.get("launch_config_sha256") != own[str(directory / "launch_config.json")]):
        raise ValueError("initialized launch seal or contract differs")
    bindings = merge(config["input_sha256"], own)
    if config["launcher_source_sha256"] != bindings[str(Path(__file__).resolve())]:
        raise ValueError("launcher source identity differs")
    runner.verify_hashes(bindings)
    base, failed = Path(config["base_plan_dir"]), Path(config["failed_run_dir"])
    separate_paths(base, failed, directory, config["run_output"], config["receipt_output"])
    _, base_hashes = snapshot(base, BASE_FILES)
    if base_hashes[str(base / "experiment_config.json")] != config["base_plan_sha256"]:
        raise ValueError("base scientific plan identity differs")
    original = runner.load_plan(base)
    if original["question_count"] != runner.EXPECTED_QUERIES or original["pair_count"] != runner.EXPECTED_PAIRS:
        raise ValueError("base scientific scope differs")
    failure, failure_hashes = diagnosed_failure(failed, config["failed_stderr"])
    if (runner.object_hash(failure_hashes) != config["prior_failure_binding_sha256"]
            or failure_hashes[str(failed / "failure.json")] != config["prior_failure_sha256"]
            or failure["elapsed_seconds"] != config["prior_failed_attempt_seconds"]):
        raise ValueError("prior failure identity differs")
    if config["runtime"] != {"torch": str(runner.torch.__version__), "transformers": runner.transformers.__version__,
                             "cuda_runtime": runner.torch.version.cuda}:
        raise ValueError("launcher runtime changed after preparation")
    return config, bindings


def output_hashes(directory):
    directory = Path(directory)
    return {path.name: runner.digest(path) for path in sorted(directory.iterdir()) if path.is_file()} if directory.exists() else {}


def run(args):
    started = time.monotonic()
    directory = Path(args.plan).resolve()
    config, bindings = load_plan(directory)  # No GPU/process action before verified inputs.
    output, receipt = Path(config["run_output"]), Path(config["receipt_output"])
    if output.exists() or receipt.exists():
        raise FileExistsError("this explicit attempt already has an output or receipt; no retry")
    receipt.mkdir(parents=True, exist_ok=False)  # Atomic single-attempt claim.
    common = {"schema": SCHEMA, "launch_plan_sha256": bindings[str(directory / "launch_config.json")],
        "base_plan_sha256": config["base_plan_sha256"], "launcher_source_sha256": config["launcher_source_sha256"],
        "input_binding_sha256": runner.object_hash(bindings), "old_failed_run": config["failed_run_dir"],
        "old_failure_sha256": config["prior_failure_sha256"], "new_run": str(output),
        "authorization": CONTRACT["authorization"], "automatic_retries": 0,
        "api_calls": 0, "paid_api_cost_usd": "0", "key_read": False, "test_payload_read": False}
    runner.write(receipt / "started.json", {**common, "status": "explicit_attempt_started", "at_utc": now().isoformat()})
    report = {**common, "status": "failed", "failure_stage": "deadline_before_admission",
        "cuda_initialized_before": None, "cuda_initialized_after": None, "cuda_init_invoked": False,
        "outer_gpu_guard_samples": [], "outer_gpu_guard_seconds": 0.0, "cuda_init_seconds": 0.0,
        "runner_invocations": 0, "runner_elapsed_seconds": 0.0,
        "experimental_scores_available": False, "scientific_outputs_independently_audited": False}
    caught = None
    try:
        check_stop()
        report["failure_stage"] = "outer_gpu_admission"
        timing = time.monotonic()
        try:
            report["outer_gpu_guard_samples"] = runner.ensure_gpu_ready()
        finally:
            report["outer_gpu_guard_seconds"] = time.monotonic() - timing
        check_stop()
        report["failure_stage"] = "cuda_initialization"
        report["cuda_initialized_before"] = runner.torch.cuda.is_initialized()
        timing = time.monotonic()
        try:
            report["cuda_init_invoked"] = True
            runner.torch.cuda.init()
        finally:
            report["cuda_init_seconds"] = time.monotonic() - timing
            report["cuda_initialized_after"] = runner.torch.cuda.is_initialized()
        if report["cuda_initialized_after"] is not True:
            raise RuntimeError("explicit CUDA initialization did not initialize PyTorch CUDA state")
        check_stop()
        report["failure_stage"] = "unchanged_runner"
        report["runner_invocations"] = 1
        timing = time.monotonic()
        try:
            result = runner.run(SimpleNamespace(plan=config["base_plan_dir"], output=str(output)))
        finally:
            report["runner_elapsed_seconds"] = time.monotonic() - timing
        if (result.get("status") != "completed" or result.get("question_count") != runner.EXPECTED_QUERIES
                or result.get("pair_count") != runner.EXPECTED_PAIRS
                or result.get("record_count") != runner.EXPECTED_QUERIES * len(runner.CONTRACT["selection_k"])):
            raise ValueError("unchanged runner did not return the complete frozen scope")
        check_stop()
        report["failure_stage"] = "postrun_source_verification"
        runner.verify_hashes(bindings)
        snapshot(directory, PLAN_FILES)
        snapshot(config["base_plan_dir"], BASE_FILES)
        snapshot(config["failed_run_dir"], ("failure.json",))
        report.update(status="completed", failure_stage=None, experimental_scores_available=True,
                      original_runner_reported_seconds=result["elapsed_seconds"])
    except BaseException as exc:
        caught = exc
        report["error_class"] = type(exc).__name__
    finally:
        try:
            runner.verify_hashes(bindings)
            report["all_bound_inputs_unchanged"] = True
        except Exception as exc:
            report["all_bound_inputs_unchanged"] = False
            if caught is None:
                caught = exc
                report.update(status="failed", failure_stage="final_source_verification",
                              experimental_scores_available=False, error_class=type(exc).__name__)
        report["total_launcher_seconds"] = time.monotonic() - started
        report["initialization_and_launcher_overhead_seconds"] = report["total_launcher_seconds"] - report["runner_elapsed_seconds"]
        report["prior_failed_attempt_seconds"] = config["prior_failed_attempt_seconds"]
        report["run_output_sha256"] = output_hashes(output)
        report["timing_limit"] = CONTRACT["timing"]
        runner.write(receipt / "receipt.json", report)
        runner.write(receipt / "receipt_manifest.json", {"schema": SCHEMA, "status": report["status"],
            "started_sha256": runner.digest(receipt / "started.json"),
            "receipt_sha256": runner.digest(receipt / "receipt.json"),
            "launch_plan_sha256": common["launch_plan_sha256"]})
    if caught is not None:
        raise caught
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare")
    for name in ("base-plan", "failed-run", "failed-stderr", "run-output", "receipt-output", "output"):
        freeze.add_argument("--" + name, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--plan", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = {"prepare": prepare, "run": run}[args.command](args)
    print(json.dumps({"status": result["status"], "api_calls": 0, "automatic_retries": 0}))
