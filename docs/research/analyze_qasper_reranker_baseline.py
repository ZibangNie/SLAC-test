"""Freeze and run complete-family paired reranker/dense development analysis.

freeze binds the specification before reranker inference. analyze first replays
the complete saved reranker outputs, then recomputes dense selection from the
same prepared candidates. No model load, GPU query, key read or provider call.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import analyze_qasper_extended_results as bootstrap
import run_qasper_extended_development as stage
import run_qasper_reranker_baseline as reranker


SCHEMA = "slac-qasper-reranker-paired-analysis-v1"
METRICS = ("official_evidence_f1", "reference_evidence_recall",
           "official_text_only_evidence_f1", "actual_evidence_tokens")
SIZES = (1, 2, 3)
METHODS = tuple(method for k in SIZES for method in (f"dense_k{k}", f"bge_reranker_v2_m3_k{k}"))
COMPARISONS = tuple({"k": k, "role": "primary" if k == 3 else "sensitivity",
                    "plus": f"bge_reranker_v2_m3_k{k}", "minus": f"dense_k{k}"} for k in (3, 1, 2))
SPECIFICATION = {
    "schema": SCHEMA, "primary_k": 3, "sensitivity_k": [1, 2], "retained_k": list(SIZES),
    "metrics": list(METRICS), "methods": list(METHODS), "comparisons": list(COMPARISONS),
    "scope": "all 77 prepared development questions in 24 families and all 1214 support pairs",
    "selection_budget_bge_tokens": 1024, "deduplication": "exact native text, first ranked fitting unit",
    "dense_ranking": "original frozen prepared ranked_ids; same candidate_ids as the reranker",
    "reranker_scores": "fixed revision raw classification logits; all saved outputs independently audited",
    "bootstrap_seed": 20260927, "bootstrap_replicates": 10000,
    "random_generator": "numpy.random.PCG64", "resampling_unit": "whole family",
    "counts_sampler": "multinomial(F, uniform family probabilities)",
    "question_weighted": "sampled question delta sum divided by sampled question count",
    "family_balanced": "mean of sampled within-family question-mean deltas",
    "interval": "two-sided percentile 95%, linear quantiles at 0.025 and 0.975",
    "same_resamples_across_all_k_and_metrics": True, "tie_absolute_tolerance": 1e-12,
    "p_values_computed": False, "multiple_comparison_adjustment": "none",
    "best_k_selected": False, "positive_only_selection": False, "query_subset_selection": False,
    "analysis_timing": "specification frozen before this reranker inference; other exposed development results already existed",
}
LIMITS = [
    "All 77 questions are exposed development data, not independent confirmation.",
    "Fixed-pool, given-document selection does not evaluate corpus retrieval, answers or a shared-relation architecture.",
    "The primary k=3 and both k=1/2 sensitivities retain every metric and every sign of the result.",
    "Percentile intervals are descriptive, unadjusted across comparisons, and do not establish significance or future performance.",
    "Whole-family resampling preserves all questions within each sampled family; larger families still affect question-weighted means more.",
    "Equal evidence caps do not imply equal actual evidence lengths; paired token differences are reported separately.",
    "Reranker logits are not calibrated probabilities; no learned threshold or winner selection is performed.",
    "Comparisons reuse one deterministic reranker run; bootstrap intervals do not include device/precision or model-run variability.",
]
RUN_FILES = (*reranker.OUTPUT_FILES, "summary.json", "run_manifest.json")
PLAN_FILES = ("experiment_config.json", "plan_manifest.json", "pair_audit.jsonl", "length_audit.json")


def code_paths():
    return {Path(__file__).resolve(), Path(reranker.__file__).resolve(), Path(stage.__file__).resolve(),
            Path(bootstrap.__file__).resolve(), Path(bootstrap.pilot_analysis.__file__).resolve()}


def hashes_for(paths):
    return {str(Path(path).resolve()): reranker.digest(path) for path in paths}


def merge_bindings(*inventories):
    result = {}
    for inventory in inventories:
        for path, value in inventory.items():
            normalized = str(Path(path).resolve())
            if normalized in result and result[normalized] != value:
                raise ValueError("same source has conflicting frozen hashes")
            result[normalized] = value
    return result


def validate_implementation_contract():
    if (bootstrap.BOOTSTRAP_SEED != SPECIFICATION["bootstrap_seed"]
            or bootstrap.BOOTSTRAP_REPLICATES != SPECIFICATION["bootstrap_replicates"]
            or bootstrap.TIE_TOLERANCE != SPECIFICATION["tie_absolute_tolerance"]
            or stage.SELECTOR["sizes"] != list(SIZES) or stage.SELECTOR["budget"] != 1024
            or reranker.CONTRACT["selection_k"] != list(SIZES)
            or reranker.CONTRACT["evidence_budget_bge_tokens"] != 1024):
        raise ValueError("shared selection or statistical implementation contract changed")


def freeze(args):
    output, plan_dir = Path(args.output).resolve(), Path(args.reranker_plan).resolve()
    if output.exists():
        raise FileExistsError("analysis specification output already exists")
    validate_implementation_contract()
    initial = hashes_for(plan_dir / name for name in PLAN_FILES)
    plan = reranker.load_plan(plan_dir)
    if plan["question_count"] != 77 or plan["pair_count"] != 1214:
        raise ValueError("analysis requires the full frozen reranker scope")
    inputs = merge_bindings(plan["input_sha256"], initial, hashes_for(code_paths()))
    reranker.verify_hashes(inputs)
    config = {"schema": SCHEMA, "status": "analysis_specification_frozen_before_inference",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "specification": SPECIFICATION,
        "specification_sha256": reranker.object_hash(SPECIFICATION), "reranker_plan_dir": str(plan_dir),
        "reranker_plan_sha256": initial[str(plan_dir / "experiment_config.json")],
        "input_sha256": inputs, "model_inference_performed": False, "api_calls": 0,
        "test_payload_read": False, "limits": LIMITS}
    output.mkdir(parents=True, exist_ok=False)
    reranker.write(output / "analysis_config.json", config)
    reranker.write(output / "analysis_manifest.json", {"schema": SCHEMA,
        "analysis_config_sha256": reranker.digest(output / "analysis_config.json"),
        "status": "analysis_specification_frozen_before_inference"})
    return config


def load_specification(directory):
    directory = Path(directory).resolve()
    initial = hashes_for(directory / name for name in ("analysis_config.json", "analysis_manifest.json"))
    seal, config = reranker.read(directory / "analysis_manifest.json"), reranker.read(directory / "analysis_config.json")
    if seal.get("analysis_config_sha256") != initial[str(directory / "analysis_config.json")]:
        raise ValueError("analysis specification seal differs")
    validate_implementation_contract()
    if (config.get("schema") != SCHEMA or config.get("specification") != SPECIFICATION
            or config.get("specification_sha256") != reranker.object_hash(SPECIFICATION)
            or config.get("status") != "analysis_specification_frozen_before_inference"):
        raise ValueError("frozen paired analysis specification differs")
    reranker.verify_hashes(config["input_sha256"])
    reranker.verify_hashes(initial)
    return config, merge_bindings(config["input_sha256"], initial)


def statistics_for(prepared, reranker_records, dense_records):
    """Generic small-fixture-testable statistics; analyze separately locks 77/24."""
    validate_implementation_contract()
    questions = bootstrap.questions_from(prepared)
    tables = bootstrap.validated_tables([*dense_records, *reranker_records], METHODS, METRICS, questions)
    groups, draws = bootstrap.family_resamples(questions)
    comparisons, private = [], []
    for comparison in COMPARISONS:
        plus, minus = comparison["plus"], comparison["minus"]
        values = {metric: [tables[plus][key][metric] - tables[minus][key][metric] for key in questions]
                  for metric in METRICS}
        comparisons.append({**comparison, "metrics": {
            metric: bootstrap.clustered_delta(values[metric], groups, draws, metric) for metric in METRICS}})
        for index, key in enumerate(questions):
            private.append({**dict(zip(bootstrap.IDENTITY, key)), **comparison,
                            "deltas": {metric: values[metric][index] for metric in METRICS}})
    means = []
    for method in METHODS:
        metrics = {}
        for metric in METRICS:
            values = np.asarray([tables[method][key][metric] for key in questions], dtype=np.float64)
            metrics[metric] = {"question_weighted": float(values.mean()),
                               "family_balanced": float(np.mean([values[group].mean() for group in groups]))}
        means.append({"method": method, "metrics": metrics})
    return {"question_count": len(questions), "family_count": len(groups),
        "questions_per_family_histogram": dict(Counter(len(group) for group in groups)),
        "method_means": means, "paired_comparisons": comparisons,
        "numpy_version": str(np.__version__), "shared_resamples_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
        "primary_k": 3, "sensitivity_k": [1, 2], "best_k_selected": False}, private


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze(args):
    started = time.monotonic()
    deadline = started + 600
    output, analysis_dir, run_dir = (Path(getattr(args, name)).resolve()
                                      for name in ("output", "analysis_plan", "run"))
    if output.exists():
        raise FileExistsError("paired analysis output already exists")
    config, bindings = load_specification(analysis_dir)
    plan_dir = Path(config["reranker_plan_dir"])
    if {path.name for path in run_dir.iterdir()} != set(RUN_FILES):
        raise ValueError("requires the exact complete reranker output inventory")
    run_bindings = hashes_for(run_dir / name for name in RUN_FILES)
    bindings = merge_bindings(bindings, run_bindings)
    verified = reranker.audit_saved_run(plan_dir, run_dir, deadline=deadline)
    if (verified.get("status") != "verified" or verified.get("queries") != 77
            or verified.get("pairs") != 1214 or verified.get("records") != 231
            or verified.get("all_pair_identities_and_encoding_hashes_match") is not True
            or verified.get("all_rankings_and_metrics_reproduced") is not True
            or verified.get("all_source_plan_output_hashes_unchanged") is not True):
        raise ValueError("requires complete audited 77-query reranker output")
    reranker.verify_hashes(bindings)  # No read after audit can silently change a saved input.
    plan = reranker.load_plan(plan_dir)
    if reranker.digest(plan_dir / "experiment_config.json") != config["reranker_plan_sha256"]:
        raise ValueError("analysis references a different reranker plan")
    prepared, manifest, documents = reranker.preparation.load_prepared(plan["prepared_dir"])
    questions = bootstrap.questions_from(prepared)
    if len(questions) != 77 or len({key[0] for key in questions}) != 24:
        raise ValueError("primary analysis requires all 77 queries and 24 families")
    annotations = reranker.selected_gold(manifest["source_paths"]["sidecar"], prepared)
    tokenizer = reranker.AutoTokenizer.from_pretrained(plan["bge_tokenizer"], local_files_only=True, trust_remote_code=False)
    dense = stage.baseline_records(prepared, documents, annotations, tokenizer)
    records = read_rows(run_dir / "per_question.jsonl")
    statistics, private = statistics_for(prepared, records, dense)
    if len(dense) != 231 or len(records) != 231 or len(private) != 231:
        raise ValueError("complete three-k metric denominator differs")
    reranker.verify_hashes(bindings)
    reranker.deadline_check(deadline)
    report = {"schema": SCHEMA, "status": "completed", "specification": SPECIFICATION,
        "specification_sha256": config["specification_sha256"], "statistics": statistics,
        "audit": {"reranker_saved_outputs": verified, "dense_candidate_ranking": "original frozen prepared ranked_ids",
            "same_prepared_candidates_and_queries": True, "all_77_queries_retained": True,
            "all_three_k_and_four_metrics_retained": True, "all_input_hashes_unchanged": True},
        "analysis_specification_config_sha256": bindings[str(analysis_dir / "analysis_config.json")],
        "reranker_plan_sha256": config["reranker_plan_sha256"],
        "reranker_run_manifest_sha256": run_bindings[str(run_dir / "run_manifest.json")],
        "input_binding_sha256": reranker.object_hash(bindings), "api_calls": 0, "key_read": False,
        "test_payload_read": False, "model_inference_performed": False, "paid_outputs_read": False,
        "independent_confirmation": False, "significance_claimed": False,
        "raw_text_or_question_ids_in_public_summary": False, "limits": LIMITS,
        "elapsed_seconds": time.monotonic() - started}
    output.mkdir(parents=True, exist_ok=False)
    reranker.write(output / "source_binding.json", {"input_sha256": bindings})
    reranker.write_rows(output / "dense_per_question.jsonl", dense)
    reranker.write_rows(output / "paired_per_question.jsonl", private)
    report["local_output_sha256"] = {name: reranker.digest(output / name)
        for name in ("source_binding.json", "dense_per_question.jsonl", "paired_per_question.jsonl")}
    reranker.write(output / "public_aggregate.json", report)
    reranker.write(output / "analysis_output_manifest.json", {"status": "completed",
        "public_aggregate_sha256": reranker.digest(output / "public_aggregate.json"),
        "local_output_sha256": report["local_output_sha256"]})
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    for name in ("reranker-plan", "output"):
        prepare.add_argument("--" + name, required=True)
    execute = commands.add_parser("analyze")
    for name in ("analysis-plan", "run", "output"):
        execute.add_argument("--" + name, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = freeze(arguments) if arguments.command == "freeze" else analyze(arguments)
    print(json.dumps({"status": result["status"], "api_calls": 0,
        "questions": result.get("statistics", {}).get("question_count"), "primary_k": SPECIFICATION["primary_k"]}))
