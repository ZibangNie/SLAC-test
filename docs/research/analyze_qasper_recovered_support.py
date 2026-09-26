"""Read-only full-denominator statistics after explicitly amended support recovery.

Uses the original pre-result comparison specification without changing a method,
metric, denominator, k, resample, or comparison direction. No network or key read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_qasper_extended_results as stats
import run_qasper_relation_pilot as pilot
from run_qasper_evidence_baselines import digest


SCHEMA = "slac-recovered-support-statistics-v1"
LIMITS = [
    "All 77 questions and 24 families are exposed development data, not independent confirmation.",
    "One explicitly documented repeat follows the old uncertain timeout; the old attempt and unknown cost remain.",
    "Support-only fixed-pool selection does not test shared document relations or complete SLAC.",
    "Every original same-k pair, metric and direction is retained; no best k or positive-only subset is chosen.",
    "Family bootstrap intervals use the original 10000 PCG64 draws and have no multiplicity correction.",
    "Equal evidence budgets do not imply equal actual lengths; token differences accompany quality differences.",
    "JEV reported scores are unnormalized and are not calibrated probabilities.",
    "This analysis performs no answer generation, model inference, training, new QA exposure or API request.",
]


def summarize(prepared, records):
    questions = stats.questions_from(prepared)
    groups, draws = stats.family_resamples(questions)
    if len(questions) != 77 or len(groups) != 24:
        raise ValueError("requires the frozen complete 77-question, 24-family denominator")
    support, local = stats.summarize_domain(
        "support", records, stats.SUPPORT_METHODS, stats.SUPPORT_METRICS,
        stats.SUPPORT_PAIRS, questions, groups, draws,
    )
    if len(records) != 1155 or support["method_count"] != 15 or support["comparison_count"] != 30:
        raise ValueError("complete original support comparison coverage is required")
    result = {
        "schema": SCHEMA, "status": "completed", "question_count": len(questions),
        "family_count": len(groups), "record_count": len(records),
        "questions_per_family_histogram": dict(Counter(len(group) for group in groups)),
        "specification": stats.SPECIFICATION,
        "specification_sha256": pilot.client.object_hash(stats.SPECIFICATION),
        "specification_scope_executed": "support_only; primary_answer_k3 has not been analyzed here",
        "bootstrap_numpy_version": np.__version__,
        "bootstrap_counts_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
        "support": support, "reported_interval_count": 240,
        "api_calls": 0, "key_read": False, "test_payload_read": False,
        "answer_generation_performed": False, "independent_confirmation": False,
        "significance_claimed": False, "multiple_comparison_control_performed": False,
        "limits": LIMITS,
    }
    return result, local


def analyze(args, *, verifier=None):
    plan, run, output = (Path(getattr(args, name)).resolve() for name in ("plan", "run", "output"))
    if output.exists():
        raise FileExistsError("analysis output already exists")
    before = {"plan": stats.snapshot(plan), "run": stats.snapshot(run)}
    bindings = {p: value for mapping in before.values() for p, value in mapping.items()}
    for module in (stats, pilot):
        bindings[str(Path(module.__file__).resolve())] = digest(module.__file__)
    bindings[str(Path(__file__).resolve())] = digest(__file__)
    if verifier is None:
        import run_qasper_primary_support_recovery as recovery
        bindings[str(Path(recovery.__file__).resolve())] = digest(recovery.__file__)
        verifier = recovery.verify_completed_run
    config, prepared, _, records, summary = verifier(plan, run)
    if summary.get("status") != "completed" or summary.get("all_results_available") is not True:
        raise ValueError("requires verified complete recovery; no partial quality analysis")
    for path, value in config["input_sha256"].items():
        if path in bindings and bindings[path] != value:
            raise ValueError("conflicting bound source identities")
        bindings[path] = value
    result, local = summarize(prepared, records)
    if stats.snapshot(plan) != before["plan"] or stats.snapshot(run) != before["run"]:
        raise ValueError("completed source directory changed during analysis")
    pilot.verify_hashes(bindings)
    result.update(
        recovery_plan_sha256=digest(plan / "experiment_config.json"),
        recovery_summary_sha256=digest(run / "summary.json"),
        input_hashes_unchanged=True,
        input_binding_sha256=pilot.client.object_hash(bindings),
        input_sha256=[{"path_sha256": hashlib.sha256(path.encode()).hexdigest(), "content_sha256": value}
                      for path, value in sorted(bindings.items())],
    )
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_rows(output / "paired_per_question.jsonl", local)
    pilot.write_json(output / "source_binding.json", {
        "input_sha256": bindings, "plan": str(plan), "run": str(run),
        "old_unknown_attempt_retained": True,
    })
    result["local_output_files_sha256"] = {
        name: digest(output / name) for name in ("paired_per_question.jsonl", "source_binding.json")
    }
    pilot.write_json(output / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "output"):
        parser.add_argument("--" + name, required=True)
    result = analyze(parser.parse_args())
    print(json.dumps({key: result[key] for key in (
        "status", "question_count", "family_count", "record_count", "reported_interval_count", "api_calls",
    )}))


if __name__ == "__main__":
    main()
