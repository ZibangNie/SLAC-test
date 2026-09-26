"""Post-hoc, complete-family descriptive intervals for the frozen corpus bridge.

This reads only completed offline artifacts and their existing validation
sources. It never reads paid execution directories or makes model requests.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import analyze_qasper_extended_results as bootstrap
import run_qasper_corpus_bridge as bridge


METRICS = (
    "source_qualified_evidence_f1", "source_qualified_evidence_recall",
    "candidate_source_qualified_evidence_recall", "official_evidence_f1",
    "actual_evidence_tokens", "source_doc_hit_at_1_units",
    "source_doc_hit_at_5_units", "source_doc_hit_at_8_units",
)
EXPECTED_FILES = {"experiment_config.json", "corpus_unit_index.json", "per_question.jsonl",
                  "public_aggregate.json", "summary.json"}
SPECIFICATION = {
    "comparison": "corpus_32 minus given_document", "metrics": list(METRICS),
    "bootstrap_seed": 20260927, "bootstrap_replicates": 10000,
    "random_generator": "numpy.random.PCG64", "resampling_unit": "whole family",
    "counts_sampler": "multinomial(F, uniform family probabilities)",
    "interval": "two-sided percentile 95%, linear quantiles at 0.025 and 0.975",
    "question_weighted": "sampled question delta sum divided by sampled question count",
    "family_balanced": "mean of sampled family question-mean deltas",
    "shared_resamples_across_all_metrics": True,
    "analysis_timing": "post-hoc descriptive analysis after aggregate bridge results were visible",
    "p_values_computed": False, "multiple_comparison_adjustment": "none",
    "positive_only_selection": False, "query_subset_selection": False,
}


def saved_artifacts(directory):
    """Hash and parse each exact artifact from the same bytes."""
    directory = Path(directory).resolve()
    if {path.name for path in directory.iterdir()} != EXPECTED_FILES:
        raise ValueError("completed bridge artifact inventory differs")
    raw = {name: (directory / name).read_bytes() for name in EXPECTED_FILES}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    parsed = {name: json.loads(value) for name, value in raw.items() if name.endswith(".json")}
    summary = parsed["summary.json"]
    if summary.get("output_sha256") != {name: value for name, value in hashes.items() if name != "summary.json"}:
        raise ValueError("completed bridge output hash mismatch")
    public = parsed["public_aggregate.json"]
    if public != {key: value for key, value in summary.items() if key not in {"input_sha256", "output_sha256"}}:
        raise ValueError("public aggregate differs from completed summary")
    records = [json.loads(line) for line in raw["per_question.jsonl"].decode("utf-8").splitlines() if line.strip()]
    return parsed, records, {str(directory / name): value for name, value in hashes.items()}


def audit_run(directory, *, deadline=math.inf):
    parsed, records, hashes = saved_artifacts(directory)
    config, public, summary = (parsed[name] for name in ("experiment_config.json", "public_aggregate.json", "summary.json"))
    if (config.get("schema") != bridge.SCHEMA or config.get("config") != bridge.CONFIG
            or summary.get("input_sha256") != config.get("input_sha256")
            or public.get("input_binding_sha256") != bridge.stable_hash(config["input_sha256"])):
        raise ValueError("bridge configuration or source binding differs")
    bridge.pilot.verify_hashes(config["input_sha256"])
    torch.set_num_threads(1)
    source = bridge.load_verified_inputs(config["prepared_dir"], config["dense_dir"], deadline=deadline)
    if source["input_sha256"] != config["input_sha256"]:
        raise ValueError("current verified source inventory differs from execution")
    expected_index = [{"global_index": i, "doc_id": doc, "unit_id": uid} for i, (doc, uid) in enumerate(source["keys"])]
    if parsed["corpus_unit_index.json"] != expected_index:
        raise ValueError("saved global/native identity mapping differs")
    scores = {}
    for q in source["prepared"]["queries"]:
        bridge.check_time(deadline)
        key = (q["doc_id"], q["question_id"])
        scores[key] = (source["candidate_vectors"] @ source["query_vectors"][source["query_positions"][key]]).tolist()
    replay = bridge.validate_within_document_replay(source["prepared"], source["documents"], source["units"],
        source["keys"], source["positions"], scores, source["rankings"], source["tokenizer"], deadline)
    rebuilt = []
    for q in source["prepared"]["queries"]:
        key = (q["doc_id"], q["question_id"])
        rebuilt.extend(bridge.evaluate_query(q, source, scores[key], replay[key], deadline=deadline))
    timing_fields = {"retrieval_wall_seconds", "packing_wall_seconds"}
    strip_time = lambda row: {key: value for key, value in row.items() if key not in timing_fields}
    if [strip_time(row) for row in records] != [strip_time(row) for row in rebuilt]:
        raise ValueError("saved scientific records differ from exact selection/metric replay")
    for row in records:
        if any(type(row[field]) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0 for field in timing_fields):
            raise ValueError("invalid recorded per-question duration")
    tables = bridge.summarize(records, source["prepared"])
    empty = [bridge.evidence_metrics([], source["qa_by_key"][(q["doc_id"], q["question_id"])]["answer_annotations"])
             for q in source["prepared"]["queries"]]
    headers = Counter(unit.unit_id for unit in source["units"])
    expected_public = {
        "schema": bridge.SCHEMA, "status": "completed", "config": bridge.CONFIG,
        "question_count": 77, "family_count": 24, "corpus_document_count": 32, "corpus_unit_count": 1850,
        "api_calls": 0, "test_payload_read": False, "model_loaded": False, "training_performed": False,
        "answer_generation_performed": False, "independent_confirmation": False,
        "gold_used_for_ranking_or_selection": False, "jev_labels_reused": False,
        "within_document_replay": {"questions": len(replay), "ranking_candidate_render_token_selection_score_parity": True},
        "all_packs_within_budget": all(row["actual_evidence_tokens"] <= 1024 and row["selected_units"] <= 3 for row in records),
        "global_rendered_id_collisions": sum(value - 1 for value in headers.values()),
        "global_identity": "separate (doc_id,unit_id) mapping; original rendered unit IDs retained",
        "empty_pack_reference": {"questions": 77,
            "official_evidence_f1_question_macro": statistics.mean(row["evidence_f1"] for row in empty),
            "reference_evidence_recall_question_macro": statistics.mean(row["evidence_recall"] for row in empty),
            "questions_with_any_empty_reference": sum(row["evidence_f1"] == 1 for row in empty)},
        "input_binding_sha256": bridge.stable_hash(source["input_sha256"]),
        "script_sha256": bridge.digest(bridge.__file__), "execution": public.get("execution"),
        **tables, "limits": bridge.LIMITS,
    }
    if public != expected_public:
        raise ValueError("scientific summary metadata differs from replay")
    execution = public["execution"]
    expected_timing_fields = {"dense_scoring_wall_seconds", "total_wall_seconds", "total_process_cpu_seconds"}
    expected_runtime = {"torch": str(torch.__version__), "device": "cpu", "dtype": "float32", "threads": 1,
        "timing_scope": "Full-corpus vector scores computed once and shared; arm timings measure scope filtering/ranking/expansion and packing, not standalone deployment latency."}
    if ({key: value for key, value in execution.items() if key not in expected_timing_fields} != expected_runtime
            or any(type(execution[field]) not in (int, float) or not math.isfinite(execution[field]) or execution[field] < 0
                   for field in expected_timing_fields)
            or not 0 < execution["total_wall_seconds"] <= config["max_seconds"] <= 600):
        raise ValueError("execution metadata violates recorded runtime contract")
    hashes.update(source["input_sha256"])
    bridge.pilot.verify_hashes(hashes)
    return source, records, public, hashes


def descriptive_statistics(prepared, records):
    if bootstrap.BOOTSTRAP_SEED != 20260927 or bootstrap.BOOTSTRAP_REPLICATES != 10000:
        raise ValueError("shared bootstrap contract changed")
    questions = bootstrap.questions_from(prepared)
    tables = bootstrap.validated_tables(records, bridge.METHODS, METRICS, questions)
    groups, draws = bootstrap.family_resamples(questions)
    comparisons, private = {}, []
    for metric in METRICS:
        values = [tables["corpus_32"][key][metric] - tables["given_document"][key][metric] for key in questions]
        comparisons[metric] = bootstrap.clustered_delta(values, groups, draws, metric)
    for key in questions:
        private.append({**dict(zip(bootstrap.IDENTITY, key)), "deltas": {
            metric: tables["corpus_32"][key][metric] - tables["given_document"][key][metric] for metric in METRICS}})
    means = {}
    for method in bridge.METHODS:
        means[method] = {}
        for metric in METRICS:
            values = np.array([tables[method][key][metric] for key in questions], dtype=np.float64)
            means[method][metric] = {"question_weighted": float(values.mean()),
                "family_balanced": float(np.mean([values[group].mean() for group in groups]))}
    return {"specification": SPECIFICATION, "question_count": len(questions), "family_count": len(groups),
            "questions_per_family_histogram": dict(Counter(len(group) for group in groups)),
            "numpy_version": str(np.__version__), "shared_draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
            "method_means": means, "paired_delta_corpus_minus_given": comparisons}, private


def structural_diagnostics(records, source):
    result = {}
    for method in bridge.METHODS:
        rows = [row for row in records if row["method"] == method]
        selected_kinds = Counter(source["units"][index].kind for row in rows for index in row["selected_global_indices"])
        result[method] = {
            "questions": len(rows), "source_doc_hit_counts": {
                str(k): sum(row[f"source_doc_hit_at_{k}_units"] for row in rows) for k in (1, 5, 8)},
            "official_vs_source_qualified_f1_different_questions": sum(row["official_evidence_f1"] != row["source_qualified_evidence_f1"] for row in rows),
            "candidate_string_vs_source_recall_different_questions": sum(row["candidate_official_string_evidence_recall"] != row["candidate_source_qualified_evidence_recall"] for row in rows),
            "duplicate_header_pack_count": sum(row["duplicate_rendered_headers_in_pack"] > 0 for row in rows),
            "candidate_cross_document_duplicate_text_question_count": sum(row["cross_document_duplicate_text_groups"] > 0 for row in rows),
            "candidate_cross_document_duplicate_text_group_count": sum(row["cross_document_duplicate_text_groups"] for row in rows),
            "same_pack_as_seed_only_count": sum(row["same_pack_as_seed_only"] for row in rows),
            "same_pack_as_full_ranking_count": sum(row["same_pack_as_dense_full_ranking"] for row in rows),
            "selected_native_unit_kind_counts": dict(selected_kinds),
            "actual_token_min_max": [min(row["actual_evidence_tokens"] for row in rows), max(row["actual_evidence_tokens"] for row in rows)],
            "empty_pack_count": sum(row["empty_pack"] for row in rows),
        }
    return result


def analyze(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("analysis output already exists")
    source, records, public, hashes = audit_run(args.run, deadline=time.monotonic() + 300)
    statistics_result, private = descriptive_statistics(source["prepared"], records)
    for path in (Path(__file__).resolve(), Path(bootstrap.__file__).resolve()):
        hashes[str(path)] = bridge.digest(path)
    bridge.pilot.verify_hashes(hashes)
    report = {"schema": "slac-qasper-corpus-bridge-analysis-v1", "status": "completed",
        "bridge_public_aggregate": public, "statistics": statistics_result,
        "diagnostics": structural_diagnostics(records, source),
        "audit": {"exact_artifact_inventory": True, "input_hashes_verified": True,
            "full_scientific_records_and_summary_replayed": True,
            "recorded_timings": "hash-bound and range-validated; historical wall-clock measurements are not re-derived"},
        "api_calls": 0, "key_read": False, "test_payload_read": False,
        "independent_confirmation": False, "significance_claimed": False,
        "input_binding_sha256": bridge.stable_hash(hashes),
        "input_sha256": [{"path_sha256": hashlib.sha256(path.encode()).hexdigest(), "content_sha256": value}
                         for path, value in sorted(hashes.items())],
        "task_construction": {
            "original_qasper": "Questions are follow-ups to a particular paper's title and abstract, with answers sought in that paper.",
            "primary_source": "https://aclanthology.org/2021.naacl-main.365/",
            "corpus_bridge": "Query-only search over 32 papers removes original paper context and is a new stress diagnostic, not the standard Qasper task.",
            "query_context_dependence": "Questions may rely on the given paper; this analysis does not label or select a self-contained subset.",
            "future_title_policy": "Adding a known source title requires an explicit common protocol and disclosure of source-document localization information."},
        "limits": [
            "Post-hoc descriptive bootstrap on all 77 exposed development questions; no independent or multiplicity-adjusted inference.",
            "All eight requested metrics are retained, with the same 10000 PCG64 family draws, irrespective of delta sign.",
            "Losses jointly reflect corpus ambiguity, source-paper context removal and retrieval behavior; they do not isolate an architecture defect.",
            "No JEV judgment, SLAC refiner/tree/reranker, or answer-generation effect is estimated.",
            "Raw string and source-qualified F1 equality in this run does not remove the general risk of matching another paper's identical heading.",
            "Repeated rendered unit IDs remain ambiguous across papers; current packs must not be passed directly to a generator.",
        ]}
    output.mkdir(parents=True, exist_ok=False)
    bridge.pilot.write_json(output / "source_binding.json", {"input_sha256": hashes})
    with (output / "paired_per_question.jsonl").open("x", encoding="utf-8") as stream:
        for row in private:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["local_output_files_sha256"] = {name: bridge.digest(output / name) for name in ("source_binding.json", "paired_per_question.jsonl")}
    bridge.pilot.write_json(output / "analysis.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    result = analyze(parser.parse_args())
    print(json.dumps({"status": result["status"], "questions": result["statistics"]["question_count"],
                      "families": result["statistics"]["family_count"], "api_calls": 0}))
