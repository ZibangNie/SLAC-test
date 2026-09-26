"""Pre-specified descriptive comparisons for complete extended development runs.

All support methods are compared within k; all six primary answer methods use
k=3. Family-cluster bootstrap intervals are exploratory development summaries,
not independent confirmation, significance tests, or multiplicity-controlled
inference. This module creates no requests and never selects a winning setting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

import analyze_qasper_relation_pilot as pilot_analysis
import run_qasper_answer_evaluation as answer_stage
import run_qasper_extended_development as support_stage
import run_qasper_relation_pilot as pilot
from run_qasper_evidence_baselines import digest


IDENTITY = ("family_id", "doc_id", "question_id")
SIZES = (1, 2, 3)
SUPPORT_BASES = ("dense", "I_jev", "I_general", "ordinal_then_p_yes", "p_yes_only")
SUPPORT_METHODS = tuple(f"{base}_k{k}" for k in SIZES for base in SUPPORT_BASES)
ANSWER_METHODS = ("empty", "dense_k3", "I_jev_k3", "I_general_k3", "ordinal_then_p_yes_k3", "p_yes_only_k3")
SUPPORT_METRICS = ("official_evidence_f1", "reference_evidence_recall",
                   "official_text_only_evidence_f1", "actual_evidence_tokens")
ANSWER_METRICS = ("official_answer_f1", "actual_evidence_tokens")
BOOTSTRAP_SEED = 20260927
BOOTSTRAP_REPLICATES = 10000
TIE_TOLERANCE = 1e-12


def pairs_for(methods):
    """Fixed orientation: later method minus earlier method in the declared list."""
    return tuple((plus, minus) for minus, plus in combinations(methods, 2))


SUPPORT_PAIRS = tuple(pair for k in SIZES for pair in pairs_for(tuple(f"{base}_k{k}" for base in SUPPORT_BASES)))
ANSWER_PAIRS = pairs_for(ANSWER_METHODS)
SPECIFICATION = {
    "schema": "slac-extended-descriptive-statistics-spec-v1",
    "support_methods": list(SUPPORT_METHODS), "support_pairs": [list(pair) for pair in SUPPORT_PAIRS],
    "answer_methods": list(ANSWER_METHODS), "answer_pairs": [list(pair) for pair in ANSWER_PAIRS],
    "support_metrics": list(SUPPORT_METRICS), "answer_metrics": list(ANSWER_METRICS),
    "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "resampling_unit": "family; sample F families with replacement and retain every question in each sampled family",
    "random_generator": "numpy.random.PCG64", "cluster_counts_sampler": "multinomial(F, uniform family probabilities)",
    "interval": "two-sided percentile, quantiles 0.025 and 0.975, linear interpolation",
    "question_weighted": "sum of all sampled question deltas divided by number of sampled questions",
    "family_balanced": "mean of sampled within-family question-mean deltas",
    "same_resamples_for_all_pairs_and_metrics": True, "tie_absolute_tolerance": TIE_TOLERANCE,
    "multiple_comparison_adjustment": "none", "p_values_computed": False, "best_k_selected": False,
}


def questions_from(prepared):
    questions = [tuple(row[name] for name in IDENTITY) for row in prepared["queries"]]
    if (not questions or len(questions) != len(set(questions))
            or len({(key[1], key[2]) for key in questions}) != len(questions)
            or any(not isinstance(value, str) or not value for key in questions for value in key)):
        raise ValueError("prepared question identities must be nonempty and unique")
    families_by_document = {}
    for family, doc, _ in questions:
        if families_by_document.setdefault(doc, family) != family:
            raise ValueError("one document cannot belong to multiple families")
    return sorted(questions)


def validated_tables(records, methods, metrics, questions):
    indexed = pilot_analysis.index_rows(records, methods, questions)
    tables = {method: {key: indexed[(method, *key)] for key in questions} for method in methods}
    for row in indexed.values():
        for metric in metrics:
            value = row[metric]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("metrics must be finite real numbers")
            if metric == "actual_evidence_tokens":
                if type(value) is not int or not 0 <= value <= 1024:
                    raise ValueError("evidence tokens violate the fixed whole-pack budget")
            elif not 0 <= value <= 1:
                raise ValueError("quality metric outside unit interval")
    return tables


def family_resamples(questions):
    families = sorted({key[0] for key in questions})
    family_indices = [np.array([i for i, key in enumerate(questions) if key[0] == family], dtype=np.int64)
                      for family in families]
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    counts = generator.multinomial(len(families), np.full(len(families), 1 / len(families)),
                                   size=BOOTSTRAP_REPLICATES)
    return family_indices, counts


def clustered_delta(values, family_indices, draws, metric):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("paired deltas must be a finite vector")
    flattened = [int(i) for group in family_indices for i in group]
    if sorted(flattened) != list(range(len(values))) or any(len(group) == 0 for group in family_indices):
        raise ValueError("family partition must cover every question exactly once")
    if (draws.shape != (BOOTSTRAP_REPLICATES, len(family_indices)) or (draws < 0).any()
            or not np.equal(draws, np.floor(draws)).all()
            or not np.equal(draws.sum(axis=1), len(family_indices)).all()):
        raise ValueError("invalid fixed-size family bootstrap draws")
    sizes = np.array([len(group) for group in family_indices], dtype=np.float64)
    sums = np.array([values[group].sum() for group in family_indices], dtype=np.float64)
    means = sums / sizes
    weighted = (draws @ sums) / (draws @ sizes)
    balanced = (draws @ means) / len(family_indices)

    def estimate(point, samples):
        lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
        return {"delta": float(point), "bootstrap_percentile_95": [float(lower), float(upper)]}

    positive, negative = int((values > TIE_TOLERANCE).sum()), int((values < -TIE_TOLERANCE).sum())
    equal = len(values) - positive - negative
    result = {"questions": len(values), "families": len(family_indices),
        "question_weighted": estimate(values.mean(), weighted),
        "family_balanced": estimate(means.mean(), balanced),
        "question_positive": positive, "question_ties": equal, "question_negative": negative}
    if metric == "actual_evidence_tokens":
        result.update(question_token_increases=positive, question_token_decreases=negative,
                      interpretation="positive means longer evidence, not better quality")
    else:
        result.update(question_wins=positive, question_losses=negative)
    return result


def summarize_domain(domain, records, methods, metrics, pairs, questions, family_indices, draws):
    tables = validated_tables(records, methods, metrics, questions)
    comparisons, local = [], []
    for plus, minus in pairs:
        deltas = {metric: [tables[plus][key][metric] - tables[minus][key][metric] for key in questions]
                  for metric in metrics}
        comparisons.append({"plus": plus, "minus": minus,
            "metrics": {metric: clustered_delta(values, family_indices, draws, metric)
                        for metric, values in deltas.items()}})
        for index, key in enumerate(questions):
            local.append({"domain": domain, **dict(zip(IDENTITY, key)), "plus": plus, "minus": minus,
                          "deltas": {metric: values[index] for metric, values in deltas.items()}})
    method_means = []
    for method in methods:
        values = {}
        for metric in metrics:
            data = np.array([tables[method][key][metric] for key in questions], dtype=np.float64)
            values[metric] = {"question_weighted": float(data.mean()),
                             "family_balanced": float(np.mean([data[group].mean() for group in family_indices]))}
        method_means.append({"method": method, "metrics": values})
    return {"method_count": len(methods), "comparison_count": len(pairs),
            "method_means": method_means, "paired_comparisons": comparisons}, local


def summarize(prepared, support_records, answer_records):
    questions = questions_from(prepared)
    groups, draws = family_resamples(questions)
    support, support_local = summarize_domain("support", support_records, SUPPORT_METHODS, SUPPORT_METRICS,
                                             SUPPORT_PAIRS, questions, groups, draws)
    answers, answer_local = summarize_domain("answer", answer_records, ANSWER_METHODS, ANSWER_METRICS,
                                            ANSWER_PAIRS, questions, groups, draws)
    result = {"schema": "slac-extended-descriptive-statistics-v1", "status": "completed",
        "question_count": len(questions), "family_count": len(groups),
        "questions_per_family_histogram": dict(Counter(len(group) for group in groups)),
        "specification": SPECIFICATION, "specification_sha256": pilot.client.object_hash(SPECIFICATION),
        "bootstrap_numpy_version": np.__version__, "support": support, "primary_answer_k3": answers,
        "api_calls": 0, "test_payload_read": False, "independent_confirmation": False,
        "significance_claimed": False, "multiple_comparison_control_performed": False,
        "limits": [
            "All intervals describe exposed development data; they are not independent confirmation.",
            "Family bootstrap resamples whole families, never treats questions within a family as independent draws.",
            "Question-weighted estimates give larger families more influence; family-balanced estimates weight families equally.",
            "Intervals are unadjusted across all comparisons and metrics; no p-values or significance claims.",
            "All 30 support same-k pairs and all 15 primary answer pairs are retained, including negative or tied outcomes.",
            "No highest k or positive-only subset is selected; answer evaluation remains fixed at k=3.",
            "Equal evidence caps do not ensure equal actual evidence lengths; paired token differences are reported.",
            "One Qwen generator is shared by all answer methods; selection and generation can share a model.",
            "Identical answer payloads share responses; bootstrap uncertainty excludes repeat-generation variability.",
            "Bounds are percentile intervals from 10000 fixed-seed replicates, not guarantees of future performance."]}
    return result, support_local + answer_local


def snapshot(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("completed artifact directory missing")
    return {str(path.resolve()): digest(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def analyze(args):
    directories = {name: Path(getattr(args, name)).resolve()
                   for name in ("support_plan", "support_run", "answer_plan", "answer_run")}
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("analysis output already exists")
    before = {name: snapshot(path) for name, path in directories.items()}
    bindings = {path: value for inventory in before.values() for path, value in inventory.items()}
    for module in (sys_module(), support_stage, answer_stage, pilot_analysis):
        bindings[str(Path(module.__file__).resolve())] = digest(module.__file__)
    # The fixed comparison specification exists before any results are loaded.
    config, prepared, _, support_records, _ = support_stage.verify_completed_run(
        directories["support_plan"], directories["support_run"])
    answer_stage.audit(argparse.Namespace(plan=directories["answer_plan"], run=directories["answer_run"]))
    answer_config_path = directories["answer_plan"] / "experiment_config.json"
    answer_config = pilot.read_json(answer_config_path)
    if (Path(answer_config["support_plan"]).resolve() != directories["support_plan"]
            or Path(answer_config["support_run"]).resolve() != directories["support_run"]
            or Path(answer_config["prepared_dir"]).resolve() != Path(config["prepared_dir"]).resolve()):
        raise ValueError("answer stage belongs to a different support execution or prepared denominator")
    bindings.update(config["input_sha256"])
    bindings.update(answer_config["input_sha256"])
    answer_path = directories["answer_run"] / "per_question.jsonl"
    raw = answer_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != bindings[str(answer_path)]:
        raise ValueError("answer records changed after completed-run verification")
    answer_records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(prepared["queries"]) != 77 or len({q["family_id"] for q in prepared["queries"]}) != 24:
        raise ValueError("requires the complete 77-question, 24-family frozen denominator")
    result, local = summarize(prepared, support_records, answer_records)
    for name, directory in directories.items():
        if snapshot(directory) != before[name]:
            raise ValueError("completed source directory changed during analysis")
    pilot.verify_hashes(bindings)
    result.update(input_hashes_unchanged=True, input_binding_sha256=pilot.client.object_hash(bindings),
        input_sha256=[{"path_sha256": hashlib.sha256(path.encode()).hexdigest(), "content_sha256": value}
                      for path, value in sorted(bindings.items())])
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_rows(output / "paired_per_question.jsonl", local)
    pilot.write_json(output / "source_binding.json", {"input_sha256": bindings,
        "directories": {name: str(path) for name, path in directories.items()}})
    result["local_output_files_sha256"] = {name: digest(output / name)
        for name in ("paired_per_question.jsonl", "source_binding.json")}
    pilot.write_json(output / "analysis.json", result)
    return result


def sys_module():
    import sys
    return sys.modules[__name__]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("support-plan", "support-run", "answer-plan", "answer-run", "output"):
        parser.add_argument("--" + name, required=True)
    report = analyze(parser.parse_args())
    print(json.dumps({"status": report["status"], "question_count": report["question_count"],
                      "family_count": report["family_count"], "api_calls": 0,
                      "input_binding_sha256": report["input_binding_sha256"]}))
