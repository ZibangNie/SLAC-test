"""Fixed five-arm confirmation statistics V2; pure functions, no I/O or QA access.

Input records contain already validated official scores and evidence token counts.
This module checks complete paired coverage; it neither scores answers nor admits
data, models, expenditure or an execution. Inferential coverage is conditional on
the declared family sampling and t-approximation assumptions.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math

import numpy as np


METHODS = ("dense_k3", "reranker_k3", "I_jev_k3", "I_general_k3", "p_yes_only_k3")
SCORE_METHOD = "p_yes_only_k3"
INFERENTIAL_COMPARATORS = ("reranker_k3", "I_jev_k3", "I_general_k3")
DESCRIPTIVE_COMPARATORS = (*INFERENTIAL_COMPARATORS, "dense_k3")
METRICS = ("official_answer_f1", "actual_evidence_tokens")
IDENTITY = ("doc_id", "question_id", "family_id")
ALPHA = 0.05
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260927
TIE_TOLERANCE = 1e-12
MAX_EVIDENCE_TOKENS = 1024
CONDITIONAL_FAMILY_TARGET = 193
NUMERICAL_DEGENERACY_TOLERANCE = 1e-12

SPEC = {
    "schema": "slac-qasper-confirmation-statistics-spec-v2",
    "statistics_implementation": "family_confirmation_v2",
    "methods": list(METHODS),
    "score_method": SCORE_METHOD,
    "inferential_comparators": list(INFERENTIAL_COMPARATORS),
    "primary_comparator": "reranker_k3",
    "descriptive_comparators": list(DESCRIPTIVE_COMPARATORS),
    "metrics": list(METRICS),
    "inferential_metric": "official_answer_f1",
    "inferential_weighting": "family_balanced",
    "inferential_interval": "paired_family_mean_t_bonferroni",
    "familywise_alpha": ALPHA,
    "simultaneous_comparisons": 3,
    "two_sided": True,
    "marginal_confidence": 1 - ALPHA / 3,
    "quantile_probability": 1 - ALPHA / 6,
    "positive_claim_rule": "unrounded_lower_bound_strictly_above_zero",
    "zero_sample_sd": "unavailable_no_confirmatory_claim",
    "numerical_degeneracy_range_inclusive": NUMERICAL_DEGENERACY_TOLERANCE,
    "numerical_degeneracy_action": "unavailable_preserve_mean_sd_range_no_claim",
    "numerical_degeneracy_used_for_interval_endpoint_comparison": False,
    "bootstrap": {"draws": BOOTSTRAP_DRAWS, "rng": "PCG64",
                  "seed": BOOTSTRAP_SEED, "unit": "whole_family",
                  "count_sampler": "multinomial_equal_family_probabilities",
                  "interval": "unadjusted_two_sided_percentile_95",
                  "quantile_method": "linear", "shared_draws": True},
    "descriptive_interval_count": 16,
    "direction_tie_absolute_tolerance": TIE_TOLERANCE,
    "token_count_range_inclusive": [0, MAX_EVIDENCE_TOKENS],
    "minimum_families_for_computation": 2,
    "conditional_planning_family_target": CONDITIONAL_FAMILY_TARGET,
    "planning_half_width": 0.05,
    "planning_sd_multiplier": 1.5,
    "planning_is_power_or_admission": False,
    "outcome_based_top_up": False,
    "student_t_quantile_implementation": "stdlib_incomplete_beta_bisection_v1",
}


def _object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


SPEC_SHA256 = _object_hash(SPEC)


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _identity(row):
    if any(type(row.get(field)) is not str or not row[field].strip() for field in IDENTITY):
        raise ValueError("identity fields must be nonempty strings")
    return tuple(row[field] for field in IDENTITY)


def validate_inputs(questions, records):
    """Return canonical private keys, family index groups and metric arrays.

    questions: list of exact {doc_id, question_id, family_id} objects.
    records: list of exact identity + method + both METRICS objects.
    Inputs are not changed. Arrays align every method to the same canonical keys.
    """
    if type(questions) is not list or not questions or type(records) is not list:
        raise ValueError("questions and records must be nonempty complete lists")
    by_question, doc_families = {}, {}
    for row in questions:
        if type(row) is not dict or set(row) != set(IDENTITY):
            raise ValueError("invalid question manifest schema")
        doc, question, family = _identity(row)
        if (doc, question) in by_question:
            raise ValueError("duplicate manifest question identity")
        if doc in doc_families and doc_families[doc] != family:
            raise ValueError("one document cannot belong to multiple families")
        doc_families[doc] = family
        by_question[(doc, question)] = family
    families = sorted(set(by_question.values()))
    if len(families) < 2:
        raise ValueError("at least two families are required")
    keys = sorted((doc, question, family) for (doc, question), family in by_question.items())
    keys.sort(key=lambda key: (key[2], key[0], key[1]))
    indices = {key: index for index, key in enumerate(keys)}
    if len(records) != len(keys) * len(METHODS):
        raise ValueError("five complete method records required for every question")
    tables = {method: {metric: np.empty(len(keys), dtype=np.float64) for metric in METRICS}
              for method in METHODS}
    seen = set()
    schema = set(IDENTITY) | {"method", *METRICS}
    for row in records:
        if type(row) is not dict or set(row) != schema:
            raise ValueError("invalid metric record schema")
        doc, question, family = key = _identity(row)
        method = row["method"]
        if type(method) is not str or method not in METHODS:
            raise ValueError("unknown method")
        if key not in indices:
            raise ValueError("unknown question or family mismatch")
        pair = (method, key)
        if pair in seen:
            raise ValueError("duplicate method-question record")
        seen.add(pair)
        f1, tokens = row[METRICS[0]], row[METRICS[1]]
        if not _finite_number(f1) or not 0 <= f1 <= 1:
            raise ValueError("official Answer F1 must be finite in [0,1]")
        if type(tokens) is not int or not 0 <= tokens <= MAX_EVIDENCE_TOKENS:
            raise ValueError("evidence tokens must be an integer in [0,1024]")
        tables[method][METRICS[0]][indices[key]] = f1
        tables[method][METRICS[1]][indices[key]] = tokens
    groups = [np.array([i for i, key in enumerate(keys) if key[2] == family], dtype=np.int64)
              for family in families]
    return keys, groups, tables


def _beta_continued_fraction(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1 / d
    result = d
    for step in range(1, 10001):
        twice = 2 * step
        for numerator in (step * (b - step) * x / ((qam + twice) * (a + twice)),
                          -(a + step) * (qab + step) * x / ((a + twice) * (qap + twice))):
            d = 1 + numerator * d
            c = 1 + numerator / c
            if abs(d) < 1e-300:
                d = 1e-300
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1 / d
            change = d * c
            result *= change
        if abs(change - 1) < 1e-14:
            return result
    raise ArithmeticError("incomplete beta failed to converge")


def _regularized_beta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1 - factor * _beta_continued_fraction(b, a, 1 - x) / b


def student_t_cdf(value, degrees_of_freedom):
    if not _finite_number(value) or type(degrees_of_freedom) is not int or degrees_of_freedom < 1:
        raise ValueError("invalid Student t arguments")
    if value == 0:
        return 0.5
    df = degrees_of_freedom
    half_tail = 0.5 * _regularized_beta(df / 2, 0.5, df / (df + value * value))
    return 1 - half_tail if value > 0 else half_tail


def student_t_quantile(probability, degrees_of_freedom):
    """Deterministic inverse CDF; no SciPy dependency or normal substitution."""
    if (not _finite_number(probability) or not 0 < probability < 1
            or type(degrees_of_freedom) is not int or degrees_of_freedom < 1):
        raise ValueError("invalid Student t quantile arguments")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -student_t_quantile(1 - probability, degrees_of_freedom)
    low, high = 0.0, 1.0
    for _ in range(1024):
        if student_t_cdf(high, degrees_of_freedom) >= probability:
            break
        high *= 2
        if not math.isfinite(high):
            raise ArithmeticError("Student t quantile bracket overflow")
    else:
        raise ArithmeticError("Student t quantile not bracketed")
    for _ in range(100):
        middle = (low + high) / 2
        if student_t_cdf(middle, degrees_of_freedom) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _family_sums(values, groups):
    return np.array([math.fsum(float(values[index]) for index in group) for group in groups])


def _descriptive_delta(values, groups, counts, sizes):
    sums = _family_sums(values, groups)
    means = sums / sizes
    samples = {"question_weighted": (counts @ sums) / (counts @ sizes),
               "family_balanced": (counts @ means) / len(groups)}
    points = {"question_weighted": math.fsum(float(v) for v in values) / len(values),
              "family_balanced": math.fsum(float(v) for v in means) / len(groups)}
    positive = int(np.count_nonzero(values > TIE_TOLERANCE))
    negative = int(np.count_nonzero(values < -TIE_TOLERANCE))
    result = {weight: {"delta": points[weight],
                       "bootstrap_percentile_95": np.quantile(sample, [0.025, 0.975], method="linear").tolist()}
              for weight, sample in samples.items()}
    result.update(question_positive=positive, question_ties=len(values) - positive - negative,
                  question_negative=negative, question_count=len(values), family_count=len(groups))
    return result


def _inferential_interval(values, groups, comparator, critical):
    family_deltas = [math.fsum(float(values[i]) for i in group) / len(group) for group in groups]
    count = len(groups)
    mean = math.fsum(family_deltas) / count
    # Floating subtraction can turn a nominal constant gain into a tiny nonzero SD.
    delta_range = max(family_deltas) - min(family_deltas)
    numerically_degenerate = delta_range <= NUMERICAL_DEGENERACY_TOLERANCE
    zero_sd = all(value == family_deltas[0] for value in family_deltas)
    sd = 0.0 if zero_sd else math.sqrt(math.fsum((value - mean) ** 2 for value in family_deltas) / (count - 1))
    result = {"plus": SCORE_METHOD, "minus": comparator,
              "role": "primary" if comparator == "reranker_k3" else "key_secondary",
              "metric": "official_answer_f1", "weighting": "family_balanced",
              "family_count": count, "degrees_of_freedom": count - 1,
              "delta": mean, "sample_sd_ddof1": sd, "critical_t": critical,
              "family_delta_range": delta_range,
              "numerical_degeneracy_threshold": NUMERICAL_DEGENERACY_TOLERANCE,
              "numerically_degenerate": numerically_degenerate,
              "marginal_confidence": 1 - ALPHA / 3,
              "familywise_nominal_confidence": 1 - ALPHA,
              "interval": None, "half_width": None,
              "positive_effect_supported": False, "negative_effect_supported": False,
              "status": "unavailable_zero_sample_sd" if sd == 0 else "unavailable_numerically_degenerate_family_differences"}
    if sd > 0 and not numerically_degenerate:
        half_width = critical * sd / math.sqrt(count)
        lower, upper = mean - half_width, mean + half_width
        result.update(status="computed", interval=[lower, upper], half_width=half_width,
                      positive_effect_supported=lower > 0, negative_effect_supported=upper < 0)
    return result


def analyze(questions, records):
    """Compute the fixed aggregate from a complete manifest and five-arm records.

    No identities, raw questions, predictions or per-question scores are returned.
    Completeness is necessary but does not replace the caller's provenance audit.
    """
    if _object_hash(SPEC) != SPEC_SHA256:
        raise ValueError("fixed statistics specification was mutated")
    keys, groups, tables = validate_inputs(questions, records)
    family_count = len(groups)
    sizes = np.array([len(group) for group in groups], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    counts = rng.multinomial(family_count, np.full(family_count, 1 / family_count), size=BOOTSTRAP_DRAWS)
    methods = []
    for method in METHODS:
        metrics = {}
        for metric in METRICS:
            values = tables[method][metric]
            metrics[metric] = {
                "question_weighted": math.fsum(float(v) for v in values) / len(keys),
                "family_balanced": math.fsum(float(v) for v in _family_sums(values, groups) / sizes) / family_count}
        methods.append({"method": method, "metrics": metrics})
    comparisons = []
    for comparator in DESCRIPTIVE_COMPARATORS:
        metrics = {}
        for metric in METRICS:
            delta = tables[SCORE_METHOD][metric] - tables[comparator][metric]
            metrics[metric] = _descriptive_delta(delta, groups, counts, sizes)
            if metric == "actual_evidence_tokens":
                metrics[metric]["direction_interpretation"] = "positive_means_longer_not_better"
                metrics[metric]["question_token_increases"] = metrics[metric]["question_positive"]
                metrics[metric]["question_token_decreases"] = metrics[metric]["question_negative"]
            else:
                metrics[metric]["question_wins"] = metrics[metric]["question_positive"]
                metrics[metric]["question_losses"] = metrics[metric]["question_negative"]
        comparisons.append({"plus": SCORE_METHOD, "minus": comparator, "metrics": metrics})
    critical = student_t_quantile(1 - ALPHA / 6, family_count - 1)
    inferential = [_inferential_interval(tables[SCORE_METHOD][METRICS[0]] - tables[comparator][METRICS[0]],
                                        groups, comparator, critical)
                   for comparator in INFERENTIAL_COMPARATORS]
    return {
        "schema": "slac-qasper-confirmation-statistics-v2", "status": "computed_complete_inputs",
        "question_count": len(keys), "family_count": family_count,
        "method_count": len(METHODS), "record_count": len(records),
        "questions_per_family_histogram": {str(size): count for size, count in sorted(Counter(map(len, groups)).items())},
        "specification": deepcopy(SPEC), "specification_sha256": SPEC_SHA256,
        "method_means": methods, "descriptive_comparisons": comparisons,
        "descriptive_interval_count": 16, "inferential_comparisons": inferential,
        "inferential_comparison_count": 3,
        "bootstrap_numpy_version": np.__version__,
        "bootstrap_draws_sha256": hashlib.sha256(np.asarray(counts, dtype="<i8").tobytes(order="C")).hexdigest(),
        "planning": {"conditional_family_target": CONDITIONAL_FAMILY_TARGET,
                     "conditional_precision_target_met": family_count >= CONDITIONAL_FAMILY_TARGET,
                     "actual_families": family_count,
                     "power_computed": False, "data_or_paid_execution_admitted": False,
                     "observed_effect_used_to_extend_sample": False,
                     "achieved_half_width_target": 0.05,
                     "achieved_by_comparison": [{"minus": row["minus"],
                         "met": None if row["half_width"] is None else row["half_width"] <= 0.05}
                         for row in inferential]},
        "limits": [
            "The three adjusted intervals concern FB official Answer F1 only; secondary claims are separate.",
            "QW, dense and token intervals are descriptive and cannot rescue the primary BGE comparison.",
            "All 16 bootstrap intervals are unadjusted descriptions, not simultaneous coverage or calibrated p-values.",
            "t intervals are an approximation requiring independent comparable families; small F is retained without a coverage guarantee.",
            "Bonferroni does not repair marginal-interval error, hidden family dependence or domain shift.",
            "An exact-zero SD or family-difference range <=1e-12 leaves t inference unavailable; mean, SD and range remain reported.",
            "The degeneracy tolerance is not applied to CI endpoints: support requires an unrounded lower bound strictly above zero.",
            "Zero-containing intervals do not prove equivalence or non-inferiority.",
            "Actual lengths can differ under a common cap; neither length-independent benefit nor relation sharing is identified.",
            "One saved response per exact payload does not measure generation variance; cache reuse creates no independent observations.",
            "The 193-family flag is conditional planning, not power, admission, an achieved precision guarantee or a top-up rule.",
            "Caller must audit source, full cohort admission, official-score provenance, model stability and failures separately.",
        ],
        "identities_or_per_question_scores_in_public_output": False,
    }
