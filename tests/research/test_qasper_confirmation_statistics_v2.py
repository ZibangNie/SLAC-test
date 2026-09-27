"""Synthetic-only tests: never open a prepared manifest or saved real scores."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[2] / "docs/research/analyze_qasper_confirmation_v2.py"
module_spec = importlib.util.spec_from_file_location("confirmation_statistics_v2_under_test", SOURCE)
stats = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(stats)


def fixture(sizes=(1, 3, 2)):
    questions, records = [], []
    for family, size in enumerate(sizes):
        for question in range(size):
            identity = {"doc_id": f"private-document-{family}", "question_id": f"private-query-{question}",
                        "family_id": f"private-family-{family}"}
            questions.append(identity)
            for index, method in enumerate(stats.METHODS):
                records.append({**identity, "method": method,
                                "official_answer_f1": (index + family % 3 + question % 2) / 10,
                                "actual_evidence_tokens": 100 + 10 * index + 3 * family + question})
    return questions, records


def assign(records, method, function, metric="official_answer_f1"):
    for row in records:
        if row["method"] == method:
            family = int(row["family_id"].rsplit("-", 1)[1])
            row[metric] = function(family)


def inferential(result, comparator):
    return next(row for row in result["inferential_comparisons"] if row["minus"] == comparator)


def test_complete_shape_counts_fixed_spec_and_all_denominators():
    q, r = fixture()
    result = stats.analyze(q, r)
    assert (result["question_count"], result["family_count"], result["method_count"], result["record_count"]) == (6, 3, 5, 30)
    assert result["questions_per_family_histogram"] == {"1": 1, "2": 1, "3": 1}
    assert [m["method"] for m in result["method_means"]] == list(stats.METHODS)
    assert [m["minus"] for m in result["descriptive_comparisons"]] == list(stats.DESCRIPTIVE_COMPARATORS)
    assert [m["minus"] for m in result["inferential_comparisons"]] == list(stats.INFERENTIAL_COMPARATORS)
    intervals = 0
    for pair in result["descriptive_comparisons"]:
        for metric, row in pair["metrics"].items():
            assert row["question_count"] == 6 and row["family_count"] == 3
            assert row["question_positive"] + row["question_ties"] + row["question_negative"] == 6
            intervals += sum("bootstrap_percentile_95" in row[w] for w in ("question_weighted", "family_balanced"))
    assert intervals == result["descriptive_interval_count"] == 16
    assert result["inferential_comparison_count"] == 3
    json.dumps(result, allow_nan=False)


def test_manifest_and_records_are_not_mutated_and_order_is_irrelevant():
    q, r = fixture()
    prior = deepcopy((q, r))
    first = stats.analyze(q, r)
    second = stats.analyze(list(reversed(q)), list(reversed(r)))
    assert first == second
    assert (q, r) == prior
    first["specification"]["methods"].append("not-real")
    assert len(stats.SPEC["methods"]) == 5


def test_output_contains_no_private_identity_or_per_question_score():
    q, r = fixture()
    result = json.dumps(stats.analyze(q, r))
    for word in ("private-document-", "private-query-", "private-family-"):
        assert word not in result


@pytest.mark.parametrize("bad", [None, {}, (), [], "questions"])
def test_invalid_manifest_container(bad):
    _, r = fixture()
    with pytest.raises(ValueError):
        stats.analyze(bad, r)


@pytest.mark.parametrize("bad", [None, {}, (), [], "records"])
def test_invalid_records_container_or_missing(bad):
    q, _ = fixture()
    with pytest.raises(ValueError):
        stats.analyze(q, bad)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown_method", "unknown_question", "wrong_family", "extra_field"])
def test_complete_record_contract_rejects_mutations(mutation):
    q, r = fixture()
    if mutation == "missing":
        r.pop()
    elif mutation == "duplicate":
        r[-1] = deepcopy(r[0])
    elif mutation == "unknown_method":
        r[0]["method"] = "best_method_after_results"
    elif mutation == "unknown_question":
        r[0]["question_id"] = "not-in-manifest"
    elif mutation == "wrong_family":
        r[0]["family_id"] = "private-family-1"
    elif mutation == "extra_field":
        r[0]["gold_answer"] = "forbidden"
    with pytest.raises(ValueError):
        stats.analyze(q, r)


@pytest.mark.parametrize("value", [True, False, None, "0.4", float("nan"), float("inf"), -0.01, 1.01, 10**400])
def test_invalid_f1(value):
    q, r = fixture()
    r[0]["official_answer_f1"] = value
    with pytest.raises(ValueError):
        stats.analyze(q, r)


@pytest.mark.parametrize("value", [True, False, None, "12", 12.0, float("nan"), -1, 1025])
def test_invalid_token_count(value):
    q, r = fixture()
    r[0]["actual_evidence_tokens"] = value
    with pytest.raises(ValueError):
        stats.analyze(q, r)


@pytest.mark.parametrize("field", stats.IDENTITY)
@pytest.mark.parametrize("value", [None, True, 7, "", "   "])
def test_invalid_identity(field, value):
    q, r = fixture()
    q[0][field] = value
    with pytest.raises(ValueError):
        stats.analyze(q, r)


def test_manifest_duplicate_question_and_cross_family_document_rejected():
    q, r = fixture()
    q.append(deepcopy(q[0]))
    with pytest.raises(ValueError, match="duplicate"):
        stats.analyze(q, r)
    q, r = fixture()
    q[1]["doc_id"] = q[0]["doc_id"]
    q[1]["question_id"] = "different-q"
    with pytest.raises(ValueError, match="multiple families"):
        stats.analyze(q, r)


def test_one_family_rejected_two_retained_without_variance_filter():
    q, r = fixture((3,))
    with pytest.raises(ValueError, match="two families"):
        stats.analyze(q, r)
    q, r = fixture((1, 1))
    result = stats.analyze(q, r)
    assert result["family_count"] == 2
    assert len(result["inferential_comparisons"]) == 3
    assert result["planning"]["conditional_precision_target_met"] is False


@pytest.mark.parametrize("probability", [.6, .9, .975, 1 - .05 / 6])
def test_student_t_df1_and_df2_against_exact_closed_forms(probability):
    expected1 = math.tan(math.pi * (probability - .5))
    y = 2 * probability - 1
    expected2 = y * math.sqrt(2 / (1 - y * y))
    assert stats.student_t_quantile(probability, 1) == pytest.approx(expected1, rel=2e-12)
    assert stats.student_t_quantile(probability, 2) == pytest.approx(expected2, rel=2e-12)


def test_student_t_known_quantile_and_symmetry():
    assert stats.student_t_quantile(.975, 30) == pytest.approx(2.0422724563012373, abs=2e-10)
    assert stats.student_t_quantile(.5, 2) == 0
    assert stats.student_t_quantile(.025, 30) == pytest.approx(-stats.student_t_quantile(.975, 30), abs=1e-12)


@pytest.mark.parametrize("df", [2, 7, 23, 95, 191, 248, 500])
def test_quantile_against_independent_simpson_density_integration(df):
    probability = 1 - .05 / 6
    upper = stats.student_t_quantile(probability, df)
    steps = 20000
    h = upper / steps
    constant = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    density = lambda x: constant * (1 + x * x / df) ** (-(df + 1) / 2)
    integral = h / 3 * (density(0) + density(upper)
                       + math.fsum((4 if i % 2 else 2) * density(i * h) for i in range(1, steps)))
    assert .5 + integral == pytest.approx(probability, abs=2e-11)


@pytest.mark.parametrize("p,df", [(0, 1), (1, 1), (True, 2), (None, 2), (.9, 0), (.9, True), (.9, 2.0)])
def test_student_t_invalid_arguments(p, df):
    with pytest.raises(ValueError):
        stats.student_t_quantile(p, df)


def test_unequal_family_weights_and_naive_cluster_bootstrap_reproduction():
    q, r = fixture((1, 3, 2))
    assign(r, stats.SCORE_METHOD, lambda f: [.1, .8, .5][f])
    assign(r, "reranker_k3", lambda f: [.5, .2, .4][f])
    result = stats.analyze(q, r)
    row = result["descriptive_comparisons"][0]["metrics"]["official_answer_f1"]
    differences = [-.4, .6, .1]
    sizes = [1, 3, 2]
    assert row["question_weighted"]["delta"] == pytest.approx((-.4 + 1.8 + .2) / 6)
    assert row["family_balanced"]["delta"] == pytest.approx(.1)
    counts = np.random.Generator(np.random.PCG64(20260927)).multinomial(3, [1 / 3] * 3, size=10000)
    weighted, balanced = [], []
    for draw in counts.tolist():
        family_values, all_question_values = [], []
        for i, multiplicity in enumerate(draw):
            family_values.extend([differences[i]] * multiplicity)
            all_question_values.extend([differences[i]] * multiplicity * sizes[i])
        balanced.append(math.fsum(family_values) / len(family_values))
        weighted.append(math.fsum(all_question_values) / len(all_question_values))
    assert row["question_weighted"]["bootstrap_percentile_95"] == pytest.approx(np.quantile(weighted, [.025, .975], method="linear"))
    assert row["family_balanced"]["bootstrap_percentile_95"] == pytest.approx(np.quantile(balanced, [.025, .975], method="linear"))
    assert (row["question_positive"], row["question_ties"], row["question_negative"]) == (5, 0, 1)
    inf = inferential(result, "reranker_k3")
    expected_sd = math.sqrt(sum((d - .1) ** 2 for d in differences) / 2)
    assert inf["sample_sd_ddof1"] == pytest.approx(expected_sd)
    assert inf["half_width"] == pytest.approx(stats.student_t_quantile(1 - .05 / 6, 2) * expected_sd / math.sqrt(3))


@pytest.mark.parametrize("delta", [0, .125, -.125])
def test_zero_sd_is_unavailable_even_for_nonzero_constant_gain(delta):
    q, r = fixture((1, 1, 1))
    assign(r, stats.SCORE_METHOD, lambda _: .5 + delta)
    assign(r, "reranker_k3", lambda _: .5)
    row = inferential(stats.analyze(q, r), "reranker_k3")
    assert row["delta"] == delta
    assert row["status"] == "unavailable_zero_sample_sd"
    assert row["interval"] is row["half_width"] is None
    assert row["positive_effect_supported"] is row["negative_effect_supported"] is False


def test_primary_failure_not_rescued_by_positive_secondary_or_dense():
    q, r = fixture((1,) * 24)
    assign(r, stats.SCORE_METHOD, lambda _: .5)
    assign(r, "reranker_k3", lambda f: .5 + (.3 if f % 2 else -.3))
    assign(r, "I_jev_k3", lambda f: .1 + .02 * (f % 2))
    assign(r, "I_general_k3", lambda f: .15 + .02 * (f % 2))
    assign(r, "dense_k3", lambda _: 0)
    result = stats.analyze(q, r)
    assert inferential(result, "reranker_k3")["positive_effect_supported"] is False
    assert inferential(result, "I_jev_k3")["positive_effect_supported"] is True
    assert inferential(result, "I_general_k3")["positive_effect_supported"] is True
    assert [row["role"] for row in result["inferential_comparisons"]] == ["primary", "key_secondary", "key_secondary"]


def test_negative_effect_is_retained_and_tokens_do_not_drive_claim():
    q, r = fixture((1,) * 24)
    assign(r, stats.SCORE_METHOD, lambda _: .2)
    assign(r, "reranker_k3", lambda f: .8 + .02 * (f % 2))
    assign(r, stats.SCORE_METHOD, lambda _: 1, "actual_evidence_tokens")
    result = stats.analyze(q, r)
    row = inferential(result, "reranker_k3")
    assert row["negative_effect_supported"] is True and row["positive_effect_supported"] is False
    length = result["descriptive_comparisons"][0]["metrics"]["actual_evidence_tokens"]
    assert length["question_negative"] == 24
    assert length["direction_interpretation"] == "positive_means_longer_not_better"


def test_inferential_decision_is_unrounded_and_zero_endpoint_does_not_pass(monkeypatch):
    groups = [np.array([0]), np.array([1])]
    # Mean=1, sd=sqrt(2); critical=1 gives lower exactly zero.
    row = stats._inferential_interval(np.array([0., 2.]), groups, "reranker_k3", 1.)
    assert row["interval"][0] == 0 and row["positive_effect_supported"] is False
    row = stats._inferential_interval(np.array([0., 2.]), groups, "reranker_k3", 1. - 1e-13)
    assert 0 < row["interval"][0] < 1e-12 and row["positive_effect_supported"] is True


@pytest.mark.parametrize("families,met", [(192, False), (193, True)])
def test_193_is_only_a_planning_flag(families, met):
    q, r = fixture((1,) * families)
    result = stats.analyze(q, r)
    assert result["planning"]["conditional_precision_target_met"] is met
    assert result["planning"]["power_computed"] is False
    assert result["planning"]["data_or_paid_execution_admitted"] is False
    assert result["record_count"] == 5 * families


def test_fixed_spec_mutation_cannot_silently_change_contract(monkeypatch):
    monkeypatch.setitem(stats.SPEC, "familywise_alpha", .5)
    with pytest.raises(ValueError, match="specification"):
        stats.analyze(*fixture())


def test_statistical_core_has_no_file_network_or_model_dependencies():
    import ast
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imports |= {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imports <= {"__future__", "collections", "copy", "hashlib", "json", "math", "numpy"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"open", "eval", "exec"}
                   for node in ast.walk(tree))


def test_nominal_constant_decimal_gain_is_not_confirmed_by_float_noise():
    q, r = fixture((1, 1, 1))
    assign(r, stats.SCORE_METHOD, lambda f: [.2, .3, .4][f])
    assign(r, "reranker_k3", lambda f: [.1, .2, .3][f])
    row = inferential(stats.analyze(q, r), "reranker_k3")
    actual = [.2 - .1, .3 - .2, .4 - .3]
    mean = math.fsum(actual) / 3
    assert row["delta"] == mean
    assert row["sample_sd_ddof1"] == math.sqrt(math.fsum((v - mean)**2 for v in actual) / 2)
    assert 0 < row["sample_sd_ddof1"] < 1e-15
    assert row["family_delta_range"] == max(actual) - min(actual)
    assert row["status"] == "unavailable_numerically_degenerate_family_differences"
    assert row["numerically_degenerate"] is True
    assert row["numerical_degeneracy_threshold"] == 1e-12
    assert row["interval"] is row["half_width"] is None
    assert row["positive_effect_supported"] is row["negative_effect_supported"] is False


def test_genuine_variation_above_frozen_tolerance_remains_inferable():
    q, r = fixture((1, 1, 1))
    assign(r, stats.SCORE_METHOD, lambda f: [.2, .3, .4 + 2e-12][f])
    assign(r, "reranker_k3", lambda f: [.1, .2, .3][f])
    row = inferential(stats.analyze(q, r), "reranker_k3")
    assert row["family_delta_range"] > 1e-12
    assert row["sample_sd_ddof1"] > 0
    assert row["numerically_degenerate"] is False
    assert row["status"] == "computed" and row["positive_effect_supported"] is True


@pytest.mark.parametrize("width,degenerate", [(0., True), (1e-13, True), (1e-12, True), (np.nextafter(1e-12, np.inf), False)])
def test_numerical_degeneracy_boundary_is_fixed_and_inclusive(width, degenerate):
    groups = [np.array([0]), np.array([1])]
    row = stats._inferential_interval(np.array([0., width]), groups, "reranker_k3", 1.)
    assert row["family_delta_range"] == width
    assert row["numerically_degenerate"] is degenerate
    assert (row["interval"] is None) is degenerate
    if width == 0:
        assert row["status"] == "unavailable_zero_sample_sd"


def test_v2_spec_and_output_explicitly_disclose_guard_without_endpoint_tolerance():
    result = stats.analyze(*fixture())
    assert stats.SPEC["schema"] == "slac-qasper-confirmation-statistics-spec-v2"
    assert stats.SPEC["statistics_implementation"] == "family_confirmation_v2"
    assert result["schema"] == "slac-qasper-confirmation-statistics-v2"
    assert stats.SPEC["numerical_degeneracy_range_inclusive"] == 1e-12
    assert stats.SPEC["numerical_degeneracy_used_for_interval_endpoint_comparison"] is False
