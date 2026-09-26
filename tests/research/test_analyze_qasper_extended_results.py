"""Synthetic-only checks of the fixed, outcome-independent comparison scheme."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import analyze_qasper_extended_results as analysis


def dataset(question_count=3, family_count=2):
    queries = [{"family_id": f"secret-family-{i % family_count}", "doc_id": f"secret-doc-{i % family_count}",
                "question_id": f"secret-question-{i}"} for i in range(question_count)]
    prepared = {"queries": queries}
    support, answer = [], []
    for query in queries:
        for index, method in enumerate(analysis.SUPPORT_METHODS):
            # The last rule is intentionally worse, so reporting cannot drop it.
            value = .1 * (index % 5) if "p_yes_only" not in method else .05
            support.append({**query, "method": method, "official_evidence_f1": value,
                "reference_evidence_recall": value, "official_text_only_evidence_f1": value,
                "actual_evidence_tokens": 100 + index})
        for index, method in enumerate(analysis.ANSWER_METHODS):
            answer.append({**query, "method": method, "official_answer_f1": .1 * index,
                           "actual_evidence_tokens": 0 if method == "empty" else 100 + index})
    return prepared, support, answer


def test_pair_set_covers_every_same_k_method_pair_and_all_primary_answer_pairs():
    assert len(analysis.SUPPORT_METHODS) == 15 and len(analysis.SUPPORT_PAIRS) == 30
    assert len(analysis.ANSWER_METHODS) == 6 and len(analysis.ANSWER_PAIRS) == 15
    assert len({frozenset(pair) for pair in analysis.SUPPORT_PAIRS}) == 30
    assert len({frozenset(pair) for pair in analysis.ANSWER_PAIRS}) == 15
    assert all(plus[-2:] == minus[-2:] for plus, minus in analysis.SUPPORT_PAIRS)
    assert all(method == "empty" or method.endswith("_k3") for pair in analysis.ANSWER_PAIRS for method in pair)
    assert analysis.BOOTSTRAP_REPLICATES == 10000 and analysis.BOOTSTRAP_SEED == 20260927


def test_unequal_family_sizes_distinguish_weighted_and_balanced_estimands():
    questions = [("a", "doc1", "q1"), ("a", "doc1", "q2"), ("b", "doc2", "q3")]
    groups, draws = analysis.family_resamples(questions)
    stats = analysis.clustered_delta([1, 0, -1], groups, draws, "official_evidence_f1")
    assert stats["question_weighted"]["delta"] == 0
    assert stats["family_balanced"]["delta"] == -.25
    assert stats["question_weighted"]["bootstrap_percentile_95"] == [-1, .5]
    assert stats["family_balanced"]["bootstrap_percentile_95"] == [-1, .5]
    assert stats["question_wins"] == stats["question_ties"] == stats["question_losses"] == 1
    assert draws.shape == (10000, 2) and np.all(draws.sum(axis=1) == 2)


def test_resampling_is_whole_family_and_one_family_interval_degenerates():
    questions = [("a", "doc", f"q{i}") for i in range(3)]
    groups, draws = analysis.family_resamples(questions)
    stats = analysis.clustered_delta([1, 0, -1], groups, draws, "official_evidence_f1")
    assert stats["question_weighted"]["bootstrap_percentile_95"] == [0, 0]
    assert stats["family_balanced"]["bootstrap_percentile_95"] == [0, 0]
    assert np.all(draws == 1)


def test_fixed_seed_determinism_zero_deltas_and_reversing_comparison():
    questions = [("a", "doc1", "q1"), ("a", "doc1", "q2"), ("b", "doc2", "q3")]
    groups, draws = analysis.family_resamples(questions)
    assert np.array_equal(draws, analysis.family_resamples(questions)[1])
    positive = analysis.clustered_delta([.8, .2, -.1], groups, draws, "official_evidence_f1")
    negative = analysis.clustered_delta([-.8, -.2, .1], groups, draws, "official_evidence_f1")
    for weighting in ("question_weighted", "family_balanced"):
        assert positive[weighting]["delta"] == -negative[weighting]["delta"]
        assert positive[weighting]["bootstrap_percentile_95"] == pytest.approx(
            [-value for value in reversed(negative[weighting]["bootstrap_percentile_95"])])
    zero = analysis.clustered_delta([0, 0, 0], groups, draws, "official_evidence_f1")
    assert zero["question_weighted"]["bootstrap_percentile_95"] == [0, 0]
    assert zero["question_ties"] == 3


def test_actual_token_difference_is_not_labeled_a_quality_win():
    prepared, _, _ = dataset()
    groups, draws = analysis.family_resamples(analysis.questions_from(prepared))
    stats = analysis.clustered_delta([10, 0, -20], groups, draws, "actual_evidence_tokens")
    assert stats["question_token_increases"] == stats["question_token_decreases"] == 1
    assert "question_wins" not in stats and "longer" in stats["interpretation"]


def test_all_comparisons_retained_and_aggregate_contains_no_actual_identities():
    prepared, support, answer = dataset()
    result, local = analysis.summarize(prepared, support, answer)
    assert result["support"]["comparison_count"] == 30
    assert result["primary_answer_k3"]["comparison_count"] == 15
    assert len(local) == 45 * 3
    negative = next(row for row in result["support"]["paired_comparisons"]
                    if row["plus"] == "p_yes_only_k3" and row["minus"] == "ordinal_then_p_yes_k3")
    assert negative["metrics"]["official_evidence_f1"]["question_losses"] == 3
    assert "secret-" not in json.dumps(result)
    assert "secret-" in json.dumps(local)
    assert result["independent_confirmation"] is result["significance_claimed"] is False
    assert result["multiple_comparison_control_performed"] is False
    assert result["specification"]["p_values_computed"] is result["specification"]["best_k_selected"] is False
    reordered = {"queries": list(reversed(prepared["queries"]))}
    assert analysis.summarize(reordered, list(reversed(support)), list(reversed(answer))) == (result, local)


@pytest.mark.parametrize("mutation", ["missing_support", "missing_answer", "duplicate", "foreign_method", "nonfinite", "invalid_tokens", "quality_range"])
def test_incomplete_or_invalid_result_tables_refused(mutation):
    prepared, support, answer = dataset()
    if mutation == "missing_support": support.pop()
    elif mutation == "missing_answer": answer.pop()
    elif mutation == "duplicate": support.append(deepcopy(support[0]))
    elif mutation == "foreign_method": support[0]["method"] = "selected_best_k"
    elif mutation == "nonfinite": support[0]["official_evidence_f1"] = float("nan")
    elif mutation == "invalid_tokens": answer[0]["actual_evidence_tokens"] = 1025
    else: answer[0]["official_answer_f1"] = 2
    with pytest.raises(ValueError): analysis.summarize(prepared, support, answer)


def test_invalid_family_partition_or_cluster_draws_refused():
    groups = [np.array([0, 1]), np.array([1, 2])]
    draws = np.ones((10000, 2), dtype=int)
    with pytest.raises(ValueError, match="partition"):
        analysis.clustered_delta([0, 0, 0], groups, draws, "official_evidence_f1")
    with pytest.raises(ValueError, match="draws"):
        analysis.clustered_delta([0, 0], [np.array([0]), np.array([1])], draws[:3], "official_evidence_f1")


@pytest.fixture
def completed_fixture(tmp_path, monkeypatch):
    prepared, support, answer = dataset(77, 24)
    directories = {name: tmp_path / name for name in ("support_plan", "support_run", "answer_plan", "answer_run")}
    for directory in directories.values(): directory.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("fixed source", encoding="utf-8")
    config = {"prepared_dir": str(tmp_path / "prepared"), "input_sha256": {str(source): analysis.digest(source)}}
    answer_config = {**config, "support_plan": str(directories["support_plan"]),
                     "support_run": str(directories["support_run"])}
    analysis.pilot.write_json(directories["support_plan"] / "experiment_config.json", config)
    analysis.pilot.write_json(directories["answer_plan"] / "experiment_config.json", answer_config)
    analysis.pilot.write_rows(directories["support_run"] / "per_question.jsonl", support)
    analysis.pilot.write_rows(directories["answer_run"] / "per_question.jsonl", answer)
    called = []

    def verified(plan, run):
        assert Path(plan) == directories["support_plan"] and Path(run) == directories["support_run"]
        called.append("support")
        return config, prepared, {}, support, {"status": "completed"}

    def audited(args):
        assert Path(args.plan) == directories["answer_plan"] and Path(args.run) == directories["answer_run"]
        called.append("answer")
        return {"status": "verified"}

    monkeypatch.setattr(analysis.support_stage, "verify_completed_run", verified)
    monkeypatch.setattr(analysis.answer_stage, "audit", audited)
    monkeypatch.setattr(analysis.pilot.client, "read_key", lambda *_: pytest.fail("analysis must never read credentials"))
    args = SimpleNamespace(**directories, output=tmp_path / "analysis")
    return args, called


def test_complete_artifacts_audited_before_statistics_and_paths_stay_local(completed_fixture):
    args, called = completed_fixture
    result = analysis.analyze(args)
    assert called == ["support", "answer"]
    assert result["question_count"] == 77 and result["family_count"] == 24
    assert len(analysis.pilot_analysis.read_rows(args.output / "paired_per_question.jsonl")) == 77 * 45
    assert result["api_calls"] == 0 and result["input_hashes_unchanged"] is True
    aggregate = (args.output / "analysis.json").read_text(encoding="utf-8")
    assert "secret-" not in aggregate and str(args.support_run) not in aggregate
    local = analysis.pilot.read_json(args.output / "source_binding.json")
    assert local["directories"]["support_run"] == str(args.support_run)
    with pytest.raises(FileExistsError): analysis.analyze(args)


def test_completed_run_verifier_failure_blocks_all_results(completed_fixture, monkeypatch):
    args, _ = completed_fixture
    def failed(*_args): raise ValueError("execution incomplete")
    monkeypatch.setattr(analysis.support_stage, "verify_completed_run", failed)
    with pytest.raises(ValueError, match="incomplete"): analysis.analyze(args)
    assert not args.output.exists()


def test_changed_source_after_audit_is_not_rebound(completed_fixture, monkeypatch):
    args, _ = completed_fixture
    original = analysis.answer_stage.audit
    def changed(value):
        result = original(value)
        (args.support_run / "per_question.jsonl").write_text("changed", encoding="utf-8")
        return result
    monkeypatch.setattr(analysis.answer_stage, "audit", changed)
    with pytest.raises(ValueError, match="changed during analysis"): analysis.analyze(args)
    assert not args.output.exists()


def test_answer_must_reference_the_same_support_execution(completed_fixture):
    args, _ = completed_fixture
    path = args.answer_plan / "experiment_config.json"
    value = analysis.pilot.read_json(path)
    value["support_run"] = str(path.parent / "different")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="different support"): analysis.analyze(args)
    assert not args.output.exists()
