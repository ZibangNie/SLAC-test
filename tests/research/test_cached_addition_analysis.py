"""Bounded synthetic contracts for cached one-unit-addition aggregation."""

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
from cached_addition_analysis import AdditionEdge, CachedEndpoint, STRATA, analyze_additions


def endpoint(key, units, score, tokens, cache=None):
    return CachedEndpoint(key, tuple(units), cache or "|".join((*key, *units, "answer")), score, tokens)


def edge(key, small, large, *, before=.2, after=.6, before_tokens=10, after_tokens=20,
         label="yes", added=None, task=None):
    new = added if added is not None else next(unit for unit in large if unit not in small)
    return AdditionEdge(key, endpoint(key, small, before, before_tokens),
                        endpoint(key, large, after, after_tokens), new,
                        task or "|".join((*key, new, "support")), label, .7)


def fixture():
    qa = ("family-a", "doc", "question-1")
    qa2 = ("family-a", "doc", "question-2")
    qb = ("family-b", "doc", "question-1")
    edges = (
        edge(qa, (), ("u1",), before=.2, after=.8, before_tokens=0, after_tokens=10),
        edge(qa, ("u1",), ("u1", "u2"), before=.8, after=.6, before_tokens=10, after_tokens=30,
             label="unknown"),
        edge(qa2, ("u1",), ("u1", "u2"), before=.8, after=.4, before_tokens=20, after_tokens=10),
        edge(qb, (), ("u1",), before=.1, after=.6, before_tokens=0, after_tokens=30, label="no"),
    )
    pool = (qa, qa2, qb, ("family-a", "doc", "uncovered"), ("family-c", "doc", "uncovered"))
    return edges, pool


def test_within_question_then_family_weights_and_original_coverage_denominators():
    edges, pool = fixture()
    result = analyze_additions(edges, pool_keys=pool, expected_edge_count=4)
    all_rows = result["strata"]["all"]
    assert result["pool"] == {"question_count": 5, "family_count": 3}
    assert all_rows["edge_count"] == 4
    assert all_rows["covered_questions"] == 3
    assert all_rows["covered_families"] == 2
    assert all_rows["question_coverage"] == .6
    assert all_rows["family_coverage"] == pytest.approx(2 / 3)
    f1 = all_rows["answer_f1_delta"]
    assert f1["covered_question_mean"] == pytest.approx((.2 - .4 + .5) / 3)
    assert f1["covered_family_balanced_mean"] == pytest.approx(((-.1) + .5) / 2)
    assert f1["edge_win_tie_loss"] == {"wins": 2, "ties": 0, "losses": 2}
    assert f1["question_win_tie_loss"] == {"wins": 2, "ties": 0, "losses": 1}
    assert f1["family_win_tie_loss"] == {"wins": 1, "ties": 0, "losses": 1}
    tokens = all_rows["evidence_token_delta"]
    assert tokens["covered_question_mean"] == pytest.approx((15 - 10 + 30) / 3)
    assert tokens["covered_family_balanced_mean"] == pytest.approx(((15 - 10) / 2 + 30) / 2)
    assert tokens["edge_increase_equal_decrease"] == {"increases": 3, "equal": 0, "decreases": 1}


def test_twelve_fixed_strata_keep_support_labels_without_outcome_filtering():
    edges, pool = fixture()
    strata = analyze_additions(edges, pool_keys=pool)["strata"]
    assert tuple(strata) == STRATA
    assert len(strata) == 12
    assert strata["all"]["edge_count"] == strata["empty"]["edge_count"] + strata["nonempty"]["edge_count"]
    for base in ("empty", "nonempty"):
        assert strata[base]["edge_count"] == sum(strata[f"{base}_support_{label}"]["edge_count"]
                                                for label in ("yes", "no", "unknown"))
    assert strata["support_yes"]["answer_f1_delta"]["edge_win_tie_loss"] == {
        "wins": 1, "ties": 0, "losses": 1}
    assert strata["nonempty_support_yes"]["answer_f1_delta"]["covered_question_mean"] == pytest.approx(-.4)
    absent = strata["empty_support_unknown"]
    assert absent["available"] is False and absent["edge_count"] == 0
    assert absent["answer_f1_delta"] is absent["evidence_token_delta"] is None
    assert absent["question_coverage"] == absent["family_coverage"] == 0.0
    # Empty and nonempty coverage overlap at qa: question counts must not be added.
    assert strata["empty"]["covered_questions"] + strata["nonempty"]["covered_questions"] > strata["all"]["covered_questions"]


def test_an_empty_edge_manifest_has_unavailable_contrasts_without_imputation():
    result = analyze_additions((), pool_keys=(("f", "d", "q"),), expected_edge_count=0)
    assert result["edge_count"] == 0
    assert all(not item["available"] and item["answer_f1_delta"] is None
               and item["evidence_token_delta"] is None for item in result["strata"].values())


def test_sign_counts_use_strict_deltas_without_tolerance_or_label_thresholds():
    key = ("f", "d", "q")
    tiny = edge(key, (), ("u",), before=.5, after=math.nextafter(.5, 1.0),
                before_tokens=0, after_tokens=0, label="unknown")
    tiny = replace(tiny, raw_yes_score=0.0)
    result = analyze_additions((tiny,), pool_keys=(key,))["strata"]["support_unknown"]
    assert result["answer_f1_delta"]["edge_win_tie_loss"] == {"wins": 1, "ties": 0, "losses": 0}
    assert result["evidence_token_delta"]["edge_increase_equal_decrease"] == {"increases": 0, "equal": 1, "decreases": 0}


@pytest.mark.parametrize("mutation", ["self", "two", "removed", "wrong_added", "reorder", "repeat_unit", "different_query"])
def test_edges_require_one_addition_same_query_and_preserved_source_order(mutation):
    key = ("f", "d", "q")
    item = edge(key, ("u1", "u3"), ("u1", "u2", "u3"))
    if mutation == "self":
        item = replace(item, superset=item.subset)
    elif mutation == "two":
        item = replace(item, superset=replace(item.superset, unit_ids=("u1", "u2", "u3", "u4")))
    elif mutation == "removed":
        item = replace(item, superset=replace(item.superset, unit_ids=("u2", "u3", "u4")))
    elif mutation == "wrong_added":
        item = replace(item, added_unit_id="u1")
    elif mutation == "reorder":
        item = replace(item, superset=replace(item.superset, unit_ids=("u3", "u2", "u1")))
    elif mutation == "repeat_unit":
        item = replace(item, superset=replace(item.superset, unit_ids=("u1", "u2", "u2")))
    else:
        item = replace(item, superset=replace(item.superset, key=("f", "different-doc", "q")))
    with pytest.raises(ValueError):
        analyze_additions((item,), pool_keys=(key,))


def test_missing_endpoint_missing_edge_and_duplicate_edge_are_rejected():
    edges, pool = fixture()
    with pytest.raises(TypeError, match="complete endpoints"):
        analyze_additions((replace(edges[0], superset=None),), pool_keys=pool)
    with pytest.raises(ValueError, match="manifest"):
        analyze_additions(edges[:-1], pool_keys=pool, expected_edge_count=4)
    with pytest.raises(ValueError, match="duplicate addition"):
        analyze_additions(edges + edges[:1], pool_keys=pool)


@pytest.mark.parametrize("field,value", [("answer_f1", .7), ("evidence_tokens", 11), ("unit_ids", ("u3",))])
def test_reused_cache_identity_rejects_conflicting_scores_tokens_and_packs(field, value):
    edges, pool = fixture()
    second = edges[1]
    changed = replace(second.subset, **{field: value})
    if field == "unit_ids":
        # Preserve nesting so the dedicated cache identity check is exercised.
        second = replace(second, superset=replace(second.superset, unit_ids=("u3", "u2")))
    second = replace(second, subset=changed)
    with pytest.raises(ValueError, match="cache identity conflicts"):
        analyze_additions((edges[0], second), pool_keys=pool)


def test_cache_identity_cannot_be_reused_across_queries_or_changed_for_same_query_pack():
    edges, pool = fixture()
    cross_query = replace(edges[2], subset=replace(edges[2].subset, cache_id=edges[0].superset.cache_id))
    with pytest.raises(ValueError, match="cache identity conflicts"):
        analyze_additions((edges[0], cross_query), pool_keys=pool)
    second_cache = replace(edges[1], subset=replace(edges[1].subset, cache_id="new-answer-same-pack"))
    with pytest.raises(ValueError, match="multiple cache"):
        analyze_additions((edges[0], second_cache), pool_keys=pool)


def test_support_task_consistency_is_checked_without_dropping_any_label():
    key = ("f", "d", "q")
    first = edge(key, (), ("new",), before_tokens=0)
    second = edge(key, ("old",), ("old", "new"))
    analyze_additions((first, second), pool_keys=(key,))
    with pytest.raises(ValueError, match="support task identity conflicts"):
        analyze_additions((first, replace(second, support_label="no")), pool_keys=(key,))
    with pytest.raises(ValueError, match="multiple support task"):
        analyze_additions((first, replace(second, support_task_id="different-task")), pool_keys=(key,))
    optional_score = replace(first, raw_yes_score=None)
    assert analyze_additions((optional_score,), pool_keys=(key,))["edge_count"] == 1


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, -.1, 1.1])
def test_missing_or_invalid_official_endpoint_scores_are_not_imputed(bad):
    edges, pool = fixture()
    changed = replace(edges[0], superset=replace(edges[0].superset, answer_f1=bad))
    with pytest.raises((TypeError, ValueError), match="Answer F1"):
        analyze_additions((changed,), pool_keys=pool)


def test_invalid_token_label_scope_and_original_pool_are_rejected():
    edges, pool = fixture()
    for bad_tokens in (None, True, 1.5, -1):
        changed = replace(edges[0], superset=replace(edges[0].superset, evidence_tokens=bad_tokens))
        with pytest.raises((TypeError, ValueError), match="tokens"):
            analyze_additions((changed,), pool_keys=pool)
    with pytest.raises(ValueError, match="support label"):
        analyze_additions((replace(edges[0], support_label="YES"),), pool_keys=pool)
    with pytest.raises(ValueError, match="original pool"):
        analyze_additions(edges, pool_keys=pool[1:])
    with pytest.raises(ValueError, match="unique"):
        analyze_additions(edges, pool_keys=pool + pool[:1])
    with pytest.raises(ValueError, match="nonempty"):
        analyze_additions(edges, pool_keys=())


def test_aggregates_expose_no_real_identity_fields_and_are_input_order_invariant():
    edges, pool = fixture()
    result = analyze_additions(edges, pool_keys=pool)
    assert result == analyze_additions(tuple(reversed(edges)), pool_keys=tuple(reversed(pool)))
    assert "family-a" not in repr(result)
    assert "question-1" not in repr(result)
    assert "cache_id" not in repr(result)
