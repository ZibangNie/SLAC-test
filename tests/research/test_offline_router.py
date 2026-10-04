"""Synthetic routing tests: no real features, answers, credentials, or API calls."""

import hashlib
import itertools
import math
from dataclasses import FrozenInstanceError, replace

import pytest

from SLAC.retrieval.routing import offline_router as router
from SLAC.retrieval.routing.offline_router import (
    RoutingInput,
    assign_family_folds,
    fit_weighted_ridge,
    low_gap_routes,
    oof_routes,
    predict_ridge,
    quota_routes,
    random_quota_expected_values,
)


def row(family, question="q", gap=0.5, *, doc="d", features=None):
    return RoutingInput((family, doc, question),
                        features if features is not None else (gap, .3, .3, .3))


def cohort():
    rows = []
    targets = {}
    for family_index, size in enumerate([6, 5, 4, 3, 2, 2, 1]):
        for question_index in range(size):
            gap = (family_index + question_index) % 11 / 10
            features = (gap, question_index % 3 / 2, family_index % 3 / 2,
                        (family_index + question_index) % 4 / 3)
            item = row(f"family-{family_index}", f"question-{question_index}", features=features)
            rows.append(item)
            targets[item.key] = (family_index - question_index) / 8
    return tuple(rows), targets


def test_family_folds_cover_each_key_once_without_family_leakage():
    rows, _ = cohort()
    keys = tuple(item.key for item in rows)
    folds = assign_family_folds(keys)
    assert len(folds) == 5
    assert set(key for fold in folds for key in fold.test_keys) == set(keys)
    assert sum(len(fold.test_keys) for fold in folds) == len(keys)
    for index, fold in enumerate(folds):
        assert fold.fold_index == index
        assert fold.test_keys and fold.train_keys
        assert not set(fold.test_families) & set(fold.train_families)
        assert not set(fold.test_keys) & set(fold.train_keys)
        assert set(fold.test_keys) | set(fold.train_keys) == set(keys)
        assert fold.quota == len(fold.test_keys) // 2
        assert fold.test_keys == tuple(sorted(fold.test_keys))
        assert set(key[0] for key in fold.test_keys) == set(fold.test_families)
    assert assign_family_folds(tuple(reversed(keys))) == folds


def test_folds_follow_fixed_hash_then_smallest_load_rule():
    rows, _ = cohort()
    keys = tuple(item.key for item in rows)
    sizes = {family: sum(key[0] == family for key in keys) for family in {k[0] for k in keys}}
    ordered = sorted(sizes, key=lambda family: (
        -sizes[family], hashlib.sha256(("slac-routing-20261004-fold-v1|" + family).encode()).hexdigest(),
        family))
    expected = [[], [], [], [], []]
    for family in ordered:
        loads = [sum(sizes[name] for name in bucket) for bucket in expected]
        expected[loads.index(min(loads))].append(family)
    actual = assign_family_folds(keys)
    assert [fold.test_families for fold in actual] == [tuple(sorted(bucket)) for bucket in expected]


def test_full_identity_ties_and_fold_hash_collision_fallback(monkeypatch):
    class EqualHash:
        def hexdigest(self):
            return "0" * 64

    monkeypatch.setattr(router.hashlib, "sha256", lambda _: EqualHash())
    keys = tuple((f"family-{i}", "d", "q") for i in reversed(range(6)))
    folds = assign_family_folds(keys)
    assert folds[0].test_families == ("family-0", "family-5")
    assert quota_routes(keys, (0.0,) * 6) == tuple(sorted(keys)[:3])


def test_quota_routing_uses_frozen_hash_ties_and_fixed_integer_count():
    keys = tuple((f"family-{i}", "d", "q") for i in range(7))
    expected = tuple(sorted(sorted(keys, key=lambda key: (
        hashlib.sha256(("slac-routing-20261004-tie-v1|" + "|".join(key)).encode()).hexdigest(), key
    ))[:3]))
    assert quota_routes(keys, (0.0,) * 7) == expected
    assert quota_routes(tuple(reversed(keys)), (0.0,) * 7) == expected
    assert quota_routes(keys[:1], (-99.0,)) == ()
    # Negative benefits still fill the frozen quota; this is not a threshold rule.
    assert quota_routes(keys[:3], (-3.0, -1.0, -2.0)) == (keys[1],)
    assert quota_routes(keys[:3], (-3.0, -1.0, -2.0), prefer_low=True) == (keys[0],)


def test_low_gap_rule_has_no_target_input():
    rows = (row("a", gap=.8), row("b", gap=.1), row("c", gap=.3), row("d", gap=.7))
    assert low_gap_routes(rows) == tuple(sorted([rows[1].key, rows[2].key]))


def test_closed_form_ridge_uses_equal_family_weight_and_alpha_one():
    rows = tuple(row("a", str(i), gap=0.0) for i in range(3)) + (row("b", gap=1.0),)
    targets = {item.key: -.8 if item.key[0] == "a" else .8 for item in rows}
    model = fit_weighted_ridge(rows, targets, feature_indices=(0,))
    assert model.training_weights == pytest.approx((1 / 6, 1 / 6, 1 / 6, 1 / 2))
    assert math.fsum(model.training_weights) == pytest.approx(1.0)
    assert model.means == pytest.approx((.5,))
    assert model.scales == pytest.approx((.5,))
    # With standardized x = +/-1 and y = .8*x: beta = .8 / (1 + alpha) = .4.
    assert model.coefficients == pytest.approx((.4,))
    assert model.intercept == pytest.approx(0.0, abs=1e-15)
    assert predict_ridge(model, rows) == pytest.approx((-.4, -.4, -.4, .4))
    objective = math.fsum(w * (prediction - targets[key]) ** 2
                          for key, w, prediction in zip(model.training_keys, model.training_weights,
                                                        predict_ridge(model, rows))) + .4 ** 2
    assert objective == pytest.approx(.32)


def test_intercept_is_not_penalized_and_constant_point_three_is_zero_column():
    rows = tuple(row(f"family-{i}", gap=i / 6) for i in range(7))
    model = fit_weighted_ridge(rows, {item.key: .75 for item in rows})
    assert model.means[1:] == (.3, .3, .3)
    assert model.scales[1:] == (0.0, 0.0, 0.0)
    assert model.coefficients == pytest.approx((0.0,) * 4, abs=1e-15)
    assert model.intercept == pytest.approx(.75)
    assert predict_ridge(model, rows) == pytest.approx((.75,) * len(rows))


def test_train_constant_features_stay_zero_when_test_features_change():
    training = (row("a", features=(.3,) * 4), row("b", features=(.3,) * 4))
    model = fit_weighted_ridge(training, {training[0].key: -1.0, training[1].key: .5})
    assert model.scales == (0.0,) * 4
    assert model.coefficients == (0.0,) * 4
    test = (row("test", features=(0.0, 1.0, 0.0, 1.0)),)
    assert predict_ridge(model, test) == (-.25,)


def test_nonconstant_variance_underflow_is_a_zero_column():
    training = (row("a", gap=0.0), row("b", gap=1e-300))
    model = fit_weighted_ridge(training, {training[0].key: -.5, training[1].key: .5},
                               feature_indices=(0,))
    assert model.scales == (0.0,)
    assert model.coefficients == (0.0,)
    assert predict_ridge(model, (row("test", gap=1.0),)) == (0.0,)


def test_prediction_uses_only_training_statistics_and_does_not_clip():
    training = (row("a", gap=0.0), row("b", gap=.25))
    model = fit_weighted_ridge(training, {training[0].key: -1.0, training[1].key: 1.0},
                               feature_indices=(0,))
    assert model.means == (.125,)
    assert model.scales == (.125,)
    assert predict_ridge(model, (row("test", gap=1.0),)) == pytest.approx((3.5,))
    assert model.means == (.125,)


def test_replicating_whole_family_preserves_family_balanced_fit():
    original = (row("a", "a0", gap=0.0), row("a", "a1", gap=.25), row("b", gap=1.0))
    targets = {original[0].key: -.6, original[1].key: -.2, original[2].key: .8}
    extra = (row("a", "a2", gap=0.0), row("a", "a3", gap=.25))
    extended_targets = {**targets, extra[0].key: -.6, extra[1].key: -.2}
    base = fit_weighted_ridge(original, targets)
    duplicate = fit_weighted_ridge(original + extra, extended_targets)
    assert duplicate.means == pytest.approx(base.means)
    assert duplicate.scales == pytest.approx(base.scales)
    assert duplicate.coefficients == pytest.approx(base.coefficients)
    assert duplicate.intercept == pytest.approx(base.intercept)
    assert predict_ridge(duplicate, original) == pytest.approx(predict_ridge(base, original))


def test_fitting_is_order_invariant_and_model_is_immutable():
    rows, targets = cohort()
    model = fit_weighted_ridge(rows, targets)
    assert fit_weighted_ridge(tuple(reversed(rows)), targets) == model
    with pytest.raises(FrozenInstanceError):
        model.intercept = 0.0
    assert type(model.means) is type(model.coefficients) is type(model.training_weights) is tuple


def test_oof_all_folds_enforce_quotas_and_explicit_train_scope():
    rows, targets = cohort()
    folds = assign_family_folds(tuple(item.key for item in rows))
    results = oof_routes(rows, targets, folds)
    assert len(results) == 5
    for result in results:
        fold = result.fold
        assert result.gap_model.training_keys == result.four_feature_model.training_keys == fold.train_keys
        assert result.gap_model.training_families == fold.train_families
        assert len(result.gap_predictions) == len(result.four_feature_predictions) == len(fold.test_keys)
        for selected in (result.low_gap_selected_keys, result.gap_ridge_selected_keys,
                         result.four_feature_selected_keys):
            assert len(selected) == fold.quota
            assert set(selected) <= set(fold.test_keys)


def test_mutating_entire_test_fold_targets_cannot_change_its_predictions_or_routes():
    rows, targets = cohort()
    folds = assign_family_folds(tuple(item.key for item in rows))
    baseline = oof_routes(rows, targets, folds)
    for index, fold in enumerate(folds):
        changed = dict(targets)
        for key in fold.test_keys:
            changed[key] = -1.0 if targets[key] >= 0 else 1.0
        rerun = oof_routes(rows, changed, folds)
        assert rerun[index] == baseline[index]
        # The mutation is real and can affect folds that train on those rows.
        assert any(rerun[i].four_feature_predictions != baseline[i].four_feature_predictions
                   for i in range(5) if i != index)


@pytest.mark.parametrize("mutation", ["overlap", "missing", "duplicate", "family", "quota", "index", "order"])
def test_oof_rejects_custom_incomplete_or_leaking_fold_membership(mutation):
    rows, targets = cohort()
    folds = assign_family_folds(tuple(item.key for item in rows))
    first = folds[0]
    if mutation == "overlap":
        edited = replace(first, train_keys=first.train_keys + first.test_keys[:1])
    elif mutation == "missing":
        edited = replace(first, test_keys=first.test_keys[1:])
    elif mutation == "duplicate":
        edited = replace(first, test_keys=first.test_keys + first.test_keys[:1])
    elif mutation == "family":
        edited = replace(first, test_families=first.test_families + first.train_families[:1])
    elif mutation == "quota":
        edited = replace(first, quota=first.quota + 1)
    elif mutation == "index":
        edited = replace(first, fold_index=1)
    else:
        edited = replace(first, test_keys=tuple(reversed(first.test_keys)))
    with pytest.raises(ValueError, match="partition"):
        oof_routes(rows, targets, (edited,) + folds[1:])


def test_random_expectation_matches_enumerating_every_uniform_subset():
    bge = (.1, .3, .7, .5, .2)
    jev = (.9, .1, .8, .5, .4)
    expected = random_quota_expected_values(bge, jev)
    allocations = list(itertools.combinations(range(len(bge)), len(bge) // 2))
    enumerated = tuple(math.fsum(jev[i] if i in subset else bge[i] for subset in allocations)
                       / len(allocations) for i in range(len(bge)))
    assert expected == pytest.approx(enumerated)
    # Arbitrary row weights (including family balance) preserve linear expectation.
    weights = (.25, .25, 1 / 6, 1 / 6, 1 / 6)
    assert math.fsum(w * value for w, value in zip(weights, expected)) == pytest.approx(
        math.fsum(math.fsum(weights[i] * (jev[i] if i in subset else bge[i]) for i in range(5))
                  for subset in allocations) / len(allocations))
    assert random_quota_expected_values((.3,), (.9,)) == (.3,)


def test_same_question_id_in_different_full_scopes_is_valid_but_duplicate_scope_is_not():
    rows = (row("a", doc="x"), row("a", doc="y"), row("b", doc="x"))
    model = fit_weighted_ridge(rows, {item.key: 0.0 for item in rows})
    assert len(model.training_keys) == 3
    with pytest.raises(ValueError, match="unique"):
        fit_weighted_ridge(rows + rows[:1], {item.key: 0.0 for item in rows})


def test_training_target_scope_cannot_include_held_out_or_missing_rows():
    training = (row("a"), row("b"))
    target = {item.key: 0.0 for item in training}
    with pytest.raises(ValueError, match="scope"):
        fit_weighted_ridge(training, {**target, row("held-out").key: .5})
    with pytest.raises(ValueError, match="scope"):
        fit_weighted_ridge(training, {training[0].key: 0.0})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, None, "0.1", -1.01, 1.01])
def test_invalid_training_targets_are_rejected(bad):
    item = row("a")
    with pytest.raises((TypeError, ValueError), match="target"):
        fit_weighted_ridge((item,), {item.key: bad})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, None, "0.1", -.01, 1.01])
def test_invalid_features_are_rejected_during_fit_and_predict(bad):
    valid = row("a")
    model = fit_weighted_ridge((valid,), {valid.key: 0.0})
    invalid = replace(valid, features=(bad, .3, .3, .3))
    with pytest.raises((TypeError, ValueError), match="feature"):
        fit_weighted_ridge((invalid,), {invalid.key: 0.0})
    with pytest.raises((TypeError, ValueError), match="feature"):
        predict_ridge(model, (invalid,))


@pytest.mark.parametrize("key", [("", "d", "q"), ("f", "  ", "q"), ("f", "d", 1),
                                ("f", "q"), ["f", "d", "q"], "fdq"])
def test_malformed_scoped_identities_are_rejected(key):
    with pytest.raises(ValueError, match="query key"):
        quota_routes((key,), (0.0,))


@pytest.mark.parametrize("indices", [(1,), (0, 1), (0, 0, 1, 2), [0], (False,), (), (0, 1, 2, 4)])
def test_feature_specification_is_frozen_without_hyperparameter_search(indices):
    item = row("a")
    with pytest.raises(ValueError, match="feature_indices"):
        fit_weighted_ridge((item,), {item.key: 0.0}, feature_indices=indices)


def test_empty_inputs_short_family_cohorts_and_mutable_features_are_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        assign_family_folds(())
    with pytest.raises(ValueError, match="five"):
        assign_family_folds(tuple((f"f{i}", "d", "q") for i in range(4)))
    with pytest.raises(ValueError, match="nonempty"):
        fit_weighted_ridge((), {})
    with pytest.raises(ValueError, match="immutable tuple"):
        fit_weighted_ridge((row("a", features=[.3] * 4),), {row("a").key: 0.0})
    with pytest.raises(ValueError, match="four"):
        fit_weighted_ridge((row("a", features=(.3,)),), {row("a").key: 0.0})
    with pytest.raises(ValueError, match="nonempty"):
        quota_routes((), ())


def test_routing_rejects_invalid_scores_length_and_flags():
    key = row("a").key
    with pytest.raises(ValueError, match="align"):
        quota_routes((key,), ())
    with pytest.raises(ValueError, match="unique"):
        quota_routes((key, key), (0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        quota_routes((key,), (float("nan"),))
    with pytest.raises(TypeError, match="boolean"):
        quota_routes((key,), (0.0,), prefer_low=1)


@pytest.mark.parametrize("bge,jev", [((), ()), ((.1,), ()), ((float("nan"),), (.1,)),
                                   ((True,), (.1,)), ((1.01,), (.1,)), ((.1,), (-.01,))])
def test_random_baseline_rejects_unavailable_or_invalid_metrics(bge, jev):
    with pytest.raises((TypeError, ValueError)):
        random_quota_expected_values(bge, jev)


def test_model_dimension_and_numerical_tampering_are_rejected():
    item = row("a")
    model = fit_weighted_ridge((item,), {item.key: 0.0})
    with pytest.raises(ValueError, match="dimensions"):
        predict_ridge(replace(model, scales=(0.0,)), (item,))
    with pytest.raises(ValueError, match="finite"):
        predict_ridge(replace(model, intercept=float("nan")), (item,))
