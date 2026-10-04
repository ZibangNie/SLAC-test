"""Frozen, local routing experiment primitives with explicit training scope.

Only caller-supplied pre-JEV features enter prediction. Quality differences are
training targets, never inference features. This module performs no I/O, model
calls, feature extraction, or tuning. Its five-fold evaluation is an offline
batch quota allocation, not an online cost guarantee or independent validation.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np


QueryKey = tuple[str, str, str]
FOLD_SALT = "slac-routing-20261004-fold-v1|"
TIE_SALT = "slac-routing-20261004-tie-v1|"
N_FOLDS = 5
RIDGE_ALPHA = 1.0
GAP_FEATURES = (0,)
FOUR_FEATURES = (0, 1, 2, 3)


@dataclass(frozen=True)
class RoutingInput:
    """A scoped identity and four features available before JEV inference.

    ``key`` is (family ID, document ID, question ID). Features, in order, are
    normalized BGE gap, Dense/BGE pack disagreement, BGE source span, and BGE
    heading share. The caller is responsible for their provenance; numeric
    validation cannot detect reference-derived or post-JEV feature leakage.
    """

    key: QueryKey
    features: tuple[float, float, float, float]


@dataclass(frozen=True)
class FamilyFold:
    fold_index: int
    train_keys: tuple[QueryKey, ...]
    test_keys: tuple[QueryKey, ...]
    train_families: tuple[str, ...]
    test_families: tuple[str, ...]
    quota: int


@dataclass(frozen=True)
class RidgeModel:
    """Immutable training statistics; coefficients use standardized features.

    Constant training columns, or columns whose floating variance underflows
    to zero, have zero scale and contribute zero to all predictions, even if
    that feature varies in a held-out fold. Training
    weights align with the canonical ``training_keys`` order, sum to one, and
    give each training family equal total weight up to floating-point rounding.
    """

    feature_indices: tuple[int, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    training_keys: tuple[QueryKey, ...]
    training_families: tuple[str, ...]
    training_weights: tuple[float, ...]


@dataclass(frozen=True)
class FoldRoutingResult:
    """Both OOF predictions align with ``fold.test_keys``; routes are key-sorted."""

    fold: FamilyFold
    gap_model: RidgeModel
    four_feature_model: RidgeModel
    gap_predictions: tuple[float, ...]
    four_feature_predictions: tuple[float, ...]
    low_gap_selected_keys: tuple[QueryKey, ...]
    gap_ridge_selected_keys: tuple[QueryKey, ...]
    four_feature_selected_keys: tuple[QueryKey, ...]


def _key(value: object) -> QueryKey:
    if (type(value) is not tuple or len(value) != 3
            or any(not isinstance(part, str) or not part.strip() for part in value)):
        raise ValueError("query key must be a tuple of three nonempty identity strings")
    return value


def _number(value: object, name: str, bounds: tuple[float, float] | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, excluding bool")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if bounds is not None and not bounds[0] <= result <= bounds[1]:
        raise ValueError(f"{name} is outside the required numeric bounds")
    return result


def _keys(keys: Sequence[QueryKey]) -> tuple[QueryKey, ...]:
    if not isinstance(keys, Sequence) or not keys:
        raise ValueError("keys must be a nonempty sequence")
    result = tuple(_key(key) for key in keys)
    if len(set(result)) != len(result):
        raise ValueError("query keys must be unique within their full identity scope")
    return result


def _rows(rows: Sequence[RoutingInput]) -> tuple[RoutingInput, ...]:
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("rows must be a nonempty sequence")
    result = tuple(rows)
    for row in result:
        if not isinstance(row, RoutingInput):
            raise TypeError("each row must be a RoutingInput")
        _key(row.key)
        if type(row.features) is not tuple or len(row.features) != 4:
            raise ValueError("features must be an immutable tuple of four pre-JEV values")
        for value in row.features:
            _number(value, "feature", (0.0, 1.0))
    _keys(tuple(row.key for row in result))
    return result


def _targets(targets: Mapping[QueryKey, float], keys: tuple[QueryKey, ...]) -> dict[QueryKey, float]:
    if not isinstance(targets, Mapping):
        raise TypeError("targets must be a mapping with exactly the training query scope")
    for key in targets:
        _key(key)
    if set(targets) != set(keys):
        raise ValueError("target keys must match the explicit row scope exactly")
    return {key: _number(targets[key], "target", (-1.0, 1.0)) for key in keys}


def _feature_indices(value: tuple[int, ...]) -> tuple[int, ...]:
    if (type(value) is not tuple or any(type(i) is not int for i in value)
            or value not in (GAP_FEATURES, FOUR_FEATURES)):
        raise ValueError("feature_indices must be the frozen gap-only or four-feature specification")
    return value


def assign_family_folds(keys: Sequence[QueryKey]) -> tuple[FamilyFold, ...]:
    """Build the frozen five-fold partition using identities and counts only.

    Families sort by descending question count, then salted SHA-256, then ID.
    Each goes to the fold with fewest questions so far, breaking ties by fold
    index. At least five families are required; no fold may be empty. Returned
    train/test keys and family IDs use canonical lexicographic order.
    """
    canonical = tuple(sorted(_keys(keys)))
    counts = Counter(key[0] for key in canonical)
    if len(counts) < N_FOLDS:
        raise ValueError("five family-disjoint folds require at least five families")
    family_order = sorted(
        counts,
        key=lambda family: (
            -counts[family],
            hashlib.sha256((FOLD_SALT + family).encode("utf-8")).hexdigest(),
            family,
        ),
    )
    assigned: list[list[str]] = [[] for _ in range(N_FOLDS)]
    loads = [0] * N_FOLDS
    for family in family_order:
        index = min(range(N_FOLDS), key=lambda i: (loads[i], i))
        assigned[index].append(family)
        loads[index] += counts[family]

    folds = []
    for index, families in enumerate(assigned):
        test_families = tuple(sorted(families))
        test_set = set(test_families)
        test_keys = tuple(key for key in canonical if key[0] in test_set)
        train_keys = tuple(key for key in canonical if key[0] not in test_set)
        folds.append(FamilyFold(
            fold_index=index,
            train_keys=train_keys,
            test_keys=test_keys,
            train_families=tuple(sorted(set(counts) - test_set)),
            test_families=test_families,
            quota=len(test_keys) // 2,
        ))
    return tuple(folds)


def fit_weighted_ridge(
    train_rows: Sequence[RoutingInput],
    train_targets: Mapping[QueryKey, float],
    *,
    feature_indices: tuple[int, ...] = FOUR_FEATURES,
) -> RidgeModel:
    """Fit weighted MSE + sum(beta**2), with an unpenalized intercept.

    ``train_targets`` must contain exactly these explicit training rows, never
    the complete cross-validation target map. Weights give each family equal
    mass and sum to one. Feature means/scales are learned only from training
    rows. There is no parameter search, prediction clipping, or interaction.
    """
    rows = tuple(sorted(_rows(train_rows), key=lambda row: row.key))
    keys = tuple(row.key for row in rows)
    targets = _targets(train_targets, keys)
    columns = _feature_indices(feature_indices)
    family_counts = Counter(key[0] for key in keys)
    raw_weights = [1.0 / (len(family_counts) * family_counts[key[0]]) for key in keys]
    normalizer = math.fsum(raw_weights)
    weights = np.array([weight / normalizer for weight in raw_weights], dtype=np.float64)
    features = np.array([[row.features[i] for i in columns] for row in rows], dtype=np.float64)
    target = np.array([targets[key] for key in keys], dtype=np.float64)

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            constant = np.max(features, axis=0) == np.min(features, axis=0)
            means = np.sum(weights[:, None] * features, axis=0)
            # Exact constant detection avoids amplifying a rounded mean residue.
            means[constant] = features[0, constant]
            variance = np.sum(weights[:, None] * (features - means) ** 2, axis=0)
            scales = np.sqrt(variance)
            scales[constant] = 0.0
            standardized = np.zeros_like(features)
            variable = (~constant) & (scales > 0)
            standardized[:, variable] = (
                (features[:, variable] - means[variable]) / scales[variable]
            )

            # Center again for numerical residuals in weighted standardization.
            # This keeps the intercept unpenalized even after rounding.
            standardized_mean = np.sum(weights[:, None] * standardized, axis=0)
            target_mean = float(np.dot(weights, target))
            centered = standardized - standardized_mean
            gram = centered.T @ (weights[:, None] * centered)
            rhs = centered.T @ (weights * (target - target_mean))
            coefficients = np.linalg.solve(gram + RIDGE_ALPHA * np.eye(len(columns)), rhs)
            intercept = target_mean - float(standardized_mean @ coefficients)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("weighted ridge produced invalid numerical statistics") from exc
    if not all(np.all(np.isfinite(value)) for value in (means, scales, coefficients, intercept)):
        raise ValueError("weighted ridge statistics must be finite")
    return RidgeModel(
        feature_indices=columns,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=float(intercept),
        training_keys=keys,
        training_families=tuple(sorted(family_counts)),
        training_weights=tuple(float(value) for value in weights),
    )


def predict_ridge(model: RidgeModel, rows: Sequence[RoutingInput]) -> tuple[float, ...]:
    """Predict in supplied row order, using only immutable training statistics."""
    if not isinstance(model, RidgeModel):
        raise TypeError("model must be a RidgeModel")
    inputs = _rows(rows)
    columns = _feature_indices(model.feature_indices)
    width = len(columns)
    if not all(type(values) is tuple and len(values) == width
               for values in (model.means, model.scales, model.coefficients)):
        raise ValueError("model statistic dimensions must match its feature specification")
    means = np.array([_number(v, "model mean", (0.0, 1.0)) for v in model.means])
    scales = np.array([_number(v, "model scale", (0.0, 1.0)) for v in model.scales])
    coefficients = np.array([_number(v, "model coefficient") for v in model.coefficients])
    intercept = _number(model.intercept, "model intercept")
    features = np.array([[row.features[i] for i in columns] for row in inputs], dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            standardized = np.zeros_like(features)
            variable = scales > 0
            standardized[:, variable] = (features[:, variable] - means[variable]) / scales[variable]
            predictions = intercept + standardized @ coefficients
    except FloatingPointError as exc:
        raise ValueError("ridge prediction overflowed") from exc
    if not np.all(np.isfinite(predictions)):
        raise ValueError("ridge predictions must be finite")
    return tuple(float(value) for value in predictions)


def quota_routes(
    keys: Sequence[QueryKey], scores: Sequence[float], *, prefer_low: bool = False,
) -> tuple[QueryKey, ...]:
    """Choose exactly floor(N/2), with the frozen hash and full-key tie break.

    Higher scores mean greater predicted JEV benefit unless ``prefer_low`` is
    true (the low-gap baseline). Negative predicted benefit is allowed: this
    is a fixed-quota experiment, not a threshold policy. Results are sorted by
    full query key, independently of rank, for deterministic serialization.
    """
    identities = _keys(keys)
    if type(prefer_low) is not bool:
        raise TypeError("prefer_low must be boolean")
    if not isinstance(scores, Sequence) or len(scores) != len(identities):
        raise ValueError("scores must align with all supplied keys")
    values = tuple(_number(value, "routing score") for value in scores)
    ranked = sorted(
        range(len(identities)),
        key=lambda i: (
            values[i] if prefer_low else -values[i],
            hashlib.sha256((TIE_SALT + "|".join(identities[i])).encode("utf-8")).hexdigest(),
            identities[i],
        ),
    )
    return tuple(sorted(identities[i] for i in ranked[:len(identities) // 2]))


def low_gap_routes(rows: Sequence[RoutingInput]) -> tuple[QueryKey, ...]:
    inputs = _rows(rows)
    return quota_routes(tuple(row.key for row in inputs), tuple(row.features[0] for row in inputs),
                        prefer_low=True)


def oof_routes(
    rows: Sequence[RoutingInput],
    targets: Mapping[QueryKey, float],
    folds: tuple[FamilyFold, ...],
) -> tuple[FoldRoutingResult, ...]:
    """Fit both frozen models outside each test fold and allocate its quota.

    The supplied partition must match ``assign_family_folds`` exactly. This
    rejects custom partitions, incomplete or repeated test keys, and family
    leakage. A fold's model receives only its explicit training target slice.
    This orchestration function has all targets for cross-validation; unlike
    the fit API, it should not be an inference-time entry point.
    """
    inputs = _rows(rows)
    by_key = {row.key: row for row in inputs}
    expected = assign_family_folds(tuple(by_key))
    if type(folds) is not tuple or folds != expected:
        raise ValueError("folds must exactly match the frozen, complete family-disjoint partition")
    validated_targets = _targets(targets, tuple(by_key))
    results = []
    for fold in folds:
        training = tuple(by_key[key] for key in fold.train_keys)
        test = tuple(by_key[key] for key in fold.test_keys)
        training_targets = {key: validated_targets[key] for key in fold.train_keys}
        gap = fit_weighted_ridge(training, training_targets, feature_indices=GAP_FEATURES)
        four = fit_weighted_ridge(training, training_targets, feature_indices=FOUR_FEATURES)
        gap_predictions = predict_ridge(gap, test)
        four_predictions = predict_ridge(four, test)
        results.append(FoldRoutingResult(
            fold=fold,
            gap_model=gap,
            four_feature_model=four,
            gap_predictions=gap_predictions,
            four_feature_predictions=four_predictions,
            low_gap_selected_keys=low_gap_routes(test),
            gap_ridge_selected_keys=quota_routes(fold.test_keys, gap_predictions),
            four_feature_selected_keys=quota_routes(fold.test_keys, four_predictions),
        ))
    return tuple(results)


def random_quota_expected_values(
    bge_scores: Sequence[float], jev_scores: Sequence[float],
) -> tuple[float, ...]:
    """Per-row exact expectation under a uniform same-quota subset of one fold.

    Each of N rows has inclusion probability floor(N/2)/N. This expectation
    can be aggregated with either question or family weights. It is not an
    actual routed subset, a variance estimate, or an independence assumption.
    """
    if (not isinstance(bge_scores, Sequence) or not isinstance(jev_scores, Sequence)
            or not bge_scores or len(bge_scores) != len(jev_scores)):
        raise ValueError("both score sequences must be nonempty and aligned within one fold")
    bge = tuple(_number(value, "BGE Answer F1", (0.0, 1.0)) for value in bge_scores)
    jev = tuple(_number(value, "JEV Answer F1", (0.0, 1.0)) for value in jev_scores)
    probability = (len(bge) // 2) / len(bge)
    return tuple(base + probability * (judge - base) for base, judge in zip(bge, jev))
