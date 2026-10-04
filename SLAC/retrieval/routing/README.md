# Offline JEV routing primitives

`offline_router.py` implements the fixed five-fold routing design in
`docs/research/CACHED_ROUTING_PROTOCOL_20261004.md`. It uses NumPy and the Python
standard library, performs no file or network I/O, and leaves the existing
retrieval pipeline unchanged. It does not compute features from model output
or claim that the proposed routing rule improves answer quality.

Each `RoutingInput` contains a `(family, document, question)` identity and an
immutable four-value feature tuple: BGE score gap, Dense/BGE pack disagreement,
BGE source span, and BGE heading share. Features must be finite in `[0,1]` and
must be available before JEV is called. The caller must establish that
provenance; numerical validation alone cannot establish it.

This complete synthetic example runs from the repository root:

```python
from SLAC.retrieval.routing.offline_router import (
    RoutingInput, assign_family_folds, oof_routes,
)

rows = tuple(
    RoutingInput((f"family-{i // 2}", "document", f"question-{i}"),
                 (i / 9, (i % 3) / 2, (i % 4) / 3, (i % 2)))
    for i in range(10)
)
# Persist feature/fold bindings before loading any real quality targets.
folds = assign_family_folds(tuple(row.key for row in rows))
# These synthetic values are training targets, never inference features.
targets = {row.key: .4 - .8 * row.features[0] for row in rows}
results = oof_routes(rows, targets, folds)
assert len(results) == 5
assert sum(len(result.four_feature_selected_keys) for result in results) == 5
assert all(not set(result.fold.train_families) & set(result.fold.test_families)
           for result in results)
```

`assign_family_folds` requires at least five families. It sorts them by
decreasing question count, then the SHA-256 of
`slac-routing-20261004-fold-v1|` plus the family ID, then the family ID. It
assigns each family to the least-loaded fold, breaking ties by fold index.
Every question is tested once, and all of its family is excluded from that
fold's training set. Return values are immutable `FamilyFold` objects with
sorted train/test keys, family IDs, and the test quota.

`fit_weighted_ridge(train_rows, train_targets, feature_indices=...)` accepts
either `(0,)` for gap-only or `(0,1,2,3)` for all four features. The target map
must contain exactly the explicit training keys; extra held-out targets are
rejected. Targets are JEV Answer F1 minus BGE Answer F1, finite in `[-1,1]`.
Weights give every training family equal total mass and sum to one. The model
uses training-only weighted means and population standard deviations. Constant
columns, and any column whose numerical variance is zero, contribute zero.
The objective is weighted mean squared error plus the sum of squared
coefficients, with penalty one and an unpenalized intercept. There is no
hyperparameter selection.

`predict_ridge(model, rows)` uses only the immutable model statistics and the
supplied pre-JEV features, returning predictions in input-row order. Predicted
benefit is not clipped to `[-1,1]`. Models include their training keys,
families, and weights for local scope verification; these identifiers and
coefficients from real experiments should remain in private artifacts.

`quota_routes(keys, scores)` selects exactly `floor(N/2)` questions by greatest
score. `prefer_low=True`, also used by `low_gap_routes(rows)`, selects the
smallest gaps. Ties use SHA-256 of `slac-routing-20261004-tie-v1|` plus the three
key parts joined by `|`, followed by the complete identity. Chosen keys are
returned in canonical key order. Negative predictions still fill the quota.
This is offline batch allocation, not an online threshold policy or a hard
dollar budget.

`oof_routes` verifies that the supplied folds exactly match the frozen
identity-derived partition. For each fold it passes only training targets to
the two models, then returns predictions aligned to `fold.test_keys`, the
low-gap selection, and both ridge selections. Its orchestration layer receives
all targets for cross-validation; the separate fit and predict functions make
the training/inference boundary explicit. Changing an entire held-out fold's
targets must leave that fold's predictions and routes unchanged.

`random_quota_expected_values(bge_scores, jev_scores)` gives each question's
exact expected Answer F1 in one fold under a uniformly random same-quota
subset. Every row has inclusion probability `floor(N/2)/N`. The caller may
aggregate those values with question or family weights. They are expectations,
not a sampled allocation or new model responses.

Verify the synthetic contracts with:

```text
python -m pytest tests/research/test_offline_router.py -q
```

Cross-validation on previously exposed development questions does not create
independent validation. Routing counts also do not directly establish token,
latency, HTTP-request, or dollar savings; those require separate measurement.
