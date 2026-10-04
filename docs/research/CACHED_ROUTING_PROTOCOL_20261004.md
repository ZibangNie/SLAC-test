# Cached SLAC/JEV routing: frozen development protocol

2026-10-04. The previous turn made progress: reusable selectors and their completed negative budget-selection experiment are committed at `013c6828`. This next stage implements [the saved routing design](NEXT_OFFLINE_ROUTING_DESIGN_20261004.md). It uses **zero model API calls**, no key access, no new answer generation, no raw-response rescan, and no new 900-question references. Historical paid-run pauses remain intact. This protocol is fixed before new routing outcomes are computed.

## Inputs and exposure

Retain all 77 already exposed development questions and 24 original families, including unanswerable questions and empty packs. The only two deployable actions are the original complete `reranker_k3` BGE evidence pack and the original `p_yes_only_k3` JEV-score evidence pack. Reuse their corresponding previously audited `official_answer_f1` and actual evidence-token fields; do not construct mixed packs. Verify the existing completed parent audits and common-generator comparison, including `qwen/qwen3.6-plus`, the same original prompt/decoding contract, and common Dense/empty bridge. Do not reopen every provider response or recompute references to repeat that audit.

Freeze identities, candidate identities, feature values, and folds into a feature-only artifact **before reading the two per-question answer-quality files**. The feature builder may parse legacy evidence-baseline containers but projects only the named pack identity, content hash and token fields. No JEV judgment, score, pack or quality enters features or folding. The cache-contract metadata records the exact source hashes; executable plans bind that metadata, consumed inputs, source and tests. Malformed, missing or changed bindings stop execution rather than shrink the denominator.

## Four pre-JEV features

Use the entire original common BGE candidate pool; do not apply JEV `no` exclusions.

1. **BGE gap:** (third minus fourth highest `raw_logit`) / (highest minus lowest `raw_logit`). Fewer than four candidates or zero range gives zero. All logits must be finite and cover exactly the saved candidate identities.
2. **Dense/BGE disagreement:** one minus the Jaccard overlap between original Dense k3 and BGE k3 selected-ID sets; an empty union gives zero.
3. **BGE source span:** (maximum minus minimum source order among BGE-selected units) / (number of native document units minus one); fewer than two selected units or at most one document unit gives zero.
4. **BGE heading share:** fraction of BGE-selected native units with `kind` equal to `title` or `heading`; empty pack gives zero.

All features are finite in [0,1]. Validate the selected IDs and exact source-order rendering hashes against the frozen prepared units. Constant or uninformative columns remain as specified; do not substitute a new feature after observing them. Task counts come only from the original pre-judgment support-task map.

## Folds, models and routing quota

Construct five folds using only families and question counts. Sort families by decreasing size, then SHA-256 of `slac-routing-20261004-fold-v1|family_id`, then family ID. Place each family into the fold with the fewest assigned questions, breaking ties by fold number. Save the complete partition before reading quality targets. Every question is tested once and its entire family is absent from that fold's training data.

For a training set with G families and n_f questions in family f, each question has weight 1/(G*n_f). The target is the saved JEV Answer F1 minus BGE Answer F1. Standardize each feature using training-only weighted mean and population variance. Detect a constant column directly from its training values; set it to zero rather than magnify floating rounding residuals. If a nonconstant column's floating variance is zero, it also receives zero scale. Fit the objective weighted mean squared error plus the sum of squared coefficients, with fixed coefficient penalty one and an unpenalized intercept. Use no feature interaction, hyperparameter search or validation-based model selection.

Fit exactly two ridge models per fold: BGE gap only and all four features. Predict held-out benefit without passing held-out targets to the fit function. In each held-out fold of size N, select exactly floor(N/2) questions for JEV. Rank by decreasing predicted benefit; the fixed low-gap control ranks by increasing BGE gap. Ties use SHA-256 of `slac-routing-20261004-tie-v1|family_id|doc_id|question_id`, then the complete identity. This is offline batch quota assignment, not an online budget guarantee.

## Six mandatory strategies and statistics

- Always BGE.
- Always JEV.
- The exact expectation under a uniform random same-quota subset per fold: each question's JEV probability is k/N. This is an expectation, not a realized random policy; do not invent selected IDs for it.
- Low-gap same-quota rule.
- Gap-only ridge at the same quota.
- Four-feature ridge at the same quota.

For all six, report full-denominator question-weighted and family-balanced Answer F1, mean evidence tokens (the random row is expected tokens), all five fold means, routed/skipped question counts and support-judgment counts. The random row reports expectations for counts where necessary. Do not infer HTTP requests, dollars or latency savings from these counts. No physical JEV calls are made in the experiment.

The primary incremental comparison is **four-feature ridge minus gap-only ridge**. Report all 15 pairwise comparisons in the fixed six-strategy order, each with both weighted differences, question-level and 24-family win/tie/loss counts. For the random strategy these compare expected values, not sampled realizations. Compare unrounded numerical deltas to zero. Do not add confidence intervals, significance tests, an oracle router, best-of-seed selection, or an independently validated/noninferiority claim.

Record a descriptive gate: pass only if four-feature ridge exceeds both gap-only ridge and random expectation under family-balanced Answer F1, and neither corresponding question-weighted difference is negative. A pass is a reason for later independent work, not proof of innovation. If the gate fails, stop this fixed feature hypothesis; do not add features, move folds, change the penalty or tune quota using these outcomes. A simple gap rule succeeding alone is not a SLAC structural contribution.

## Execution separation and QA

1. Prepare immutable cache/feature/fold bindings without reading answer-quality targets.
2. Load only the two bound complete parent per-question answer records. Verify original identities, fixed method/pack mappings, returned-model audit metadata and metric ranges. Fit training-only fold models and save all out-of-fold predictions and routing decisions before aggregate held-out scoring.
3. Replace each fold's held-out targets with `-original_target` when nonzero and `1.0` when zero, assert that every held-out value changed, and require that fold's predictions and selected sets to stay exactly unchanged. This is five mechanical train/test-isolation checks, not another quality search. Also use synthetic tests for weighted regression, constants, quotas, ties, folds, and random expectations.
4. Score every fixed strategy using its saved action or exact random expectation. Preserve all outcomes and report the two original endpoints unchanged.

Content and independent numerical QA follow the previous deterministic sample: up to eight families by SHA-256 of `slac-budget-selection-20261004-sample-v1|family_id`, then up to two questions per family by SHA-256 of `slac-budget-selection-20261004-sample-v1|family_id|doc_id|question_id`, with identity tie breaks. This is the same fixed 8-family/14-question sample, not a newly chosen favorable subset. Recompute feature and prediction math for that sample with an independent implementation. Mechanical complete identity/count/hash checks and full-denominator experimental scoring are allowed; a new full row-by-row content or response audit is not.

Publish only source, synthetic fixtures, aggregates and hashes. Real identities, texts, features, targets, per-question outcomes, model coefficients tied to training identities and local paths remain under the ignored research artifact directory. Existing cached answers provide one saved realization per original payload; reused predictions and cross-validation do not create independent model samples. The whole study remains exposed-development research.
