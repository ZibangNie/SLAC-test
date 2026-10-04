# Next offline question: can cheap SLAC features target JEV use?

2026-10-04. **Design only; not executed or validated.** The current budget-selection gate changes only 2/77 packs. A different, more informative next experiment can reuse existing complete BGE and JEV answer caches without generating new answers or opening the 900-question confirmation references. Verify the cache and feature contracts before freezing an executable plan; do not change the design in response to new test outcomes.

## Hypothesis and novelty boundary

Existing BGE score-gap routing -> add cheap SLAC evidence structure and retrieval disagreement -> test whether those features identify questions where JEV improves the final answer, at the same fixed routing quota.

Routing expensive models and proxy cascades have direct precedents, including [RouteLLM](https://arxiv.org/abs/2406.18665) and [ScaleDoc](https://arxiv.org/abs/2509.12610v2). Neither substituting JEV nor saving logical calls establishes novelty. The narrower hypothesis is whether existing SLAC structure supplies predictive information beyond ordinary score ambiguity. A positive result would remain an exposed-development signal requiring independent validation. This experiment cannot establish cross-question semantic relation reuse.

## Frozen candidate design for review before execution

Retain all 77 old questions and 24 families. Actions are exclusively the original complete BGE reranker k3 pack or the original complete JEV raw-score k3 pack, with their previously audited answer responses. Never mix their units or create new packs. Cached answer identity and actual returned-model contracts must be verified from existing audited artifacts, without rescanning every raw response. If complete two-arm cache coverage is unavailable, stop this design rather than evaluate only favorable or matched rows.

Use only four features available **before any JEV judgment**:

| Feature | Fixed definition |
|---|---|
| BGE ambiguity | (third-highest minus fourth-highest score) / (highest minus lowest score); zero for fewer than four candidates or zero range |
| Local retrieval disagreement | 1 minus Jaccard overlap of saved Dense and BGE pack unit-ID sets; zero for an empty union |
| Source span | (maximum minus minimum source order of BGE-selected units) / (document unit count minus one); zero for fewer than two selected units |
| Heading share | fraction of BGE-selected units whose kind is title or heading; zero for an empty pack |

JEV labels, scores, packs, JEV/BGE disagreement, references and answer quality are forbidden as inference-time features. Per-question cached Answer F1 difference, JEV minus BGE, is a training target only. Reject unavailable features rather than invent a replacement after evaluation.

Construct five family-disjoint folds using only identities and counts: sort families by descending question count, break ties by a predeclared salted SHA-256, and assign each to the fold with fewest questions so far, breaking ties by fold index. Save membership before loading quality targets. Five folds provide larger test batches than leave-one-family-out for quota assignment; they are not newly independent data.

Fit weighted ridge regression with an unpenalized intercept and no interactions. Training sample weights sum to one and give every training family equal total weight. Use training-only weighted feature centering/scaling; constant columns become zero. Fix the objective as weighted mean squared error plus the sum of squared coefficients, with regularization strength one. Do not search features, penalties, folds, seeds or thresholds.

For each held-out test fold, route exactly floor(N_test / 2) questions to JEV by predicted benefit, with frozen identity-hash tie breaking. This is offline batch allocation, not an online hard budget guarantee. Actual routed counts will reflect integer rounding. Do not convert a question fraction into a dollar-saving claim: batching and request sizes also matter.

Keep all six strategies:

1. Always BGE.
2. Always JEV.
3. Exact expected performance of a uniformly random same-quota routing subset in each test fold, using each question's inclusion probability k/N_test; no Monte Carlo search.
4. Same-quota low-BGE-gap rule.
5. Same-quota ridge using only BGE gap.
6. Same-quota ridge using all four features.

The primary incremental comparison is four-feature ridge minus gap-only ridge. Also compare with the same-quota random expectation and report both full-use endpoints. Do not promote a different contrast after seeing outcomes.

Report all 77 questions' question-weighted and family-balanced Answer F1, paired deltas, 24-family win/tie/loss, all five fold results, actual evidence tokens, routed/skipped query counts, and candidate-judgment counts if the frozen task map supports them. Logical requests are estimates; this cached experiment spends zero model API calls. No noninferiority or independent-confirmation claim follows from a descriptive gain-retention ratio.

Leakage checks must replace an entire test fold's quality targets and confirm that that fold's predictions and routing decisions are unchanged. Freeze features/folds before target access, and save out-of-fold decisions before final held-out metric aggregation. Synthetic tests cover target isolation, train-only scaling, quotas, hash ties and exact random expectation. Content/numerical QA uses a predeclared fixed sample, not a full raw-response audit.

## Decision rule

If the four-feature router's family-balanced incremental value fails to exceed both gap-only ridge and random expectation, or its question-weighted direction is opposite, stop this feature hypothesis. Do not respond by adding features, moving folds or tuning a threshold on the same outcomes. If only the simple gap rule works, attribute the result to ordinary routing rather than SLAC structure. Either outcome should reduce uncertainty before any future paid experiment.
