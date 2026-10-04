# Cached one-unit addition diagnostic

2026-10-04. Execution specification, fixed before this diagnostic opens endpoint quality or added-unit judgments. This implements [the previously saved design](NEXT_CACHED_ADDITION_DIAGNOSIS_20261004.md).

The preceding metadata inventory identifies exactly 32 directed one-unit additions among 454 original query-packs: 22/77 questions and 15/24 families, with 54 distinct endpoint packs. Nine edges have an empty base; 23 have a nonempty base. There are no complete two-by-two inclusion squares. Freeze every eligible edge using mappings alone; any missing or conflicting endpoint causes failure of the diagnostic, never selective exclusion.

## Fixed question and estimands

Does a saved per-unit JEV `yes` judgment coexist with a negative saved Answer F1 difference after that unit is added to an existing evidence pack? Report the empty-base contrasts separately. This question concerns compatibility of two observed signals; it does not test whether the support label is incorrect. The support prompt asks whether a unit is directly useful, without requiring it to contain the complete answer. It does not promise a positive marginal answer-quality change in every evidence set.

Before opening quality or judgments, independent design review requested the full Cartesian table of base type (`all`, `empty`, `nonempty`) and support label (`all`, `yes`, `no`, `unknown`): **12 fixed cells**, including unavailable cells. This expands the six marginal cells in the initial implementation draft to locate positive-support decreases within empty versus nonempty bases. No cells are selected after observing outcomes.

For every cell, average larger-minus-smaller official Answer F1 and evidence-token differences first across edges within each question. Report the mean across covered questions and the equal-family mean of the covered-question means. Coverage denominators remain 77 questions and 24 families; absent questions have no imputed score. Report edge, question and family sign counts descriptively. Edges and cells overlap and are not independent experiments. Raw JEV yes scores are retained unchanged in private projected records for traceability, with no thresholds, calibration or confidence interpretation.

## Data and execution boundary

Use the four completed answer-cache sources: local, owner, recovered-primary and relation. Inherit their pinned complete audits and model/prompt contracts. Relation intermediate packs require `pack_records.jsonl`; its `per_question.jsonl` alone is insufficient. Match original payload-cache identity, question, ordered units, pack hash and token count, and require every duplicate score for a required cache identity to agree. Old JSON containers may include answer strings; project only identities, pack metadata and saved numeric scores, without inspecting, retaining or publishing answer text. Do not regenerate answers, score references again or open provider responses.

After the full metadata manifest is frozen, join the previously saved JEV support labels and raw scores by the added candidate's exact support-task identity. Labels never affect edge admission. Record file hashes and immutable output receipts. The executable uses only standard-library file operations and pure aggregation; its command-line entry point disables sockets.

Validate synthetic weighting, nesting, duplicate identities, missing endpoints and conflicts. Independent numerical QA uses only the intersection of the existing fixed 14-question sample with these edges. Do not expand the sample. Full identity/count/hash integrity checks and execution of the experiment are distinct from all-row numerical QA.

## Interpretation and stopping rule

These are posthoc, exposed development caches with one saved generated response per payload. A difference can reflect generation or evidence-position changes as well as content; it is not a causal semantic-harm estimate. Means across yes/no strata involve different questions and do not measure a label treatment effect. No confidence intervals, significance tests, oracle selection, interaction estimate, new selection algorithm or quality-gain claim is supported.

Record the result once. A positive-support decrease would establish only that standalone support does not guarantee improvement in these observed additions, providing a reason to study a separately controlled set-conditioned decision. Its absence would leave that motivation unsupported by this small cache. Neither outcome proves a set-conditioned judge works. This stage has zero model API calls and does not resume the paused 900-question paid plan.
