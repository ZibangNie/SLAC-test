# Cached JEV budget selection: frozen offline protocol

2026-10-04. The user clarified that research should continue without large API workloads. This stage uses **zero API calls**, no key access, no training, and no new 900-question references. Historic paid-run pause and accounting files stay intact. This protocol is written before new selection results are inspected.

## Question and scope

Does exact optimization of the existing JEV reported-score utility change the evidence packs produced by score-ranked greedy selection under the existing token budget? This is a mechanism diagnostic and reusable framework component, not a claim that knapsack optimization is novel or that raw scores are calibrated.

Use all 77 already exposed development questions / 24 families, their existing candidate pools (at most 16 native units), cached JEV labels and raw scores. Keep `no` exclusions, exact native-text duplicate exclusion, full native units, source-order rendering, at most three selected units, and 1,024 actual BGE rendered-evidence tokens. No score renormalization, threshold sweep, changed candidate construction, relation bonus, or near-duplicate rule is introduced.

## Fixed selectors

1. **Score greedy:** reproduce `p_yes_only_k3`. Priority is descending raw `yes` score, then saved retrieval rank, then source order. Scan in priority order, skipping an already represented native text or an over-budget addition. Stop at three units.
2. **Score exact:** enumerate all subsets of up to three eligible candidates, with at most one representative of each exact native text. Maximize `math.fsum` of their raw `yes` scores under the complete rendered-pack token cap. Preserve the greedy pack if its utility is optimal. Other exact ties prefer inclusion of earlier candidates in the fixed score priority. This tie rule avoids reporting arbitrary tie changes as optimization gains.
3. **Density greedy:** priority is raw `yes` score divided by the actual complete single-unit rendered token count, then the original score priority. Use the same scan, duplicate exclusion and complete-pack feasibility checks. A nonempty zero-token unit is rejected as an invalid tokenizer contract.
4. **Matched-resource exact:** same objective and tie rule as score exact, but per question tokens and maximum units are bounded by the original greedy pack's actual tokens and unit count. This separates a utility improvement from simply consuming more resources.

The tokenizer counts the full rendering, including unit identifiers, separators and special tokens. Native text is used for duplicate identity; the saved canonical unit text is used for rendering. Unit costs must not be added together, and no infeasibility or monotonicity assumption is used to prune subsets. The empty pack costs zero by the inherited contract. For 16 candidates and k=3 there are at most 697 subsets including empty; an explicit enumeration limit refuses accidental combinatorial expansion.

## Execution and evidence boundaries

The first execution is an opportunity gate with **no reference input and no quality scoring**. Record all methods for all 77 questions, baseline replay agreement, exact utility regret, changed packs, token and unit deltas, budget blocking, and enumeration/cost counts. If exact selection has no strict utility improvements, do not evaluate new quality merely to search among ties. If improvements exist, a separately saved evaluation may use the existing 77-question references; never change these selectors after seeing that evaluation.

No new answer generation is permitted. New packs without complete exact request-cache coverage have no comparable full-denominator Answer F1. Cached historical Answer F1 is context, not a measured result of these new selectors. Utility improvement does not establish evidence or answer improvement. Score additivity is an additional modeling assumption beyond rank ordering; monotone score transformations can change the exact optimum.

QA uses a deterministic sample: choose up to eight families by SHA-256 of `slac-budget-selection-20261004-sample-v1|family_id`, then up to two questions per family by SHA-256 of `slac-budget-selection-20261004-sample-v1|family_id|doc_id|question_id`. Break family hash ties by family ID and question hash ties by the complete (family, document, question) identity. Do not inspect all raw responses or manually check every data row. Full mechanical identity/count checks and the actual all-question experiment are distinct from content/numerical QA. Independently recompute the exhaustive optimum only for the sampled questions, with a separate implementation. Reports expose aggregates, code and content hashes, not question/document identifiers, raw texts, predictions, keys or local model weights.

Success for this stage means a tested reusable selector, reproducible cached execution, and a clear positive or negative opportunity result. It does not complete the broader SLAC/JEV research objective or independent confirmation.
