# Cached single-unit additions: support and marginal answer change

2026-10-04. Complete zero-API diagnostic; independent fixed-sample verification passed. This is a descriptive result on exposed development caches, not a new retrieval method or an independent confirmation experiment.

## Main observation

Among **16 additions to nonempty packs whose added unit had a saved JEV `yes` support label**, 15 saved Answer F1 contrasts were exactly tied, one decreased, and none increased. The covered-question mean difference was **−0.0230769231** and the covered-family-balanced difference was **−0.025**. This stratum covers 13 questions and 12 families; its negative mean comes from one observed decreasing contrast.

Thus positive standalone support and a lower saved answer score can coexist in this cache. This does not establish that the support judgment was wrong: the support prompt asks whether a unit is useful, not whether adding it to any existing evidence set guarantees higher Answer F1. Nor does one cached response identify causal semantic harm or prove that a set-conditioned judge could prevent it.

## Complete frozen diagnostic

The metadata-only manifest was frozen before loading these endpoint scores or added-unit judgments. It retained all 32 eligible directed additions, 54 distinct endpoint packs and 32 distinct added-unit support tasks, covering **22/77 questions and 15/24 families**. No endpoint was missing or conflicting; none was removed after observing quality. Every required original-source occurrence agreed with the same saved cache identity and score. The sources include relation intermediate pack records, not just its final method table.

The [execution protocol](CACHED_ADDITION_PROTOCOL_20261004.md) fixes 12 crossed cells: base all/empty/nonempty × label all/yes/no/unknown. The following table shows the main margins and nonempty label cells; the [complete aggregate JSON](results/qasper_cached_additions_20261004.json) includes all 12, coverage, token differences and question/family sign counts.

| Observed additions | Edges | Covered questions / families | Edge higher / tied / lower F1 | Covered-question ΔF1 | Covered-family ΔF1 |
|---|---:|---:|---:|---:|---:|
| All | 32 | 22 / 15 | 5 / 25 / 2 | +0.090664 | +0.105376 |
| Empty base | 9 | 9 / 7 | 4 / 5 / 0 | +0.299663 | +0.313853 |
| Nonempty base | 23 | 17 / 14 | 1 / 20 / 2 | +0.008596 | +0.021152 |
| Nonempty, support yes | 16 | 13 / 12 | 0 / 15 / 1 | −0.023077 | −0.025000 |
| Nonempty, support no | 6 | 3 / 3 | 1 / 4 / 1 | +0.148709 | +0.148709 |
| Nonempty, support unknown | 1 | 1 / 1 | 0 / 1 / 0 | 0 | 0 |

All nine empty-base additions had support `yes`; the empty/no and empty/unknown cells are **unavailable**, not zero effects. Across all support-yes additions, there are 25 edges over 18 questions/14 families, with 4 higher, 20 tied and 1 lower F1. The support-no and support-unknown margins equal their nonempty cells above.

Each cell first averages its edges within a question, then averages covered questions, or gives covered families equal weight after averaging their covered questions. These conditional contrasts are not full-77 method F1. Cells overlap; their question/family counts cannot be added. The yes and no cells involve different questions, so their means must not be compared as a causal label effect.

All 32 additions increased actual evidence tokens. The nonempty/yes cell added a covered-question mean of **133.346154 tokens**, despite its 15 tied and one decreasing saved quality outcome. This is an observed resource/quality mismatch within these cached endpoints, not a demonstrated deployment saving or a recommendation to discard all positive-support additions.

## Evidence boundary and decision

The existing packs were produced by earlier policies and are not a random or exhaustive sample of possible additions. The cache has **zero complete two-by-two inclusion squares**, so it cannot estimate second-order complementarity. One generated answer per payload also leaves content, source-order position and generation variability entangled. Edge sign counts are dependent descriptive observations; no confidence interval or significance claim is made.

The narrowly stated compatibility question is answered, with only one positive-support decrease. This motivates inspecting set-conditioned usefulness as a different target from standalone support, but is too weak to justify a claimed algorithm improvement, feature search on these 77 exposed questions or a large API campaign. A [primary-source novelty check](SET_CONDITIONED_JEV_GATE_20261004.md) has now confirmed direct precedents for conditional selection; the next step is a fixed synthetic integration contract that separates ordinary conditioning from verifiable SLAC source relations. Do not choose only the decreasing case, tune a score cutoff, or infer that the JEV model's recency constitutes innovation.

The zero-API budget selector and structural router hypotheses remain stopped after their negative results. The broader framework research continues. The paused 900-question paid plan remains separate and unexecuted.

## Reproducibility and verification

- [Metadata preparation](prepare_cached_additions.py) and [pure aggregation](cached_addition_analysis.py) are separate from the [numeric endpoint join](run_cached_additions.py).
- The frozen preparation contains all 32 edge identities, both original cache bindings, added support-task identity and the original 77-question coverage denominator. Preparation SHA-256: `777686edc433967883d3251e538181b3d00435439681775db534df46903d96a7`.
- The once-executed result summary SHA-256 is `a631cbf7138ccf9c4e0d6b7ef0b99933cc708bdab0a580ffb734f493197b6abf`.
- 51 synthetic tests passed: 13 metadata-preparation, 25 aggregation and 13 numeric-join tests. Correctness tests are not evidence of retrieval quality.
- Independent synthetic verification additionally checked 1,364 edges across a hand-worked fixture and 48 seeded small graphs, all 12 cells, reversed input order and 12 invalid-binding cases. Mean tolerances were 1e-14 relative/1e-15 absolute; sign counts had zero tolerance.
- The original fixed 14-question QA sample intersects these additions at **five questions/four families/eight edges**. Independent verification passed for 13 endpoints, eight support tasks, 59 duplicate source-score occurrences and the 12 sample aggregation cells. The remaining 24 edges were not numerically rechecked; complete identity/count/hash integrity checks were performed separately. No outcome-based sample expansion occurred.
- New model API calls: **0**. No key, reference answer file or raw provider response was opened. Legacy score containers were parsed and immediately projected; saved answer strings were not inspected, retained in new records or used in analysis.

Private identities and per-edge records remain under ignored `artifacts/research-foundation/offline-20261004/addition-prepared-01/` and `addition-run-01/`. Public code and aggregate hashes do not redistribute the underlying private cache or by themselves provide its data.
