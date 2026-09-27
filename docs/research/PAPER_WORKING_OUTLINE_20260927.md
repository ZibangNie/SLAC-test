# Working paper outline: budgeted evidence selection in SLAC

**Status, 27 September 2026:** a research and writing outline, not a completed paper or a claim of publication readiness. It now includes the complete, audited primary support and answer experiments, both independently checked, alongside local baselines, owner-order and candidate-oracle results. All quality observations are exposed development evidence; the cross-stage relation contribution remains unproved.

The [English manuscript draft](MANUSCRIPT_DEVELOPMENT_DRAFT_20260927.md) now provides a continuous methods/results/discussion narrative. The [complete BGE posthoc comparison](POSTHOC_RERANKER_ANSWER_RESULTS_20260927.md), [501-mask relation opportunity check](RELATION_OPPORTUNITY_RESULTS_20260927.md), and [conditional precision appendix](CONDITIONAL_PRECISION_PLANNING_20260927.md) are finished. The latter two establish behavior and hypothetical precision scales, not new quality or power claims. A [source review of the single overlap flag](VALIDATION_SOURCE_RELATION_REVIEW_20260927.md) found author-declared material reuse and additional research; data admission remains unresolved.

编辑说明：明早首先决定论文要检验哪个贡献，而不是给现有所有模块找一个统一的成功叙事。当前建议以“候选覆盖如何转化为有限预算内的有效证据”为问题主线；关系共享、JEV 后端和 Refiner 分别接受可能失败的检验。本提纲不修改既有协议、指标、预算或数据准入，也不授权新的调用或训练。

## 1. The problem and the claim we could eventually defend

Retrieving more potentially useful paragraphs does not ensure that a small evidence pack contains the information needed to answer a question. A hierarchical retrieval pipeline introduces a second decision: projecting retrieved groups back to complete source units and selecting a budget-feasible subset. Candidate coverage, selection quality and answer quality must therefore be measured separately.

The original architectural question remains useful: **can source-grounded document relations improve budgeted evidence selection beyond independent relevance/support ranking, and can the same relations usefully inform an earlier chunking stage?** The current experiments identify a selection problem worth studying. They do not yet establish either relation-content value or cross-stage sharing value. [Original experimental draft](EXPERIMENT_PROTOCOL_DRAFT.md), [mechanism direction review](MECHANISM_RESEARCH_DIRECTION_20260927.md).

Two defensible paper directions are conditional on further evidence:

| Direction | Required contribution | Present decision |
|---|---|---|
| A. Evidence-selection mechanism | A specified relation representation and selection rule improve downstream answers beyond strong ranking and structural controls; a separate contrast identifies any benefit of using the relation in both stages. | Preferred research question, **unproved method contribution**. First test relation content on a fixed candidate pool. |
| B. Quality–cost systems study | A bounded typed-judgment workflow achieves a reproducible quality–cost trade-off against reasonably batched general models and local rerankers, under real document/query workloads. | Viable only if complete results and measured workload costs support it. A lower advertised price or response-cache reuse alone is insufficient. |

A careful empirical failure analysis may remain useful if neither direction succeeds. The present development sample alone does not establish its novelty or sufficient scope for submission. Do not present a baseline correction as a new framework, or claim that JEV's recent release supplies novelty.

## 2. Separate the hypotheses before writing a method section

| Question | Intervention and fixed control | Evidence that would falsify or narrow the claim |
|---|---|---|
| **Selection:** does relation content help choose evidence? | Same query-visible information, candidate identities, backend judgments, packer and generator; compare no relation, content relations, native adjacency and a frozen relation placebo with matched effective edge opportunity. Include BGE reranking and independent support/score ranking. | Content relations fail to improve answers beyond these controls, or the change is explained by candidate expansion, selected count or length. An evidence-only improvement is insufficient. |
| **Cross-stage sharing:** does one relation representation usefully alter two stages? | Same backend; independent stages (I), exact-call cache only (C), and shared relation decisions (S). Trace relation → boundary/group change and relation → selected pack. Separate fixed-pool attribution from an index/retrieval experiment in which candidate pools can change. | S has no incremental behavior or answer benefit over C; only the count of identical calls differs. This supports caching at most, not a shared decision mechanism. |
| **Backend substitution:** is JEV a better judgment backend for this task? | Same task definition, candidate information, selection rule, reasonable batching and complete evaluation; JEV against the fixed general-model backend and local reranking. Report actual returned model identities and failures. | Gains disappear under a matched rule/length analysis, or quality–cost advantage fails under complete workloads. API typing is not evidence of semantic accuracy or calibration. |
| **Boundary editing:** is the Refiner useful independently? | Same admissible training data, encoder, input state including `b0`, capacity and training budget; editing model against a final-boundary classifier and simple segmentation. | No incremental downstream gain, or results depend on repaired legacy labels that have not passed provenance and split review. Keep this module outside the core contribution until these conditions are met. |

Query-conditioned support or dependency judgments must include the query in their identity and cost. They cannot be called query-independent reusable document relations. Reusing an exact completed response in an experiment is also different from transferring relation information between stages.

## 3. Positioning against relevant primary sources

This section combines the repository's existing review with a bounded official-source update; it is not an exhaustive literature search or an assertion of priority. The original six entries were recorded in [the mechanism review](MECHANISM_RESEARCH_DIRECTION_20260927.md). The newly checked ETS paper and two June 2026 preprint abstracts are documented with reading scope and limitations in [recent related work](RECENT_RELATED_WORK_20260927.md). Broader chunking comparisons remain in [related-work gaps](RELATED_WORK_GAPS_20260927.md).

| Primary source | Relevant prior capability | What our paper cannot claim merely by implementing it |
|---|---|---|
| [RAPTOR](https://arxiv.org/html/2401.18059v1) | Hierarchical construction and retrieval; tree traversal versus collapsed-tree querying under a context budget. | Hierarchical retrieval or flattening a hierarchy is new. Our native owner-to-leaf projection is not a reproduction of RAPTOR's generated summaries. |
| [FILCO](https://arxiv.org/abs/2311.08377) | Learning to filter retrieved context for downstream generation. | Relevance and generation usefulness differ, or filtering context is itself a new contribution. |
| [RankRAG](https://arxiv.org/abs/2407.02485) | An instruction-tuned model performs context ranking and answer generation. | LLM-based ranking or using a model across tasks demonstrates shared document relations. |
| [DPS: From Ranking to Selection](https://arxiv.org/abs/2508.09497) | Dynamic paragraph selection with paragraph dependencies. | Moving beyond independent top-k or considering dependencies is new without a specific, tested distinction. |
| [ETS, ACL 2025](https://aclanthology.org/2025.acl-long.1175/) | Context-conditioned sentence-set search models dependencies. | Dependency-aware evidence selection alone establishes novelty. |
| [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) | Effects of relevant-information position in long inputs. | Our owner-priority intervention isolates position: it changes selected units and length, then renders them in source order. A position experiment would hold the selected evidence fixed. |
| [TypeSafe's JEV introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Typed judgments, reported scores and a structured workflow. | Vendor descriptions prove task-specific calibration, latency, cost advantage or our method's novelty. |

Novelty review must eventually compare the final implemented mechanism, rather than a general phrase such as “semantic sharing.” For direction A, the proposed distinction is **explicit relation information reused to change both boundary/group decisions and evidence selection**, with incremental value over selection-only and exact-cache controls. This distinction remains a hypothesis, not an established gap or result. A submission needs an ETS-informed comparison: reproduce or transparently adapt a relevant dependent-selection baseline under a compatible unit/budget contract, or justify the limitation without claiming experimental superiority. Its published scores cannot be imported into our 77-question table.

The two preprints in [the update](RECENT_RELATED_WORK_20260927.md) also narrow direction B: adaptive budget allocation and confidence-guided retrieval are already proposed. A systems contribution needs a measured distinction in our workflow, full decision overhead and a matched quality–cost comparison. Their abstracts do not establish task-specific JEV calibration or validate our native-paragraph setting. None of these literature entries is a reproduced experimental baseline in the current results.

## 4. Experimental setting and boundaries

The main development denominator is **77 questions from 24 families**, selected by the frozen family-based procedure. The earlier relation pilot contains **15 questions from 8 other families**. Keep it in a separate development-history table; do not pool it into 92 questions, compare its mean against the 77-question mean, or select the primary `k` from its best result. Earlier 104-question baseline/oracle results are also separate. The 77-question pool had prior retrieval exposure and is not an independent test set. [Pilot protocol](RELATION_PILOT_PROTOCOL_20260926.md), [development separation](RELATION_MECHANISM_DIAGNOSIS_20260926.md).

Qasper asks questions associated with a particular paper; question authors saw its title and abstract. Our given-document evaluation retains that scope. Searching the same 32-paper corpus with the question string alone is an additional stress diagnostic: deictic or paper-dependent questions may lack source context. Do not filter these questions after observing results. Adding a source title in a future protocol would provide document-location information and must be disclosed for every method. [Qasper paper](https://aclanthology.org/2021.naacl-main.365/), [corpus diagnostic](CORPUS_BRIDGE_RESULTS_20260927.md).

All current packs contain at most **three complete native units** and **1,024 actual BGE evidence tokens**. Equal caps are not equal actual lengths. Report source-qualified `(document, exact native text)` evidence metrics for cross-document diagnostics, and retain the official string metric under its own name. A retrieved fragment cannot count as a complete gold paragraph unless every method returns the full native paragraph and pays its actual token cost; otherwise partial and full coverage need distinct measures.

Use the frozen [official-metric implementation](QASPER_METRICS.md) and its [upstream evaluator revision](https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/scripts/evaluator.py). Preserve all original references, reference-list duplicate denominators and FLOAT annotations. Evidence F1 and recall independently maximize over references; an empty evidence prediction can score 1 against an empty evidence reference. Answer F1 is the maximum token F1 over answer references and has different empty-string semantics. The actual empty-evidence generator control uses the same evidence-only prompt: all 77 outputs in the completed local experiment were `Unanswerable`. Its Answer F1 measures this abstention behavior, not unrestricted closed-book knowledge.

Keep the original `k=3` primary comparison. Report all pre-fixed `k=1/2` sensitivity results without selecting a winning `k`. Paired intervals resample the **24 whole families**, using PCG64 seed 20260927 and 10,000 shared draws. Publish question-weighted and family-balanced point estimates and percentile 95% intervals, signed wins/ties/losses and actual lengths. These exploratory intervals have no multiplicity adjustment and do not measure run-to-run generator variation.

## 5. Completed development findings that may enter the working draft

The following are verified observations within the exposed development setting, not confirmed general improvements. Full precision and all contrasts remain in the linked complete reports.

### 5.1 Baselines establish a selection problem, not a winning architecture

Given-document means; `QW / FB` means question-weighted / family-balanced. Evidence F1 is source-qualified; in this given-document setting it agrees with the official string metric.

| Method | Candidate recall QW | Evidence F1 QW | Answer F1 QW / FB | Actual evidence tokens QW / FB |
|---|---:|---:|---:|---:|
| Dense, `k=3` | 0.679046 | 0.202453 | 0.413368 / 0.461764 | 420.844 / 430.114 |
| BGE reranker, same frozen candidate pool | Same pool as dense | 0.238374 | 0.437147 / 0.472624 | 510.247 / 507.618 |
| BM25, its own candidate retrieval | Separate pool; see source report | 0.185323 | 0.341302 / 0.365962 | 469.442 / 467.876 |
| Leaf owner, original priority | 0.716378 | 0.143845 | 0.293052 / 0.344455 | 331.078 / 335.815 |
| Dual owner, original priority | 0.730880 | 0.139207 | 0.317545 / 0.372038 | 343.039 / 349.410 |
| Empty evidence | Not a retrieval method | Not used to claim retrieval ability | 0.090909 / 0.119444 | 0 / 0 |

Sources: [reranker](RERANKER_BASELINE_RESULTS_20260927.md), [BM25](LEXICAL_BASELINE_RESULTS_20260927.md), [native dual index](NATIVE_DUAL_INDEX_RESULTS_20260927.md), [complete local answers](LOCAL_BASELINE_ANSWER_RESULTS_20260927.md). Do not treat the varying candidate pools as a single controlled ranking experiment.

- Reranker−dense Evidence F1 is +0.035920 QW, CI [+0.002893,+0.064668], but +0.027127 FB, CI [−0.001029,+0.054872]. Its Answer F1 contrasts are +0.023779 [−0.043356,+0.085758] QW and +0.010860 [−0.066095,+0.078062] FB. The answer intervals both cross zero; packs are longer. Retain both weightings.
- Original leaf owner and dual owner have negative Answer F1 contrasts against dense under both weightings. BM25−dense crosses zero under QW, while its FB interval is negative. This does not license dropping BM25 or the owner baselines.
- Dual−leaf has slightly lower Evidence F1 but higher Answer F1 point estimates; the answer intervals cross zero. Evidence scores cannot stand in for answers.

### 5.2 A controlled priority change repairs much of the owner deficit

Holding each owner candidate pool fixed and replacing owner priority by cached leaf scores restores both given-document Evidence F1 means to 0.202453. Candidate recall is unchanged. The downstream controls both reach Answer F1 **0.410526 QW / 0.459485 FB**, compared with dense **0.413368 / 0.461764**. Each control versus dense has 0 wins, 76 ties and 1 loss: QW Δ−0.002842 [−0.009245,0], FB Δ−0.002279 [−0.006838,0]. The upper bound is exactly zero; this is not a crossing interval, equivalence test or non-inferiority result.

Both controls improve over their weaker original owner versions, with positive intervals under both weightings, while increasing mean evidence length by 86.558 and 82.701 tokens. This intervention changes selected evidence and length, not just rendering position. Exact dense payloads are reused on 75/77 and 70/77 questions respectively. Shared responses are not independent generation replications. In the cross-corpus evidence diagnostic the priority change goes in the opposite point-estimate direction, with intervals crossing zero; no new cross-corpus answers were generated. [Evidence-order report](NATIVE_OWNER_ORDER_RESULTS_20260927.md), [complete owner-order answers](OWNER_ORDER_ANSWER_RESULTS_20260927.md).

### 5.3 The exact candidate oracle separates candidate opportunity from realized selection

All six native configurations, 77 questions each, were exhaustively checked over subsets of size 0–3: **311,028 combinations**, 1,150 over budget and 309,878 feasible. Every actual selection is a valid witness; the oracle dominates or ties its Evidence F1. It uses gold and is never sent to the generator.

| Scope and metric, QW | Direct | Leaf owner | Dual owner |
|---|---:|---:|---:|
| Given-document actual Evidence F1 | 0.202453 | 0.143845 | 0.139207 |
| Given-document oracle Evidence F1 | 0.801105 | 0.816724 | 0.833298 |
| Given-document oracle tokens | 152.779 | 157.364 | 159.403 |
| Corpus stress actual Evidence F1 | 0.082127 | 0.104359 | 0.093143 |
| Corpus stress oracle Evidence F1 | 0.419317 | 0.415147 | 0.429989 |
| Corpus stress oracle tokens | 64.247 | 60.636 | 63.974 |

The large oracle−actual gaps do not show that a deployable method can close them. The smaller extra-candidate contrasts answer another question: given-document leaf−direct is +0.015619 QW and +0.018424 FB; dual−leaf is +0.016574 and +0.017427. All four corresponding intervals cross zero. Cross-corpus dual−leaf has lower CI endpoints exactly zero under both weights and only 2 wins, 74 ties and 1 loss; do not call it a stable gain.

All groups retain **11 questions with at least one empty reference**. Oracle empty selections range from 19 to 53: they also arise when every budget-feasible subset has zero F1 and the fewest-token tie-break chooses empty. These are not answerability predictions. The oracle's much shorter packs and gold-informed cardinality contribute to the diagnostic gap. It cannot be attributed entirely to semantic ranking. Full FB means, all four contrasts and lengths are in [the oracle report](CANDIDATE_ORACLE_RESULTS_20260927.md).

### 5.4 Earlier pilot evidence constrains, rather than confirms, the mechanism

Keep the 15-question/8-family relation pilot separate. Its tested S variants did not change per-question Evidence F1 or recall relative to same-support I; swapping relation backends, changing direction or relaxing the merge threshold did not establish a quality benefit. This falsifies success claims for those particular implementations/settings, not every possible use of relations. Reported-score ranking is an exploratory ranking signal, not calibrated probability or a sharing result. [Complete historical diagnosis](RELATION_MECHANISM_DIAGNOSIS_20260926.md).

## 6. Complete primary support and answer evidence

The complete sources are [primary support](PRIMARY_SUPPORT_RESULTS_20260927.md) and [primary answers](PRIMARY_ANSWER_RESULTS_20260927.md). Their reports retain every pre-fixed comparison and hash-bound audit. Support covers 15 methods × 77 questions, all k=1/2/3 and 240 intervals; answers cover six groups × 77 questions and all 60 intervals. The fixed primary k remains 3.

| Primary k=3 method | Evidence F1 QW / FB | Answer F1 QW / FB | Evidence tokens QW / FB | Unanswerable / 77 |
|---|---:|---:|---:|---:|
| Dense | .202453 / .209361 | .413368 / .461764 | 420.844 / 430.114 | 30 |
| JEV ordinal I | .350834 / .379660 | .484088 / .526805 | 467.052 / 450.906 | 20 |
| General ordinal I | .371904 / .382876 | .486695 / .532401 | 444.597 / 458.134 | 19 |
| JEV ordinal then raw yes score | .391755 / .420270 | .513325 / .559005 | 478.247 / 455.743 | 17 |
| JEV raw yes score only | .391755 / .420270 | .513325 / .559005 | 478.247 / 455.743 | 17 |
| Empty evidence | Not a retrieval method | .090909 / .119444 | 0 / 0 | 77 |

Raw-score ranking minus JEV ordinal increases Evidence F1 by .040921 QW [.003472,.083598] and .040610 FB [.001457,.087435]. However, its **Answer F1** differences are .029237 QW [−.033753,.089437] and .032201 FB [−.024981,.094369], both crossing zero. Against general-model judgments, both evidence and answer intervals cross zero. These are different outcomes; a support improvement cannot be relabeled a confirmed answer improvement.

Raw-score ranking minus dense increases Answer F1 by .099957 QW [.031072,.166471] and .097241 FB [.034031,.160985], with 31 wins, 38 ties and 8 losses. JEV ordinal minus dense has a positive QW interval but a zero-crossing FB interval; general ordinal minus dense has positive intervals under both weights. These observations justify further research, but do not establish backend superiority, multiplicity-adjusted significance, a shared-relation mechanism or generalization.

The [separate posthoc BGE comparison](POSTHOC_RERANKER_ANSWER_RESULTS_20260927.md) is now complete, retaining all four contrasts and 16 intervals. Raw-score ranking minus BGE has Answer F1 differences .076178 QW [.013165,.140602] and .086382 FB [.016376,.164718], with 22 wins, 48 ties and 7 losses. Both ordinal methods have zero-crossing intervals under both weights. Raw-score packs are shorter by 32.000 QW / 51.875 FB tokens, with a zero-crossing QW length interval and a negative FB interval. Contrasts were fixed after parent means were visible: this remains a posthoc development observation, not a pre-fixed primary contrast or held-out confirmation. Different actual lengths prevent a length-independent causal interpretation, and the two score rules share all 77 responses rather than independent replications.

The two raw-score methods have identical selected packs for every question at every k, and identical evidence, complete payloads and shared answers for all 77 primary questions. Their equal means and zero differences reflect the same observation, not independent replication, equivalence or a benefit from the ordinal component. Actual lengths differ across methods. JEV returned `probabilities` are untreated reported scores, with no calibration test. No static shared relation was added by this primary experiment.

The [operational recovery amendment](PRIMARY_SUPPORT_RECOVERY_AMENDMENT_20260927.md) remains part of the result. The original run preserved 164 successes and one unknown attempt; one explicitly linked replacement and 143 untouched requests then completed. Support has 308 valid logical responses and 309 physical attempts, retaining the historical unknown cost. This is an amended execution, not an uninterrupted no-retry run.

The [complete answer plan](RECOVERED_PRIMARY_ANSWER_PROTOCOL_20260927.md) kept the same evidence-only prompt, Qwen3.6 Plus/Alibaba, temperature 0, reasoning off and 512 output limit. From 367 + 7 historical successful requests it inherited 168 exact unique payloads, covering 223 logical predictions; 120 new unique requests covered the remaining 239. Full endpoint/prompt/canonical bytes, actual model and responses were audited, including unused ancestors. Dense and empty match their parents exactly on all 77 questions. Shared responses do not estimate repeated-generation variation or production cache speedup.

The full night has 803 physical attempts, 802 successful responses and one retained unknown: known fees total $0.487667787, conservative attempted reservations $4.6053728675, with $0.3946271325 left under the fixed $5 cap. These amounts include completed local and owner-control answers as well as primary support and answers; they are not a standalone deployable-method cost. Support's backend known subtotals are $0.023254812 plus one unknown for JEV, and $0.378657500 for the general model. This is an observed route/task cost contrast, not proof of equal quality or end-to-end latency superiority. No reservation is refunded from a lower reported fee.

Official answer-type summaries stay descriptive: each prediction's best reference can change its assigned type/denominator. All 77 questions, including empty-reference and unanswerable cases, remain in the main denominator. Retain the complete negative and zero contrasts, response-sharing limitations, family bootstrap specification and development exposure in the paper.

## 7. Proposed paper structure and evidence responsibilities

1. **Introduction:** motivate the candidate-to-pack gap; state a narrow testable contribution only after its experiment succeeds. Do not open with an assumed “unified framework improvement.”
2. **Related work:** the six-source distinctions above, plus the final mechanism's closest works. Clearly mark methods discussed but not reproduced.
3. **Task and representation:** source spans, native-unit projection, relation visibility, query-dependent versus static judgments, candidate pool, renderer and hard budget. Specify exactly which stage uses each relation.
4. **Method:** deterministic selection rule, tie-breaking, unknown/no eligibility and optional boundary operation. Separate the backend interface from the proposed mechanism; isolate the Refiner unless it has independent valid evidence.
5. **Experimental design:** data exposure and family split, complete denominators, strong local/general-model controls, I/C/S contrasts, operational amendments and all pre-fixed statistical comparisons. Preserve a development chronology without rewriting posthoc decisions as preregistration.
6. **Results:** completed baseline/ordering/oracle diagnostics first; then the complete audited primary tables. An oracle subsection must be visibly gold-guided and non-deployable. Report negative and sensitivity results beside favorable estimates.
7. **Mechanism and resource analysis:** edge opportunity → eligibility → changed selection → changed answer; separately candidate supply, length/cardinality, actual response reuse and real workload amortization.
8. **Limitations and reproducibility:** exposed development data, reference incompleteness, context-dependent corpus stress, model/version constraints, unknown charges, unmeasured latency scopes and unresolved training provenance. Link immutable configurations and aggregate artifacts; keep raw licensed QA and credentials out of the public repository.

## 8. The next minimal falsifiable experiment

**Recommended next scientific test, following the complete primary tables:** freeze one given-document candidate pool and test whether relation **content** improves a common budgeted selector beyond strong independent ranking. This is a new development experiment, not another way to score the current oracle.

Use the same backend and the same independent support scores in every arm. Compare (a) no relation, (b) explicitly defined source-grounded dependency relations, (c) native adjacency, and (d) a seeded document-fixed placebo controlling relation count and effective edge opportunity. Match the decision rule, candidate text, accepted-edge/eligibility accounting, maximum units and actual-token budget; record effective degrees/opportunities rather than assuming a randomization preserves them. Keep the fixed BGE reranker and independently ranked support/score baselines. Freeze the seed, edge proposal rule, direction, eligibility behavior and all pair directions before calls. Missing judgments for a changed pool require a complete new judgment plan; old labels cannot fairly cover only a favorable subset.

Do not silently make static document relations query-conditioned. If the proposed mechanism requires “A is needed to interpret B's support for this query,” define that task separately, include the query in the request/cache key and bill it per query. First demonstrate an incremental benefit; only then test whether any query-independent component can be shared across stages or amortized across queries.

Primary outcome: official Answer F1 on **all** admitted questions, with paired family uncertainty and complete method coverage. Auxiliary outcomes: source-qualified evidence scores, actual length/count, failure rate and relation-trigger traces. The primary experiment remains cap-matched, not automatically length-matched. Before seeing outcomes, define a secondary same-cardinality/token-band control using only a reference-free baseline's pack statistics; preserve all questions with a stated deterministic fallback and report infeasible matches. If no such control is implemented, limit conclusions to the combined selection-and-length change.

**Falsification rule:** if content relations do not improve answers beyond no-relation and structural/placebo controls, or any apparent benefit disappears when the planned count/length control is applied, do not claim a semantic relation-selection contribution. If a content effect survives, a subsequent I/C/S experiment must still establish the separate cross-stage sharing claim. If only API cost improves, use direction B and measure a real workload. Do not start Refiner training to compensate for a failed selection hypothesis.

Independent human review should distinguish support, complementarity and directional dependency, and allow alternative valid evidence. Qasper non-matches are not semantic-negative labels. The existing disagreement-enriched material is for error discovery, not a representative accuracy denominator; no unperformed human annotations may be implied.

## 9. Conditions for a blinded confirmation stage

- Complete an exposure ledger before opening new QA: prior development, prompt examples, training/teacher labels, family/version links and document-level overlap. The remaining-validation document screening does not itself admit an independent set. In particular, the 249 screened documents and one overlap flag require exposure/family adjudication, not automatic acceptance or removal. [Screening report](REMAINING_VALIDATION_SCREEN_20260927.md), [overlap review](VALIDATION_OVERLAP_REVIEW_20260927.md).
- Freeze the final method, checkpoint, input/prompt contract, main metric/pairs, sample-selection rule, all exclusions and failure handling before a separate evaluator reveals held-out answers. Select families without using observed method disagreement, gold attainability or the oracle gap. Retain unanswerable, figure-dependent and empty-reference cases in the declared complete denominator.
- Determine sample size or precision targets from development assumptions before confirmation; do not assert that 77 questions or 24 families are sufficient power. Predeclare multiplicity handling and any equivalence/non-inferiority margin rather than interpreting a zero-crossing or zero-width interval afterward.
- Require a second task/domain and a clearly specified corpus setting before broad RAG claims. Resolve partial-native coverage and header/source ambiguity before cross-document answer generation. Do not give source-title information to only one method.
- Keep Refiner training closed until label provenance and cross-split source problems are resolved. Mechanical repaired-label replay, small synthetic-noise recovery or a historical thesis claim is not training admission or independent evidence. Preserve the original `cleared_for_training=false` boundary where applicable.
- For a systems claim, measure construction, embedding/indexing, query judgments and generation separately; run cold/warm and multi-query workloads with fixed concurrency, complete failures and actual costs. The existing request-cycle p50/p95 are not end-to-end RAG latency, and saved-response replay cannot supply missing cold-cache measurements. [Measured local-answer resources](LOCAL_ANSWER_RESOURCES_20260927.md).

The completed [finite demand compilation](RELATION_DEMAND_COMPILATION_RESULTS_20260927.md) is a computational diagnostic, not a positive relation-quality result. It proves exact outputs on all 501 assignments and 23,915 cache paths. Only 19 of 101 unique eligible edges are essential, with 19 occurrences and no repeated essential edge across queries. Per-query empty-cache minimum and maximum depths are identical (0:60, 1:15, 2:2), so the present result supports static pruning, not observed adaptive stopping or cross-query amortization. Classic ordered finite-terminal diagrams and an exponential truth-table reference do not establish a new scalable algorithm. Any content-plus-placebo reduction must consider the full original permutation before pruning; reconstructing a null on the essential subset changes the control.

The [complete 96-pack answer study](RELATION_PACK_ANSWER_RESULTS_20260927.md) adds a concrete negative constraint on this rule. Full adjacency minus I has QW/FB differences -0.003589/+0.002083 with zero-crossing intervals. Per-query maxima over the fixed saved predictions average 0.505481/0.552162, below score ranking 0.513325/0.559005; that realization-specific ceiling is descriptive, not an expected-generation bound or a deployable policy. The [original-domain placebo protocol](RELATION_PLACEBO_DEMAND_PROTOCOL_20260927.md) was frozen before these new generations and must not be redesigned using them. If its full functions coincide, stop the semantic paid phase as prescribed. Broader optimization novelty must also address [existing semantic-operator systems](SEMANTIC_OPERATOR_RELATED_WORK_20260927.md), rather than relying on the novelty of JEV or lazy calls.

## 10. Morning decision record to complete

Choose one primary contribution candidate, name its strongest fixed comparator, and name the result that would cause us to remove that contribution from the paper. Current recommendation: investigate relation content in fixed-pool evidence selection; keep cross-stage sharing as an additional hypothesis and Refiner as a deferred component. The complete primary results support studying selection, while the incremental answer benefit of reported-score ranking over ordinal judgments remains uncertain. A relation experiment must pass a reference-free opportunity check and the frozen budget gate; if it cannot, narrow the next step rather than add modules or paid calls without a testable intervention.

No abstract should claim improved shared reasoning, calibrated judgments, full SLAC superiority, independent generalization or amortized latency until the corresponding evidence above exists. The undergraduate report and migration notes provide engineering history and hypotheses; unverifiable claims from them are not promoted to results in this outline.

新增[原域placebo需求结果](RELATION_PLACEBO_DEMAND_RESULTS_20260927.md)：完整501符号赋值及独立复核表明只有1/77题的content/placebo函数可能不同，不能将这组控制描写为广泛区分语义效用的证据。U为20条唯一边，预算仅作新合同投影，实际关系标签和质量尚未产生。

[上下界直接选择器](RELATION_LAZY_BOUNDS_RESULTS_20260927.md)将完整真值表移出运行策略：所有501空cache及23915缓存路径正确，独立真实分词与全completion证书验证通过。应完整报告空缓存326次逻辑读取、22次非必要读取及两题1–2条的自适应范围，不把有限验证写成最优算法、新理论、实测HTTP节省或部署benchmark。
