# Typed Judgments for Budgeted Evidence Selection: A SLAC Development Study

**Working manuscript, 27 September 2026.** This is a draft of the completed development study, not a submission-ready paper. It does not establish a novel shared-relation framework. All quality results use exposed development data. Protocols and complete aggregates linked below are authoritative when this condensed narrative omits details.

## Abstract

Retrieving relevant candidates and selecting useful evidence under a small context budget are distinct problems. We study this distinction in a SLAC research pipeline using complete native paragraphs from Qasper. On a fixed development sample of 77 questions from 24 document families, we compare dense ranking, a local cross-encoder reranker, typed support judgments, and two rules using JEV's reported support scores. Evidence packs contain at most three complete units and 1,024 BGE tokens, and a common generator answers from each pack. JEV score ranking obtains question-weighted Answer F1 of 0.5133, compared with 0.4134 for dense ranking. A separately declared posthoc comparison against BGE reranking yields a difference of 0.0762, with positive exploratory family-bootstrap intervals under two weighting schemes. However, score ranking versus ordinal JEV or general-model judgments remains uncertain for Answer F1, and the two score rules yield identical predictions. Additional diagnostics show that greater hierarchical candidate coverage can coexist with worse selected evidence, while a reference-guided oracle leaves substantial unrealized selection headroom. These observations motivate further tests of budgeted selection. They do not demonstrate held-out generalization, calibrated scores, backend superiority, or useful relation sharing across pipeline stages.

## 1. Introduction

A retrieval pipeline must decide both which source units to consider and which of them to put into a generator's context. These decisions interact with evidence length, duplicate content, and the way grouped or hierarchical retrieval results are projected back to source text. Increasing candidate recall may therefore fail to improve the final answer.

This study examines that failure mode in the SLAC research pipeline. We first establish comparable native-paragraph evaluation and strong ranking controls, then test whether typed support judgments and their reported scores improve selection within a fixed pool. We also diagnose hierarchical owner projection and distinguish candidate availability from realized selection. The study's contribution at this stage is a reproducible development analysis with explicit controls and failure accounting; novelty and sufficient empirical scope for publication remain open.

Our eventual architectural hypothesis is stronger: reusable, source-grounded document relations might influence both boundary/group decisions and evidence selection. That hypothesis requires interventions at both stages and controls for ordinary response caching. The score-ranking experiments reported here do not implement or establish it.

## 2. Task and experimental design

Qasper questions are associated with a research paper; their authors saw its title and abstract, and separate annotators supplied answers and evidence. We retain the given-document setting for the main comparisons. A separate question-only search across the 32-paper development corpus is a stress diagnostic, not an interchangeable evaluation protocol. [Dasigi et al., 2021](https://aclanthology.org/2021.naacl-main.365/).

The main denominator is 77 questions from 24 families after excluding all families used by an earlier 15-question pilot. Excluding those pilot families does not make the remaining sample unseen: retrieval and subsequent development analyses have already exposed it. We retain every admitted question, including empty evidence references and unanswerable cases. Earlier 104-question baselines and the 15-question pilot remain separate; they are not pooled with the 77-question results.

For the main ranking comparison, dense top-eight candidates and the frozen adjacent-unit expansion provide at most 16 native units per question, totaling 1,214 question–unit judgments. The candidate identities and text are shared by dense, BGE reranker, and judgment-based methods. The local reranker is `BAAI/bge-reranker-v2-m3` at revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. BM25 and owner-based retrieval use different candidate construction and are reported as additional pipeline baselines, not fixed-pool ranking interventions.

The primary limit is three accepted complete native units and 1,024 actual BGE evidence tokens. A candidate is skipped if its complete rendered addition would exceed the limit or duplicate already selected native text from the same document. Headers and source-ordered rendering are counted, and units are not truncated. Equal maximum budgets do not imply equal realized lengths. The fixed primary cardinality limit remains three; all pre-fixed one- and two-unit sensitivity results remain in the [complete support report](PRIMARY_SUPPORT_RESULTS_20260927.md).

All answer conditions use the same Qwen3.6 Plus endpoint and Alibaba provider contract, evidence-only prompt, temperature zero, disabled reasoning and maximum 512 output tokens. Exact full-payload matches inherit audited historical responses. This reduces new calls while preserving the original prediction; it does not constitute an independent generation replicate or a measured production cache speedup. Actual returned model identities and request provenance are bound in the execution artifacts. [Answer protocol](RECOVERED_PRIMARY_ANSWER_PROTOCOL_20260927.md).

## 3. Selection rules

For each candidate, a support judgment takes one of three values: yes, unknown or no. The ordinal rule permanently excludes no, orders yes before unknown, and breaks ties by the frozen dense rank and source order. The same rule is applied to JEV and the general-model judgments, separating backend choice from selection logic.

The two score rules also exclude no. **Ordinal-then-score** orders by the yes/unknown tier, descending reported yes score, dense rank and source order. **Score-only** removes the ordinal tier from that ordering. Both feed the same complete-unit packer. The raw values are accepted only when all three choice scores are finite and in [0,1], with a sum in the frozen [0.985,1.015] reported-score range. They are not normalized, treated as calibrated probabilities, or replaced by a confidence field. No missing-score fallback or threshold search is used. This contract is an engineering acceptance rule, not a calibration result.

TypeSafe describes JEV as a model for typed judgments and structured workflows. Such an interface motivates separating semantic judgments from deterministic budget enforcement, but vendor claims do not establish task-specific accuracy, calibration or a research contribution. Our primary experiments are ranking interventions over query-conditioned support judgments; they do not reuse query-independent document relations. [TypeSafe introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

## 4. Evaluation and execution integrity

Answer F1 follows the frozen official Qasper evaluator, maximizing token F1 over the original references. Evidence metrics retain their separate reference semantics. Source-qualified evidence identities are used in cross-document diagnostics so that identical strings from unrelated documents do not silently receive credit. Empty-reference behavior is preserved. The empty-evidence generator condition uses the same prompt and returns Unanswerable on all 77 questions; its score measures that abstention control, not unconstrained closed-book knowledge. [Metric contract](QASPER_METRICS.md).

We report question-weighted (QW) means and family-balanced (FB) means. Paired intervals use 10,000 shared bootstrap draws of the 24 complete families, PCG64 seed 20260927, and linear 2.5th/97.5th percentiles. FB first averages questions within each family; QW weights all sampled questions equally. These exploratory intervals are not multiplicity-adjusted tests and do not include repeated-generation variance. Complete fixed comparisons, not only positive intervals, are retained.

The original support run stopped after a timeout with no response for one attempt. A publicly recorded operational amendment preserved that unknown attempt and performed one explicitly linked replacement plus the remaining scheduled calls. The recovered support stage completed all 308 logical calls across 309 physical attempts. Main answers comprise 462 logical predictions, using 168 inherited unique payloads and 120 new unique calls. Full replay audits and independent numerical implementations verified all 240 support intervals and 60 primary-answer intervals. Failure history is part of the experiment, not erased by recovery. [Recovery amendment](PRIMARY_SUPPORT_RECOVERY_AMENDMENT_20260927.md), [complete answers](PRIMARY_ANSWER_RESULTS_20260927.md).

## 5. Results

### 5.1 Fixed-pool judgments and downstream answers

| Method, primary k=3 | Evidence F1 QW | Answer F1 QW | Answer F1 FB | Actual evidence tokens QW |
|---|---:|---:|---:|---:|
| Dense | 0.202453 | 0.413368 | 0.461764 | 420.844 |
| BGE reranker | 0.238374 | 0.437147 | 0.472624 | 510.247 |
| JEV ordinal | 0.350834 | 0.484088 | 0.526805 | 467.052 |
| General ordinal | 0.371904 | 0.486695 | 0.532401 | 444.597 |
| JEV ordinal-then-score | 0.391755 | 0.513325 | 0.559005 | 478.247 |
| JEV score-only | 0.391755 | 0.513325 | 0.559005 | 478.247 |
| Empty evidence | — | 0.090909 | 0.119444 | 0 |

Sources: [support](PRIMARY_SUPPORT_RESULTS_20260927.md), [primary answers](PRIMARY_ANSWER_RESULTS_20260927.md), [reranker](RERANKER_BASELINE_RESULTS_20260927.md), [local answers](LOCAL_BASELINE_ANSWER_RESULTS_20260927.md). The two score rules select identical evidence and share predictions on every question; their two rows are not independent replications.

Score ranking minus dense has Answer F1 differences of +0.099957 QW [0.031072,0.166471] and +0.097241 FB [0.034031,0.160985], with 31 wins, 38 ties and 8 losses. Score ranking minus ordinal JEV improves Evidence F1 under both weighting intervals, but its Answer F1 differences remain uncertain: +0.029237 QW [−0.033753,0.089437] and +0.032201 FB [−0.024981,0.094369]. Score ranking minus general ordinal judgments also has zero-crossing answer intervals. Evidence gains should therefore not be substituted for demonstrated downstream gains against those judgment baselines.

### 5.2 Explicit posthoc comparison with BGE reranking

After parent means were visible, a separate protocol fixed four judgment-method comparisons against BGE. The analysis reused complete audited predictions and validated exact dense/empty bridging across the parent experiments. It is not retrospectively included among pre-fixed primary contrasts.

| Method minus BGE | Answer F1 difference QW [95%] | Answer F1 difference FB [95%] | Wins / ties / losses |
|---|---|---|---:|
| JEV ordinal | +0.046941 [−0.019932,0.115232] | +0.054181 [−0.009500,0.121733] | 20 / 46 / 11 |
| General ordinal | +0.049547 [−0.035074,0.133885] | +0.059777 [−0.029738,0.157519] | 21 / 45 / 11 |
| JEV ordinal-then-score | +0.076178 [0.013165,0.140602] | +0.086382 [0.016376,0.164718] | 22 / 48 / 7 |
| JEV score-only | +0.076178 [0.013165,0.140602] | +0.086382 [0.016376,0.164718] | 22 / 48 / 7 |

Score packs average 32.000 fewer tokens QW and 51.875 fewer FB than BGE packs. The QW token interval crosses zero and the FB interval is negative. Because content and realized length differ, the answer contrast does not isolate a length-independent effect. Both ordinal methods have zero-crossing Answer F1 intervals against BGE. All four contrasts and all 16 answer/length intervals appear in the [posthoc report](POSTHOC_RERANKER_ANSWER_RESULTS_20260927.md).

### 5.3 Candidate coverage does not ensure useful packing

Given-document candidate recall rises from 0.679046 for direct native retrieval to 0.716378 for leaf-owner projection and 0.730880 with an additional chunk channel. Nevertheless, the original owner methods obtain Evidence F1 of only 0.143845 and 0.139207, and Answer F1 of 0.293052 and 0.317545 QW. Both owner answer contrasts against dense have negative intervals under both weighting schemes.

Holding each owner candidate pool fixed and changing priority to cached leaf scores restores Evidence F1 to 0.202453. Both resulting answer conditions reach 0.410526 QW, slightly below dense, with zero wins, 76 ties and one loss against dense. Their intervals end at exactly zero; this is not equivalence. Most payloads equal dense payloads, and evidence lengths change. The intervention identifies a consequential selection choice, while neither exceeding dense nor isolating rendering position. [Owner evidence](NATIVE_OWNER_ORDER_RESULTS_20260927.md), [owner answers](OWNER_ORDER_ANSWER_RESULTS_20260927.md).

An exhaustive reference-guided oracle over complete subsets of size zero to three provides a complementary diagnostic. Across six configurations, all 311,028 combinations were checked against the true token limit. Given-document oracle Evidence F1 is 0.801105, 0.816724 and 0.833298 for direct, leaf-owner and dual-owner pools. This large headroom is not an achievable answer prediction: gold, cardinality choice and shorter packs contribute to it, and no oracle pack is sent to the generator. The incremental oracle gains from the larger pools have zero-crossing intervals under both weights. [Complete oracle results](CANDIDATE_ORACLE_RESULTS_20260927.md).

## 6. Relation mechanisms and prior work

Dependence-aware evidence selection is already studied. DPS proposes dynamic passage selection with paragraph dependencies, and ETS searches evidence sets conditioned on accumulated context. Neither has been reproduced under this native-paragraph protocol; their reported benchmark scores are not comparable entries in our table. The general idea of using dependencies cannot establish novelty for SLAC. [DPS](https://arxiv.org/abs/2508.09497), [ETS](https://aclanthology.org/2025.acl-long.1175/).

The earlier pilot did not demonstrate incremental evidence benefit from its tested shared-relation variants. A new, narrower rule is examined separately: selecting a paragraph can increase the priority of its immediately preceding paragraph when a directed interpretation dependency is active. Its frozen CPU gate enumerates all 501 eligible binary relation masks across the 77 questions, without gold or quality scoring. Seventeen questions have any possible change in the final pack; the other 60 have none. Full adjacency changes the same 17 questions. All masks retain the baseline selected-unit count in this sample, but some change actual length. Complete replay and an independent selector implementation agree. This establishes behavioral opportunity only: it does not validate relations, select the best mask or establish useful sharing. The [gate protocol](RELATION_OPPORTUNITY_PROTOCOL_20260927.md) and [complete results](RELATION_OPPORTUNITY_RESULTS_20260927.md) remain separate from the quality tables above.

A subsequent finite-function compiler tests which relation inputs can affect the complete output: source-qualified ordered identities, exact rendered bytes and actual token count. Independent verification matches all 501 assignments and all 23,915 assignment/cache combinations, covering 4,075 partial cache states. Of 101 unique eligible edges, 19 are essential; these also form only 19 question-edge occurrences. Sixty questions need zero relation reads, fifteen need one and two need two. The minimum and maximum empty-cache depths are equal for every question. Thus this sample demonstrates removal of irrelevant inputs, but provides no observed empty-cache adaptive-depth saving or repeated essential edge across questions. The bound of 19 logical reads is not an HTTP count, a measured cost reduction or a computed global cache worst case. The compiler applies established ordered finite-terminal decision diagrams; the cache-aware reference still uses truth-table cofactors. It is exponential in eligible variables and has only been verified up to seven here. Output equivalence assumes a fixed binary oracle and does not preserve the full-domain placebo label statistics automatically. [Compilation protocol](RELATION_DEMAND_COMPILATION_PROTOCOL_20260927.md), [complete results](RELATION_DEMAND_COMPILATION_RESULTS_20260927.md), [decision-diagram and control review](RELATION_DEMAND_DESIGN_REVIEW_20260927.md).

A separate complete reachable-pack answer experiment covers all 96 distinct question-pack outputs, with 82 exact historical responses and 14 new generations. Every required response completes before reference scoring. Full adjacency obtains Answer F1 0.480499 QW and 0.528888 FB. Its differences from ordinal JEV are -0.003589 [-0.029640, 0.022111] and +0.002083 [-0.025438, 0.036661], with 3 improvements, 68 ties and 6 declines. Its packs are shorter by 15.429 QW and 13.643 FB tokens, with negative length intervals. Against dense the answer intervals are positive, but against BGE and score ranking both cross zero. These are structural-rule outcomes; no learned dependency labels have been obtained. All 16 fixed intervals remain in the [complete report](RELATION_PACK_ANSWER_RESULTS_20260927.md).

The mean per-question maximum observed Answer F1 over these saved pack responses is 0.505481 QW / 0.552162 FB, below the score-ranking means of 0.513325 / 0.559005. Any selection confined to these same packs and saved predictions cannot exceed that descriptive weighted maximum. This is a statement about the completed response realization, not expected quality under new generations, an implementable gold-free selector, or a bound on a changed relation rule. The corresponding minimum is 0.457535 QW / 0.502690 FB. No directional interval or significance claim is attached to either extremum.

A future relation-content experiment must preserve the pool, independent support judgments, generator and packer across no-relation, content-relation, adjacency and fixed-placebo controls. A further experiment must isolate using the same relation in both stages from selection-only use and exact-call caching. Strong ranking controls remain necessary. The current development result supports conducting those tests, not reporting their outcome in advance.

Semantic-operator optimization also limits a systems novelty claim. LOTUS and Palimpzest optimize semantic workloads, while the June 2026 Larch preprint explicitly studies adaptive semantic-predicate ordering and short-circuit evaluation. Our compiler does not establish a competing scalable optimizer or measured deployment advantage. These systems have not been reproduced here. [Primary-source comparison and inspected versions](SEMANTIC_OPERATOR_RELATED_WORK_20260927.md), [Larch problem formulation](https://arxiv.org/html/2606.07923v1#S3.SS1).

## 7. Costs, limitations and next validation

The six paid stages in the overnight accounting chain contain 817 physical attempts: 816 succeeded and one historical support attempt has unknown cost. Known reported charges total $0.490420862, plus that unknown charge. Conservative attempted reservation totals $4.6586873300; it is a budget safeguard rather than measured expenditure. These totals combine different experimental stages and cannot be interpreted as a per-query deployed-system cost. Local computation, downloads and hardware are not monetized, and no end-to-end or multi-query amortization advantage is established. [Execution record](OVERNIGHT_RESEARCH_20260927.md).

The main limitations are the small exposed development sample, numerous exploratory comparisons, one generation per unique payload, shared responses, and unequal realized evidence lengths. Vendor model aliases do not establish immutable underlying weights. Query-conditioned support scores are neither calibrated probabilities nor reusable query-independent relations. The corpus diagnostic changes the information setting, while the native owner projection uses rule groups rather than a newly validated learned Refiner. Existing Refiner label provenance and cross-split concerns remain unresolved; no formal training contribution is claimed.

Document-only screening of 249 remaining validation papers found one moderate train–validation lexical-overlap flag. A subsequent [official-source review](VALIDATION_SOURCE_RELATION_REVIEW_20260927.md) found an explicit author declaration of reused earlier material and additional research, with the reference resolving to the flagged partner. This establishes a documentary relationship, not QA leakage, model contamination or an automatic family decision. Screening is not family independence, sample admission or an unseen-test declaration, and no new QA was opened for it. Future confirmation requires a frozen method, provenance/exposure decisions before outcomes, an independently admitted sample and a separate analysis protocol. The [conditional precision appendix](CONDITIONAL_PRECISION_PLANNING_20260927.md) illustrates scale under explicit variance assumptions; it supplies neither power nor a sufficient-sample decision.

## 8. Conclusion

On this development sample, JEV reported-score ranking provides a positive answer signal against dense ranking and, in an explicitly posthoc comparison, a strong BGE reranker. Its incremental answer value over ordinal model judgments remains uncertain. Hierarchical candidate coverage alone is insufficient, and deterministic packing choices materially affect observed quality. The next research step is independent confirmation and a controlled test of relation content; cross-stage sharing and publication-level novelty remain hypotheses.

## Reproducibility and availability note

Public code, frozen protocol metadata, complete aggregate results and figure sources are linked from the [research index](README.md). Credentials, raw questions/documents, per-question identities, provider responses and model weights are excluded. Hashes link local audited artifacts but do not make private response data reconstructible from Git alone. A submission must specify a lawful, usable data/model acquisition path and a redacted reproducibility package; the present repository should not be described as sufficient for exact independent replication of every stored API response.
