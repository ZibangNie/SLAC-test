# Finite relation demand compilation: development results

2026-09-27. Completed serial CPU run, full audit and independent mathematical verification; **zero API calls and no quality evaluation**. The [frozen protocol](RELATION_DEMAND_COMPILATION_PROTOCOL_20260927.md) and [unchanged public aggregate](results/qasper_relation_demand_compilation_20260927.json) define the complete result. This is an exploratory follow-up to the [501-mask opportunity protocol](RELATION_OPPORTUNITY_PROTOCOL_20260927.md), using the same exposed **77 questions / 24 families**.

**Nineteen unique relation variables are necessary for the exact output functions in this batch.** They occur in 17 questions: 15 need one variable and two need two. All 501 hypothetical assignments preserve the audited eager output exactly. Across the 101 unique eligible static edges represented in these cubes, 82 are irrelevant to every covered query's output. These findings concern a fixed selector and fixed input pool; relation labels, their correctness and downstream quality have not been evaluated.

The stronger performance interpretations are unsupported here. The essential union and essential edge–query occurrence count are both 19, so the retained necessary variables show no repeated essential dependency across queries in this batch. Each question's minimum and maximum empty-cache depth are equal, so the observed reduction comes from removing statically irrelevant variables; it does not show label-dependent early stopping reducing empty-cache accesses. Full partial-cache checks establish correctness under hypothetical pre-existing knowledge, not measured cache reuse.

## Exact scope and logical-access counts

The target `F(mask)` includes ordered source-qualified selected identities, exact rendered evidence bytes, their SHA-256 and actual evidence tokens. Its preservation is stronger than text-only equality and does not imply selection-trace equality. The original support, candidates, direction, bonus, ordering, deduplication and packing are inherited unchanged. This stage loads no tokenizer or model; token values come from the already independently verified full-pack counts.

| Quantity | Result |
|---|---:|
| Complete questions / families | 77 / 24 |
| Complete hypothetical query assignments | 501 |
| Eligible edge–query occurrences | 107 |
| Unique eligible static edges | 101 |
| Essential edge–query occurrences | 19 |
| Unique essential static edges | 19 |
| Locally irrelevant edge–query occurrences | 88 |
| Globally irrelevant edges **within the eligible union** | 82 |
| Reduced decision nodes / full-output terminals | 21 / 96 |
| Distinct partial-cache subcubes verified, summed across queries | 4,075 |
| Full-assignment × known-subset paths verified | 23,915 |
| Separate-query maximum-depth sum: safe upper bound | 19 |
| Shared-cache unique-label access: safe upper bound | 19 |

“Access” or “request” in these results means one logical Boolean relation-label read. It does not mean an HTTP request, model invocation or provider charge. The two bounds coincide here; neither is presented as an exact worst-case shared-cache schedule. No concrete global label assignment or 77-query cache schedule was executed. The original 562-edge preparation is a different inventory and remains an unexecuted paid stage; this compilation does not complete its label-coverage contract.

## All depth distributions

| Number of essential variables or reads | Questions: essential count | Questions: minimum empty-cache depth | Questions: maximum empty-cache depth |
|---|---:|---:|---:|
| 0 | 60 | 60 | 60 |
| 1 | 15 | 15 | 15 |
| 2 | 2 | 2 | 2 |
| Total | 77 | 77 | 77 |

| Additional logical reads | Empty-cache full assignments | Assignment × initially known subset paths |
|---|---:|---:|
| 0 | 213 | 14,367 |
| 1 | 256 | 9,420 |
| 2 | 32 | 128 |
| Total | 501 | 23,915 |

These are complete combinatorial inventories. The 501 assignments weight queries by their cube size; the 23,915 paths additionally enumerate every initially known subset. Neither column describes a real relation-label distribution, a cache hit rate, an expected API cost, or a uniform-question average. No favorable assignment or variable ordering was selected.

The input eligibility distribution remains 31, 19, 8, 10, 6, 1, 1 and 1 questions with respectively 0–7 variables. Thus the cache-state counts `Σ3^m = 4,075` and path counts `Σ4^m = 23,915` were fixed from metadata before compilation.

## Verification and implementation boundary

The formal implementation computes full-cube one-bit influence, chooses the first conditional-essential native boundary, stops at a constant subcube and interns equal full terminals/ordered decision triples. Independent pointer traversal matches all 501 eager outputs. Its cache-aware reference scheduler first restricts **all** known bits, then recomputes truth-table cofactors; the independent callback checks each requested variable, absence of repeated/known reads, label consistency and constant-subcube termination over all 23,915 paths.

A separate stdlib verifier imports none of the formal compiler, influence, cache or aggregate functions. It enumerates all ternary restrictions using ordered left/right cofactor vectors, independently derives full-cube essential variables, validates and traverses the saved DAG, and recomputes every per-mask/per-question record, depth histogram, union count, public aggregate and summary. All values match. It also checks the full inventory and input/output hashes before and after verification.

Empty-cache execution uses DAG pointer traversal. The cache-aware checks use a truth-table reference runtime; efficient restriction of a compiled DAG and production inference are **not implemented**. Consequently these checks must not be called a measured warm-cache DAG deployment. This is an application of established ordered finite-terminal decision-diagram methods, not a new BDD algorithm; see the [primary-source design review](RELATION_DEMAND_DESIGN_REVIEW_20260927.md).

| Validation / timing | Result |
|---|---:|
| Formal synthetic tests, also independently rerun | 52 passed |
| Independent-verifier synthetic tests, also independently rerun | 33 passed |
| Run: source loading / compilation plus complete verification / post-computation before summary | 0.062 / 0.766 / 0.015 s |
| Run: recorded total before summary | 0.843 s |
| External run process wall time | 0.922 s |
| External full-audit process wall time | 0.984 s |
| Independent verification, function elapsed time | 0.219 s |

Run and audit exited successfully without the external 300-second watchdog firing. The internal deadline is cooperative. The timings include validation work and have different boundaries; they are single local measurements, not model latency, a throughput benchmark or an eager-versus-demand speed comparison. This stage incurred zero API charges and does not revise the earlier night's reservation or unknown-cost ledger.

## Provenance and research limits

Direct consumption is restricted to nine gate/receipt/protocol files plus the new source, tests and protocol: 12 bindings. The source independent receipt's 33-file commitments and the upstream 2,564 commitments remain inherited provenance; unused ancestor QA, request, tokenizer and weight files were not reread. The independent mathematical receipt binds 23 directly used files. Its final binding check was also independently reviewed. Private cubes, native evidence, identities, diagrams and traces remain in ignored artifacts; the public JSON is a byte-for-byte copy of the approved aggregate.

| Artifact | SHA-256 |
|---|---|
| Frozen plan | `b3b12ccb509bbb67ede1ee66f0b9b01ba2b505cd8d5e797663670f0f23fd4f46` |
| Direct input commitment | `cd73fcdce2205163a31e1437282dfb473f4ff69b7d1d6f3c234a37f4194591c9` |
| Independent mathematical verification receipt | `164199b78cdb3475693e99ec42f5612d374507ce3f0a5f2a46873b9410897c4c` |
| Public aggregate | `d3189d6d3d30ce61435cb2bc35cd1668fa4239afd7d1455740f59d3bd61ade19` |

The equality guarantee assumes one fixed binary relation oracle. It does not imply equal predictions from different batches, prompts, models or stochastic provider executions. It also does not establish correct relation semantics, better evidence/answers, exact API savings, scalable inference or cross-stage sharing. Truth-table size is exponential; the observed maximum is seven variables.

Content-essential variables cannot silently replace the original placebo label domain. If `F(a,b)=a`, a swap placebo yields `F(P(a,b))=b`; a content-irrelevant source label can therefore be needed by the control. Unread labels remain unobserved. Any later limited-label content experiment must freeze a new coverage/control contract, or derive the requirements of the **original full-domain** placebo composition and its required statistics. This exploratory development result neither admits new spending nor changes the original 562-edge protocol.
