# Lazy rank bounds: complete exact-pack validation

The new selector reproduced every frozen eager pack in all **501 empty-cache assignments** and all **23,915 assignment × initially-known-subset paths**. Its runtime uses candidate/support/rank inputs and observed Boolean labels, without a truth-table or essential-variable set. At most two new logical edge reads occurred in a path. This establishes exact output on the fixed development domain; it does not establish API savings, relation accuracy or answer-quality improvement.

The [execution-before-results protocol](RELATION_LAZY_BOUNDS_PROTOCOL_20260927.md) was published before evaluation. Source and plan were released in commit `87235a6e4c172cce75f471d0fb003c3f881790bf`. Formal execution, complete replay audit and independent eager-completion/tokenizer verification all succeeded. The [public JSON](results/qasper_relation_lazy_bounds_20260927.json) is a byte-exact copy of the original public aggregate, with all 25 top-level fields retained.

## Fixed domain and output contract

All 77 exposed development questions / 24 families remain included. Original native units, candidate pools, JEV coarse support labels, dense ranks and the local BGE tokenizer are unchanged. The eligible-edge occurrence histogram is `0:31, 1:19, 2:8, 3:10, 4:6, 5:1, 6:1, 7:1`, totaling 107 occurrences. Candidate pools have at most 16 units. The pack remains at most three complete native units / 1,024 actual tokenizer tokens, with original headers, source-order rendering and exact native-text deduplication.

The selector bounds the score of each pending candidate. It certifies the next candidate when that candidate's worst ranking key precedes every competitor's best key; otherwise it asks the first native-order unknown active edge attached to the incumbent or a conflicting candidate. The accepted/skipped candidates are still decided by exact complete-pack tokenization, without a monotonic-length shortcut. It does not fabricate an actual bonus for an unread edge.

Every saved path was checked against the original complete gate for selected identities, rendered pack bytes/hash and actual tokens. Independent verification enumerated consistent completions for each certificate, checked the eager winner and requested-edge order, and reproduced full-pack token counts using the bound BGE tokenizer. Full initially known caches made no new requests. The independent verifier imports no formal SLAC selector, certificate validator or aggregation function.

## Complete logical-read results

The all-subset suite **includes the same 501 empty-cache paths**. They are two reported test suites, not 24,416 distinct scientific cases or independent replications. The exhaustive assignment counts are combinatorial, not observed label probabilities.

| Measure | Empty cache | All initially known subsets |
|---|---:|---:|
| Complete paths | 501 | 23,915 |
| Exact-output equivalent paths | 501 | 23,915 |
| Reads if every initially unknown eligible edge were read | 2,166 | 76,390 |
| Actual logical reads | 326 | 9,682 |
| Reads avoided against that reference | 1,840 | 66,708 |
| Maximum new reads per path | 2 | 2 |
| Paths with fewer reads than that reference | 460 | 23,072 |
| Paths with equal reads | 41 | 843 |
| Paths with 0 / 1 / 2 new reads | 191 / 294 / 16 | 14,297 / 9,554 / 64 |
| Reads of output-nonessential variables | 22 | 70 |
| Paths reading an output-nonessential variable | 22 | 70 |
| Unique source-qualified edges requested | 24 | 24 |
| Unique query-edge occurrences requested | 24 | 24 |
| Winner certificates | 1,552 | 72,294 |
| Consistent-completion/event checks | 80,880 | 1,130,866 |

Essentiality was calculated **after selection** from the original output cube: 19 essential occurrences, each on a different source-qualified edge, matching the earlier compilation. Those identities did not guide the lazy strategy. The additional nonessential reads above are retained; output equivalence does not imply an optimal number of questions.

Across the 77 questions, the empty-cache minimum-read histogram is `0:55, 1:22`; its maximum-read histogram is `0:55, 1:20, 2:2`. Thus two questions have assignment-dependent counts of one or two reads. The strategy adapts within a path to observed Boolean labels. No real cross-question cache schedule or global relation-label assignment was executed.

For a **posthoc descriptive comparison only**, the earlier native-order compiled decision diagram used 320 reads on its 501 empty-cache assignments and 9,676 on its 23,915 all-subset paths, calculated from its [published complete histograms](results/qasper_relation_demand_compilation_20260927.json). The new totals are six higher in each suite. These are different policies on the same finite domain: the earlier compiler consumes full output cubes, while the new runtime works from rank bounds. The comparison neither establishes optimality nor transfers the old strategy's constant per-question depth to this one. It is not an expected API-cost or real-label result.

## Tokenization, CPU work and cache limits

| Measure | Empty cache | All initially known subsets |
|---|---:|---:|
| Selector complete-pack counter calls | 2,053 | 96,209 |
| Certificate validator counter calls | 2,053 | 96,209 |
| New tokenizer encodes during selection | 253 | 0 |
| New tokenizer encodes during certificate validation | 0 | 0 |

Across both executed suites there were 196,524 counter calls, 253 actual tokenizer encodes and 330 cache entries. The entries include one pre-cached empty pack for each of 77 questions. The largest question cache contained 15 entries. The sum of the fixed candidate-subset bounds was 51,596; the general per-question bound is `1 + C(n,1) + C(n,2) + C(n,3)`, at most 697 when `n = 16`.

The exact token-count cache was shared across a question's replay paths and suites. Consequently, the all-subset suite's zero additional encodes reflects warming by the preceding empty-cache suite, not elimination of token checks. Relation observations were independently reset for each path. These CPU measurements are validation with warm token caches, not separate cold-start serving latency or a production throughput benchmark.

| Recorded phase | Seconds |
|---|---:|
| Formal loading and source validation | 0.906 |
| Formal path/certificate validation | 21.922 |
| Formal writing and source verification before summary | 1.359 |
| Formal internal total before summary | 24.187 |
| External formal run wall time | 27.547 |
| External complete replay audit wall time | 32.234 |
| Independent verification internal wall time | 25.484 |
| Independent verification external wall time | 26.218 |

All processes completed within their 300-second limits. Timing values are measurements of those runs; the independent verifier checks scientific outputs and token/counter statistics, not bit-for-bit reproduction of elapsed time. The runtime was Python 3.12.10, Transformers 4.57.3 and Tokenizers 0.22.1, with tokenizer parallelism disabled. No encoder weights or GPU were used.

## Provenance and verification

The metadata-only plan binds 17 directly consumed inputs. The independent verification receipt binds 30 files, all rehashed again for this publication. Original ancestors not directly consumed remain inherited provenance commitments. No frozen source, plan or result was edited. The tables below distinguish public implementation files from private artifact commitments; hashes do not publish the original identities, paragraphs or per-path packs.

| Public file | SHA256 |
|---|---|
| [Protocol](RELATION_LAZY_BOUNDS_PROTOCOL_20260927.md) | `afda965c49e86984a2493b476f147bbe4f3f2819283f18ad6afd313b74e9c9d5` |
| [Selector / prepare / run / audit](run_qasper_relation_lazy_bounds.py) | `bb3662b6fd9fa8f979b99987c1e956ba4e23ebc6df3c553b343681ae14e529a5` |
| [49 synthetic tests](../../tests/research/test_qasper_relation_lazy_bounds.py) | `e73e31ececfb1b486b12d7fd33075224e3d42d9e923fb8923aa7bb9224d17f6c` |
| Public aggregate / byte-exact public JSON | `f5b86686f774dca7111b674caa6d9114232a1361748ab57acaead7df38861de1` |
| Earlier compiled aggregate, used only for the posthoc comparison | `d3189d6d3d30ce61435cb2bc35cd1668fa4239afd7d1455740f59d3bd61ade19` |

| Private artifact commitment | SHA256 |
|---|---|
| Plan | `e0a15722fea215328df995d2343145ade7f88a15056741adc9fcede2976a26cf` |
| Projected core inputs | `77ceb488d551804327e30039e81023cffa801f22c680e74081ed869e397111af` |
| Plan seal | `21d5d91199e11c95b90b593c2f5bbea914a070077d622ee311b28ca791039965` |
| Input binding object | `00f7ca4f9647a6f7e810330457d62421afa5fa48ea9fc8686c45d26b69b9e226` |
| Complete run summary | `8379a13c0d48ebc4a471fdc3d8c0e2b504c1583b3c4423828951528216042ca8` |
| Complete formal replay audit | `91c8ff9d675eed760cc04dc6a2aa780c050793cdefcf79a25088a70bc4f13b0a` |
| Root completed-execution receipt | `9aec128ce822dee0f32ba2aefbbd084c08a17cd8d1054ba1d31b971e5ac49ca2` |
| Independent verifier source | `1975649330ebd2ce526b504e99656afbe13e15a1dbfd9a74ca9fbd85b50f54ef` |
| Independent verifier tests, 16 synthetic cases | `90669373362b822e5e25b84c37f3c22eae510719ca69417881c3df3cccb94669` |
| Complete independent verification receipt | `e8c69eea1f90f4bd0662971f3f36f5a298c5633c6ebf0e324a401f747798a57e` |
| Full 501-path local records | `c237c5397bdf2256e0f74cb171c5d19359746bcb066385984a83ad1d91de76e9` |
| Full 23,915-path local records | `6878ed399df05efe4565b7efd77d3c8a6bc7af74164899b0c884de311acc57cb` |
| Per-question local aggregates | `40c9d2a8d9adaa02c5ed1e8e4c47e2ac272f766d82c5917c698899931c4182e2` |

## Interpretation boundary

This is an exact-output reference policy under a fixed hypothetical Boolean oracle. Runtime selection does not require truth-table enumeration, but the exhaustive validation here is exponential and has at most seven eligible variables per question. No optimal interrogation strategy, new classical algorithm, compact optimal decision diagram or broad scalability benchmark is claimed.

The scope is the existing fixed candidate/support/packer configuration. Changing support, candidates, dense ordering, native text, tokenizer or packing policy requires renewed verification. Identical judgment text sent in different real API batches can yield different labels, so oracle-preserving equivalence does not itself establish provider-label equivalence. Logical reads cannot be converted into API requests or fees without a separate frozen execution design.

There were **zero API calls, no key access, no new QA/reference or generated-answer input, and no quality scoring** in this study. It does not supply JEV-specific relation evidence, cross-stage sharing gains, retrieval improvements or answer improvements. The 77 questions remain exposed development data. The result does not alter data admission, family membership, held-out confirmation or the separate full-domain content/placebo protocol.
