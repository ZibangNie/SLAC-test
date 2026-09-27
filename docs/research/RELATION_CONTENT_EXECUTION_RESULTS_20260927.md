# Twenty demanded relation judgments: complete development results

The completed 20-judgment stage does **not support a semantic-content benefit for the current +1 relation-bonus rule**. Rcontent and the original-domain Rplacebo produce identical packs, exact cached answer payloads and Answer F1 on all 77 questions. Every non-placebo Answer F1 contrast has a 95% exploratory interval crossing zero under both weightings. The observed equality is not a statistical equivalence or noninferiority result, and it does not generally reject JEV or other relation rules.

This report preserves all seven methods, 539 method-question records, six contrasts and 24 intervals. It is posthoc exploration on 77 already exposed development questions / 24 families. The [execution protocol](RELATION_CONTENT_EXECUTION_PROTOCOL_20260927.md) and reviewed source/plan were frozen before the 20 new judgments; earlier development results were already known. The original complete 562-edge labeling stage remained unadmitted and unexecuted.

The demanded union contains 19 content edges and one additional original-domain placebo edge. Twenty original singleton prompts were executed with `slac-local-decision-v1`, Typesafe and actual model `typesafe/jev-1.13-20260917`: 7 `dependent`, 13 `independent`, and no legal `unknown`. These counts describe **U20 only**, not the full original strata or 562 edges. Unrequested labels remained unobserved; all consistent completions had to resolve to unique saved content/placebo packs. The original-domain placebo was structurally different for one question on four of 501 assignments, but this observed label realization yielded no downstream content/placebo contrast.

There were **zero new answer-generation calls**. All required exact payloads were covered by 262 complete cached Qwen answers, including the complete 96 query-pack study and the four baselines. The actual inherited generator is `qwen/qwen3.6-plus`; identical payloads reuse the same saved response. R0 is original I-JEV; Radjacent activates full eligible adjacency; Rcontent uses observed dependent judgments; Rplacebo uses the frozen original-domain permutation. All preserve the fixed candidate and packing contracts, at most three whole units and 1024 actual BGE evidence tokens.

QW denotes question-weighted means; FB gives equal weight to each of 24 family means. Every comparison resamples whole families with the same 10,000 PCG64 draws, seed `20260927`. All signs and ties are retained, with no multiple-comparison control. The [aggregate JSON](results/qasper_relation_content_20260927.json) preserves every original public summary field at full numeric precision; only the local output inventory is omitted.

| Method | Answer F1 QW | Answer F1 FB | Evidence tokens QW | Evidence tokens FB |
|---|---:|---:|---:|---:|
| R0 | 0.484087860 | 0.526804598 | 467.051948 | 450.905556 |
| Rcontent | 0.474248308 | 0.519768514 | 461.129870 | 446.040972 |
| Radjacent | 0.480499134 | 0.528887986 | 451.623377 | 437.262500 |
| Rplacebo | 0.474248308 | 0.519768514 | 461.129870 | 446.040972 |
| dense_k3 | 0.413367996 | 0.461763839 | 420.844156 | 430.113889 |
| reranker_k3 | 0.437147329 | 0.472623599 | 510.246753 | 507.618056 |
| p_yes_only_k3 | 0.513325328 | 0.559005332 | 478.246753 | 455.743056 |

All contrasts below are **Rcontent minus the named comparator**. The first three are the protocol's primary comparisons; dense, BGE reranker and p-yes-only are secondary. F1 tables use nine decimals so the very small positive R0 interval upper bounds remain visible.

| Comparator | Role | F1 QW delta [95% interval] | F1 FB delta [95% interval] | Wins / ties / losses |
|---|---|---:|---:|---:|
| R0 | Primary | -0.009839552 [-0.024602265, +0.000057391] | -0.007036084 [-0.017723314, +0.000087283] | 1 / 74 / 2 |
| Radjacent | Primary | -0.006250826 [-0.036193908, +0.022639035] | -0.009119472 [-0.043260951, +0.018327618] | 4 / 71 / 2 |
| Rplacebo | Primary | +0.000000000 [+0.000000000, +0.000000000] | +0.000000000 [+0.000000000, +0.000000000] | 0 / 77 / 0 |
| dense_k3 | Secondary | +0.060880312 [-0.005969423, +0.126253418] | +0.058004675 [-0.017032335, +0.128513362] | 20 / 50 / 7 |
| reranker_k3 | Secondary | +0.037100979 [-0.030190965, +0.107860260] | +0.047144915 [-0.016419399, +0.115857860] | 19 / 45 / 13 |
| p_yes_only_k3 | Secondary | -0.039077020 [-0.102311336, +0.028379643] | -0.039236818 [-0.102682243, +0.020041045] | 6 / 54 / 17 |

Rcontent versus R0 has upper bounds **+0.0000573914852557696 (QW)** and **+0.00008728288382648293 (FB)**; neither is negative or zero. The all-zero placebo interval arises from identical observed per-question outcomes. It does not establish population equivalence or adequate power.

| Comparator | Token QW delta [95% interval] | Token FB delta [95% interval] | Longer / same / shorter |
|---|---:|---:|---:|
| R0 | -5.922078 [-15.545538, +1.445792] | -4.864583 [-13.704948, +2.020833] | 2 / 70 / 5 |
| Radjacent | +9.506494 [+2.988221, +16.782749] | +8.778472 [+1.096788, +17.532865] | 10 / 66 / 1 |
| Rplacebo | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 77 / 0 |
| dense_k3 | +40.285714 [-7.292002, +86.298295] | +15.927083 [-54.240972, +75.154601] | 36 / 16 / 25 |
| reranker_k3 | -49.116883 [-92.933544, -8.756048] | -61.577083 [-120.662604, -12.175451] | 29 / 5 / 43 |
| p_yes_only_k3 | -17.116883 [-52.793096, +13.770788] | -9.702083 [-40.125191, +15.359601] | 19 / 34 / 24 |

Positive token deltas mean longer evidence, not improved quality. Rcontent is longer than Radjacent and shorter than BGE under both interval estimates. These length differences prevent a length-independent semantic interpretation; a common maximum budget is not a matched-length control.

The [complete paired figure](results/qasper_relation_content_20260927.svg) displays all 24 intervals from the full-precision aggregate; its [plot source](plot_qasper_relation_content.py) is included. Consult the numeric table for the very small positive R0 upper bounds.

| Rcontent versus | Exact payload same | Payload changed |
|---|---:|---:|
| R0 | 70 | 7 |
| Radjacent | 66 | 11 |
| Rplacebo | 77 | 0 |
| dense_k3 | 15 | 62 |
| reranker_k3 | 5 | 72 |
| p_yes_only_k3 | 34 | 43 |

Seven payloads change versus R0, but only three Answer F1 values change (one win, two losses). All reported metrics and intervals retain the full 77-question denominator. The 77 identical content/placebo packs are an observed deterministic identity under this saved label/answer realization, not 77 independent generator replications. No new permutation, favorable subset or alternative threshold is selected after seeing these results.

All 20 requests completed without retry or new unknown cost. Stage accounting below is a snapshot after this complete stage; reservations are commitments rather than observed charges and are not refunded by lower actual fees.

| Accounting | Attempts | Observed known cost subtotal (USD) | Reserved (USD) | Unknown-cost attempts |
|---|---:|---:|---:|---:|
| Inherited night ledger | 817 | 0.490420862 | 4.6586873300 | 1 |
| New static stage | 20 | 0.000712740 | 0.100 | 0 |
| Cumulative at this stage | 837 | 0.491133602 | 4.7586873300 | 1 |

The cumulative ledger contains 836 successful attempts and one historical unknown-cost attempt. The unchanged $5 cap leaves $0.2413126700 uncommitted reservation headroom at this stage. Cached generation was not charged again; new generation calls and automatic retries are both zero. The new cost subtotal describes these 20 demanded judgments, not a full 562-edge or end-to-end cost comparison. Root external run/audit elapsed times were 21.203 / 11.797 seconds.

The full formal audit replayed raw responses, event-linked request/response bytes, strict task IDs, legal choices, dated model/provider, usage/cost, resolution, all saved answers, 539 records and all 24 intervals. A separately implemented verifier passed 52 synthetic tests and independently recomputed the new raw-response chain, completion resolution, official Answer F1, family draws and every aggregate. It checked 385 parent baseline records, 154 content/placebo pack-score bridges and all 262 exact cached predictions. The old 817-attempt raw-transport review was inherited through its independently audited receipt and freshly checked hashes, not silently represented as a second full parsing of every old response. Independent receipt bindings cover 3,306 files and were separately rehashed successfully.

Publication source commit: `a6a0b60f3a4cfde7a95e464d35b51b3bb4c39c25`. Public scientific sources and private artifact receipts are distinguished below; hashes authenticate local evidence without publishing its identities or text.

| Evidence | SHA-256 |
|---|---|
| Public frozen protocol | `a32ac6352f7706aaed047df05ff4493c13124e50b11900711f1f864def28d154` |
| Public frozen runner | `2cf469af998e5f8f5376b693bb1f7c1664a59b4490355288e66ca2948dbd9dd7` |
| Public runner tests | `c34b2d7ddb57a147768f5a9186ced47f27ce35f4e8b438c7dbc744cfb42e18b5` |
| Private frozen plan config | `84eb8ca9b4ba63ce3e565088ce97e48968563a9690007a1970ad8ba2b98ef4c6` |
| Private frozen jobs | `e9d79b00bf5d1243458dee7856eec2af7f58107d6f82f4778dc579be3b77557e` |
| Private frozen manifest | `505c2a965281e0b23b5b068c169a23dd58c723bfae0ce0ac8e127ccdd37b53ce` |
| Private complete formal summary | `de98e60faa67b624e78cc1275bb0048a796182b5478b17eb77d70f322cacf174` |
| Private complete formal audit | `ea14c1ca725e05451eb8e083a5a2dff8ea459abd449e44b3aa7ce74306310854` |
| Private Root complete-review release | `72420d8ed8c0a0fdd4f7657291e4cd4b07eb5ea5df557206314e44e945cb3f63` |
| Private completed execution receipt | `96251eafc977496e90486bc4657ff43e2cbe71a7dbb4f37846d2a3239487343c` |
| Private independent verifier source | `03c1b74a7b33a886e8766c6711cc9b14b07ff3b84d4d21c31def6bad203dcd8a` |
| Private independent verifier tests | `a3795fafba7999c380e59f456bbc02b4a63e4430ec68a7a69102f46661d4466d` |
| Private independent verification receipt | `ca2edfa6e981bf829f931423c8f07819d16d7cee3aa0a1a532cad288cffe823c` |
| Private independent source-binding receipt | `a4b822f0415062abff7abbcfda27baab2d474389897a0d0b3f2fa13cec79e9f2` |

The current experiment stops the semantic-benefit claim for this fixed +1 rule: the realized content control does not separate from the placebo, and no non-placebo F1 interval establishes superiority. This narrow result does not show that JEV is generally ineffective, that all relation-aware selectors fail, or that semantic information cannot help another adequately controlled rule. Exposed development data, limited demanded-edge coverage, weak realized placebo separation, unequal lengths, one saved response per payload and unadjusted exploratory intervals all constrain the inference. Future scientific claims require a distinct justified mechanism and independently admitted confirmation data; this report makes neither change.
