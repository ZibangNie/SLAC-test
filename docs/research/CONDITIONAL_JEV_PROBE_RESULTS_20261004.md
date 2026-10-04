# JEV conditional-semantics probe: two authored pairs

2026-10-04. The single frozen six-request run completed. On both authored pairs, JEV returned `added=yes` when the candidate fact was absent from the current pack and `added=no` when it was already covered, including the paraphrase case. All four predeclared added-information expectations matched. This is a small positive semantic sanity result, not a real-data retrieval gain or a novel method.

## Complete observations

The [protocol](CONDITIONAL_JEV_PROBE_PROTOCOL_20261004.md) fixed the six requests, source hashes and expectations before inference. There were no retries, prompt changes, extra examples, generation calls or training. Inputs came only from the already published [authored fixture](fixtures/conditional_jev_contract_v1.json); no real research dataset was inspected.

| Pair | Standalone support | Added: fact absent | Added: already covered | Predeclared pattern |
| --- | --- | --- | --- | --- |
| P01: literal redundancy | yes | yes (C01) | no (C02) | matched |
| P02: paraphrase redundancy | yes | yes (C03) | no (C04) | matched |

Coverage was 6/6 physical requests and 10/10 typed decisions. Added-information coverage was 4/4, with 4/4 expectation matches and 2/2 complete `yes → no` patterns. The two standalone labels were `yes`; all four conflict labels were `no`. Those auxiliary dimensions are descriptive because the frozen fixture supplied expected labels only for added information. No confidence threshold, calibration score, confidence interval or population accuracy is estimated.

## Fees and execution

All six responses reported `typesafe/jev-1.13-20260917` and the TypeSafe provider. Provider-reported usage totaled 3,819 input tokens and 384 output tokens. The exact reported cost was **USD 0.000160398**, with six known-cost attempts, zero unresolved attempts and no in-flight request remaining. The local reservation was USD 0.030; it was an admission allowance, not the amount charged or an account spending limit.

The worker took about 4.329 seconds; the outer controller took about 4.437 seconds and confirmed termination. These include local processing and are not a model latency benchmark. The 180-second deadline was not reached. The pinned request route and price constraints were accepted during this run; future availability is not guaranteed. The observed fee is consistent with the [current published input rate](https://openrouter.ai/typesafe/jev-1.13), but account billing beyond the reported request costs was not inspected.

## Verification and provenance

Before inference, 56 client tests, 14 runner tests and four watchdog tests, including actual-process termination, passed. Independent fake preflight checked six wire commitments, 14 failure/success scenarios and preservation of an unresolved ledger after a harmless worker was killed. The formal plan matched the independent preflight plan byte for byte.

The frozen plan SHA-256 is `707f3edf4ac481e3836f3b50c691e26f069292389d463ed27cee5ea04190080b`. The immutable local summary SHA-256 is `0c20d9a1ad1975c35578fb6d9415cfaccb5f5092839dc99db8f8337fdd4299cb`. Independent post-run verification passed for all six request/response pairs, ten typed labels, fourteen source commitments, reported usage/fees and model/provider identities. It made no additional API calls. The [public aggregate](results/conditional_jev_probe_20261004.json) retains the complete readout and verification hashes. Saved provider response objects, response identifiers and the live ledger remain in the ignored artifact directory. Credentials were not persisted in probe artifacts or published.

## Consequence for the research

The result justifies using the frozen conditional distinction as a candidate signal in further controlled work. It does not show that a selector changes the evidence beneficially, that source relations improve a plain conditional judge, or that the model generalizes to realistic scientific documents. These two easy, exposed examples are insufficient for those claims. The [novelty gate](SET_CONDITIONED_JEV_GATE_20261004.md) remains unchanged.

This run is closed; its unused reservation is not permission to append requests or retry cases. The next work is offline design of native-text relation provenance and controls, so a future comparison can distinguish source-structure effects from ordinary conditioning. Any additional inference needs its own frozen scope and accounting. The old 900-question paid plan remains paused.
