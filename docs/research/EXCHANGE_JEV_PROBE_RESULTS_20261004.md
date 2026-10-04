# JEV exchange probe: loss checks change the decision on authored cases

2026-10-04. The single frozen six-request probe completed. The complete gain/loss/conflict rule accepted both authored beneficial exchanges and rejected all four authored undesirable exchanges. The same-output gain-only ablation accepted three of those four undesirable exchanges. This is a small positive capability observation on deliberately constructed examples, **not natural RAG improvement or established novelty**.

One of the eighteen predicted dimensions did not match its frozen expectation: in the removed-mapping case X06, JEV still predicted new information. Its separate loss prediction rejected the exchange. The error remains part of the result; neither the label nor the prompt was changed or rerun.

## Every planned case

The [protocol](EXCHANGE_JEV_PROBE_PROTOCOL_20261004.md), [inputs](fixtures/exchange_probe_inputs_v1.json) and [readout-only expectations](fixtures/exchange_probe_expectations_v1.json) were fixed and published as commit `1e94630e93962c1b2e8c2c47a02d5c706e0d956a` before inference. A separate AI reviewer, given only inputs and question semantics, agreed with all authored labels. This is not human annotation, an unseen test set, or six independent natural questions. Two controlled pairs intentionally share much of their text.

| Case | Observed gain / loss / conflict | Frozen expectation | Full rule | Same-output gain-only |
|---|---|---|---|---|
| X01: remove irrelevant detail | yes / no / no | Same | Accept | Accept |
| X02: remove unique sampling period | yes / yes / no | Same | Reject | Accept |
| X03: recover removed material and add color | yes / no / no | Same | Accept | Accept |
| X04: recover material, color already covered | no / no / no | Same | Reject | Reject |
| X05: conflict remains between retained units | yes / no / yes | Same | Reject | Accept |
| X06: delete the sole code-to-name mapping | **yes** / yes / no | **no** / yes / no | Reject | Accept |

Coverage was 6/6 physical requests and 18/18 typed dimensions; 17/18 dimension labels matched the authored expectations. There were zero `unknown` predictions and zero missing observations. Both predeclared paired patterns appeared: loss changes from no to yes when only the removed-unit role changes in X01/X02, and gain changes from yes to no when original coverage changes in X03/X04. Neither pair establishes robustness across different wording, order, documents or repeated model executions.

Against the fixed authored decision labels, full-rule agreement was 6/6 and gain-only agreement was 3/6. Both rules accepted the two intended beneficial cases. Full-rule acceptance among the four intended rejections was 0/4, versus 3/4 for the ablation. The three differences are a lost fact, an existing retained conflict and a lost mapping, respectively; they are not measured answer-score degradation. The no-op was rejected by both rules.

The ablation reuses the exact same observed full-pack gain dimension and removes the other vetoes in local code. It is **not** a separately prompted single-passage judge, an independent added-only model, a BGE baseline, or a comparison at matched inference cost. Only six model requests were made, not twelve. All cases are retained and no favorable subset was selected after execution.

## The mismatch constrains the interpretation

X06 originally maps code F4 to Luma. The candidate gives Luma's wavelength, while the proposed pack removes the sole mapping. Under the frozen independent-pack interpretation, the proposed pack cannot attach that wavelength to F4, so the expected new-information label was no. JEV returned yes for gain and yes for original-information loss. The loss veto therefore prevented acceptance even though the gain judgment disagreed with the intended semantics.

This supports keeping gain and loss distinct in this particular example. It does not establish why the gain prediction failed: the returned typed labels do not reveal whether the model borrowed a removed premise, treated topical information as sufficient relevance, or interpreted the comparison differently. It also does not prove that loss will reliably compensate for other gain errors. No post-output prompt repair, new example or repeat request was performed.

None of the cases expects `unknown`, so the zero-unknown output provides no calibration evidence. The simple examples contain no long scientific context, noisy source extraction, multiple competing mappings or realistic retrieval uncertainty. No generated answers, Answer F1, Evidence F1, source-relation treatment or dataset-quality result was measured.

## Cost, process and verification

All six responses reported `typesafe/jev-1.13-20260917` and the TypeSafe provider. Usage totaled **6,993 input tokens and 708 output tokens**. The provider-reported total cost was **USD 0.000293706**. All six attempt costs are known; there are no in-flight or unresolved attempts. The USD 0.030 local reservation was an admission allowance, not the amount charged or an account-level cap. Account billing beyond these response-reported charges was not inspected.

The worker reported 3.969 seconds, and the outer controller measured 4.125 seconds including process startup and handling. These are whole-run durations, not a pure inference latency benchmark. The controller verified normal worker termination. No retries, fallback, training or additional API calls occurred. The probe is closed; its unused reservation does not permit appended requests.

Before execution, 132 focused tests passed across the new runner/client/watchdog and relevant existing components. Independent preflight matched all six request commitments and checked complete fake output, semantic unknowns and a partial timeout with preserved cost uncertainty. It bound eighteen source files to plan SHA-256 `a6690829177dfcd35c527599639ad46316d6987e34f02d07c44086866df704bd`. Expected labels and authored case IDs were absent from every model payload. The final request bodies were 3,808–3,922 bytes; the evidence counter used declared artificial UTF-8-byte units, while the token usage above comes from actual responses.

Independent post-run verification passed for all six saved request/response pairs, eighteen observed labels, source bytes in both the working tree and the published commit, served model/provider identities, token usage and fees. It independently recomputed both policy readouts and preserved the X06 mismatch. The audit also confirmed the worker process was absent; it made no new API call. The immutable run-summary SHA-256 is `fe00c22cebf8307d1f8cb7448c37e74bbfdb93fd16d9f83edc50496438e73584`.

The [public aggregate](results/exchange_jev_probe_20261004.json) retains all six observed label triples, all dimensions and mismatches, both paired patterns, both policy decisions and complete cost coverage. Raw response objects, provider identifiers, the ledger and controller receipt remain in ignored artifacts. The credential was loaded only for this explicit live run and was not included in published files.

## Research consequence

The useful next question is whether the loss/conflict dimensions reject bad exchanges **without suppressing useful ones on a small natural sample**, and whether their value survives ordinary plain-text controls. This probe provides a reason to investigate that question, not to expand API usage or claim the method works generally. The [prior-art gate](EVIDENCE_EXCHANGE_GATE_20261004.md) remains in force: replacement, gap repair and information preservation have established precedents.

Continue first with offline selection and source review of a bounded natural sample, keeping candidate proposals independent of answer outcomes. Keep the X06 failure visible when designing any future mapping or relation comparison. No large confirmation experiment is resumed, and the old 900-question paid plan remains paused.
