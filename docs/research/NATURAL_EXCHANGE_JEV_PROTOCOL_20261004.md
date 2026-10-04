# Six fixed natural exchanges: bounded JEV semantic probe

2026-10-04. Freeze before inference. The active user objective permits tiny JEV experiments while prohibiting large API runs. This is a separate probe with **at most six physical requests, eighteen Choice questions, USD 0.03 local reservation, a 180-second watchdog, no retry and no extension**. The old 900-question paid plan and both closed authored probes remain unchanged.

## Question and sample

The [offline six-case review](NATURAL_EXCHANGE_SAMPLE_RESULTS_20261004.md) found one shared possible strict information improvement, two bare-heading candidates, and two unresolved question-scope cases. Does the unchanged exchange judgment agree with the determinate source annotations on these natural passages, and does it retain uncertainty on the unresolved cases? This reduces uncertainty about transferring the authored contract to real evidence; it is not an answer-quality, accuracy, calibration or publication-novelty experiment.

Reuse **all six fixed proposals** from three previously exposed development documents. Do not change candidates, victims, order, questions, prompts or source text; do not filter headings, add missing context or read gold/generated answers. No new search, reranking, training, reference resolution or full-dataset scan is involved. The packet is `artifacts/research-foundation/offline-20261004/natural-exchange-sample-01/review_packet.json`, SHA256 `bd4e903a2de589d0f823288a9274428e434b5f57dab86840db00f72bec7984f1`.

The previously saved A/B annotations stay distinct. A SHA256 is `f6ad210b3f767d6912a09f6b591b4058e878e3cdd255c736dac61ff2da4b430f`; B is `a414d6d9e10aafa14304926ecd2e1588b53a87e5a4942b48cee92c3d96a94c9b`. They are AI source judgments, not ground truth or fully context-naive independent annotators. No third adjudication or label revision is allowed after observing JEV outputs.

## Fixed readout

Order below is gain / loss / conflict. Keep every dimension and every planned row even if execution fails.

| Case | Review A | Review B | Predeclared interpretation |
|---|---|---|---|
| 1 | no / no / no | no / no / no | No additional evaluation-dataset information |
| 2 | yes / no / no | yes / no / no | Shared possible partial model-detail gain; only shared accept case |
| 3 | no / no / no | no / no / no | Bare heading adds no supervision information |
| 4 | no / yes / no | no / yes / no | Bare heading adds nothing and loses a dataset caveat |
| 5 | no / no / no | unknown / no / no | Gain scope unresolved: collected versus illustrative topics |
| 6 | unknown / no / no | unknown / yes / no | Gain target unresolved; training-description loss disputed |

The primary descriptive readout is correspondence to the **15 agreed determinate dimensions**, split into matched, opposite yes/no, model unknown, and unobserved. Report the dimension-specific fixed denominators: gain 4, loss 5, conflict 6. Keep the **one agreed-unknown dimension** (case 6 gain) separate, reporting its exact model choice; one example cannot calibrate abstention. Keep the **two reviewer-disagreement dimensions** (case 5 gain, case 6 loss) separate and show both annotations without treating either as the correct label. Do not pool all eighteen into an accuracy headline.

Those fifteen labels contain **thirteen no and only two yes**. Always show the two positive-reference observations separately: case 2 gain and case 4 loss. A constant-no response would agree on 13/15 dimensions while rejecting the only shared beneficial exchange; this arithmetic reference needs no API call and is not a useful selection policy.

Report all six exchange gate statuses using the unchanged [contract](../../SLAC/retrieval/decision/exchange.py): accept only yes/no/no; unknown takes priority over negative judgments and keeps S. Report failure to accept case 2 as well as acceptance of the three shared reject cases (1,3,4), so universal rejection cannot appear successful. Cases 5 and 6 remain a separate unresolved category. Also show full-gate correspondence to A and B separately, with no new ground-truth label.

Replay gain-only from the **same observed gain label**, accepting if gain=yes and ignoring loss/conflict. Report both status differences and actual accept/keep differences, which are distinct when reject and abstain both keep S. This is a policy ablation sharing one inference, not an independently prompted baseline, matched-cost comparison or RAG evaluation. The source annotations themselves produced no incremental full-gate accept/keep benefit; preserve that null result when interpreting any model-output difference.

No natural cases form a controlled causal pair. An output consistent with a topic/target distinction does not prove the model used that reasoning; a mismatch does not establish its internal cause. All six shared conflict annotations are no, so conflict-detection sensitivity cannot be measured here.

## Request surface and budgets

Project only the fixed query and complete unit text, unit ID, document ID and source order into the existing full-pack exchange contract. Include the original pack, candidate, removed ID and proposed IDs exactly as fixed. Exclude review labels, reasons, facets, unresolved-context notes, source scores, ranking, old gaps and expected gate decisions. Annotations are bound in the plan solely for readout and never enter model payloads.

Use the pinned local BGE evidence tokenizer, verifying its four files against the preceding sample plan; load offline, without remote code. Count the actual exchange-core renderer `[doc/id]` with whole-pack special tokens and no truncation. Require S and T each to fit **1,024 BGE tokens and three units**, and their comparison inventory S union candidate to fit **2,048 BGE tokens**. These new counts use a different header format from the old sample renderer `[unit_id]`; report that difference instead of treating counts or hashes as identical. BGE evidence counts are not provider billed input counts. The final JSON wire independently fits 24,000 UTF-8 bytes and the response 65,536 bytes. If any fixed request fails admission, stop and report it without choosing another case or silently increasing a cap.

The [official Decisions endpoint documentation](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request) and [Jev 1.13 page](https://openrouter.ai/typesafe/jev-1.13) were checked again on 2026-10-04. Use `https://openrouter.ai/api/alpha/decisions`, requested model `typesafe/jev-1.13`, and expected served model `typesafe/jev-1.13-20260917`. Pin TypeSafe; disable provider fallback and redirects; retain price ceilings USD 0.042 per million input tokens, zero output and zero request price. Provider rejection stops the run without changing settings and resubmitting.

Reuse the unchanged [exchange client](exchange_probe_client.py), transport, accounting and process watchdog. A new, small natural-input runner freezes exact source hashes, tokenizer hashes, request bytes and order, limits and both review matrices before credential access. Use a fresh exclusive plan, run directory and controller receipt. Do not reopen any prior run. The user-supplied credential is loaded only by the live worker; it is never printed, committed, copied into results or sent anywhere except the pinned API authentication header.

Reserve USD 0.005 per physical request, at most USD 0.03 total. This is conservative admission accounting, not measured expense or an account-level hard spending cap. Persist a reservation and in-flight unknown-cost record before dispatch. Preserve any valid reported cost even if a later validation fails. Unknown cost is never zero. Stop on transport, HTTP, deadline, size, malformed response, model/provider identity, answer-ID/schema or usage/fee failure. Semantic unknown is a valid observation and is not retried. An upstream accepted request may still finish after local termination; inspect the same worker handle and retain unresolved reservations, never restart on a polling timeout.

## Decision after this probe

Fake-transport tests, exact source/request reconstruction and independent review must pass before the one bounded run. Publish complete anonymous outputs, the predeclared comparison partitions, measured usage/known costs and unresolved attempts. Keep raw questions, source passages, credentials, provider identifiers and responses private. No automatic follow-on generation, prompt tuning, repeat requests or scale-up follows a favorable result.

If natural gain/loss or uncertainty judgments disagree with the fixed annotations, inspect those existing passages and requests offline without expanding the sample or relabeling the original references. If correspondence is favorable, it supports only a narrow semantic feasibility signal. Novelty, gain over strong retrieval baselines and end-to-end answer benefit remain open; JEV's recent release alone does not resolve them.
