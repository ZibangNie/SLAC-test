# Six-request JEV conditional-semantics probe

2026-10-04. Fixed before model execution. The user permits small-scale JEV exploration while excluding large API runs. This separate probe uses at most six requests and a local reservation of USD 0.03. It does not resume the old 900-question plan, its ledger, reservation or automation.

## Question and fixed inputs

Does the frozen conditional added-information judgment respond differently when the same candidate fact is absent from versus already covered by the current evidence? Use only the first two pairs in the previously published [authored fixture](fixtures/conditional_jev_contract_v1.json). P01 covers literal redundancy; P02 covers a paraphrase. These are simple, exposed development examples, not an independent benchmark or a representative sample.

| Physical order | Pair / case | Arm | Requested dimensions | Primary expected label |
| --- | --- | --- | --- | --- |
| 1 | P01 / C01 candidate | standalone | support | Unscored |
| 2 | P01 / C01 | plain conditional | added information, conflict | added = yes |
| 3 | P01 / C02 | plain conditional | added information, conflict | added = no |
| 4 | P02 / C03 candidate | standalone | support | Unscored |
| 5 | P02 / C03 | plain conditional | added information, conflict | added = yes |
| 6 | P02 / C04 | plain conditional | added information, conflict | added = no |

There are six physical requests and ten typed decisions. The standalone candidate is identical inside each pair, so request it once per pair. Reuse the existing `project_state` whitelist and `conditional.py` questions verbatim. Case/pair IDs, supervision and expected labels stay outside the provider payload. No relation arm, real dataset, generation, training or additional sample is admitted.

## Endpoint and request identity

Use HTTPS `https://openrouter.ai/api/alpha/decisions`, request model `typesafe/jev-1.13`, expected served model `typesafe/jev-1.13-20260917`. Pin the TypeSafe route with fallback disabled and maximum input/output/request prices of 0.042 per million input tokens / 0 / 0. The [current model page](https://openrouter.ai/typesafe/jev-1.13) and [official Decisions example](https://openrouter.ai/blog/insights/what-is-jev/) were checked on 2026-10-04. Provider routing fields follow the [official routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection). A rejected routing setting is a stop, not permission to remove it and retry.

Use a new wire namespace binding the complete provider-augmented payload bytes, endpoint, core request and all of its schema/prompt/question/renderer/counter/model identities, expected served revision and provider constraints. The internal binding is not transmitted as evidence. Never invoke the old task-level `submit`, which rebuilds a different prompt. Do not reuse historical response caches or mutate prompts after observing outputs.

## Execution and accounting limits

- Freeze all source hashes, request commitments, sequence, reservations and expected labels before starting. Verify the complete plan again before creating the fresh run directory or reading the designated key. Source or request drift aborts execution.
- At most 6 attempts and 10 typed questions, with no retries, hidden fallback, continuation or duplicate submission. Each final payload must fit 24,000 bytes; each response fits 65,536 bytes. The existing synthetic evidence counter remains a contract counter, not measured provider usage.
- Compute reservation from the final wire payload using the existing conservative byte/overhead allowance and a minimum USD 0.005 per attempt. The frozen total must be at most USD 0.03. This is a client admission rule, not an account-level hard monetary cap or a tokenizer proof. Actual provider-reported cost, reservation and unknown cost must remain separate.
- Persist each attempt as `in_flight` with cost unknown and its full reservation before transport. Keep a legitimate reported cost even if subsequent usage/schema/model checks fail. Missing or invalid cost, usage beyond its allowance or cost beyond reservation stops further requests. Unknown expense is never recorded as zero.
- Use a 180-second outer process deadline. Each network operation has at most 30 seconds, additionally bounded by remaining time. Kill the worker at the outer deadline; retain any in-flight attempt and reservation as unresolved. Never start a second worker because observation timed out.
- Stop on the first transport/HTTP/size/JSON/schema/identity/usage failure. Missing or extra answer IDs, invalid type/choice, wrong served revision and a reported provider that differs from the pinned route are failures. A valid `choice="unknown"` is a semantic response, not a parsing failure. A missing provider field is explicitly reported as unreported while the request route remains pinned.
- Load the user-designated credential only during explicit live execution. Never serialize credentials, request headers, arbitrary exception text or undecodable response bodies. Saved response objects are redacted. Only aggregate results and authored identifiers can be published.

The runner and transport must pass fake-only tests and independent preflight checks before the single admitted run. A dedicated outer controller uses an exclusive receipt path and never resumes a used run directory. No key contents, key fingerprint or authenticated account inspection is needed for planning.

## Complete readout and decision

Retain all six planned rows even if execution stops, with completed, failed and unattempted states. For the four conditional observations, report each added-information label beside the fixture's pre-existing expectation; keep unknown, incorrect and unobserved cases distinct. Count expectation matches out of all four planned cases and complete `yes → no` patterns out of both planned pairs. Also report observation coverage so unattempted items cannot inflate agreement. Preserve the two standalone labels and four conflict labels descriptively; this fixture did not predeclare those dimensions' expected labels, so do not compute their accuracy.

Report known provider costs, unresolved attempt count/reservation, total reserved amount, returned model identities, request/answer counts and elapsed time. Do not call total expense known if any attempt lacks valid cost evidence. Independently verify this tiny artificial run in full, including fees and all labels. This is not a full-data content audit.

If both pairs follow the expected pattern, the result supports executability and the intended conditional distinction on these two examples only. If either fails, retain the failure and diagnose it without fitting a replacement prompt or repeating the example. Either outcome leaves source-structure benefit, relation controls, decision calibration, retrieval/Answer F1 and publication-level novelty unproven. Do not expand this run based on favorable outputs.
