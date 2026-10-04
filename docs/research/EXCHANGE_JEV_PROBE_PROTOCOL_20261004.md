# Six-request JEV evidence-exchange probe

2026-10-04. Freeze before inference. The user permits small JEV probes while excluding large API runs. This is one separate six-request probe, with USD 0.03 local reservation, no retry or extension. Old paid plans and the previous closed conditional probe remain unchanged.

## Scientific question and fixed sample

Can the current full-pack exchange questions distinguish new information from loss caused by replacement? Use the six [authored inputs](fixtures/exchange_probe_inputs_v1.json) in order X01–X06, with the unchanged questions from [exchange.py](../../SLAC/retrieval/decision/exchange.py). Independent AI review checks labels without first reading the separate [expectations](fixtures/exchange_probe_expectations_v1.json). This is an artificial development challenge, not independent human annotation or a representative benchmark. Any ambiguity must be recorded and resolved before inference; no labels, prompts or cases change after outputs are observed.

| Case | Controlled circumstance | Expected gain / loss / conflict | Gate |
|---|---|---|---|
| X01 | Remove irrelevant maintenance detail, add battery color | yes / no / no | Accept |
| X02 | Same query and evidence inventory as X01; remove unique sampling period | yes / yes / no | Reject |
| X03 | Candidate restores removed housing material and adds light color | yes / no / no | Accept |
| X04 | Same query, candidate and removal as X03; retained statement already gives light color | no / no / no | Reject |
| X05 | Candidate adds build year, while retained evidence already conflicts | yes / no / yes | Reject |
| X06 | Remove the sole code-to-name mapping needed to attach a wavelength to the queried instrument | no / yes / no | Reject |

The primary paired observations are loss `no → yes` for X01/X02 and gain `yes → no` for X03/X04. Other dimensions remain explicit per-case predictions. X06 interprets each pack independently: the original mapping cannot be borrowed merely because it remains visible in the judge's comparison inventory. Failure on that case is not repaired by relabeling after inference.

Pre-inference blinded AI review agreed with all eighteen authored labels without changes. X03/X04 assume ordinary equivalence of enclosure/housing and status indicator/light; X05 preserves original statements despite their contradiction. None of the six cases expects `unknown`, so this set cannot validate uncertainty calibration or correct abstention on genuinely indeterminate inputs. Any observed unknown still remains in the denominator.

## Request, cost and execution boundaries

The [official Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request) and [current model page](https://openrouter.ai/typesafe/jev-1.13) were checked on 2026-10-04. Send to `https://openrouter.ai/api/alpha/decisions` with requested model `typesafe/jev-1.13` and expected served revision `typesafe/jev-1.13-20260917`. Pin TypeSafe, disable fallback and retain price ceilings of USD 0.042 per million input tokens, zero output price and zero per-request price. Rejected routing settings stop execution without alteration and retry.

Each request contains exactly three Choice questions, eighteen total. Case IDs, expectations, reasoning notes and pair definitions are readout-only and absent from model payloads. There is no source-relation arm, answer generation, dataset access, training or BGE execution. The generation evidence cap is three units and 2,048 units of an explicitly named UTF-8-byte counter; the judge union cap is 4,096 byte-counter units. These are artificial contract limits, **not tokenizer measurements or equal-length guarantees**. The complete wire payload separately fits 24,000 bytes; responses fit 65,536 bytes.

The new transport adapter reuses the existing verified transport/accounting methods while keeping its constructor, wire namespace and ledger namespace separate. Old source files and old probe entries remain unchanged. Bind and verify all source hashes, request bytes, model/provider identities, budgets, sequence and expectations before any credential load. Use one fresh plan, fresh run directory and exclusive controller receipt; never resume an existing run.

Reserve at least USD 0.005 per physical request using the existing conservative final-wire allowance, at most USD 0.03 total. This is admission accounting, not an account-level hard cap or actual expenditure. Before every send, persist an in-flight record with unknown cost and its reservation. Preserve any valid reported cost even when later schema or identity checks fail. Unknown cost is never zero. Stop subsequent calls on transport, HTTP, deadline, size, JSON, schema, identity, missing/invalid usage, excessive usage or fee failure. Valid semantic `unknown` is retained and does not cause a retry.

The inherited process watchdog gives the new worker at most 180 seconds; individual network operations have at most thirty seconds within the remaining deadline. Parent termination cannot cancel a request already received upstream. Preserve unresolved reservations after termination and inspect the same worker handle rather than restarting. Fake-only client/runner tests, source-bound preflight and independent review must pass before execution. The credential is read only in explicit live mode, never printed, committed or stored in artifacts. Request headers and arbitrary exception bodies are not persisted.

## Complete readout and decision boundary

Retain all six planned rows and eighteen dimensions after any partial failure. Report observed yes/no/unknown, missing/unattempted dimensions, matches and mismatches, both paired patterns, gate decisions and expected beneficial acceptance (X01/X03) versus undesired acceptance (X02/X04/X05/X06). Unknown dimensions abstain. Distinguish information loss, no-op and retained conflict rather than calling them all measured answer degradation.

Replay an ablation that accepts whenever the **same observed full-pack gain** is yes, ignoring the loss/conflict dimensions. This is a policy ablation using shared model outputs, not a separately prompted added-only model, candidate-only baseline, matched-cost benchmark or RAG evaluation. Also report the full gate's failure to retain beneficial exchanges, so universal rejection cannot look successful. No probabilities are calibrated or tuned and no population accuracy/confidence interval is estimated.

Publish complete authored-case labels and aggregate fee/usage/coverage; keep provider response identifiers and raw response objects in ignored artifacts. Independently check all six small authored cases, wire/source bindings, labels and costs. If simple cases fail, preserve the failure and investigate offline without prompt-fitting or repeated submissions. If they pass, the result supports only this tiny semantic capability check; it does not authorize scale-up or establish natural-query utility, source-structure benefit or publication novelty.
