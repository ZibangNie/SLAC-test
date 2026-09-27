# Confirmation provider and budget review — 2026-09-27

**Status: bounded read-only review; pricing input for a future execution plan.** This does not admit paid requests, reopen the completed overnight budget, or change a frozen scientific rule. Two model-specific public metadata endpoints were fetched without authentication at **2026-09-27 01:59:53 UTC / 09:59:53 Asia/Shanghai**. No key, new QA, paid inference, full model inventory, or model download was used. The user requested sampled checks: this review inspected the two relevant clients, one old endpoint snapshot, and two small historical failure ledgers rather than re-auditing all earlier responses.

## Current official contract

| Use | Fixed request model | HTTP endpoint | Provider restriction | Advertised endpoint revision | Base input / output USD per million |
|---|---|---|---|---|---:|
| Support decisions | `typesafe/jev-1.13` | `POST https://openrouter.ai/api/alpha/decisions` | `typesafe` | `typesafe/jev-1.13-20260917` | 0.042 / 0 |
| General support and answer generation | `qwen/qwen3.6-plus` | `POST https://openrouter.ai/api/v1/chat/completions` | `alibaba` | `qwen/qwen3.6-plus-04-02` | 0.325 / 1.95 |

The JEV route accepts `model`, `state`, and typed `questions`; Choice decisions return a label and reported probabilities. Its official example returns the dated model string. Our existing yes/no/unknown Choice contract fits this interface. Retain these values as **reported scores**, without claiming independently established calibration. Do not substitute the moving `jev-latest` alias or send JEV to chat completions. [Official JEV guide](https://openrouter.ai/docs/guides/community/jev), [Decisions API reference](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request).

Both public metadata responses listed one endpoint and returned HTTP 200. JEV advertised a 32,000-token context; Qwen advertised 1,000,000 tokens and 65,536 maximum completion tokens. Both reported 100% uptime for the preceding 5 and 30 minutes; Qwen's daily value was 99.98422254895011%. Both returned raw `status: 0`, with recent latency and throughput fields null. These are provider metadata, not an authenticated account probe, an execution guarantee, or measurements of our workload. The snapshot retains the raw status code without assigning an undocumented enum meaning. [JEV endpoint metadata](https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints), [Qwen endpoint metadata](https://openrouter.ai/api/v1/models/qwen/qwen3.6-plus/endpoints).

Qwen's price object also contains a **256,000 prompt-token threshold** with input/output rates of **1.30 / 3.90 USD per million**. Our existing 24,000-byte payload ceiling and 26,048 input allowance do not plan for that tier; measured usage and price checks must still reject unexpected overruns. The byte allowance is not a proof about the provider's tokenizer. No cache discount is assumed. [Qwen endpoint metadata](https://openrouter.ai/api/v1/models/qwen/qwen3.6-plus/endpoints).

The safe machine-readable record is [provider snapshot](results/qasper_confirmation_provider_snapshot_20260927.json). It includes retrieval times, raw-response hashes, the relevant current metadata, and hashes of the sampled local contracts. Raw public responses and the fetch receipt remain in the ignored provider-review artifact directory. Account credits, access restrictions, rate limits, and the model string of a new inference response remain untested.

## Compatibility findings from the existing implementation

- [`openrouter_decision_client.py`](openrouter_decision_client.py) still has a historical GPT-4.1-mini default for `general`. A new plan must explicitly choose **`qwen36plus-json`** before payload construction; an omitted profile would change the comparator. The working contract uses `json_object`, `reasoning.enabled=false`, temperature zero, exact output-key validation, and full visible unit text.
- [`run_qasper_answer_evaluation.py`](run_qasper_answer_evaluation.py) supplies the fixed evidence-only answer prompt, JSON `answer`, 512 output-token cap, and the same Qwen route. The support output allowance is 1,024. Preserve prompt/version and whole evidence-pack rendering; do not change the output length to fit a budget.
- Provider routing remains `only`, `allow_fallbacks=false`, and price ceilings; Qwen also requires parameter support. OpenRouter documents prompt/completion `max_price` in **USD per million tokens**, unlike the endpoint API's per-token price strings. This filter constrains eligible provider rates; it does not enforce our total account budget. [Official provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).
- The historical Qwen endpoint snapshot has the same complete price object as this fresh snapshot. Two small earlier diagnostic ledgers record HTTP 403 and HTTP 400, each with unknown actual cost. This is enough to preserve the lesson that listed availability does not prove account/parameter acceptance; it is not a new measured failure rate. The old working JSON profile must not silently revert to the unsuccessful strict-schema profile merely because current metadata lists structured outputs.
- The completed recovery contracts locked observed response identities to **`typesafe/jev-1.13-20260917`** and **`qwen/qwen3.6-plus`**. The advertised Qwen endpoint suffix differs from its observed response alias. Freeze the expected response strings in the new plan; stop on drift rather than broadening the allowed set after observing it. An unchanged alias still cannot prove unchanged provider weights or infrastructure.

## Full-cohort accounting formula

Let `Q` be the complete admitted **question** count, `c_q` the frozen candidate count for question `q`, and `B = sum_q b_q` the actual common batch count after full-text byte checks. The proposed 248 documents are not 248 questions. With a candidate cap of 16, `C = sum_q c_q <= 16Q`; each backend judges all `C` items. Retain one query-local batch grouping for both backends, at most eight items per batch. Long requests can split further, so `B` cannot be replaced with `ceil(C/8)` without constructing the payloads. A singleton that exceeds the ceiling stops preparation; it does not authorize text truncation or question removal.

For each exact UTF-8 canonical support payload, let `L_b` be its byte length and `n_b` its item count. Each backend has its own `L_b`. The existing conservative reservation is:

```text
A_JEV,b  = (L_JEV,b  + 2048) * n_b
A_Qwen,b =  L_Qwen,b + 2048
R_JEV,b  = max(0.005, 1.5 * (A_JEV,b * 0.042) / 1,000,000)
R_Qwen,b = max(0.005, 1.5 * (A_Qwen,b * 0.325 + 1024 * 1.95) / 1,000,000)
R_support = sum_b (R_JEV,b + R_Qwen,b)
```

The JEV multiplier conservatively allows repeated state processing across typed questions. It is an inherited budgeting assumption, not a claim that the server bills precisely this amount. A local allowance overrun or cost overrun is detected after the attempt; reservation is not a remote billing hard cap.

For the fixed five arms, `I_jev_k3` and `p_yes_only_k3` reuse one set of JEV decisions; `I_general_k3` needs one Qwen support set. Dense and BGE reranking add no OpenRouter support calls. Thus support requires **`2B` planned HTTP attempts**, not a fresh judge for every arm. GPU/runtime costs remain separate from API credits. This plan contains no static-relation stage.

For answer generation, all five arms retain all `Q` questions: **`5Q` logical predictions**. Let `G` be the exact number of new unique full payloads after matching endpoint, prompt version, canonical JSON bytes, and any explicitly admissible completed cache sources. Equal selected IDs or equal F1 do not establish a cache hit. No reuse rate from the development cohort should be projected onto these new questions. For each new payload `g`:

```text
A_g = L_g + 2048
R_g = max(0.001, 1.5 * (A_g * 0.325 + 512 * 1.95) / 1,000,000)
R_new = R_support + sum_g R_g
G <= 5Q
```

Before support outcomes exist, reserve for every possible required answer call using a bound, rather than assuming favorable future deduplication. Under the unchanged 24,000-byte ceiling the local formula gives:

| Reservation-formula bound | USD |
|---|---:|
| One JEV support batch, at most eight items | 0.013128192 |
| One Qwen support batch | 0.015693600 |
| One unique answer payload | 0.014196000 |

Consequently `R_new <= 0.028821792 * B + 0.070980000 * Q` is a deliberately loose **reservation-formula** bound when every payload fits the contract. It is not expected spend, a server billing guarantee, or a completed cohort estimate. Exact support payload reservations plus a fixed full-answer allowance provide an actionable first gate. Once all support decisions pass audit, exact answer deduplication can reduce the planned requests within that already declared scope; it cannot refund earlier attempted reservations or justify dropping arms.

## What the next execution plan still needs

1. Bind the admitted question manifest, complete candidate inventory, two sets of exact support payloads, common batch mapping, five-arm selector contracts, and answer rendering. Report `Q`, families, `C`, `B`, exact support reservation, and the answer allowance. These counts were not derived by this review, which did not read the new QA.
2. Declare a new execution window, duration/attempt limits, per-attempt deadline, and an explicit total budget including the full answer stage. Old runners contain the elapsed **2026-09-27 09:00 Asia/Shanghai** cutoff, 77-question-era caps, and the old USD 5 night ledger. Reuse their pure payload/parser contracts through a new versioned runner; do not mutate or silently reset those old limits. This review neither starts a fresh budget nor authorizes spending the difference between old reservation and actual charges.
3. Record reservation before dispatch, keep unknown-cost failures and their full reservations, and stop on timeout, refusal, rate/route/model drift, malformed output, or usage overrun. Preserve completed prefixes for audit; never turn a partial cohort into the primary result. A later explicitly planned recovery must retain earlier unknown attempts rather than silently retrying them.
4. Recheck only these model endpoints when sealing an execution plan if this snapshot has become stale. Preserve exact expected rates/provider/model, with no automatic fallback. Any new paid canary must itself be predeclared and included in the same accounting; the public GETs here consumed no inference requests.

Provider status supports preparing the intended JEV/Qwen route. It does not establish statistical readiness, a cohort cost, successful future access, or any scientific advantage of using JEV.
