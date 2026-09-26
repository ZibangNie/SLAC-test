# Original-domain placebo composition and label-demand protocol

2026-09-27. **Frozen before the new all-pack answer generation is started.** This protocol fixes a new exploratory coverage contract; its CPU implementation and any later paid phase require separate review and release. No paid execution is authorized here. The original 562-edge static-label stage remains closed and unexecuted. This protocol does not modify any existing source, plan, result or budget limit.

## Fixed inputs and complete population

Use all **77 exposed development questions / 24 families** from the completed [opportunity gate](RELATION_OPPORTUNITY_PROTOCOL_20260927.md), its **501 complete query assignments**, and the independently verified [demand compilation](RELATION_DEMAND_COMPILATION_RESULTS_20260927.md). No query is filtered for answerability, evidence availability, support, variable count, pack change, quality or placebo degeneracy. No new QA or all-pack answer/quality file is an input to preparation, permutation construction, demand derivation or budget estimation.

Keep the original candidate pool, JEV coarse support, eligibility, selected-B-to-preceding-A direction, nonaccumulating +1 bonus, dense/source tie order, deduplication, full source-order rendering, 1,024 actual BGE-token budget and at most three native units. Define `F_q(mask)` using the existing complete tuple: ordered source-qualified selected identities, exact rendered pack bytes, their SHA-256 and actual evidence tokens. Tokens are inherited from the audited complete cube, not recomputed with an approximate length model. Let `S_q` be the full-cube essential variables of this exact function, never an empirical relevance score or an observed-label subset.

## Exact full-domain permutation

Use the **entire original `E*_q`**, not the 19 content-essential edges or any requested-label subset. `E*_q` comprises original candidate-adjacent native boundaries whose endpoints are both support-eligible and have distinct exact native text. Its established histogram is `0:31, 1:19, 2:8, 3:10, 4:6, 5:1, 6:1, 7:1`, with 107 edge–query occurrences and 101 unique source-qualified edges. This is the domain specified in the [original NEXT draft](NEXT_RELATION_EXPERIMENT_DRAFT_20260927.md).

For every edge `(A,B)`, form the JSON-array stratum

```text
[A.kind, B.kind, support[A], support[B], baseline_trigger_class]
```

Support values are the frozen `yes`, `no`, `unknown` strings; eligible endpoints cannot be `no`. Obtain `baseline_trigger_class` from the original zero-mask acquisition trace: count successful selections, not attempted/skipped candidates. If B is the first or second successfully selected unit and A is absent from `selected_before` at that selection, use respectively 1 or 2; otherwise use 0. This is fixed before any relation labels, permutation outputs or new answer quality are observed.

Within each stratum, `source_order` is ascending `(A.order,B.order)`. The exact source-qualified `edge_identity` is the JSON array `[doc_id,A.unit_id,B.unit_id]`. Compute

```text
SHA256(canonical_json([
  "SLAC-local-dependency-placebo-v1", 20260927,
  family_id, question_id, stratum, edge_identity
]))
```

Canonical JSON uses UTF-8, `ensure_ascii=False`, sorted object keys, compact separators and rejects nonfinite numbers. `target_order` sorts the same full stratum by this hash (ascending digest), breaking a hash tie by `(A.order,B.order)`. Define **source → target** assignment exactly as `placebo[target_order[i]] = original[source_order[i]]`. No seed search, resampling, alternative stratum, forced nonidentity permutation or subset reshuffle is allowed. Singleton/identity strata remain unchanged.

Let `g_q(t)=s` denote the original source variable whose value is sent to target variable t. Then `(P_q z)_t = z[g_q(t)]`. Consequently the original-label requirements of the placebo output are the inverse image of essential target positions: `Ess(F_q ∘ P_q) = g_q(S_q)`. The implementation must establish this equality by independent complete-cube influence checks, rather than merely assume it from notation.

## Pure CPU composition, demand and stopping conditions

For all 501 original query masks z, construct the full permuted mask Pz, look up `G_q(z)=F_q(Pz)`, and independently verify source→target assignment, stratum bijection, inverse-image essentiality and exact output parity. No source bit is dropped before this composition. Report identity/nonidentity strata, fixed/moved eligible positions, content-versus-placebo function equality over the **entire** cube, all-query/assignment coverage, required-edge counts and source bindings. These are structural facts; no actual labels, gold, answer scores or favorable masks are used.

The new finite label demand is

```text
U = union over all q of (S_q union g_q(S_q)), using (doc_id,A.unit_id,B.unit_id).
```

The already established 19 essential occurrences / 19 unique content edges imply `19 <= |U| <= 38`; the actual count is deliberately not selected in advance. Retain all 77 queries even when their functions are constant. For every complete original-label assignment, content and placebo outputs must be determined by U alone; verify this by comparing all assignments that agree on the required coordinates. Unobserved bits remain **unobserved**. Do not fill them with independent, unknown, no, false or a guessed class in saved label data. CPU completion-invariance checks may enumerate hypothetical completions but cannot present one as an observed labeling.

The original full-domain permutation is structurally count preserving within every stratum for each complete labeling. Nevertheless this subset experiment does **not** observe the full positive/negative/unknown counts, label accuracy, all 562 static labels, or the complete stratum statistics. None may be inferred or reported from the demanded subset. Raw placeholder completions must never enter a classifier-quality denominator. If the complete 501-assignment check shows content and placebo are the same function for every query, the semantic-content control is structurally degenerate: report it, **do not start this semantic-content paid phase or its admission-oriented payload estimation**, and stop a content-versus-placebo efficacy claim. Do not redesign P, change the seed or relax strata after seeing that result. Only a nondegenerate full-population result permits a separate all-U payload/reservation assessment; it is not spending admission. A later actual labeling can also produce identical packs despite distinct functions: if all content/placebo packs then coincide, stop a semantic-content benefit claim and inherit identical answer payloads without duplicate generation. All 77 questions / 24 families remain in every report and denominator.

Prepare binds sources, code, tests and this protocol and validates metadata only. It must not derive the real permutation, actual U, functional degeneracy, new fees or scientific quality. A separately released CPU run computes the symbolic results and optional budget projection; its complete audit recomputes all public/private outputs. Use a new fixed single-use directory, serial CPU, a 300-second cooperative deadline and before/after hashes. Incomplete, malformed, conflicting or changed sources, missing assignments, nonbijections, parity failures, timeout or interruption produce no complete result. No GPU, training, model loading, key access or API call is permitted.

## Prespecified scientific experiment if later separately admitted

The four scientific arms are fixed as:

| Arm | Fixed definition |
|---|---|
| R0 / independent | Original zero-mask `I_jev_k3` |
| Rcontent | `F_q(z)` for the completely observed demanded original labels |
| Radjacent | Original full-mask native-adjacency output |
| Rplacebo | `F_q(P_q z)` using the original full-domain P |

Legal dependent activates a bit; legal independent and legal unknown do not. A missing result, refusal, malformed response, model/provider drift, timeout or uncertain attempt is **not** a legal unknown: stop the paid stage, preserve its accounting and do not score a completed subset. This protocol does not choose a concrete label realization. Static relation judgments remain query-independent; the diagnostic permutation is query-conditioned through eligibility/strata and is not a reusable document annotation.

Primary contrasts, fixed now, are **Rcontent − R0**, **Rcontent − Radjacent**, and **Rcontent − Rplacebo**. The three fixed secondary contrasts are **Rcontent − dense_k3**, **Rcontent − BGE reranker_k3**, and **Rcontent − p_yes_only_k3**, using the existing complete audited given-document baselines. Do not substitute the better of score variants or a different k. All seven method means and all six contrasts retain the complete 77-question denominator; the old 15-question pilot remains separate.

Report official Answer F1 and actual evidence tokens for every contrast with the same 24-family clustered bootstrap: PCG64 seed 20260927, 10,000 shared family draws, both question-weighted and family-balanced estimates. This fixes **6 contrasts × 2 metrics × 2 weights = 24 exploratory intervals**, with all signs, zero bounds, cross-zero intervals and ties retained. No result-dependent comparison selection, multiplicity-based discovery claim, best-mask selection or parameter tuning is allowed. Equal caps do not ensure equal actual length; any favorable answer change remains a total strategy effect, not a length-independent semantic effect.

Answers, if required after a separate admission, use the existing exact question/evidence prompt, Qwen3.6 Plus / Alibaba route, temperature 0, no reasoning and 512 max output. Actual returned model must match the inherited `qwen/qwen3.6-plus` realization. Inherit only audited responses whose endpoint, prompt version and complete canonical request payload bytes match. Later availability of all-pack answers may reduce new calls; it must not change U, P, the selector or these comparisons. Evidence-empty and unchanged questions remain in every relevant denominator; an existing empty-evidence generator control is abstention-oriented, not unrestricted closed-book QA.

## Optional metadata-only budget projection; no admission

For the new U only, the allowed projection uses **one complete original static-prompt task per request**, with the original frozen JEV provider/model route. At most 38 complete singleton payloads may be constructed for estimation. Singleton batching is a new execution realization; it is not assumed to reproduce labels from the unexecuted old batches. Reject an oversized payload rather than truncate evidence. Bind the full prompt, task bytes, route, provider, prices, input/output allowances and reserve formula; count exact payloads and sum their reserves. Do not extrapolate fees proportionally from 19/562 or 38/562. No request is submitted by this CPU tool.

The historical reservation before the new 14-answer stage was USD 4.6053728675, leaving at most USD 0.3946271325 under the USD 5 night cap. That is **not** current spendable headroom once newer plans commit funds. Any later paid proposal must include the new 14-answer reservation and every subsequent commitment/attempt, preserve the historical unknown-cost attempt, reserve all required relation calls and any unmatched answer payloads, and pass the complete global budget and deadline gate. Actual charges do not refund conservative reservations. Failure of the gate means no execution; do not delete methods/questions/required edges, shorten answers or change batching to force admission.

This is development exploration following observed behavioral structure, not independent or blinded confirmation. It tests a narrow relation-content selection hypothesis if and only if later fully executed and audited. It does not establish new BDD theory, cross-stage sharing, relation-label correctness, exact online cache savings, deployment speed, or superiority over the strong baselines. The current original 562-edge experiment stays closed regardless of these symbolic results.
