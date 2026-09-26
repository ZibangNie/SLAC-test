# Exact pack selection by lazy rank bounds — execution-before-results protocol

This protocol fixes a new CPU reference selector before its 501-assignment evaluation. It does not authorize execution: source, tests and a metadata-only plan must be sealed, independently reviewed and released by Root first. No API, GPU, new QA, answer, reference, evidence-quality or answer-quality input is permitted. Existing opportunity-gate results are used only by the validation harness after selection, never by the runtime policy.

## Fixed problem and input boundary

Use every one of the original 77 exposed development questions / 24 families, the frozen candidate identities, native units, JEV coarse support labels, dense ordering and the original BGE tokenizer. The maximum candidate pool is 16. No candidate, question, family or label is selected using a downstream result. Eligible relation variables are the original native-order adjacent pairs whose endpoints both have support other than `no` and different exact native text: 107 query occurrences, with per-question counts `0:31, 1:19, 2:8, 3:10, 4:6, 5:1, 6:1, 7:1`.

The runtime input type contains only units, candidates, coarse labels and dense ranks. It excludes baseline witnesses, cube outputs, compiled decisions, essential-variable sets, question strings and quality. Source-qualified identity is added by the outer harness. Runtime relations are available solely through a callback returning a strict Python `bool` and an optional initially known mapping with strict integer variable indices and strict Boolean values. Unknown variables are not inferred from text, outcomes or the callback's closure. The test harness supplies a fixed hypothetical oracle; this is not a real provider invocation.

No demand-compilation function is imported by the strategy. The existing `Unit`, exact `render_pack` and `PackCounter` helpers may be reused; imports do not license calling their QA or metric functions. Only local tokenizer files are loaded, not encoder weights.

## Fixed algorithm

Represent scores in integer half-units: `yes = 2`, `unknown = 1`; exclude `no`. A selected succeeding endpoint B can add 2 to a pending preceding endpoint A. Bonuses do not accumulate. For this fixed adjacency graph, each A has at most one eligible outgoing edge. Before B is selected, that edge is inactive regardless of its label.

For each remaining candidate, form a lower and upper score. An inactive edge contributes `[0,0]`; an active known-dependent edge contributes `[2,2]`; an active known-independent edge contributes `[0,0]`; an active unknown edge contributes `[0,2]`. Each key is `(-score, frozen_dense_rank, native_source_order)`. Smaller keys rank first. Consult all initially known and subsequently cached labels whenever bounds are recomputed.

1. Let e have the smallest lower-score key. Its lower-score key is its worst possible key.
2. If e's worst key is no greater than every other remaining candidate's upper-score key (its best possible key), certify e as the next eager winner. Its own active unknown edge need not be read.
3. Otherwise define C as all other candidates with an upper-score key strictly smaller than e's lower-score key. Collect the still-unknown active edges belonging to C or e. Request the first edge in the original native variable order, strictly validate the callback result, cache it, and recompute bounds. Asking an already known edge, silently coercing `0`/`1` or another object to Boolean, and repeated requests are errors.
4. Remove the certified winner from pending. Apply the original exact-native-text duplicate skip; otherwise render and tokenize the complete proposed pack and accept precisely when its actual token count is at most 1024. Continue until 3 items are selected or pending is empty. Render the final pack in native source order with the unchanged unit headers.

Do not skip an unresolved candidate using a presumed monotone token cost. Token counts can depend on the complete rendering. A budget rejection does not terminate the scan. No actual bonus or original eager priority-change trace is invented for unread labels. Instead save interval certificates, requests, selected-before state, certified winner, action and proposed-token count. These records establish output equivalence, not relation-trace equivalence.

## Correctness argument and computational boundary

Every completion consistent with the cache gives each candidate a key between its best and worst bounds. At a certificate, `worst(e) <= best(c) <= actual(c)` for each other c and `actual(e) <= worst(e)`, so e is the unique eager winner under the fixed total tie order. If certification fails, C is nonempty. If no active unknown edge existed on C or e, their relevant bounds would be exact, contradicting e's minimal lower key; therefore the next query is defined. Each query removes one unknown variable and never changes an observed label, so the inner procedure terminates after at most m initially unknown eligible variables.

Induct on the candidate-removal steps: the same winner under every consistent completion, together with the same selected set and exact full-pack counter, gives the same duplicate skip, budget skip or acceptance. Thus the final selected identities, source-ordered rendered bytes, SHA256 and actual tokens equal the eager output. This argument does not require a truth-table, an essential-variable oracle or monotonic tokenization. Cache values must represent the same fixed relation oracle as the completion being compared.

For n pending candidates and m eligible variables, a straightforward implementation recomputes bounds in linear scans and sorts no decision diagram; at most n candidate removals and m new relation reads occur in a path. It is a reference implementation, not an optimal question policy or a new classical decision-diagram algorithm. No non-exponential optimality claim is made. The exhaustive validation below is exponential, but is outside runtime selection.

With n at most 16 and max selected size 3, tokenization memoization has at most `1 + C(n,1) + C(n,2) + C(n,3) <= 697` distinct subset keys per question, including the pre-cached empty pack. The harness shares this exact-text token-count cache across a question's replay paths for efficiency; it resets relation observations separately for every path. Report both counter invocations and actual tokenizer encodes/cache entries. This CPU timing is warm token-cache replay timing, not independent cold-start serving latency or a real global relation-cache schedule.

## Frozen validation and report

Metadata-only preparation validates and binds the original complete gate, original cases, tokenizer files, source/tests/protocol, runtime versions, run directory and a 300-second cooperative CPU deadline. It performs no new selector path. Runtime and verification inputs remain separately represented. The completed gate's full outputs are validation targets only. A new plan and run directory are exclusive and never overwrite old plans/results. Failures retain an explicit failure marker and publish no complete summary; there is no automatic retry.

Run all 501 complete assignments with empty relation cache. Also run all 23,915 `(assignment, initially-known-subset)` paths: `sum_q 4**m_q`. This second set includes the same 501 empty-cache paths; report the two test suites separately, not as 24,416 distinct scientific cases. For a given assignment, each known subset contains exactly that assignment's Boolean values. No hypothetical assignment is treated as a random or equally probable real label outcome.

For every path compare the final selected indices, source-qualified identities, exact rendered pack, SHA256 and actual tokens to the complete saved cube output. Independently check every saved bound certificate and conflict/request event against all completions consistent with the observations so far; check the eager winner for every such completion, fixed native query ordering, absence of known/repeated reads, and exact replay of duplicate/budget actions. The validator may inspect cubes/assignments; the selector may not. Verify that a full initial cache makes zero requests. Preserve all paths and request/certificate records locally, plus aggregate completeness and failure counts.

After all paths finish, derive essentiality only from the saved complete gate output and compare the request identities to that set; the already reported 19 essential occurrences are a contextual cross-check, not a policy input or a target count. Report, separately for empty-cache and all-subset suites: request-count histogram, maximum and total, reads saved versus reading every initially unknown eligible edge, equal/fewer-read path counts, requests of globally nonessential edges, distinct requested edge occurrences/unique source-qualified edges, request/certificate event counts, final equivalence counts, CPU timing, tokenizer/counter/cache counts and cache bounds. Retain any extra nonessential reads or zero savings. Do not select successful questions or suppress negative efficiency observations.

The audit replays all paths from the sealed original inputs and saved hypothetical assignments, recomputes certificates and aggregates, and checks beginning/end source hashes. Root independently verifies full outputs before publishing a result. No saved summary can establish correctness by merely being re-sealed after alteration. Measured runtime is checked for finite nonnegative range and binding, not claimed independently reproduced bit-for-bit.

## Scope of claims

This is an exact-output lazy rank-bound reference policy under a fixed relation oracle and fixed support, candidates, dense ranks, tokenizer and packer. Changing any of those inputs requires fresh validation. Equivalent judgments sent through different real API batches need not return equal labels. Logical callback reads and token-cache behavior are not API requests, batching fees, end-to-end cost savings, relation accuracy, JEV-specific evidence, retrieval improvement or answer-quality evidence. The 77 questions are already exposed development data. No independent confirmation, new holdout or data-admission decision follows.

The existing [decision-diagram review](RELATION_DEMAND_DESIGN_REVIEW_20260927.md) explains the classical background and why content-essential pruning does not automatically preserve permuted-placebo outputs or original stage label-count strata. This selector does not change those limitations and is not a replacement experiment for real content/placebo judgments. No new generated-answer result enters its implementation or validation.

## Planned commands

After independent source review and publication, prepare only:

```powershell
python docs/research/run_qasper_relation_lazy_bounds.py prepare --output artifacts/research-foundation/qasper-relation-lazy-bounds-plan-01 --run-output artifacts/research-foundation/qasper-relation-lazy-bounds-run-01
```

Only after Root's explicit one-time release:

```powershell
python docs/research/run_qasper_relation_lazy_bounds.py run --plan artifacts/research-foundation/qasper-relation-lazy-bounds-plan-01
python docs/research/run_qasper_relation_lazy_bounds.py audit --plan artifacts/research-foundation/qasper-relation-lazy-bounds-plan-01 --run artifacts/research-foundation/qasper-relation-lazy-bounds-run-01
```
