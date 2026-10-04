# Conditional JEV integration: offline contract protocol

2026-10-04. This stage implements the next step in [the novelty gate](SET_CONDITIONED_JEV_GATE_20261004.md). It performs **zero model inference**. Passing it establishes request, response, cache and resource-accounting behavior; it cannot establish semantic accuracy, answer quality or a novel method.

## Fixed scope

Use eight human-authored pairs, sixteen cases total. Within each pair the question and candidate text are identical; the current evidence changes. Cover missing versus already supplied facts, paraphrase redundancy, relevance altered by a resolved reference, explicit conflict, unresolved references, definitions and scope qualifiers. Use invented entities and facts, without opening the real 77-question or 900-question data. Expected semantic judgments and case-design descriptions remain outside model-visible input.

Build three arms:

- Standalone support sees only the query and candidate. Its cache identity stays constant when only the hidden current pack changes.
- Plain conditional judgment sees the query, current evidence and candidate. It separately asks whether the candidate adds answer-relevant information and whether it conflicts with the current evidence. Neither question asks the model to predict Answer F1.
- Relation-conditioned judgment sees the same text and explicit source-relation metadata. For relation-bearing cases, generate a predetermined endpoint-permutation control with identical visible text/order, relation count and serialized metadata byte length. Byte equality does not imply equal model token counts or equal inference costs. Do not add relations to cases that have no applicable authored relation.

Source relations are hypotheses backed by visible endpoint identities and anchor offsets. The adapter checks those syntactic bindings; it cannot certify the semantic relationship. A permuted control intentionally tests an altered relationship. Neither the arm name nor the state may tell a model that a relation is the true or placebo relation, or disclose expected decisions. A future semantic test must verify how the model handles both valid and inconsistent relations instead of trusting metadata unconditionally.

## Mechanical requirements

1. Preserve whole native text, immutable unit identities and source order. Reject duplicate identifiers, duplicate native text, missing relation endpoints, self-loops, duplicate relations and invalid anchors. Never truncate to meet a budget.
2. Use an injected counter for the exact rendered evidence: candidate-only for standalone; the source-ordered union of current pack and candidate for conditional arms. Accept exactly the evidence-token budget and reject above it. Reject boolean, negative or noninteger counter outputs. A separate canonical payload byte cap covers request serialization. Neither measurement is a claim about total billed model tokens.
3. Bind endpoint, requested and expected returned model, arm, schema, prompt, predicate/question and renderer versions, plus the complete effective visible state, to a canonical immutable request/cache identity. Current-pack, query, candidate, order and relation mutations must not reuse conditional results. Standalone has a separate namespace and cannot supply a conditional cached label.
4. Follow the documented OpenRouter Decisions payload shape: `model`, `state`, and typed `questions`. The response body uses `model` and an `answers` dictionary containing typed choice records. Internal request-key receipts wrap that body and are not part of model-visible state. This shape is documented by [OpenRouter](https://openrouter.ai/blog/insights/what-is-jev/); this stage does not submit it to the endpoint or verify live response behavior.
5. Validate each requested output independently. A missing or malformed dimension becomes unknown with a reason; it must not contaminate a well-formed sibling dimension. A wrong request/model binding, extra or duplicate identity, or an invalid envelope cannot be used as a valid cached decision. Raw probabilities and confidence are not thresholds, quality estimates or calibration evidence in this stage.
6. The adapter and contract runner have no credential loading or live transport path. Execution requires explicit transport injection; use fixed fake response values and separately authored malformed-response probes. Do not manufacture fake responses from the fixture's semantic expected labels or report an accuracy score against them.

## Verification and readout

Run all sixteen artificial cases through the mechanical contract. Check pairwise standalone cache reuse, conditional cache invalidation and relation-control isolation. Expected-label mutation must leave the request bytes unchanged. Verify cache-hit transport suppression and stale/misaligned response rejection. Independent QA may inspect all synthetic cases; the user's sampled-real-data constraint remains unchanged because no real research data is read here.

Publish aggregate counts, source commitments, test results and the authored fixture, clearly identifying all returned labels as simulated. Preserve an immutable run record. If a contract fails, repair it before any semantic inference; do not turn successful serialization into a research-effect claim.

A subsequent semantic probe requires its own frozen model/runtime binding and resource limit. Local decision models may serve as additional baselines if they fit the available hardware, but their predictions are not JEV predictions. This stage neither schedules such inference nor resumes the paused 900-question paid evaluation.
