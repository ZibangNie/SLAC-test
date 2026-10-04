# Set-conditioned JEV: novelty gate and next offline contract

2026-10-04. **The generic set-conditioned selection idea does not pass the novelty gate.** This note records a focused primary-source check after the [cached addition diagnostic](CACHED_ADDITION_RESULTS_20261004.md). No new model calls were made. A narrowly testable source-structure contribution remains a research hypothesis, not an established innovation.

## What already exists

| Primary source | Consequence for this project |
|---|---|
| [Evidence Tree Search, ACL 2025](https://aclanthology.org/2025.acl-long.1175/) | Context-aware expansion of accumulated evidence sets already models dependencies among evidence sentences. Conditioning the next selection on the existing set is not a new contribution. Its search tree represents selection paths; that is a different object from the original document's hierarchy. |
| [Dynamic Passage Selector, 2025](https://arxiv.org/abs/2508.09497) | Joint passage dependencies and dynamic selection already address limitations of independent ranking and fixed top-k. Replacing independent ranking with set selection is insufficient differentiation. |
| [Sufficient Context, ICLR 2025](https://arxiv.org/abs/2411.06037) | Context sufficiency and selective generation already have explicit study. Sufficiency is distinct from a guarantee that a particular generated answer is correct. |
| [TypeSafe JEV documentation](https://docs.typesafe.ai/introduction) | Typed questions in one call are evaluated independently against a shared state. A set-conditioned decision must explicitly include the question, current evidence and candidate in that state; batching standalone questions does not supply conditioning. The documentation recommends narrow, atomic judgments. |

This targeted check is not an exhaustive prior-art proof. It rules out several broad novelty claims; it does not establish that the narrower proposal below is original.

## A possible testable difference

The research question is whether **verifiable source relations** in SLAC help a judge identify a particular unresolved definition, reference or qualifier in the current evidence set, beyond what a plain set-conditioned judge can infer from the same visible text. A parent heading or adjacent paragraph alone is not proof of such a relation. Avoid inventing a semantic edge from mere document proximity.

At fixed original candidates, complete source units and actual evidence-token budget, distinguish:

1. Standalone support: `J(q, c)`.
2. Plain conditional judgment: `J(q, S, c)`.
3. The same visible text with explicit, source-verifiable relation metadata.
4. Matched adjacency or predetermined relation-permutation controls, with the same relation count and visible text.

If true relation metadata does not outperform the plain conditional arm and the relation controls, reject the source-structure benefit hypothesis. If only conditional judgment helps, attribute the result to conditional selection. If only physical calls decrease, report that separately as an engineering measurement. Neither source structure, JEV recency nor combining existing components automatically constitutes a publishable method.

Do not run these arms on the single decreasing cached example and call it validation. The 77 questions and their outcomes are already exposed, and the 32-edge cache lacks complete two-by-two counterfactuals. Any future quality comparison needs a separately frozen outcome-independent sample and a primary comparison that separates source relations from ordinary conditioning.

## Next executable step: synthetic contract, zero API

Build a pure adapter and a small, human-authored contract fixture before considering inference. Keep expected semantic labels outside model-visible state. Reuse no real question text or observed failure. The suite must include paired states with the same question/candidate and different current evidence: one where a specified fact is absent, one where it is already present. Also include irrelevant content, an explicit conflict and an unresolved reference. A conflict requires its own atomic judgment; do not collapse every criterion into one ambiguous “good addition” label.

The offline checks should establish that:

- The exact current pack, candidate, source order, relation metadata and question version participate in the canonical state/cache identity. Old standalone labels cannot masquerade as conditional labels.
- Every relation endpoint resolves to a visible source unit; changing only a relation has a distinct binding. Controls preserve visible text and metadata size.
- The builder preserves whole native units and uses an injected exact rendered-token counter to reject over-budget states. Synthetic checks validate the budget contract, not actual production token measurements.
- Typed outputs are validated independently; malformed or unresolved decisions remain unknown. No threshold is fitted from the exposed cache and no simulated success is presented as model accuracy.
- Explicit transport injection supports a fake offline transport; the default contract runner has no credential loading or live-network execution path.

This step would make the integration concrete and testable, but cannot show semantic decision accuracy, retrieval gains or novelty. Keep the real API path disabled during this next step. A later small JEV contract probe, if needed, would require its own fixed cases, precise request/cost bounds and current OpenRouter contract verification; the old paid plan and reservations cannot authorize it by inheritance. The user permits small-scale exploration, but no new paid experiment is scheduled by this note.
