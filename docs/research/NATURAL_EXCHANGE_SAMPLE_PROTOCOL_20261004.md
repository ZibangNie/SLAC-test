# Bounded natural exchange opportunity review

2026-10-04. Offline only. The preceding [six-case authored JEV probe](EXCHANGE_JEV_PROBE_RESULTS_20261004.md) showed useful loss/conflict vetoes and one gain error. Before any new inference, inspect whether a simple, outcome-independent proposal rule yields plausible information-preserving improvements in a small natural sample. This protocol is fixed before constructing or reviewing the selected new exchanges.

## Fixed scope and proposal rule

Reuse exactly the six questions and three documents already fixed in the [natural definition diagnostic](NATURAL_DEFINITION_OPPORTUNITY_20261004.md), in their saved order. These are exposed development inputs, two questions per document, not independent test questions. Reuse its 96 cached local BGE scores and original 16-candidate pools. Do not read the full prepared dataset, original QA sidecar, reference answers/evidence, generated answers, old API responses, or outcome scores. No new document or query is selected.

Inputs are the immutable local files `definition-pack-opportunity-01/selected_inputs.json`, `definition-sample-reranker-run-01/scores.json`, and `definition-pack-opportunity-01/opportunities.json` under the existing `offline-20261004` artifact root. Bind their exact published/audited hashes, this protocol and the new builder source before reading the projected exchanges. The existing input file contains three documents; document records are used only to resolve IDs in these six cases, not to search for favorable passages or inspect every source unit.

For each saved BGE evidence pack S:

1. Traverse its saved BGE rank order and choose the first candidate c that is not in S and whose exact native text differs from every selected unit. Do not rerank, tune scores, search for semantic gaps or use definition links.
2. Choose r as the selected unit with the worst position in that same rank order. Ranking is already a frozen total order; do not apply a new score tie rule.
3. Form T = S minus r plus c. Count both complete rendered packs and retain this one proposed exchange, even if T violates a budget or has no useful semantic gain. If no distinct candidate exists, retain a no-proposal row.

There are at most six proposed exchanges. Do not try the next candidate, choose another victim or replace an unsuccessful case after observing token feasibility or source meaning. This is a diagnostic of a particular simple proposal policy, not an optimized search or a claim of novelty. It deliberately tests the boundary between BGE's chosen pack and its remaining pool; it cannot estimate opportunities under arbitrary proposals.

## Source fidelity and budget

Use the original generator-evidence surface: complete retrieval text, source order and `[unit_id]` headers. Preserve exact native-text hashes separately; retrieval text and native paragraph text need not be identical. The evidence limits remain **three units and 1,024 BGE tokens**. Reuse the pinned, locally available evidence tokenizer, verify its four metadata/model-file hashes against the old manifest, and load with local files only and remote code disabled. Tokenize each whole S and T without truncation, including the original special-token setting. Check S against its saved token count. Do not use additive passage counts, the artificial byte counter from the synthetic probe, or the newer core renderer with different headers.

The new work performs local tokenization, not neural reranking, answer generation or training. Cached score/ranking reuse must not be described as a new model pass. Retain explicit over-budget and missing-candidate outcomes. A budget-feasible T only establishes that the comparison is mechanically possible.

## Independent source review

Produce an input-only review packet with each query, original units, proposed units, candidate and removed-unit role. Omit BGE scores/ranks, previous judgments, reference answers, expected labels and the earlier definition-gap review. Inspect only the visible units for each proposed exchange; unresolved references or missing tables remain uncertainty rather than prompting further document searches.

Two separate AI reviews should record the three complete-pack judgments from the fixed exchange contract: proposed information gain, original information loss and proposed internal conflict. Each non-unknown judgment needs a short query-specific explanation; claimed gain/loss should identify the relevant exact source statement or span and its unit ID. Interpret each pack independently, including facts requiring combinations of its own units. Do not borrow a removed premise from the comparison inventory. Record unknown when the visible text does not settle the judgment. These reviews are qualitative annotations, not semantic ground truth or JEV predictions.

Retain both reviews before discussing disagreements. Report complete agreement and disagreement by dimension, expected beneficial exchanges (gain yes, loss no, conflict no), loss cases, no-op/no-gain cases and unresolved cases. Distinguish overlap between these categories. Budget-infeasible proposals stay in the mechanical denominator and are not silently replaced. No statistical inference, population accuracy, calibration or answer-quality score is estimated from six cases in three exposed documents.

## Decision after review

If both reviews find a clear strict improvement, it is a candidate for a later separately bounded natural semantic probe, not automatic approval for new API calls. If no such case is found, do not submit these cases simply because the software or authored test passed. A negative result constrains this six-case proposal policy; it does not prove that information-preserving exchange is impossible or that JEV is ineffective.

Any next change to the proposal rule needs its own reason and must acknowledge exposure to this sample. The current study makes **zero API calls**, does not load credentials, and does not reopen either closed six-request probe or the old 900-question paid plan.
