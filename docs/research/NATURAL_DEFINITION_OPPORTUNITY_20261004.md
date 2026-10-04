# Natural-question definition gaps: bounded offline diagnostic

2026-10-04. Six fixed natural questions show why a source relation alone is an insufficient retrieval trigger. Dense packs omit two linked definition targets, and locally reranked BGE packs omit four. All six method/target occurrences can be accommodated by at least one single-unit replacement within the original budget, but source review does not establish that the questions need the missing acronym expansions. This phase provides no reason to spend JEV calls on these particular gaps.

This is a source and budget diagnostic, not an answer-quality experiment. No reference answers, reference evidence, model-generated answers or old provider responses were opened. **API calls: zero.** A fresh, bounded local cross-encoder pass was performed; it must not be described as zero model inference.

## Sample and comparison

The previous [definition feasibility sample](NATIVE_DEFINITION_FEASIBILITY_20261004.md) fixed three development documents. Metadata inspection showed all three belong to the old pilot families and none belongs to the later 77-question cohort. Consequently the later BGE result cannot be reused. We retained these same documents and selected the first two existing pilot input queries per document, in original prepared order, before inspecting their question text or pack outcomes. All six were retained, with 16 original candidates each.

The [sampler](prepare_definition_pack_sample.py) parsed the existing input-only prepared JSON to index IDs; validation and retained content were restricted to the selected queries and three documents. It did not read the original QA sidecar or outcome files. This was not a new unseen sample: the pilot and source documents have prior exposure. The selection was independent of this diagnostic's outcomes, not a claim of independent confirmation.

The original candidate pool already includes eight dense seeds followed by immediate native neighbors up to 16 candidates. Both arms use exactly this same pool, complete canonical retrieval text, exact native-text deduplication, the original source-ordered `[unit_id]` renderer, at most three units and at most 1,024 **BGE whole-evidence tokens**. Query, instructions and any later JEV metadata are outside this evidence budget.

The [bounded local runner](run_definition_sample_reranker.py) scored exactly 96 original query/passage pairs with the existing `BAAI/bge-reranker-v2-m3` snapshot `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. It used local files only, CUDA FP16, SDPA, microbatch four and complete untruncated pairs. Ranking uses raw logits, then original dense rank and native source order; logits are not calibrated probabilities. Model and tokenizer hashes were checked, GPU admission passed, and no download, training, retry or CPU fallback occurred.

## Mechanical opportunities, with the full denominator

A gap means that a selected unit contains a retained explicit-acronym mention but its linked definition unit is absent, with no identical native definition text already present elsewhere in the pack. This is an omitted **source unit**, not proof that the pack lacks the relevant information or cannot answer. Multiple mentions and acronyms pointing to the same target within a pack count once.

| Quantity | Dense | Local BGE reranker |
|---|---:|---:|
| Complete packs | 6 | 6 |
| Packs with at least one linked definition target omitted | 2 | 3 |
| Omitted target occurrences across packs | 2 | 4 |
| Targets already in the original candidate pool | 1 | 2 |
| Targets among the original eight dense seeds | 0 | 0 |
| Targets adjacent to any selected native unit | 1 | 2 |
| Targets selected by the other baseline for the same question | 0 | 0 |
| Exact linked acronym appears in the question | 0 | 0 |
| Appending the target fits the token cap | 2 | 4 |
| Appending also fits the three-unit cap | 0 | 0 |
| At least one admissible single-unit replacement | 2 | 4 |

There are four distinct question/target cases across the two methods, not six independent opportunities. Four of the six questions have no dense gap; three have no BGE gap. BGE's larger gap count is not evidence of worse answers. Dense pack lengths are 155–458 BGE tokens, and BGE pack lengths are 277–813; equal caps do not imply equal lengths.

Adjacency here means an immediate neighbor of **any actual selected unit in the old nonempty native-unit sequence**. It does not use distance from the mention alone or the previous sample's raw block index, which can include empty fields. Thus half the omitted target occurrences are already reachable through simple adjacency; the other half also lie outside the original candidate pool. This sample does not show a nonadjacent, in-pool target opportunity beyond that adjacency control.

The [analyzer](analyze_definition_pack_opportunity.py) counts every whole appended or replacement pack with the pinned evidence tokenizer. It does not add individual token estimates. Replacement retains every selected mention that triggered the target and drops exactly one other selected unit. It enumerates feasibility for each target individually, not joint feasibility for several targets: no replacement is selected or scored for answer quality. Appending a fourth unit would violate the existing protocol even though all those packs fit the token cap.

## Source-context review changes the next action

The fixed six questions and four distinct target cases were reviewed without answer labels. The missing definitions concern general knowledge-base terminology, out-of-vocabulary terminology, topic-feature names and a classifier name. The associated questions concern dataset coverage, model baselines, or benchmark construction/quality. None explicitly requests an acronym expansion. Exact acronym absence from a question is only a lexical observation; it is not a general irrelevance rule.

The classifier-definition paragraph also contains evaluation-protocol facts. If it later helped an answer, the benefit could come from those other facts rather than from resolving the acronym. Conversely, preserving the selected mention while inserting this definition can require dropping units about benchmark construction or preprocessing, which are directly related to that question. A budget-feasible replacement therefore has a real information tradeoff.

This is AI-assisted, post-output qualitative source review, not human annotation, an independent semantic label set, or measured answer degradation. It supports a conservative research decision: keep explicit-definition links as a traceable structural baseline, but **do not treat every missing definition as an unresolved answer requirement** and do not start a paid comparison on these cases merely because their source bindings pass.

The next framework step should require a query-specific missing fact or unresolved condition before activating a relation. Ordinary same-text conditional JEV remains the required comparator: any eventual gain from added paragraph content must be separated from gain attributable to relation metadata. The current acronym rule is neither tuned to these six questions nor promoted as the paper's central contribution.

## Verification and resources

Six focused tests passed for the analysis: token-only versus three-unit feasibility, preservation of all triggering mentions, repeated-mention denominators, equal native text at different IDs, whole-pack token counting and adjacency to another selected unit. Independent verification checked all 96 score/input bindings and ranking rules, all 12 packs using the old renderer/packer, 51 saved whole-pack token measurements, all gap and replacement records, and the complete aggregate. It did not repeat model inference. Scope and hashes are retained in the [public aggregate](results/natural_definition_opportunity_20261004.json). Raw questions, source text, identities, logits and review notes remain in ignored artifacts.

The local pass used 11,412 unpadded pair tokens, 12,080 padded input tokens, maximum pair length 416, and zero truncations. Recorded peak PyTorch allocation was 1,186,919,936 bytes; peak reservation was 1,222,639,616 bytes. The parent measured 21.860 seconds for the run process, including interpreter startup, source/model hash checks, GPU admission, loading and output handling; this is not pure model-forward latency. The process exited normally, with no timeout or in-flight work.

There was no answer generation or quality-score calculation. Existing API usage ledgers are unchanged, the six-request conditional probe remains closed, and the old 900-question paid plan remains paused.
