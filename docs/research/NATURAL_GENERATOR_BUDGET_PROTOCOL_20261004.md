# Frozen natural exchanges under the generator evidence renderer

2026-10-04. Zero API. The preceding source-generation integration is implemented
and tested, but it does not establish a useful selection mechanism. This bounded
diagnostic asks whether the actual LLM evidence template, applied to the same
six exposed natural exchanges, changes their previously reported feasibility.
The sole shared positive reference is case 2; an unchanged budget cannot explain
or repair its rejection by the earlier JEV loss judgment.

## Fixed input and identity boundary

Use all six cases, in ordinal order, from the immutable
`natural-exchange-sample-01/review_packet.json` and `proposals.json`. Retain all
twelve S/T packs, three units per pack, including headings and ambiguous cases.
No new question, candidate, victim, document, label, or reference answer may be
selected. No source-text rewriting, normalization, truncation, reranking, cap
adjustment, or search for an alternative pack is allowed.

A metadata identity check of these cases against the existing two-document
Refiner export precedes this diagnostic. The document sets have no overlap.
Do not construct a Refiner source mapping or transfer paragraph labels to
partial/merged chunks. Here "preserve source text" means preserving the frozen
retrieval-text field byte for byte; it does not certify original-document text
or native spans. The report must say **frozen-text renderer diagnosis**, not
Refiner or FinalIntegrator end-to-end evaluation.

## One declared projection, three rendered surfaces

For every saved S and T, preserve its source-order list. Map only known fields to
the production LLM `EvidenceItem`: unit ID to `chunk_id`, case document ID to
`doc_id`, complete frozen `text` to `passage_text`, and the existing question ID
and question to `query_id` and `query_text`. Leave optional ranks, scores, paths,
roles, token estimates, views, and expansion metadata unset. Source order is
not a retrieval rank. This is one declared minimal projection, not every
possible output of the FinalIntegrator normalizer; additional metadata changes
the measured surface and must not be silently inferred.

Measure exactly three complete strings per pack:

1. The previous `[unit_id]` evidence surface, whose hash/count must match the
   frozen proposal.
2. The exchange core's production `[doc_id/unit_id]` rendering, whose count must
   match the published six-request JEV result.
3. The production LLM renderer with passage-preserving policy, including its
   instructions, numbering, declared identity fields, and separators.

Reuse the pinned local BGE-M3 tokenizer, fast mode, `add_special_tokens=True`,
`truncation=False`. Bind all five tokenizer/config files, including the four
bound by the old sample and `config.json` bound by its later JEV plan. Load local
files only with remote code disabled, networking blocked and telemetry off.
Do not load model weights or perform neural inference. There are 36 primary
whole-string counts: six cases times S/T times three surfaces. Never sum
individual passage estimates.

The fixed cap remains **three units / 1,024 BGE proxy tokens**. BGE token counts
permit comparison with this study's old budget; they do not certify the answer
model's context window, API usage, or cost. Count only evidence, excluding the
system prompt, question, memory, provider framing, and output.

For the generator projection, create the existing six-field budget receipt and
compile a request using the production compiler, with an invalid placeholder
endpoint and no client. When its count exceeds 1,024, require the compiler to
reject it and preserve that outcome; do not enlarge the receipt limit. Otherwise
require the final compiled message to equal the counted block and require
serialization/recompilation to preserve the payload. Store all text/request
artifacts privately and publish only anonymous counts and hashes.

## Fixed decision readout

Use the already published JEV labels, separate A/B labels, and their frozen gate
statuses solely as historical readouts. Do not ask for new judgments or decide
the ambiguous cases. Copy the previous full-gate and gain-only choices without
reprompting a model. Report S feasibility and T feasibility separately:

- If S is over budget, report `invalid_start` for the new projection; do not
  describe keeping S as a feasible selection or repair it by dropping a unit.
- If S fits but T does not, retain S for a budget reason and show any old accept
  action that this would block.
- If both fit, budget adds no veto; preserve the old choice and unknown/reject
  distinction. An unknown state is not an accepted replacement.

The three original surfaces are not three new model runs. Previous semantic
labels remain tied to their old JEV request; this rendering diagnostic does not
claim a new model would return the same labels for a differently rendered prompt.

## Decision and verification

If feasibility and actions are unchanged, close this budget-explanation branch
for the declared projection and return to query-related loss semantics. If a
specific pack becomes infeasible, identify it and separately state whether case
2 remains feasible. Such a change motivates only a bounded integration choice;
it does not prove answer gains or a novel algorithm. Do not tune metadata/caps or
expand the sample after seeing counts.

Bind this protocol, the runner, production source, immutable input files, label
results, identity review, tokenizer files, and runtime package versions before
tokenization. Preserve failed runs instead of overwriting. Independent checking
may recount the saved 36 strings with the same local tokenizer, verify exact
frozen-text inclusion and hashes, and inspect compiler records, without rerunning
the proposal algorithm or any model. Do not retest the unrelated full suite.
Old paid probes and the 900-question plan remain closed/paused.
