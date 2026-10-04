# Native definition links: source-only feasibility

2026-10-04. A conservative explicit-abbreviation linker found usable cross-unit source links in a fixed **three-document development sample**, without API calls, model inference or reading question/answer files. This is evidence that a restricted source relation can be constructed from real native text. It does not establish semantic necessity, JEV improvement, RAG gain or a novel extraction method.

## Mechanism and source binding

The [extractor](../../SLAC/retrieval/decision/explicit_definition.py) accepts an ASCII acronym of two to six capital letters in parentheses when the final N preceding words on that line have exactly matching initials. It preserves original strings and character positions. It links later exact acronym mentions in other units to the definition; repeated definition occurrences, including identical expansions, cause abstention for that document/acronym. It does not infer a definition from proximity, a heading, or a model score.

Each link retains the exact definition span and mention span, both complete native-text SHA-256 values, document identity and source positions. `link.verify(native_units)` checks both endpoints and rule syntax before projecting a `SourceRelation`. This verifies the retained local witnesses; it does **not** re-establish global uniqueness if other units have changed or been added. Re-extract whenever the supplied document snapshot changes. Hashes attest equality to the supplied strings, not authenticity of an unseen original archive.

Existing SLAC tree expansion uses structural neighbors and ancestors. The new module supplies a textual witness that those expansions lack, but it is a research component and is not enabled in the default retrieval policy. Abbreviation lookup itself is a baseline mechanism, not a claimed paper contribution.

## Fixed bounded sample and results

The sample is the first three document records in the existing development native sidecar, selected by file order before root viewed their text. The [sampler](sample_native_definition_sources.py) parsed exactly three documents and their 193 source fields, under a preset 400-block cap. It verified field hashes and spans, complete document reconstruction, and retrieval/native string equality for all 151 selected paragraphs. Original block labels repeat between documents, so the runner uses an unambiguous JSON pair of document ID and block ID as its source-qualified unit ID.

Full sidecar files were read only for mechanical SHA-256 checks; their other document contents were not parsed by the sampler. The original archive was not reopened. These are already exposed development documents, not an independent holdout, a random population sample or an outcome evaluation.

The original plan used paragraph-only uniqueness. A source reviewer pointed out that definitions may repeat in an abstract or heading. Before extraction, an explicit scope amendment retained the original result and added the same rule over all nonempty native fields, with final mentions restricted to paragraphs. The reviewer had already seen the selected documents; this review is not blind. In this sample the wider scope removed zero links and added zero links; it is a correctness safeguard, not an observed quality gain.

| Sample document | Definition occurrences in full scope | Ambiguous acronym keys | Paragraph mention occurrences linked | Distinct unit/acronym links | Nonadjacent unit/acronym links |
|---|---:|---:|---:|---:|---:|
| 1 | 4 | 0 | 51 | 33 | 31 |
| 2 | 6 | 1 | 32 | 14 | 14 |
| 3 | 1 | 0 | 1 | 1 | 1 |
| Total | 11 | 1 | 84 | 48 | 46 |

The ambiguous key had three definition occurrences, all excluded from linking. The 84 mention occurrences reduce to **48 distinct directed unit/acronym links and 44 distinct directed unit pairs**. They involve seven definition keys in six definition units and 40 mention units. These repeated mentions are not 84 independent observations. “Nonadjacent” means source block-order distance greater than one, not a measured retrieval failure or a beneficial long-range dependency.

## Matched controls and validation

The [offline runner](run_native_definition_feasibility.py) constructed controls for the first 12 distinct unit/acronym links in the frozen ordering. Each has the same complete native definition, mention and distractor units, query, candidate/current-pack roles and source order across plain, source-linked and wrong-link arms. The wrong-link arm changes only the prerequisite ID; relation kind, dependent anchor and relation count remain fixed. The distractor is selected by a fixed nearest-source-order rule from units not containing that acronym.

All 12 triples were built successfully. True/wrong relation metadata have identical UTF-8 byte length and all three arms have identical rendered evidence; each arm has a distinct request binding. Evidence sizes range from 1,514 to 2,467 UTF-8 bytes. These are byte counts, **not model tokens or charges**. The runner uses an explicit byte counter for a mechanical construction limit and does not establish feasibility under the production token budget.

The constructed lookup question asks what the acronym stands for. It is useful for checking inputs, but its answer is already recoverable from the definition span. These 36 request objects were **not sent or assigned synthetic predictions**. They do not constitute a natural-question quality experiment. An incorrect-link control tests sensitivity to incorrect metadata; it is not an independent estimate of semantic benefit.

The new extractor's 32 authored tests passed, including repeated definitions, unchanged offsets after text mutation, Unicode/whitespace preservation, cross-document mismatches and forged partial-word mentions. Together with the existing conditional and native-bridge tests, **105 tests passed**.

A separate reviewer checked the frozen first 12 mention occurrences directly against source strings and context. All matched an explicit lexical definition, but they cover only **five distinct links, one acronym and one document**; this is not a precision estimate for all 84 occurrences, a human annotation study, or a blind review. The reviewer independently checked all 12 control triples without importing the production extractor, including source text, roles, the single-field metadata swap and recomputed byte counts. Request keys were distinct; independent cache-key recomputation was not performed because binding bytes were not in the review artifact.

Source-review details and bindings are recorded in the [aggregate result](results/native_definition_feasibility_20261004.json). Raw source text, source identities, generated requests and review notes remain in ignored local artifacts.

## Research decision

The narrow extractability gate passes: the sample contains nonadjacent explicit definition links with source witnesses. The algorithm is deliberately incomplete: lowercase or mixed-case acronyms, stopword initials, nonmatching expansions, earlier mentions and repeated definitions are excluded. Absence of a link does not establish absence of a semantic relation. Presence does not establish that an answer needs the definition.

The next useful offline test is whether a fixed small set of existing natural-question evidence packs actually contains a mention while omitting its definition, and whether this creates opportunities beyond ordinary retrieval or adjacency. Choose cases without looking at answer outcomes, preserve the full selected denominator and measure budget feasibility on whole evidence. Only after that test should a tiny JEV comparison assess whether relation metadata changes conditional judgments beyond the same visible text. Do not score generated acronym lookup questions as RAG gains or tune the rule against this sample.

This phase made **zero API calls** and changes no previous usage ledger. The six-request semantics probe remains closed; the old 900-question paid plan remains paused.
