# Evidence exchange: prior-art gate and an offline decision contract

2026-10-04. Fixed-budget evidence replacement and query-gap repair already have close precedents. The remaining question is narrower: can a cheap typed judge distinguish an information-preserving improvement from an attractive candidate that evicts necessary information? This phase defines that comparison and tests software behavior. It does **not** establish a new algorithm, JEV accuracy, or better RAG answers. No API calls, model inference, training or new natural-data reading occurred in this phase.

## The novelty claim must become narrower

| Primary source | Relevant mechanism | Consequence for this project |
|---|---|---|
| [SEAL-RAG, v1](https://arxiv.org/html/2512.10787v1), Sections 3.2–3.5 | Source-span-grounded entity/relation/qualifier ledger, sufficiency assessment, missing-fact retrieval, then fixed-k candidate/victim replacement by a weighted utility difference | Structure, explicit gaps and replacement at fixed k cannot independently carry our novelty claim. Its stated swap rule does not separately specify the full original-information-preservation veto below; that is a design difference to investigate, not proof of novelty. |
| [Context-Picker, v1](https://arxiv.org/html/2512.14465v1), Section 3.2 and Algorithm 1 | Offline leave-one-out deletion followed by regenerated-answer judging against a reference, retaining deletions that preserve correctness | Testing information removal and preserving answerability are established ideas. Online reference-free typed judgments have different supervision and costs, but their value remains unmeasured. |
| [Evidence Tree Search, ACL 2025](https://aclanthology.org/2025.acl-long.1175/) | Evidence combination and search informed by the accumulated evidence state | Conditioning selection on a set is already studied; swapping the judge backend for JEV does not establish method novelty. |

This focused check is not an exhaustive literature review. Publication value would require a useful, independently measured improvement over these overlapping ideas and appropriate cheaper controls. JEV's release date does not supply that evidence.

## Compare complete packs, with independent interpretation

The [six-question source diagnostic](NATURAL_DEFINITION_OPPORTUNITY_20261004.md) found feasible replacements that could evict query-related content. Let the original pack be S, removed unit r, retained pack A = S minus r, candidate c, and proposed pack T = A plus c. Let Fq(X) denote query-relevant information supported by the complete pack X.

The experimental strict-improvement gate asks three independent Choice questions:

1. Does T support information absent from S: Fq(T) minus Fq(S) is nonempty?
2. Is information originally supported by S lost from T: Fq(S) minus Fq(T) is nonempty?
3. Does any part of T directly conflict with another part of T?

Accept only `yes / no / no`. An unknown or malformed dimension abstains and retains S; an explicit no-gain, loss or conflict rejects and retains S. A judge must interpret each pack only through its own units. Although the comparison input contains both roles, T cannot borrow a removed definition or reference antecedent from S. Conflict assessment includes retained-versus-retained conflicts, not only the candidate.

With accurate, consistent information judgments, the first two conditions imply strict information inclusion, Fq(S) is a proper subset of Fq(T). They do not guarantee generator accuracy or answer-score improvement. This conservative gate excludes potentially worthwhile tradeoffs, efficiency-only paraphrase compression, and some corrections of erroneous original facts. It is an experimental baseline, not a universal selection objective.

## Counterexamples and reproducible algebra

The following are authored fact sets, with all atoms relevant to a fixed artificial question. They are neither natural samples nor model-labeled examples.

| Retained A | Removed r | Candidate c | Complete-pack gain / loss | Strict gate, assuming no conflict |
|---|---|---|---|---|
| {a} | {b} | {b,d} | yes / no | Accept |
| {a} | {b} | {d} | yes / yes | Reject: loses b |
| {a} | {b} | {b}, expressed differently | no / no | Reject: no information improvement |
| {a,b} | {b} | {d} | yes / no | Accept: removed fact retained elsewhere |
| {a,d} | {b} | {d} | no / yes | Reject |
| {a} | empty relevant fact set | {d} | yes / no | Accept |

Rows two and six hold the query, retained evidence and candidate constant while changing only what is removed. Candidate usefulness alone cannot distinguish them. Row three exposes a subtler issue: candidate gain relative to A can mistake replacement of the removed fact for a net improvement. Exact-text deduplication does not prevent this when the candidate paraphrases r.

The [standalone algebra audit](audit_exchange_algebra.py) enumerates the 512 triples of subsets of three artificial atoms. This tiny mathematical enumeration is not a dataset sweep. Its [saved aggregate](results/evidence_exchange_algebra_20261004.json) records:

| Ideal fact-set rule | Accepted | True strict improvements | Accepted with information lost | Accepted semantic no-ops |
|---|---:|---:|---:|---:|
| Original-relative added information alone | 169 | 127 | 42 | 0 |
| Retained-relative gain plus direct no-loss | 218 | 127 | 0 | 91 |
| Complete-pack gain plus direct no-loss | 127 | 127 | 0 | 0 |
| Full-closure forward gain plus reverse no-loss | 127 | 127 | 0 | 0 |

These counts illustrate definitions under exact set union; they are not model accuracy, estimated natural prevalence or new theoretical results. Conflict and combination-dependent inference are absent from this toy model.

The last row prevents an overclaim about existing conditional predicates. With U = S plus c = T plus r and monotone Fq, if Fq(U) minus Fq(S) is nonempty and Fq(U) minus Fq(T) is empty, then Fq(S) is a proper subset of Fq(T). That ideal forward/reverse test is sufficient, and is equivalent in this simple union model. For a merely monotone query projection it can be more conservative when r and c jointly unlock information not available in either S or T. Current candidate-contribution wording does not establish such a full-closure oracle; direct S/T questions specify the desired object explicitly.

Two additional future semantic challenges matter: a conflict already inside retained evidence, and a candidate whose pronoun or formula only becomes meaningful using the removed unit. Neither is modeled by the fact-set table. Software prompt checks cannot prove a real judge handles them.

## Contract and budget boundaries

The new [exchange module](../../SLAC/retrieval/decision/exchange.py) is separate from the existing frozen conditional interface and paid probe. It binds the original and proposed roles, removed ID, question/schema versions and budgets into immutable request identity. Typed results from another role assignment or old conditional request cannot authorize replacement. Relation metadata is outside this plain-text phase.

Original S and proposed T each satisfy the same explicit generation evidence token cap and unit cap, measured by the injected counter on the complete renderer. The judge sees their inventory union under a separate judge-evidence token cap, plus a complete-payload byte cap. Neither cap claims to be a provider's full prompt token count; instructions and the query are outside the evidence-token surface. Test counters are artificial, not BGE or JEV tokenizers. This implementation does not attach itself to a live retriever or open a network transport. The previous six-request paid client rejects its new arm.

The [33 focused exchange tests](../../tests/research/test_exchange_decision.py) and existing core/client regression tests passed: **130 tests total**. Coverage includes separate complete-pack budgets, both original/proposed overflows, final payload-byte limits, changed removal roles, stale responses, malformed/unknown dimensions and altered plans. Independent review found no blocking defect and separately enumerated all 27 yes/no/unknown triples: exactly one accepts, seven reject, and nineteen abstain. Every nonaccepted decision retains S. Three removed-unit roles on an identical judge inventory had three distinct cache keys; reversing original/candidate roles also changed identity. These are software checks with synthetic responses, not semantic accuracy tests.

## Next falsifiable research step

Keep the same q/S/r/c/T across controls: BGE ordering, added-information-only acceptance, explicit gain/loss/conflict acceptance, and only later the same text with source relations. Require beneficial acceptance as well as harmful acceptance and unknown rates; rejecting everything cannot count as an improvement. For semantic probes, first freeze a very small paired challenge with removable irrelevant, fully recoverable and uniquely necessary evidence. Keep expected labels outside model inputs and retain failures without retuning the same examples.

A useful result must reduce harmful acceptance while retaining beneficial exchanges, then survive a small natural-query check. Relation claims additionally require equal text and budget with true/no/permuted metadata. If choices barely change, useful swaps disappear, or no advantage survives the plain-text control, do not enlarge the study on the strength of interface tests. No API experiment is launched by this report. The old 900-question paid confirmation remains paused; the previous six-request conditional probe remains closed.
