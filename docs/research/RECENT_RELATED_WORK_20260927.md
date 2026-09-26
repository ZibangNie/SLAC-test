# Recent related work: evidence sets and retrieval budgets

Checked 27 September 2026 using the official sources linked below. This is a bounded addition to the [working outline](PAPER_WORKING_OUTLINE_20260927.md), not a systematic review. No new QA data, model calls or primary JEV results were read for this note. Each source summary is under 200 English words; author-reported findings are not replications.

## ETS — archival ACL paper

**Enhancing Retrieval-Augmented Generation via Evidence Tree Search**, ACL 2025. Reading scope: official abstract, introduction, Sections 2–3, experiment setup in Sections 4.1–4.3, and Appendices A–B.

ETS represents candidate evidence sets as sentence paths, conditioning expansion on accumulated evidence. MCTS annotations use reader correctness and correct-answer likelihood; policy/value training supports early-terminating beam search. Its five LongBench datasets include Qasper. Sentence units, training, readers and benchmark selection differ from our fixed 77-question native-paragraph development protocol; no reported score is directly comparable. [Official paper](https://aclanthology.org/2025.acl-long.1175/), [PDF](https://aclanthology.org/2025.acl-long.1175.pdf).

**Our inference:** dependency-aware set selection is prior work. A possible SLAC distinction requires tested reuse of explicit relations across stages, beyond a selector alone. This review does not establish that such reuse is unprecedented. ETS has not been reproduced here.

## Know Before You Fetch — preprint, abstract-level review

**Know Before You Fetch: Calibrated Retrieval-Budget Allocation for Retrieval-Augmented Generation**, arXiv:2606.29959v1, 29 June 2026. Only the official abstract and metadata were checked; this note treats it as a preprint.

The authors propose calibrating uncertainty signals into correctness probabilities to choose closed-book answering, one or five retrieved passages, or abstention. They report experiments on TriviaQA, Natural Questions and MS MARCO, including held-out threshold selection and model-dependent latency trade-offs. These adaptive QA settings differ from our fixed native-paragraph protocol. [Official abstract](https://arxiv.org/abs/2606.29959v1).

**Our inference:** neither a probability interface nor budget allocation alone establishes our novelty. Reported JEV scores need independent calibration evidence before probability claims. Any cost advantage must include decision overhead. We have not checked the full experimental implementation or reproduced the results; the abstract's claims are contextual evidence, not a performance comparator.

## AB-RAG — preprint, abstract-level review

**AB-RAG: Adaptive Budgeted Retrieval-Augmented Generation for Reliable Question Answering**, arXiv:2606.29090v1, 27 June 2026. Only the official abstract and metadata were checked; this note treats it as a preprint.

The authors describe a training-free loop that answers, estimates confidence and decides whether to retrieve more within a budget. The estimator combines model certainty, answer–evidence agreement and retrieval-score variance; closed APIs use self-consistency for certainty. The abstract reports three backbones and two datasets, including mixed findings. This adaptive stopping task differs from selecting a fixed native-paragraph pack. [Official abstract](https://arxiv.org/abs/2606.29090v1).

**Our inference:** combining confidence signals with bounded retrieval is already proposed. Our contribution would require a distinct, tested mechanism or workload trade-off. Do not infer comparable budgets, correctness calibration, reproducibility or superiority from this abstract; no AB-RAG experiment was reproduced here.

## Consequence for the next experiment

Keep the first causal test small: fixed candidate pool, same backend and generator, relation content versus independent scoring and structural controls. If it succeeds, test whether reusing that relation information across stages adds value over selection-only behavior and exact-call caching. Measure selected identities, actual lengths, stage-specific effects and full costs. An improvement attributable only to the selector, a changed pack length or a cheaper backend cannot establish cross-stage sharing value.

No primary JEV quality table is filled by this literature update. Audit-gated results and a later blinded evaluation remain separate requirements.
