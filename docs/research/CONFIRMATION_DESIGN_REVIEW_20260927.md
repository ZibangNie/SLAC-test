# Confirmation design review: one frozen reported-score rule

**DRAFT — 2026-09-27.** This is an independent design recommendation, not an executed confirmation protocol, dataset admission receipt, sample-size guarantee or permission to send paid requests. It uses only published development aggregates, existing method contracts and source/exposure metadata. No new QA, reference answer, relation label, model output or API key was read. No old frozen file was changed.

The recommended next question is narrow: **on a separately fixed project cohort, does one reported-score selector improve downstream Answer F1 over the same-pool BGE reranker and over coarse support-label selectors?** A positive answer would support this selection procedure under the specified generator and data policy. It would not establish calibrated probabilities, JEV-specific superiority across model families, cross-stage relation sharing or the full SLAC architecture.

## 1. Why this design, and what is already exposed

The [completed main answers](PRIMARY_ANSWER_RESULTS_20260927.md) and [posthoc BGE comparison](POSTHOC_RERANKER_ANSWER_RESULTS_20260927.md) use the same exposed 77 questions / 24 families. Reported-score minus BGE has a family-balanced development difference of +0.086382; differences against JEV and Qwen ordinal selection are +0.032201 and +0.026605, with intervals crossing zero. These are motivations, **not target effects for a power calculation**. The two old score rules used identical packs and responses; selecting `p_yes_only_k3` now does not retroactively turn either into an independent replication.

The [conditional precision appendix](CONDITIONAL_PRECISION_PLANNING_20260927.md) estimates four development SDs from 24 family means. Its normal, unadjusted precision scenarios neither control multiple comparisons nor establish prospective power. This review adds a proposed multiplicity family and corresponding conditional planning arithmetic; it does not rewrite that appendix.

Keep the 15-question pilot, early 104-question baseline and exposed 77-question main study separate from the future cohort. Do not pool their rows or count inherited responses as new independent evidence.

## 2. Five arms that can be locked before new outcomes

Use one given-document pool per question: the existing dense top-8 plus native-neighbour expansion, cap 16, with the exact existing ordering and expansion algorithm. All five arms receive identical query and complete candidate text. Freeze encoder/tokenizer revisions, candidate identities, original ranks and source spans before support calls. No owner expansion, relation bonus, oracle, Refiner training or cross-document retrieval enters this confirmation.

| Arm | Frozen selection contract |
|---|---|
| `dense_k3` | Original dense ranking of the common complete pool. |
| `reranker_k3` | `BAAI/bge-reranker-v2-m3`, existing revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`; full query–unit pair, no truncation, raw-logit descending; ties by original rank, then native order. Freeze the numerical/runtime profile as well as weights. |
| `I_jev_k3` | One JEV support judgment per candidate; exclude `no`, prioritize `yes` over `unknown`, then original rank/native order. |
| `I_general_k3` | Same ordinal policy using the fixed Qwen support judgments and complete candidate coverage. |
| `p_yes_only_k3` | Reuse the **same** JEV judgments as `I_jev_k3`; exclude `no`, sort by descending reported `p_yes`, then original rank/native order. No ordinal tier, threshold fitting, score normalization or confidence-field substitution. |

The score schema is fixed: exact `yes/no/unknown` fields, finite values in [0,1], sum in the inherited [0.985,1.015] interval. A malformed score is an execution failure, not a low score or `unknown`. These are reported scores, with no calibration claim. Freeze the inherited prompt and support batch policy; batch membership is part of what the model sees, not an interchangeable transport detail.

All arms use the same exact-native-text deduplication and whole-pack counter: scan rank order, skip an item whose complete rendered pack would exceed 1,024 BGE tokens, continue scanning, stop after at most three accepted units, render in native order. Do not truncate units or sum separate token counts. Empty packs remain valid outputs. No new k=1/2 search or second score arm is needed for this minimal confirmation.

Use the same `slac-qasper-answer-v1` evidence-only prompt, Qwen generator, Alibaba route, temperature 0, reasoning disabled, JSON answer contract and maximum output 512. Gold appears only in final scoring. An empty evidence pack still goes through the generator; do not hard-code its answer. There is no sixth empty-only arm in this five-arm design. The old empty control measured evidence-only abstention, not closed-book knowledge; it supplies context, not new-cohort control data.

Score–JEV ordinal holds the judgments and eligible set fixed, making it the closest comparison of score ordering with coarse ordering. Score–Qwen ordinal changes judgment backend and possibly eligibility as well; it is a comparison of selection procedures, not an isolated score-format effect. Score–BGE also compares complete procedures. All are cap-matched, **not actual-length matched**. Report actual length and count; do not condition, filter or regress away method-induced length after observing answers. No length-independent causal claim is proposed.

## 3. Cohort admission and exposure: operational, not absolute blindness

Before any new QA is revealed to selection or evaluation, freeze a source/component policy and take its **entire eligible cohort**. A workable policy can exclude all 32 previously exposed development documents and quarantine any connected source component intersecting exposed/training material under the declared version/material-reuse rules. Apply the same rule to every metadata flag. The known train-material-reuse pair requires an explicit recorded disposition under this policy; the review does not decide its family identity merely from shared text.

The screened 249 are **documents**, not 249 certified independent families. Zero lexical flags are an operational pass under the declared screen, not proof of independence or absence of pretraining contamination. Source screening, exposure registration and component formation can be executed and audited without inventing a requirement for manual inspection of every document. Record actual machine-assisted decisions honestly; do not describe them as completed human review. If family components contain several documents, average all admitted questions in the component before assigning it one family weight.

Root identified a further provenance issue during this review: the old exporter read and decoded the complete validation archive member before writing the 32-document subset. The bound source/report check must enter the exposure register. Thus **do not claim that remaining QA bytes have never been read or decoded**. Distinguish machine parsing, persisted/exported QA, researcher visibility, selector/model input and use for tuning/evaluation. A later cohort may be new to project selection/evaluation without being strictly untouched. Model-training exposure remains unknown. If that distinction cannot support the intended claim, label the study a fixed new-project-cohort replication rather than a pristine holdout test.

Include all questions belonging to admitted families; freeze question inventory after source admission and before predictions. Preserve unanswerable, figure/table, empty-reference, difficult and evidence-unreachable cases. No reference-based sampling, chosen question cap, positive-gain filtering or replacement of failed questions. Structural integrity issues must be handled uniformly before any method outcome is available, with their effect on the cohort recorded. The QA schema can be validated by a restricted exporter/evaluator without exposing reference content to method selection.

Freeze the complete cohort even if its size exceeds the precision target below; do not stop enrollment at a favorable count or choose a subset with lower estimated variance. If whole-cohort execution is unaffordable, do not shrink arms/questions after seeing outcomes. A revised scope would be a separately dated design, set before outcome exposure; no such revision is authorized by this draft.

## 4. One inferential estimand and three predeclared claims

For family f with n_f admitted questions, define `d_f(A,B) = mean_q(official_AnswerF1(A,q) - official_AnswerF1(B,q))`. The primary estimand is the mean of these family differences: `Delta_FB = sum_f d_f / F`. Qasper official max-over-reference Answer F1 is the outcome; Evidence F1 is supplementary and cannot substitute for it. Families, not questions, candidate decisions, requests or cached responses, are the independent units assumed for inference.

| Comparison, direction fixed | Role | Interpretation of a positive adjusted lower bound |
|---|---|---|
| `p_yes_only_k3 - reranker_k3` | Primary scientific comparison | Replicated improvement over the fixed strong reranker within this setting. |
| `p_yes_only_k3 - I_jev_k3` | Key secondary | Incremental downstream value of score ordering over coarse ordering of the same judgments. |
| `p_yes_only_k3 - I_general_k3` | Key secondary | Improvement over this fixed general-model ordinal procedure; not general backend superiority. |

Use **one simultaneous family of three FB Answer F1 intervals**, with two-sided alpha 0.05 and Bonferroni allocation 0.05/3 to each. Proposed implementation:

```text
s_j = sample SD of family differences for comparison j (ddof=1)
C_j = mean(d_fj) +/- t_quantile(1 - 0.05/(2*3), df=F-1) * s_j/sqrt(F)
```

Each marginal interval is 98.333333…%; the three form a nominal 95% simultaneous set under the marginal interval assumptions. The pairing and Bonferroni principles follow the [NIST paired-observation treatment](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm) and [NIST simultaneous-interval treatment](https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm). Applying them to **family-mean differences** is this protocol's proposed analysis. Independent, comparable families and a sufficiently accurate t approximation are assumptions; bounded/skewed F1 differences are not exactly normal. Bonferroni does not repair a bad marginal interval or hidden family dependence.

Report all three adjusted intervals whether or not the primary succeeds. A lower bound strictly greater than zero supports only that specified directional claim; an upper bound below zero is evidence in the opposite direction. A zero-containing interval is inconclusive, not equivalence/non-inferiority. An endpoint equal to zero does not pass. No positive QW or dense comparison rescues a failed BGE primary claim. If both ordinal comparisons remain inconclusive, do not claim that score detail adds confirmed answer value over coarse judgments. No practical-effect or non-inferiority margin is inferred from development means.

For a zero sample SD, report the observed degenerate distribution and mark the proposed t-inference unavailable; do not use a collapsed positive interval as an automatic confirmation or switch tests after seeing it. Independently verify the final analysis arithmetic. Any source/component problem affecting independence invalidates the affected inferential claim rather than being fixed by a smaller p-value.

## 5. Full reporting without an additional multiplicity trap

Publish all five arm means for Answer F1 and actual evidence tokens under both FB and QW weights, selected-count/empty-pack/abstention counts, and all four score-minus-comparator differences including dense. For these **4 contrasts × 2 metrics × 2 weights**, publish 16 descriptive 95% paired cluster-bootstrap intervals using 10,000 shared PCG64 family draws, seed `20260927`, linear quantiles. Resample F whole families, retaining all their questions and shared predictions; recompute the QW denominator for each draw. The three adjusted FB Answer F1 intervals above are an additional explicitly labeled inferential table. This preserves continuity with development reporting without pretending the 16 descriptive intervals have simultaneous coverage.

No choice between QW/FB, t/bootstrap, Answer/Evidence F1, or corrected/uncorrected intervals after seeing which is positive. Do not report bootstrap tail frequency as a calibrated p-value. Bootstrap sampling also requires an appropriate sampling population and can be unreliable for small samples; see [Dror et al., ACL 2018](https://aclanthology.org/P18-1128.pdf). Freeze synthetic tests for unequal family sizes, duplicate cache use, zero differences, negative differences and adjusted endpoints before real scoring.

The fixed-cohort mean itself is descriptive and exact for the saved predictions. Intervals extrapolate to a posited population of comparable eligible families; they do not make a fully enumerated source cohort random, cover domain shift or demonstrate model-training independence. One saved generator response per unique payload does not estimate repeat-generation variance. Exact-payload response sharing must remain paired, not treated as extra observations. Any later multi-seed or second-generator study needs a separately specified estimand.

## 6. Conditional precision target, not a power claim

Propose a planning half-width of **0.05 Answer F1** for each of the three adjusted FB intervals. This is a reporting-resolution choice, not a minimum useful effect or a guarantee that small ordinal differences will be detected. Use the development SD multiplied by **1.5** as the designated planning scenario, retaining 1 and 2 as sensitivity scenarios. This multiplier is a design assumption, not an SD confidence bound.

With no new outcomes, solve the monotone integer inequality:

```text
H_j(F,m) = t_quantile(1 - 0.05/(2*3), F-1) * m * s_development,j / sqrt(F)
F_required(h,m,j) = smallest integer F >= 3 with H_j(F,m) <= h
```

| Contrast | Development family SD | m | F for h=0.075 | F for h=0.05 |
|---|---:|---:|---:|---:|
| Score − BGE | 0.19129806752817607 | 1 | 41 | 88 |
| Score − BGE | same | 1.5 | 88 | **193** |
| Score − BGE | same | 2 | 153 | 339 |
| Score − JEV ordinal | 0.15121066036447847 | 1 | 27 | 56 |
| Score − JEV ordinal | same | 1.5 | 56 | 122 |
| Score − JEV ordinal | same | 2 | 97 | 214 |
| Score − Qwen ordinal | 0.14934004585448446 | 1 | 27 | 55 |
| Score − Qwen ordinal | same | 1.5 | 55 | 119 |
| Score − Qwen ordinal | same | 2 | 95 | 208 |

Thus the designated **conditional** target is at least 193 independent, comparable families for all three comparisons, rather than the old unadjusted normal scenario's 127 for BGE. At F=192, the BGE half-width is 0.050016499; at F=193 it is 0.049884447. Hypothetical F=249 gives 0.043830717 / 0.034645785 / 0.034217185 for BGE/JEV ordinal/Qwen ordinal. **There is no assertion that 249 independent families are available.** Under m=2, even that hypothetical size does not meet the BGE 0.05 target.

The SDs come only from the [published scenarios](results/qasper_conditional_precision_scenarios_20260927.json); their observed mean differences never enter these formulas. New numbers were computed using a standard-library incomplete-beta t CDF and bisection, checked at known df=1/2 quantiles and against separate Simpson integration at df=95/191/248 (CDF error <1e-10). This is arithmetic validation, not empirical coverage validation or new evaluation.

After metadata admission, record `conditional_precision_target_met = (F >= 193)` without changing the cohort. If false, label the plan precision-limited before QA/model outcomes; do not choose m=1 or a wider target to mark it passed. This flag is a planning warning, not a theorem prohibiting a fixed-cohort estimate. A smaller admitted cohort could still be reported as a prospective estimate if that limited objective is declared before outcomes; this review does not silently authorize that change or paid execution.

At final reporting, record all achieved adjusted half-widths and whether each is <=0.05. Report wide or negative results unchanged. **Do not add families, change a stopping time or unseal another dataset based on the observed width or direction.** A fixed simultaneous interval may support its narrow claim even when a planning target was missed, but the precision limitation must remain visible. No optional top-up or efficacy/futility look is part of this design.

## 7. Model stability, complete coverage and failure rules

Freeze the request endpoint, provider policy, prompt/body, tokenizer/model files, batch membership/order, runtime precision, allowed returned-model identifier and generator contract in a new plan. Historical actual identifiers are JEV `typesafe/jev-1.13-20260917` and Qwen `qwen/qwen3.6-plus`; a current availability/schema check using old development or synthetic material is a separate operational task. An alias string is not a cryptographic model-weight pin. If an immutable provider revision is unavailable, disclose the observed identifier/time-window limitation instead of claiming perfectly fixed weights.

Within a run, a provider or returned-model change, missing/duplicate IDs, malformed scores, refusal, truncated generator output, missing candidate judgment, source/hash change or timeout halts the run. No fallback model, response repair, candidate deletion or automatic retry. A failure is not a semantic `no`, not a model `Unanswerable`, and not a zero-quality imputation. Do not issue primary complete-cohort quality from a successful subset. Preserve all completed responses and failed/uncertain attempts, report coverage and operational failure; an explicit amendment before any quality disclosure would be necessary for a later recovery. It must not be relabeled an uninterrupted run.

Keep single-use registration, append-before-dispatch attempt accounting, exact full-payload cache checks and the inherited 65-second request bound. Freeze a **new absolute execution deadline** and whole-cohort conservative budget separately; the expired overnight window is not extended by this document. Preserve historical known and unknown costs separately, with no refund of attempted reservations. Freeze support cost from all query–candidate judgments; before generation, exact-deduplicate all five-arm complete payloads and admit the entire new-generation set or stop. Do not claim that a support-only reservation already covers every possible answer payload. Same-payload reuse checks endpoint, prompt version, canonical bytes and locked model provenance, not selected IDs or saved F1.

The Qwen support and answer roles belong to the same model family, so this remains generator-conditional evidence. It cannot settle a broad JEV-versus-general-LLM ranking. Measure support/generation costs and latency separately with all failures retained; neither these comparisons nor saved-response reuse establishes end-to-end SLAC efficiency.

## 8. Machine-protocol handoff and unresolved fields

The following proposed values can be copied into a new versioned protocol without looking at new quality. They are recommendations until that protocol and input manifest are sealed:

```json
{
  "status": "draft_not_executed",
  "arms": ["dense_k3", "reranker_k3", "I_jev_k3", "I_general_k3", "p_yes_only_k3"],
  "setting": "given_document_fixed_candidate_pool",
  "max_units": 3,
  "max_evidence_tokens": 1024,
  "primary_metric": "official_answer_f1",
  "primary_weighting": "family_balanced",
  "inferential_comparators": ["reranker_k3", "I_jev_k3", "I_general_k3"],
  "inferential_interval": "paired_family_mean_t_bonferroni",
  "familywise_alpha": 0.05,
  "two_sided": true,
  "descriptive_bootstrap_draws": 10000,
  "descriptive_bootstrap_rng": "PCG64",
  "descriptive_bootstrap_seed": 20260927,
  "planning_half_width": 0.05,
  "planning_sd_multiplier": 1.5,
  "conditional_family_target": 193,
  "cohort_selection": "all_families_passing_pre_QA_source_policy",
  "outcome_based_top_up": false,
  "automatic_retries": 0,
  "complete_cohort_required_for_primary_results": true,
  "new_paid_execution_admitted": false
}
```

Still unresolved: final exposure/source-policy evidence and component map; admitted document/family/question counts; exact candidate and task manifests; current usable provider/model contract and file hashes; synthetic statistical implementation checks; complete support and generator budget; new absolute deadline and one-use execution receipt. None is marked passed here. No mandatory new user-permission step or universal per-document human review is introduced; these are concrete data/execution checks that the authorized project can perform. The root data review owns cohort admission, and its findings must be reflected without converting operational screening into a contamination-free claim.

The next zero-API deliverable is therefore a frozen cohort/exposure manifest plus a machine-readable design with unresolved checks explicit. Only then can a separately bounded exporter expose the required question inputs, keep references out of selection, and calculate an executable whole-cohort plan. This draft creates no paid plan and opens no new QA.

## Source snapshot for this review

These hashes identify the public versions used here; they do not freeze the root's evolving handoff/status documents.

| File | SHA-256 |
|---|---|
| `CONDITIONAL_PRECISION_PLANNING_20260927.md` | `6722cc85ff923cee662ccb9239c229438a95119cb8d53012814205487cd39bbd` |
| `results/qasper_conditional_precision_scenarios_20260927.json` | `1cc780dc984b0700ae423e0702e2789005104b72266cd3c6a5ae4397df90c3c9` |
| `MORNING_RESEARCH_HANDOFF_20260927.md` | `f92d9b6737adb797b233ea45725526bf86829aee34474b7973a88bf1d41a825f` |
| `RECOVERED_PRIMARY_ANSWER_PROTOCOL_20260927.md` | `56055c8b1c79a7b73a52c796d56abc3fff067099a73631e3074cbb6b788a20c9` |
| `POSTHOC_RERANKER_ANSWER_RESULTS_20260927.md` | `ee578097819e217ea59d5d0ef5627f4676ad94014c87c9e3bf4eb0e66daaf284` |

The archive-decoding exposure observation was supplied by Root during this review and awaits its dedicated bound provenance receipt; no raw archive was opened here to investigate it. External references above are NIST's statistical handbook and the original ACL statistical-methods paper, accessed 2026-09-27. They support the general procedures and limitations, not validation of this dataset or the proposed precision target.
