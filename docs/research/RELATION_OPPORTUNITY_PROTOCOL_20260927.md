# Relation opportunity gate: fixed CPU protocol

Status: implementation and offline preparation only; the real mask enumeration requires a later explicit Root release. This protocol freezes only the zero-API gate proposed in [the next-experiment draft](NEXT_RELATION_EXPERIMENT_DRAFT_20260927.md). It does not admit static judgments, answer generation, training or any new data.

## Scope and input contract

All 77 exposed development questions and 24 families remain. Use the original support pool: 1,214 query-unit tasks, 562 unique document-adjacency pairs and 844 edge-query occurrences. Consume the completed primary-support recovery audit and its external execution receipt, both with fixed source SHA anchors. Require their expected versions, complete statuses, 77 questions, 1,155 upstream records and matching summary/plan/output hashes.

Rehash every directly consumed prepared/manifest, recovery plan/manifest, audit/execution/summary, JEV labels and per-question file. Require exact top-level source directory inventories; the provider directory itself is not reopened. Verify tokenizer files and legacy helper source against the audited plan's input commitments. Bind the new source, tests, this protocol, runtime versions and fixed output path. The full upstream input-hash commitment is retained as provenance; unused QA, model weights and old request/response bodies are not reopened or independently rehashed. The already completed audit supplies that inherited assurance. This is not a new provider/quality audit.

Old prepared/per-question JSON containers include query strings or quality fields. The adapter parses those containers and immediately projects identity, complete native units, candidate/rank, support-label mappings and saved I selection/pack/token witnesses. **It must not claim the original containers lacked quality fields.** The core `Case` has an exact field whitelist; query strings, references, answers, quality metrics and oracle witnesses cannot enter it. No old QA loader, scorer, `verify_completed_run` or generation helper is called. Prepare checks identities and counts but does not load the tokenizer or execute any selector.

## Fixed selector and edges

Native A immediately precedes B; the hypothetical relation means B depends on A. An already selected B boosts pending A by exactly one. There is no forward boost, recursive closure, cumulative bonus, whole-pair requirement or chunk merge gate. Support scores are yes=1 and unknown=0.5; no is never eligible. The active-edge domain requires both endpoints eligible and their exact native text different.

At each step choose the pending candidate with greatest base+Boolean bonus; break ties by frozen dense rank, then source order. Remove it from pending. Skip an already selected `(document, native_text)` duplicate; otherwise count the full source-ordered rendered pack with the pinned BGE tokenizer and accept only at ≤1,024 actual tokens. Stop at three accepted units or exhausted pending. Skipped candidates are not revisited. Preserve original `[unit_id]` headers, full text, tokenizer special tokens and no truncation. Final source-order rendering and empty-pack zero-token semantics match the existing packer.

The core does not search parameters or use outcome feedback. Native adjacency permits at most one such successor per A; duplicate edges are rejected rather than providing additional bonus.

## Enumeration and independent checks

For every question enumerate every binary subset of eligible edges, including empty and full. Frozen `m` histogram: 0:31, 1:19, 2:8, 3:10, 4:6, 5:1, 6:1, 7:1. This gives exactly 107 eligible edge-query occurrences and **501 masks**, with all 31 zero-edge questions retained.

Before enumerating each question, compare zero-mask output and every normalized attempt-trace field to the unchanged legacy **pure** I replay (which does not load data or score). Also match the saved audited `I_jev_k3` selected identities, exact pack SHA and actual token count. Independently implement full-adjacency selection as a separate list-scan algorithm, without calling the mask selector; require full-mask final output and complete trace parity. The trace includes candidate index, selected-before, base score, Boolean bonus, active triggers, changed priority, actual proposed tokens, acceptance and skip reason.

The complete enumeration must reproduce both endpoints. Each hypothetical pack must satisfy identity, eligibility, deduplication, quantity and actual-token constraints. Mask order and source edge order are deterministic. A per-query mask need not be jointly realizable as one shared document labeling across questions: the envelope is deliberately permissive. Neither the longest/shortest pack nor any mask is chosen as a method or forwarded for generation.

## Execution and completion

Prepare/run/audit use independent new artifacts. The plan directory has exactly `plan.json`, `cases.json`, `seal.json`; cases contain only the private whitelist. Root must review source/tests/protocol and the offline plan before running. Run output is fixed in that plan and must not exist. Any created output directory is single-use; failure is not resumed or overwritten.

Each run or audit is limited to 300 seconds of wall time from function entry. CPU mask enumeration is serial; tokenizer parallelism is disabled, `local_files_only=True`, `trust_remote_code=False`; no encoder/GPU is loaded. Cooperative deadline checks surround source loading, hashing, tokenization/selection steps, each mask, output writing and final complete-summary creation. This is not an operating-system kill guarantee for a blocked external library call. A deadline detected after summary creation removes the complete marker and writes failure; no timed-out run can pass audit.

Complete output has exactly `per_mask.jsonl`, `per_question.jsonl`, `public_aggregate.json`, `summary.json`. Failure has no valid complete summary. Audit independently re-enumerates every mask and both reference checks, compares complete saved traces and all aggregates, validates every summary field and output hash, checks timing for finite nonnegative values within the limit, and rehashes inputs/outputs. Saved timing is a measurement record, not independently reproduced latency.

## Output and interpretation

Private mask/query outputs retain source identities, selected identities, masks and traces in ignored local storage. Public output contains only denominator/inventory, parity counts, questions with any possible pack change, full-adjacency changes, unique-pack-count histogram, masks changed/unchanged, added/removed-count histogram, selected-count changes, actual-token ranges/differences, timings, runtime and source-binding hashes. It has no private identifiers, raw text, F1/recall/Answer F1, best mask or ranking of masks.

If every hypothetical mask yields the baseline pack for every question, this selector has no behavioral opportunity on this pool and the proposed static payment should stop. A nonzero opportunity count only warrants considering a later experiment; it does not establish correct relations, better evidence, improved answers, calibration, independence, cross-stage sharing or novelty. The static API plan remains outside this gate and outside current budget admission.

## Commands after preparation

From the repository root with the pinned research Python environment:

```text
python docs/research/run_qasper_relation_opportunity.py prepare --output artifacts/research-foundation/qasper-relation-opportunity-plan-01 --run-output artifacts/research-foundation/qasper-relation-opportunity-run-01
python docs/research/run_qasper_relation_opportunity.py run --plan artifacts/research-foundation/qasper-relation-opportunity-plan-01
python docs/research/run_qasper_relation_opportunity.py audit --plan artifacts/research-foundation/qasper-relation-opportunity-plan-01 --run artifacts/research-foundation/qasper-relation-opportunity-run-01
```

Only the first command is authorized during this implementation task. The latter commands document a future reviewed release, not automatic continuation.
