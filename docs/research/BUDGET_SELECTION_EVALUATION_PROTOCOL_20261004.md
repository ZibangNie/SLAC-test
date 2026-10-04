# Cached budget selection: frozen evidence evaluation

2026-10-04. This protocol is written before the new selections are scored against references. It implements the conditional evaluation in `BUDGET_SELECTION_PROTOCOL_20261004.md`: the completed reference-free opportunity gate found strict reported-score utility improvements. The selectors and their outputs are now fixed. This is an exposed-development diagnostic, not independent confirmation or an Answer F1 experiment.

## Frozen inputs and denominator

Use every one of the same 77 development questions and 24 original families. Evaluate exactly four methods: `score_greedy`, `score_exact`, `density_greedy`, and `matched_resource_exact`, producing 308 question-method records. No question, family, method, figure evidence, unanswerable question, empty evidence list, or unfavorable outcome may be removed or replaced. A malformed or missing required reference stops the whole evaluation; it does not create a smaller denominator.

The completed opportunity inputs are frozen by these SHA-256 values:

- `budget-run-01/plan.json`: `dca4932ad2462d6ab2263a682a8faff13ab45511a2e62757e8f96f025f938c8a`
- `budget-run-01/selections.json`: `57043ecf73ad3d025ce6fa46062d2c74256805a34a09b4a6f912629ad22a9df1`
- `budget-run-01/summary.json`: `753522d6defa19b37c5aba7e81896739ee22b5653bffe891382c56756324a467`
- `offline-20261004/cache_contract.json`: `dd1fe7cab7c176be82a27965e4bffc75bcc1f79d384df72f52efb81990d3e20c`

The first three files are inside `artifacts/research-foundation/offline-20261004/budget-run-01`. The reference source is the existing `qasper-alignment-v2/native_qa_sidecar_v2.jsonl`, SHA-256 `929f5cbdaf05e0c86d6e51a5c90265e729886c59b528824ddd014ca94d06a97f`. The prepared units, historical `p_yes_only_k3` records, and scoring implementation must also match their previously frozen commitments. No new 900-question references, answer cache, raw provider responses, API key, network request, tokenizer, model, or training process is used.

## Metric and comparisons

For each saved selection, pass the selected units' exact `native_text` strings and the question's original `answer_annotations` to `qasper_metrics.evidence_metrics`, leaving `text_evidence_only=False`. This preserves the official exact-string Evidence F1 semantics and maximum over annotators. Do not substitute canonical rendered text, normalize evidence strings, deduplicate reference evidence, filter figure markers, or average over annotations. Unanswerable annotations retain the official empty-reference behavior. Reference annotations are loaded only after the evaluation plan has frozen the selections and scorer hashes.

Report all four Evidence F1 means in two weightings: question weighted (the arithmetic mean over all 77 questions) and family balanced (mean within each original family, then mean across all 24 families). The question-weighted Evidence F1 is the official aggregate; family balancing is a separate descriptive statistic.

Compare each of `score_exact`, `density_greedy`, and `matched_resource_exact` against `score_greedy`. Report question-weighted and family-balanced paired mean differences, question-level win/tie/loss counts, and family-level win/tie/loss counts. Compare unrounded deltas to zero without a fitted tolerance. Do not compute confidence intervals, significance tests, threshold searches, best-method selection, answer metrics, or a selectively matched answer-cache result. Publish negative or zero effects alongside positive effects.

The complete 77-question `score_greedy` selections, pack hashes, and token counts must match the saved `p_yes_only_k3` baseline. Every recomputed baseline Evidence F1 must exactly equal that historical per-question Evidence F1 before any complete result is published. This is the experiment's baseline reproduction, not a renewed audit of raw JEV responses. If it fails, stop without a partial aggregate and investigate the binding or scoring discrepancy.

## Execution and reporting

This is a single evaluation of already frozen methods; its results must not retroactively change their selection rules. All inference code is absent. CLI execution disables ordinary Python socket connections as an additional zero-network guard. Private output contains an evaluation plan, all 308 per-question scores, and a summary. The summary exposes only aggregates, counts, hashes, and limitations. It contains no question or document IDs, source text, references, predictions, or local model paths.

The evaluation reads and hashes the bound existing sidecar bytes, mechanically projects the 77 required identities and their annotations, and computes the experiment across the complete denominator. It does not manually inspect every data row or run a new full response audit. Additional content or independent numerical QA remains restricted to the previously fixed sample if needed.
