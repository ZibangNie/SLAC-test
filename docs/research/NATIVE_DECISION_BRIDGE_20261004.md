# Native SLAC input bridge for conditional decisions

2026-10-04. The [input bridge](../../SLAC/retrieval/decision/native_bridge.py) connects existing SLAC retrieval record types to the [conditional decision contract](CONDITIONAL_JEV_CONTRACT_RESULTS_20261004.md). Its 32 synthetic tests and independent checks passed. This stage uses authored records only, with no model inference or real-dataset inspection.

## Why a direct field copy is insufficient

The index builder loads `ChunkRecord` objects, then [enriches their text](../../SLAC/retrieval/preprocess/anchor_fields.py) before writing the lookup used by retrieval. Enrichment includes normalization and can change Unicode characters, whitespace and line endings. The [build script](../../SLAC/retrieval/run/run_build_index.py) separately copies the original input to `data/refined_chunks.jsonl`. Only that original source, or records captured before enrichment, can supply the text intended by a whole-native-unit comparison. A `ChunkRecord` type alone cannot prove this provenance.

The [packer](../../SLAC/retrieval/pack/evidence_packer.py) assigns `PackedEvidenceItem.order` from retrieval selection order. Source order is `ChunkRecord.chunk_index`. Scores, pack order, paths and display anchors must not silently become the source order or source text of a conditional request.

## Fixed implementation scope

- Capture a frozen snapshot containing only document ID, chunk ID, source index and the exact text string, before mutable enrichment. This preserves the decoded string, not the original JSON file's serialization bytes.
- Resolve only the selected current-pack and candidate IDs against the caller's native snapshot mapping. Reject missing references, document/ID mismatches and invalid selected views. Do not scan or certify the whole corpus.
- Require an explicit text policy. `exact` rejects display/native differences. `reconstruct` creates a new state from complete native units and reports every changed-text ID; it is not an equivalent replay of the old displayed pack.
- Recompute the budget over the exact source-ordered rendering through the core's injected counter. Do not reuse `token_est`, truncate units, or silently keep a cheaper normalized rendering.
- Accept only explicit source relations. Parent, neighbor, expansion and path fields do not automatically establish a definition, reference or qualifier. In reconstruction mode, reject relations touching changed-text units because old character offsets may remain in bounds while referring to a different span. Re-anchoring with native-text provenance is outside this first implementation.
- Keep rank, retrieval score, role, path, metadata and supervision outside model-visible inputs. Bind bridge version and text policy in the internal state-version identity. Fake transport execution remains explicit.

## Verification and limits

The [32 tests](../../tests/research/test_native_decision_bridge.py) passed. Independent verification used three authored `ChunkRecord` objects and the existing enrichment function. It separately checked the expected state, source order and rendered text, Unicode/line-ending preservation, post-capture mutation isolation and metadata exclusion. A mapping that rejects enumeration was accessed exactly three times for the selected IDs; unrelated records were neither read nor validated.

Five invalid capture cases, twelve invalid builds and four invalid counters were rejected. Both changed relation-endpoint cases were rejected, including an offset that remained in bounds after NFKC expansion but referred to the wrong original characters. Reconstruction accepted the exact complete-render budget and rejected an insufficient one. Exact and reconstruction modes have distinct internal cache versions; standalone projection still isolates hidden current-pack content. The injected counter was a synthetic UTF-8 byte counter, not a model tokenizer or billed-cost measurement.

Both examples in the [module README](../../SLAC/retrieval/decision/README.md) executed with socket access blocked. The [aggregate verification record](results/native_decision_bridge_20261004.json) preserves source and independent-check hashes. Existing conditional-contract source bindings remain unchanged. These checks cover the selected artificial view and adapter behavior, not corpus-wide validation.

No new selection policy, semantic accuracy, Answer F1 or source-structure benefit follows from this bridge. It does not alter the default retrieval pipeline, prove that a caller supplied the original source, or validate all corpus identities. Before a real-data semantic pilot, source provenance and any new relation anchors still need a fixed sampled check. The old 900-question paid plan remains paused.
