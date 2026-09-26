# Boundary edit contract v1

`slac_refiner.label_contract` is a standard-library-only source of truth for
canonical supervision. The contract version is `slac-boundary-edit-v1`.

For `T` atoms, `b0` and the desired final boundary vector have exactly
`max(T - 1, 0)` entries. Gap `g` separates atoms `g` and `g + 1`; there is no
terminal boundary at `T - 1`. Binary values are integer `0` or `1`.

Each initial boundary gets exactly one action, ordered by source gap:

| Action | Meaning | Cost |
| --- | --- | --- |
| `KEEP` | Emit the original gap | 0 |
| `DEL` | Emit no gap | 1 |
| `SHIFT:k` | Emit source gap plus nonzero `k`, with `abs(k) <= K` | `0.25 * abs(k)` |
| `INSERT` | Emit an unmatched final gap | 1 |

Surviving edit targets must be strictly increasing. A `DEL` does not reset the
last emitted target. INSERT may be adjacent to another INSERT or edit target;
it cannot duplicate an edit target. `SHIFT:0` is spelled `KEEP` in canonical
labels. Far moves can always be represented by DEL plus INSERT.

`derive_canonical_labels(b0, b_final, K=6)` finds the minimum-cost monotone
alignment. Ties maximize KEEP count, then total matched boundaries, then
prefer the first candidate path encountered in ascending source/target order.
The implementation uses sparse weighted matching with a Fenwick prefix
maximum: a match saves `8 - abs(k)` quarter-cost units compared with DEL+INSERT.
Updates for a source are delayed until all of its candidate targets are scored,
preventing source reuse. This has the same optimization objective as dense
edit-distance DP, without allocating a full source-by-target matrix.

`replay_labels(b0, labels, K=6)` strictly validates and replays the decomposition.
Its `require_monotone=False` and `require_canonical_spelling=False` options are
for auditing legacy traces; exported canonical labels use neither opt-out.
`validate_boundary_vector(vector, num_atoms=T)` checks the exact gap domain.

The alignment objective describes edit economy. It does not validate the
semantics of `b_final`, recover noise ancestry, or prove a research benefit.
Action accuracy must only compare targets derived using the same contract;
the same final boundaries can have several valid edit decompositions.

## Legacy diagnostic export

Run from the isolated workspace, using explicitly named files:

```powershell
C:/Environment/python/venvs/slac-research/Scripts/python.exe SLAC/refiner/scripts/repair_refiner_labels.py --train-input D:/code/Github/SLAC-test/SLAC/refiner/data/refiner_real_dataset_canonical/refiner_train.jsonl --dev-input D:/code/Github/SLAC-test/SLAC/refiner/data/refiner_real_dataset_canonical/refiner_dev.jsonl --output-dir artifacts/research-foundation/labels
```

An existing output directory is refused. Input wildcards and test filenames
are refused. Only the supplied train/dev files are opened. The exporter checks
every original replay, vector shape, source coverage, radius, identity fields,
and declared split, then derives and validates canonical labels. It preserves
all text, `b0`, `b_gold`, identity and source metadata. The original noise
metadata remains historical information and no longer defines the actions.

Each repaired row records the source file SHA-256, original line number,
original-label SHA-256, contract/schema versions and lineage flags. The
manifest records input/output hashes, operation counts and split identity
overlap. Inputs are hashed before and after export. `status=complete` is
required; failed runs retain a failed manifest and partial artifacts.

The output is **legacy development diagnostics only**. Every row and the
manifest explicitly set `cleared_for_training=false`,
`independent_evaluation=false`, `semantic_gold_verified=false`, and
`source_split_lineage_unresolved=true`. A small optimization smoke check on
these rows can diagnose code execution, but cannot establish generalization,
independent dev performance, or a publishable improvement.

On 2026-09-26, the complete recovered inputs produced:

| Check | Train | Dev |
| --- | ---: | ---: |
| Rows exported and strictly replayed | 8,443 | 1,056 |
| Original nonmonotone action rows | 5,784 | 746 |
| Canonical action assignments changed | 8,045 | 1,006 |
| Rows with legacy `SHIFT:0` spelling | 8,443 | 1,056 |
| Repaired INSERT labels | 35,326 | 4,997 |
| Repaired INSERT labels adjacent to edit targets | 21,919 | 3,355 |
| Rows whose retained `orig_split` is `test` | 857 | 91 |

There is one shared `doc_id` across the exported train/dev inputs and zero
identical atom-text arrays. Exact text hashes do not exclude near duplicates
or shared source families. These are source lineage limitations, not repaired
by changing action labels. The named test split file was not opened.

The export manifest and serialized verification report are at
`artifacts/research-foundation/labels/manifest.json` and
`artifacts/research-foundation/labels/serialized_verification.json`.

## Verification

```powershell
C:/Environment/python/venvs/slac-research/Scripts/python.exe -m pytest SLAC/refiner/tests/test_label_contract.py SLAC/refiner/tests/test_repair_refiner_labels.py -q
```

The focused suite covers an independent dense-DP optimum for 5,120 small
vector/radius combinations, strict replay, crossing ancestry, DEL transitions,
adjacent INSERT, far moves, deterministic ties, malformed vectors, non-destructive
export, lineage retention, input filename restrictions, and failed manifests.
