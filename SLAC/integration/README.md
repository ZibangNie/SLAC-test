# Final integration with verified source passages

`FinalIntegrator(source_mode=...)` optionally restores verified Refiner source
text before selecting evidence. It budgets the complete evidence block rendered
by `SLAC.llm.service.renderers`, then binds that block to the compiled request.
Omitting `source_mode` retains the existing normalization, estimate-based
selection, and rendering defaults.

## Configuration

```python
from SLAC.integration.evidence.source_mode import SourceModeConfig
from SLAC.integration.orchestrator.final_integrator import FinalIntegrator

source_mode = SourceModeConfig(
    chunk_lookup=chunk_lookup,
    source_indexes=source_indexes,
    text_policy="reconstruct",
    token_counter=count_generator_evidence,
    counter_version="your-counter-and-version",
)
integrator = FinalIntegrator(llm_adapter=your_adapter, source_mode=source_mode)
response, artifacts = integrator.run_with_artifacts(integration_request)
```

The example's mappings, counter, adapter, and request are caller-supplied.
`chunk_lookup` contains `ChunkRecord` objects with the existing validated
`meta.refiner_source` envelope; `source_indexes` maps document IDs to
`NativeCoverageIndex` objects. See the [source loader and selection
documentation](../retrieval/decision/README.md) for their preparation. The source
mapping is accessed by candidate identity; this step does not search a dataset.
It restores every candidate in the chosen integration input and revalidates the
selected items before building the LLM request.

`exact` requires each candidate to contain the full source text already.
`reconstruct` also permits the matching current normalized lookup text and
restores the full original. Both reject unrelated summaries, truncation, stale
text, missing identities, or invalid source envelopes. If both `text` and
`passage_text` are supplied, they must agree exactly. Original leading/trailing
whitespace, CRLF, tabs, and Unicode are preserved in each passage. Paths and
other metadata remain separate rendered fields.

The injected counter must be deterministic and return a nonnegative Python
integer (not a boolean) for the **whole rendered evidence block**. Its unit must
match `pipeline_config.max_evidence_tokens`; production token accounting needs
an appropriate generator counter. There is no fallback to an estimate. Existing
per-item `token_est` values remain unchanged because the renderer includes those
values in the measured text. The separate budget receipt contains the actual
whole-block count. This budget excludes system instructions, queries, memory,
provider framing, and generated output; it is not a total model-context limit.

## Selection and compilation

The optional `pack_cost` path in the existing selector keeps its deduplication,
global stable ordering, direct-first quota, and remaining-candidate phase. A
trial is measured in its final global order. A candidate rejected during the
direct phase can still be considered by the existing remaining phase; costs are
not assumed additive or monotone. This is the existing greedy strategy, without
a global optimality guarantee. Empty packs, seeds, item limits, and the final
selected pack are checked as well.

In source mode, the integration preview equals the actual compiled evidence
block. The request explicitly uses `append_as_source_context_block` and carries
`meta.source_evidence_budget` with these six fields:

- `schema`: `slac-source-evidence-budget-v1`
- `renderer_version`: `slac-source-llm-evidence-v1`
- `rendered_sha256`: SHA-256 of the exact UTF-8 block
- `counter_version`: the caller's nonblank counter identifier
- `count`: the nonnegative whole-block count
- `max_tokens`: the configured nonnegative limit

The compiler checks the receipt structure, count limit, and actual rendered
digest. A source policy requires a valid receipt; a receipt requires the explicit
source policy. Removing or downgrading the policy is rejected. A text or rendered
metadata change after counting also invalidates the digest. Empty evidence has
a receipt for the empty string and adds no evidence message.

The receipt binds the caller's count to particular text. It does not authenticate
the caller, prove the counter is honest, or independently verify source
provenance. Source verification is performed by the integration layer using the
source indexes. JEV request construction shares that verified source content,
but its renderer and budget are separate from the generator's.

## Offline example and scope

```text
python docs/research/probe_source_generation.py
```

This fixed, authored example uses a compile-only adapter, toy UTF-8 byte counts,
and an invalid placeholder provider address. It blocks network and model imports
and never invokes a model. Its default output directory must be absent; results
are retained rather than overwritten. The script imports helper definitions from
the synthetic integration test, so the development dependency `pytest` is needed.

The normal `FinalIntegrator` still invokes its configured adapter after request
preparation. Source mode itself is not an offline or dry-run switch; use a
compile-only adapter for offline work, as the example does.

The example exercises `retrieval_packed_evidence`. Synthetic tests also cover
`reranker_pack_bridge` priority; they do not execute an actual reranker, retrieval
model, or answer model. See [the measured integration
results](../../docs/research/SOURCE_GENERATION_RESULTS_20261004.md) for limits and
checks. These are text preservation and budget checks, not answer-quality or
research-novelty results.
