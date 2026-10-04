# Conditional decision contracts

`conditional.py` builds immutable requests for standalone support, conditional information addition and separately judged conflict. Relation metadata can be supplied as a separate treatment. This is a research adapter: it provides typed signals, not a tested evidence-selection policy.

The module has no credential loader or network client. A caller must explicitly inject a transport. During the offline contract stage only fixed fake transports are used; their labels are not model predictions.

```python
import json
from SLAC.retrieval.decision.conditional import (
    ConditionalState, Unit, build_request, execute_offline,
)

state = ConditionalState(
    query="How many ports does the amber console have?",
    current_pack=(Unit("u001", "The amber console is indoors.", 0),),
    candidate=Unit("u002", "The amber console has four ports.", 1),
)
request = build_request(
    state, arm="plain_conditional",
    endpoint_id="offline-example", model_id="synthetic-request-model",
    expected_response_model="synthetic-response-model",
    token_counter=lambda text: len(text.split()),  # contract counter, not BGE
    counter_version="whitespace-example-v1", max_tokens=128,
)

def fixed_fake(envelope_bytes):
    envelope = json.loads(envelope_bytes)
    return {
        "request_key": envelope["request_key"],
        "response": {
            "model": "synthetic-response-model",
            "answers": {
                "conditional_added_information": {"type": "choice", "choice": "unknown"},
                "conflict": {"type": "choice", "choice": "no"},
            },
        },
    }

result = execute_offline(request, transport=fixed_fake)
assert result.envelope_valid
# These are explicitly simulated choices, not answers inferred from the text.
```

`request.payload()` returns a new dictionary, so changing it cannot mutate the frozen bytes. `freeze_response` and `decode_cached` check the request binding; a standalone response cannot be reused as a conditional response. Output dimensions are validated separately, and malformed or missing ones become unknown with a reason. Invalid envelope/model/identity bindings invalidate the whole response.

The injected token counter sees the whole source-ordered evidence rendering: candidate only for standalone, current pack plus candidate otherwise. Query, questions and relation metadata are outside that evidence budget and inside the separate full-payload byte cap. Neither synthetic word counts nor payload byte counts are actual provider usage or cost.

See the [protocol](../../../docs/research/CONDITIONAL_JEV_CONTRACT_PROTOCOL_20261004.md) and [authored fixtures](../../../docs/research/fixtures/conditional_jev_contract_v1.json). A future live connector must separately pin model/runtime/provider settings and enforce request/cost limits; the absence of a live default here is intentional. Current correctness checks do not establish semantic accuracy, retrieval benefit or novelty.

## Existing SLAC records

`native_bridge.py` resolves selected `PackedEvidenceItem` and `RetrievalCandidate` identities against explicitly captured source snapshots. Capture before `enrich_chunk_record`, or from the original input snapshot: the usual retrieval lookup has normalized text. A record's Python type cannot certify that its text came from the original source.

```python
from SLAC.retrieval.schemas.records import ChunkRecord, RetrievalCandidate
from SLAC.retrieval.preprocess.anchor_fields import enrich_chunk_record
from SLAC.retrieval.decision.native_bridge import (
    capture_native_chunk, build_native_request,
)

source = ChunkRecord(
    doc_id="invented-doc", chunk_id="invented-unit", chunk_index=4,
    atom_start=0, atom_end=1, num_atoms=1, path=[], depth=0,
    text="The panel has \uff14 ports.\r\n  Its label is amber.",
)
native = capture_native_chunk(source)
enrich_chunk_record(source)  # Existing preprocessing mutates source.text.
candidate = RetrievalCandidate(
    chunk_id=source.chunk_id, doc_id=source.doc_id,
    text=source.text, path=[], depth=0,
)
wrapped = build_native_request(
    "How many ports does the panel have?", [], candidate, {native.id: native},
    text_policy="reconstruct", arm="plain_conditional",
    endpoint_id="offline-example", model_id="synthetic-request-model",
    expected_response_model="synthetic-response-model",
    token_counter=lambda text: len(text.encode("utf-8")),
    counter_version="toy-utf8-byte-counter-v1", max_tokens=512,
)
assert wrapped.changed_text_ids == (native.id,)
assert wrapped.request.payload()["state"]["candidate"]["text"] == native.text
# No transport or model call occurs in this example.
```

`text_policy` is required: `exact` rejects differing display text; `reconstruct` creates a new native state and reports changes. Budgeting uses the complete final rendering, not old token estimates or pack order. Relations involving changed-text endpoints are rejected to prevent reusing stale character offsets. The adapter does not infer relations from parent/neighbor/path fields or change the default retrieval pipeline. See the [bridge verification and boundaries](../../../docs/research/NATIVE_DECISION_BRIDGE_20261004.md).

## Refiner source exports

`refiner_bridge.py` accepts exact source-mode chunk dictionaries together with a validated `NativeCoverageIndex`. Capture before retrieval loading or text normalization: the adapter checks the supplied row against the immutable document source view and does not certify the identity of an entire export file. It copies the actual source rendering into a frozen `Unit`, with `order=atom_start` so order stays relative to the common atom basis.

```python
from SLAC.refiner.pipeline.assemble.source_coverage import build_native_coverage_index
from SLAC.retrieval.decision.refiner_bridge import (
    capture_refiner_source_chunk, build_refiner_source_request,
)

# source_view is a validated DocumentSourceView. Each native unit supplies
# native_unit_id, source_span, and exact source_text in that document's coordinates.
index = build_native_coverage_index(source_view, native_units)
snapshot = capture_refiner_source_chunk(source_chunk, index)
wrapped = build_refiner_source_request(
    query, (), snapshot, arm="standalone",
    endpoint_id="offline-example", model_id="synthetic-request-model",
    expected_response_model="synthetic-response-model",
    token_counter=caller_token_counter, counter_version=caller_counter_version,
)
assert wrapped.request.payload()["state"]["candidate"]["text"] == source_chunk["text"]
receipt = wrapped.provenance_receipt()
assert receipt["request_key"] == wrapped.request.cache_key
# Constructing this object does not send a request or access a response cache.
```

Coverage preserves each native interval's `full` or `partial` intersection. A complete unit plus extra whitespace is full coverage but not `exact_native_unit_id`; a merged chunk can cover several full units; a leaf can cover only part of one. These are coordinate facts, not support labels, scores or a rule for reusing judgments. Blank native units are rejected and whitespace outside nonblank units stays explicit gap coverage.

The adapter reuses the core `build_request`, cache and response contracts. Only `standalone` and `plain_conditional` are supported; no relations are inferred. Source SHA256, coordinate system, atom/character spans and native coverage remain in a separate fresh provenance receipt. Same-document snapshots must agree on source version, coordinates and base model atoms/spans. Different chunk boundaries and deliberate overlaps on that basis are allowed; core duplicate-ID, duplicate-position and duplicate-complete-text checks still apply.

Changing visible text changes the existing request key. A change elsewhere in an unselected source document may leave the complete visible request unchanged and retain that key, while the new receipt identifies the current source version. Both cases require the full existing request binding, not merely a source membership or text-hash match. The evidence budget counts the complete core rendering with headers; Refiner's raw chunk token count is a different surface. No live transport/provider policy or default retrieval-pipeline integration is added here. See the [fixed two-document contract result](../../../docs/research/REFINER_JEV_SOURCE_CONTRACT_RESULTS_20261004.md).

### Source records through retrieval preprocessing

Source-mode records now require an explicit registry when loaded. Its JSON schema is
`{"schema":"slac-refiner-source-indexes-v1","documents":[{"view":{...},"native_units":[...]}]}`:
`view` contains the fields of `DocumentSourceView`; each native unit has `native_unit_id`,
`source_span` and exact `source_text`. The registry is validated when loaded, including
duplicate document IDs and JSON fields. It identifies the supplied source version; it
does not authenticate a publisher or retrieve an external original.

```python
from SLAC.retrieval.dataio.source_records import (
    load_source_indexes, source_snapshot_from_chunk_record,
)
from SLAC.retrieval.dataio.readers import load_chunk_records
from SLAC.retrieval.preprocess.anchor_fields import enrich_chunk_record

indexes = load_source_indexes("source_indexes.json")
chunks = load_chunk_records("refined_chunks.jsonl", source_indexes=indexes)
enrich_chunk_record(chunks[0])  # Retrieval uses normalized text.
snapshot = source_snapshot_from_chunk_record(chunks[0], indexes)
assert snapshot.unit.text == chunks[0].meta["refiner_source"]["text"]
# Pass the restored snapshot to build_refiner_source_request as above.
```

The JSON-safe `meta.refiner_source` retains raw text, identity, spans and hashes through
lookup serialization. Restoring it rechecks those fields against the supplied registry
and requires current retrieval text to equal either the original or its exact supported
normalization. Persisted dictionaries are not trusted proofs. Chunk and leaf loaders
retain the same signatures for legacy input; the extra `source_indexes` argument is
keyword-only. Anchors carry IDs and require a lookup to recover source text.

`python -m SLAC.retrieval.run.run_build_index` accepts `--source_indexes_json` for
source-mode inputs and copies the registry to `meta/refiner_source_indexes.json`.
Its optional `--metadata_only` action requires an absent or empty output directory,
runs validation, enrichment and metadata serialization, and returns before importing
embedding/index runtimes. It writes `indexes_built: false`; these files cannot serve
dense retrieval. Both retrieval entrypoints reject that stage before model loading
and reload a saved registry for full builds. Their module imports may still load runtime
libraries, so the model-free guarantee applies to the metadata build action.

This preserves source inputs through real preparation and reload. A query-time selector
must still explicitly restore snapshots and budget the final JEV/generator rendering;
the default selector does not automatically invoke JEV. See the
[bounded loader result](../../../docs/research/REFINER_SOURCE_LOADER_RESULTS_20261004.md).

### Selected candidates and an existing pack

`resolve_refiner_source_selection` resolves existing retrieval objects through the
validated chunk lookup. It accesses only the selected IDs; it does not enumerate
the lookup, select evidence or read scores/token estimates.

```python
from SLAC.retrieval.dataio.source_records import resolve_refiner_source_selection

current, candidate_source, changed_ids = resolve_refiner_source_selection(
    packed_items, candidate, chunk_lookup, indexes, text_policy="reconstruct",
)
wrapped = build_refiner_source_request(
    query, current, candidate_source, arm="plain_conditional",
    endpoint_id="offline-example", model_id="synthetic-request-model",
    expected_response_model="synthetic-response-model",
    token_counter=caller_token_counter, counter_version=caller_counter_version,
    max_tokens=1024,
)
# No request is sent. The builder counts its complete restored evidence render.
```

`exact` requires each selected object's text to equal the full source text.
`reconstruct` additionally permits the current, validated lookup text and reports
changed IDs in sorted order. Both accept a previously restored raw pack even when
the lookup text is normalized. Both reject a third form such as a summary,
truncation or stale display text; matching chunk IDs alone is insufficient.
Neither mutates the selected objects. All selected identities are checked before
the builder projects standalone/conditional visibility. Joint source-version and
atom-contract checks remain in the existing builder.

The resolver adds no request/cache version: after restoration, identical complete
source requests retain the existing source-bridge binding. The returned change
list describes local text restoration, not a model judgment. It also does not
certify the final answer model's budget: that renderer has its own headers and
must be counted separately from JEV's evidence render and the whole prompt.
See the [selection and renderer diagnosis](../../../docs/research/REFINER_SELECTED_SOURCE_RESULTS_20261004.md).
