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
