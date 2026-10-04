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
