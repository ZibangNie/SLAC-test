"""Bridge explicitly supplied native SLAC snapshots to offline decision requests.

Capture records before retrieval enrichment, or load an original input snapshot.
This adapter preserves supplied text; it cannot certify its upstream provenance.
It performs no file access, model execution, relation inference or score reading.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from SLAC.retrieval.decision.conditional import (
    COUNTER_VERSION, PROMPT_VERSION, QUESTION_VERSION, RENDERER_VERSION,
    SCHEMA_VERSION, ConditionalState, FrozenRequest, SourceRelation, Unit,
    build_request,
)
from SLAC.retrieval.schemas.records import ChunkRecord, PackedEvidenceItem, RetrievalCandidate


BRIDGE_STATE_VERSION = "slac-native-bridge-state-v1"
TextPolicy = Literal["exact", "reconstruct"]


@dataclass(frozen=True)
class NativeRequest:
    request: FrozenRequest
    changed_text_ids: tuple[str, ...]
    text_policy: str


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validate_unit(unit: Unit) -> None:
    if not isinstance(unit, Unit):
        raise TypeError("native snapshots must be Unit records")
    _identity(unit.id, "native unit ID")
    _identity(unit.doc_id, "native document ID")
    _identity(unit.text, "native text")
    if type(unit.order) is not int:
        raise TypeError("native source order must be an integer, excluding bool")
    if unit.order < 0:
        raise ValueError("native source order must be nonnegative")


def capture_native_chunk(record: ChunkRecord) -> Unit:
    """Copy four fields before mutable retrieval preprocessing changes them.

    No trimming, normalization, summarization or metadata fallback occurs. The
    caller is responsible for choosing a pre-enrichment/original input record;
    enriched ChunkRecord objects have no reliable native-provenance marker.
    """
    if not isinstance(record, ChunkRecord):
        raise TypeError("native capture requires a ChunkRecord")
    unit = Unit(id=record.chunk_id, text=record.text, order=record.chunk_index,
                doc_id=record.doc_id)
    _validate_unit(unit)
    return unit


def build_native_request(
    query: str,
    current_pack: Sequence[PackedEvidenceItem],
    candidate: RetrievalCandidate,
    native_units: Mapping[str, Unit],
    *,
    text_policy: TextPolicy,
    arm: str,
    endpoint_id: str,
    model_id: str,
    expected_response_model: str,
    token_counter: Callable[[str], int],
    relations: tuple[SourceRelation, ...] = (),
    max_tokens: int = 1024,
    max_payload_bytes: int = 65536,
    prompt_version: str = PROMPT_VERSION,
    schema_version: str = SCHEMA_VERSION,
    question_version: str = QUESTION_VERSION,
    renderer_version: str = RENDERER_VERSION,
    counter_version: str = COUNTER_VERSION,
) -> NativeRequest:
    """Resolve only selected identities and build a complete native request.

    ``exact`` rejects any selected display/native text mismatch. ``reconstruct``
    explicitly rebuilds S and c from whole native units; it reports changed IDs
    in lexicographic order. The policy is bound in the internal state version,
    outside model-visible state, even when no text changes. Both policies check
    all selected identities before core arm projection, including standalone.

    Packed order, estimated tokens, paths, scores and expansion roles are never
    read. Relations must be explicitly supplied; core validates their visible
    endpoints/anchors. If reconstruction changes either relation endpoint's
    text, reject the relation instead of reusing potentially stale offsets;
    callers must supply newly native-bound anchors in a separate explicit step.
    The injected counter measures the final core rendering.
    No lookup iteration occurs, so unrelated native records are neither loaded
    nor validated. Core rejects selected duplicate text/source positions.
    """
    if text_policy not in ("exact", "reconstruct") or not isinstance(text_policy, str):
        raise ValueError("text_policy must explicitly be exact or reconstruct")
    if not isinstance(current_pack, Sequence) or isinstance(current_pack, (str, bytes, bytearray)):
        raise TypeError("current_pack must be a sequence of PackedEvidenceItem records")
    if not isinstance(candidate, RetrievalCandidate):
        raise TypeError("candidate must be a RetrievalCandidate")
    if not isinstance(native_units, Mapping):
        raise TypeError("native_units must be a mapping of IDs to captured Unit records")
    seen, changed = set(), set()

    def resolve(item: PackedEvidenceItem | RetrievalCandidate) -> Unit:
        chunk_id = _identity(item.chunk_id, "selected chunk ID")
        doc_id = _identity(item.doc_id, "selected document ID")
        if chunk_id in seen:
            raise ValueError("selected chunk IDs must be unique across current pack and candidate")
        seen.add(chunk_id)
        if not isinstance(item.text, str):
            raise TypeError("selected display text must be a string")
        try:
            unit = native_units[chunk_id]
        except KeyError as error:
            raise ValueError("selected chunk is missing from the native snapshot mapping") from error
        _validate_unit(unit)
        if unit.id != chunk_id or unit.doc_id != doc_id:
            raise ValueError("native lookup key or document identity does not match selected chunk")
        if item.text != unit.text:
            if text_policy == "exact":
                raise ValueError("display text differs from native text under exact policy")
            changed.add(chunk_id)
        return unit

    current = []
    for item in current_pack:
        if not isinstance(item, PackedEvidenceItem):
            raise TypeError("current_pack must contain PackedEvidenceItem records")
        current.append(resolve(item))
    native_candidate = resolve(candidate)
    if type(relations) is not tuple:
        raise TypeError("relations must be an immutable tuple")
    for relation in relations:
        if not isinstance(relation, SourceRelation):
            raise TypeError("relations must contain SourceRelation records")
        dependent = _identity(relation.dependent_id, "relation dependent ID")
        prerequisite = _identity(relation.prerequisite_id, "relation prerequisite ID")
        if dependent in changed or prerequisite in changed:
            raise ValueError("changed-text relations require native-bound re-anchoring")
    state = ConditionalState(query, tuple(current), native_candidate, relations,
                             version=f"{BRIDGE_STATE_VERSION}:{text_policy}")
    request = build_request(
        state, arm=arm, endpoint_id=endpoint_id, model_id=model_id,
        expected_response_model=expected_response_model, token_counter=token_counter,
        max_tokens=max_tokens, max_payload_bytes=max_payload_bytes,
        prompt_version=prompt_version, schema_version=schema_version,
        question_version=question_version, renderer_version=renderer_version,
        counter_version=counter_version,
    )
    return NativeRequest(request, tuple(sorted(changed)), text_policy)
