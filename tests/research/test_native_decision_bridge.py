"""Native bridge verification using invented records only; no model execution."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
import json
import socket

import pytest

from SLAC.retrieval.decision.conditional import SourceRelation, Unit, execute_offline
from SLAC.retrieval.decision.native_bridge import (
    BRIDGE_STATE_VERSION, build_native_request, capture_native_chunk,
)
from SLAC.retrieval.preprocess.anchor_fields import enrich_chunk_record
from SLAC.retrieval.schemas.records import ChunkRecord, PackedEvidenceItem, RetrievalCandidate


def chunk(chunk_id="a", text="Original alpha fact.", order=0, doc_id="doc"):
    return ChunkRecord(doc_id, chunk_id, order, order, order + 1, text, 1, [], 0)


def packed(record, **changes):
    result = PackedEvidenceItem(999, record.chunk_id, record.doc_id, "direct", "chunk_direct",
                                [], 1, record.text)
    return replace(result, **changes)


def candidate(record, **changes):
    return replace(RetrievalCandidate(record.chunk_id, record.doc_id, record.text, [], 0), **changes)


def config(**changes):
    return {"arm": "plain_conditional", "endpoint_id": "synthetic-only",
            "model_id": "synthetic-choice", "expected_response_model": "synthetic-choice-fixed",
            "token_counter": len, "max_tokens": 10000, **changes}


def inputs():
    a, b, c = chunk(), chunk("b", "Original beta fact.", 1), chunk("c", "Original gamma fact.", 2)
    return [packed(b), packed(a)], candidate(c), {x.chunk_id: capture_native_chunk(x) for x in (a, b, c)}


def build(pack, proposed, native, *, policy="exact", **changes):
    return build_native_request("What fact is supplied?", pack, proposed, native,
                                text_policy=policy, **config(**changes))


class SelectedLookupOnly(Mapping):
    def __init__(self, entries):
        self.entries = entries
        self.lookups = []

    def __getitem__(self, key):
        self.lookups.append(key)
        return self.entries[key]

    def __iter__(self):
        raise AssertionError("bridge must never scan the native mapping")

    def __len__(self):
        raise AssertionError("bridge must never inspect the whole native mapping")


def test_capture_survives_actual_enrichment_and_later_record_mutation():
    original = "  Ａ\t\tvalid fact.\r\n\r\n\r\nNext line.  "
    record = chunk(text=original)
    snapshot = capture_native_chunk(record)
    enriched = enrich_chunk_record(record)
    assert enriched is record
    assert enriched.text != original
    record.doc_id, record.chunk_id, record.chunk_index = "changed", "changed", 50
    record.text = "Replacement summary."
    assert snapshot == Unit("a", original, 0, "doc")
    with pytest.raises(FrozenInstanceError):
        snapshot.text = "changed"


@pytest.mark.parametrize("field,value", [
    ("chunk_id", ""), ("chunk_id", 2), ("doc_id", " "), ("doc_id", None),
    ("text", ""), ("text", 3), ("chunk_index", True), ("chunk_index", 1.0), ("chunk_index", -1),
])
def test_capture_rejects_invalid_native_identity_text_and_order(field, value):
    record = chunk()
    setattr(record, field, value)
    with pytest.raises((TypeError, ValueError)):
        capture_native_chunk(record)


def test_capture_requires_record_and_does_not_certify_native_provenance():
    with pytest.raises(TypeError, match="ChunkRecord"):
        capture_native_chunk({"chunk_id": "a"})
    # There is no trustworthy marker: capturing an enriched record preserves
    # its supplied text, but cannot recover or certify the original bytes.
    enriched = enrich_chunk_record(chunk(text=" Ａ  fact. "))
    assert capture_native_chunk(enriched).text == enriched.text


def test_selected_only_lookup_uses_native_order_and_ignores_unselected_records():
    pack, proposed, native = inputs()
    native["unselected"] = object()  # Must never be accessed or validated.
    lookup = SelectedLookupOnly(native)
    result = build(pack, proposed, lookup)
    assert lookup.lookups == ["b", "a", "c"]
    state = result.request.payload()["state"]
    assert [item["id"] for item in state["current_pack"]] == ["a", "b"]
    assert [item["order"] for item in state["current_pack"]] == [0, 1]
    assert result.request.rendered_evidence.startswith("[doc/a]\nOriginal alpha fact.")
    assert result.changed_text_ids == ()
    reverse = build(list(reversed(pack)), proposed, native)
    assert reverse.request.cache_key == result.request.cache_key


def test_exact_rejects_enriched_text_and_reconstruct_restores_full_native_text():
    record = chunk("c", "  Ｃ\t fact.\r\nNext full paragraph.  ", 2)
    pack, _, native = inputs()
    native["c"] = capture_native_chunk(record)
    enriched = enrich_chunk_record(record)
    proposed = candidate(enriched)
    with pytest.raises(ValueError, match="exact policy"):
        build(pack, proposed, native)
    reconstructed = build(pack, proposed, native, policy="reconstruct")
    assert reconstructed.request.payload()["state"]["candidate"]["text"] == native["c"].text
    assert reconstructed.changed_text_ids == ("c",)
    assert reconstructed.text_policy == "reconstruct"


def test_changed_ids_stable_and_policy_binds_cache_outside_visible_state():
    pack, proposed, native = inputs()
    changed_pack = [replace(item, text="short") for item in pack]
    changed_candidate = replace(proposed, text="short")
    a = build(changed_pack, changed_candidate, native, policy="reconstruct")
    b = build(list(reversed(changed_pack)), changed_candidate, native, policy="reconstruct")
    assert a.changed_text_ids == b.changed_text_ids == ("a", "b", "c")
    assert a.request.cache_key == b.request.cache_key
    exact = build(pack, proposed, native)
    unchanged_reconstruct = build(pack, proposed, native, policy="reconstruct")
    assert exact.request.payload_bytes == unchanged_reconstruct.request.payload_bytes
    assert exact.request.cache_key != unchanged_reconstruct.request.cache_key
    binding = json.loads(a.request.binding_bytes)
    assert binding["state_version"] == f"{BRIDGE_STATE_VERSION}:reconstruct"
    assert "text_policy" not in a.request.payload()["state"]


@pytest.mark.parametrize("problem", ["missing", "wrong_key", "wrong_doc", "wrong_value_type"])
def test_selected_native_binding_is_strict(problem):
    pack, proposed, native = inputs()
    if problem == "missing":
        del native["a"]
    elif problem == "wrong_key":
        native["a"] = replace(native["a"], id="other")
    elif problem == "wrong_doc":
        native["a"] = replace(native["a"], doc_id="other-doc")
    else:
        native["a"] = chunk()
    with pytest.raises((TypeError, ValueError)):
        build(pack, proposed, native)


@pytest.mark.parametrize("problem", ["duplicate_pack_id", "candidate_in_pack", "duplicate_source_order", "duplicate_native_text"])
def test_selected_duplicates_fail_closed(problem):
    pack, proposed, native = inputs()
    if problem == "duplicate_pack_id":
        pack.append(pack[0])
    elif problem == "candidate_in_pack":
        proposed = candidate(chunk("a"))
    elif problem == "duplicate_source_order":
        native["b"] = replace(native["b"], order=0)
    else:
        native["b"] = replace(native["b"], text=native["a"].text)
        pack[0] = replace(pack[0], text=native["b"].text)
    with pytest.raises(ValueError):
        build(pack, proposed, native)


def test_same_order_across_distinct_documents_is_allowed_but_identity_swap_rejected():
    pack, proposed, native = inputs()
    native["b"] = replace(native["b"], doc_id="doc-b", order=0)
    pack[0] = replace(pack[0], doc_id="doc-b")
    assert build(pack, proposed, native)
    pack[0] = replace(pack[0], doc_id="doc")
    with pytest.raises(ValueError, match="document identity"):
        build(pack, proposed, native)


def test_metadata_tampering_cannot_enter_model_state_or_cache_identity():
    pack, proposed, native = inputs()
    expected = build(pack, proposed, native)
    tampered_pack = [replace(item, order=-50, role="HIDDEN_ROLE", hit_type="HIDDEN_HIT",
                             path=["HIDDEN_PATH"], token_est=-100, retrieve_rank_fused=-1,
                             source_views=["HIDDEN_VIEW"], expansion_from="HIDDEN_PARENT") for item in pack]
    tampered_candidate = replace(proposed, retrieve_score_raw={"HIDDEN_SCORE": 1.0},
                                 meta={"gold": "HIDDEN_GOLD"}, token_est=0,
                                 anchor_text="HIDDEN_ANCHOR", path=["HIDDEN_PATH"])
    actual = build(tampered_pack, tampered_candidate, native)
    assert actual.request.payload_bytes == expected.request.payload_bytes
    assert actual.request.cache_key == expected.request.cache_key
    assert b"HIDDEN" not in actual.request.transport_bytes()


def test_reconstruction_recounts_full_evidence_and_refuses_truncation():
    pack, proposed, native = inputs()
    native["c"] = replace(native["c"], text="Full native evidence. " * 30)
    proposed = replace(proposed, text="short", token_est=1)
    reconstructed = build(pack, proposed, native, policy="reconstruct")
    required = len(reconstructed.request.rendered_evidence)
    assert required > 500
    assert build(pack, proposed, native, policy="reconstruct", max_tokens=required)
    with pytest.raises(ValueError, match="max_tokens"):
        build(pack, proposed, native, policy="reconstruct", max_tokens=required - 1)
    assert reconstructed.request.payload()["state"]["candidate"]["text"] == native["c"].text


def test_relations_are_explicit_and_core_validates_missing_endpoints():
    pack, proposed, native = inputs()
    no_relations = build(pack, proposed, native, arm="relation_conditioned")
    assert no_relations.request.payload()["state"]["relations"] == []
    relation = SourceRelation("c", "a", "reference", 0, 8)
    explicit = build(pack, proposed, native, arm="relation_conditioned", relations=(relation,))
    assert explicit.request.payload()["state"]["relations"][0]["anchor"]["text"] == "Original"
    assert explicit.request.cache_key != no_relations.request.cache_key
    with pytest.raises(ValueError, match="endpoints"):
        build(pack, proposed, native, relations=(replace(relation, prerequisite_id="absent"),))


@pytest.mark.parametrize("changed_endpoint", ["dependent", "prerequisite"])
def test_reconstructed_relation_endpoints_require_native_reanchoring(changed_endpoint):
    # NFKC expands a ligature: display span [4:8] is 'term', whereas the same
    # in-bounds native offset points at 'rm f'. Core bounds alone cannot catch it.
    record = chunk("c", "ﬃ term follows.", 2)
    pack, _, native = inputs()
    native["c"] = capture_native_chunk(record)
    proposed = candidate(enrich_chunk_record(record))
    relation = SourceRelation("c", "a", "reference", 4, 8)
    assert proposed.text[4:8] == "term"
    assert native["c"].text[4:8] == "rm f"
    assert 8 <= len(native["c"].text)
    if changed_endpoint == "prerequisite":
        relation = SourceRelation("a", "c", "reference", 0, 8)
    with pytest.raises(ValueError, match="changed-text relations require native-bound re-anchoring"):
        build(pack, proposed, native, policy="reconstruct", arm="relation_conditioned", relations=(relation,))


def test_unrelated_reconstructed_unit_does_not_block_unchanged_relation():
    pack, proposed, native = inputs()
    proposed = replace(proposed, text="short")
    relation = SourceRelation("b", "a", "reference", 0, 8)
    assert build(pack, proposed, native, policy="reconstruct", arm="relation_conditioned", relations=(relation,))


def test_explicit_native_reanchoring_requires_consistent_display_identity():
    pack, proposed, native = inputs()
    native["c"] = replace(native["c"], text="ﬃ term follows.")
    proposed = replace(proposed, text=native["c"].text)
    relation = SourceRelation("c", "a", "reference", 2, 6)
    result = build(pack, proposed, native, relations=(relation,), arm="relation_conditioned")
    assert result.request.payload()["state"]["relations"][0]["anchor"]["text"] == "term"


def test_policy_is_required_and_identity_types_are_checked():
    pack, proposed, native = inputs()
    with pytest.raises(TypeError, match="text_policy"):
        build_native_request("q", pack, proposed, native, **config())
    with pytest.raises(ValueError, match="text_policy"):
        build(pack, proposed, native, policy="guess")
    with pytest.raises(TypeError, match="PackedEvidenceItem"):
        build([proposed], proposed, native)
    with pytest.raises(TypeError, match="RetrievalCandidate"):
        build(pack, pack[0], native)
    with pytest.raises(TypeError, match="display text"):
        build(pack, replace(proposed, text=None), native, policy="reconstruct")
    with pytest.raises(ValueError, match="chunk ID"):
        build(pack, replace(proposed, chunk_id=4), native)


def test_fake_only_end_to_end_and_no_implicit_transport(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no network is allowed")
    monkeypatch.setattr(socket, "socket", forbidden)
    result = build(*inputs())
    def fake(data):
        receipt = json.loads(data)
        return {"request_key": receipt["request_key"], "response": {
            "model": "synthetic-choice-fixed", "answers": {
                "conditional_added_information": {"type": "choice", "choice": "unknown"},
                "conflict": {"type": "choice", "choice": "no"},
            }}}
    decoded = execute_offline(result.request, transport=fake)
    assert decoded.envelope_valid
    assert {item.id: item.choice for item in decoded.decisions} == {
        "conditional_added_information": "unknown", "conflict": "no"}
    with pytest.raises(RuntimeError, match="transport"):
        execute_offline(result.request)
