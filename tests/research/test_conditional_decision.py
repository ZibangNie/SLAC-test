"""Synthetic adapter contracts. No live transport, secrets, or research data."""

import json
from dataclasses import replace

import pytest

from SLAC.retrieval.decision.conditional import (
    CachedResponse, ConditionalState, SourceRelation, Unit, build_request,
    decode_cached, decode_response, execute_offline, freeze_response, render_evidence,
)


def state():
    base = Unit("u001", "The lamp uses the reserve cell.", 0, "d001")
    candidate = Unit("u003", "The reserve cell is blue.", 2, "d001")
    relation = SourceRelation("u001", "u003", "reference", 18, 30)
    return ConditionalState("What color is the cell used by the lamp?", (base,), candidate, (relation,))


def request(value=None, **changes):
    options = dict(arm="relation_conditioned", endpoint_id="offline:decisions",
                   model_id="synthetic-request", expected_response_model="synthetic-fixed",
                   token_counter=len, max_tokens=10000)
    options.update(changes)
    return build_request(value or state(), **options)


def envelope(req, answers=None):
    return {"request_key": req.cache_key, "response": {
        "model": "synthetic-fixed",
        "answers": answers if answers is not None else {
            "conditional_added_information": {"type": "choice", "choice": "yes"},
            "conflict": {"type": "choice", "choice": "no"},
        },
    }}


def by_id(decoded):
    return {item.id: item for item in decoded.decisions}


def test_arms_expose_only_their_inputs_and_conditioning_changes_only_conditional_cache():
    original = state()
    changed = replace(original, current_pack=(replace(original.current_pack[0], text="The lamp has a blue reserve cell."),),
                      relations=())
    standalone = request(original, arm="standalone")
    assert set(standalone.payload()["state"]) == {"query", "candidate"}
    assert standalone.cache_key == request(changed, arm="standalone").cache_key
    plain = request(original, arm="plain_conditional")
    assert set(plain.payload()["state"]) == {"query", "current_pack", "candidate"}
    assert plain.cache_key != request(changed, arm="plain_conditional").cache_key
    assert len({standalone.cache_key, plain.cache_key, request(original).cache_key}) == 3
    assert standalone.expected_ids == ("standalone_support",)
    assert plain.expected_ids == ("conditional_added_information", "conflict")


def test_relations_change_binding_without_changing_visible_text_or_evidence_budget():
    original = state()
    alternative = replace(original, relations=(replace(original.relations[0], kind="qualifier"),))
    real, control = request(original), request(alternative)
    assert real.cache_key != control.cache_key
    assert real.rendered_evidence == control.rendered_evidence
    assert real.evidence_tokens == control.evidence_tokens
    left, right = real.payload()["state"], control.payload()["state"]
    assert {k: v for k, v in left.items() if k != "relations"} == {k: v for k, v in right.items() if k != "relations"}
    assert request(original, arm="plain_conditional").cache_key == request(alternative, arm="plain_conditional").cache_key
    anchor = left["relations"][0]["anchor"]
    assert anchor["text"] == original.current_pack[0].text[anchor["start"]:anchor["end"]]
    assert not {"truth", "placebo", "expected", "supervision"} & set(left)


@pytest.mark.parametrize("field", ["endpoint_id", "model_id", "expected_response_model", "prompt_version",
                                  "schema_version", "question_version", "renderer_version", "counter_version"])
def test_model_endpoint_and_all_contract_versions_bind_cache(field):
    assert request().cache_key != request(**{field: "different-binding-v2"}).cache_key


def test_query_unit_content_source_identity_order_and_state_version_bind_cache():
    original = state()
    changes = [replace(original, query="A different query?"),
               replace(original, candidate=replace(original.candidate, text="The reserve cell is green.")),
               replace(original, candidate=replace(original.candidate, order=3)),
               replace(original, candidate=replace(original.candidate, doc_id="d002")),
               replace(original, version="different-state-v2")]
    assert all(request(changed).cache_key != request(original).cache_key for changed in changes)


def test_source_rendering_preserves_complete_text_and_exact_budget_boundary():
    original = state()
    earlier = replace(original.candidate, order=0)
    later = replace(original.current_pack[0], order=2)
    rearranged = replace(original, current_pack=(later,), candidate=earlier)
    expected = render_evidence((later, earlier))
    calls = []

    def counter(rendered):
        calls.append(rendered)
        return len(rendered)

    result = request(rearranged, token_counter=counter, max_tokens=len(expected))
    assert calls == [expected]
    assert expected.index(earlier.text) < expected.index(later.text)
    assert result.evidence_tokens == len(expected)
    with pytest.raises(ValueError, match="never truncated"):
        request(rearranged, token_counter=counter, max_tokens=len(expected) - 1)
    candidate_only = render_evidence((original.candidate,))
    assert request(original, arm="standalone").evidence_tokens == len(candidate_only)


@pytest.mark.parametrize("bad", [True, False, 1.0, -1, None])
def test_counter_values_are_strict_and_payload_has_separate_byte_cap(bad):
    with pytest.raises((ValueError, TypeError), match="token count"):
        request(token_counter=lambda _: bad)
    with pytest.raises(ValueError, match="payload"):
        request(max_payload_bytes=10, token_counter=lambda _: pytest.fail("byte cap must reject first"))


@pytest.mark.parametrize("mutation", ["self", "invisible", "start_negative", "empty_anchor", "past_end", "duplicate", "kind"])
def test_relation_endpoints_and_source_spans_are_checked(mutation):
    original = state()
    relation = original.relations[0]
    changes = {
        "self": dict(prerequisite_id=relation.dependent_id),
        "invisible": dict(prerequisite_id="absent"),
        "start_negative": dict(anchor_start=-1),
        "empty_anchor": dict(anchor_end=relation.anchor_start),
        "past_end": dict(anchor_end=10000),
        "kind": dict(kind="ground_truth"),
    }
    relations = (relation, relation) if mutation == "duplicate" else (replace(relation, **changes[mutation]),)
    with pytest.raises(ValueError):
        request(replace(original, relations=relations))


def test_duplicate_ids_native_text_order_and_mutable_collections_are_rejected():
    original = state()
    for candidate in (replace(original.candidate, id=original.current_pack[0].id),
                      replace(original.candidate, text=original.current_pack[0].text),
                      replace(original.candidate, order=original.current_pack[0].order)):
        with pytest.raises(ValueError):
            request(replace(original, candidate=candidate))
    with pytest.raises(TypeError, match="immutable"):
        request(replace(original, current_pack=list(original.current_pack)))
    with pytest.raises(TypeError, match="immutable"):
        request(replace(original, relations=list(original.relations)))


def test_payload_copy_and_frozen_bytes_cannot_mutate_cache_identity():
    req = request()
    visible = req.payload()
    visible["state"]["candidate"]["text"] = "mutated"
    visible["questions"].clear()
    assert req.payload()["state"]["candidate"]["text"] == state().candidate.text
    assert req.payload()["questions"]
    with pytest.raises(ValueError, match="binding mismatch"):
        replace(req, payload_bytes=req.payload_bytes.replace(b"blue", b"pink")).payload()
    with pytest.raises(ValueError, match="binding mismatch"):
        replace(req, cache_key="legacy-standalone-key").payload()


@pytest.mark.parametrize("answer,reason", [(None, "invalid_answer_type"), ({}, "invalid_answer_type"),
    ({"type": "score", "score": 1}, "invalid_answer_type"),
    ({"type": "choice", "choice": True}, "invalid_choice"),
    ({"type": "choice", "choice": "maybe"}, "invalid_choice"),
    ({"type": "choice", "choice": "unknown"}, "model_unknown")])
def test_malformed_dimension_does_not_contaminate_valid_dimension(answer, reason):
    req = request()
    response = envelope(req)
    response["response"]["answers"]["conditional_added_information"] = answer
    parsed = decode_response(req, response)
    values = by_id(parsed)
    assert parsed.envelope_valid
    assert values["conditional_added_information"].choice == "unknown"
    assert values["conditional_added_information"].reason == reason
    assert values["conflict"].choice == "no" and values["conflict"].reason is None


def test_missing_dimension_is_unknown_optional_numbers_do_not_drive_choice_or_policy():
    req = request()
    parsed = decode_response(req, envelope(req, {"conflict": {
        "type": "choice", "choice": "yes", "confidence": 0.0,
        "probabilities": {"yes": 0.0, "no": 1.0, "unknown": 0.0}}}))
    values = by_id(parsed)
    assert values["conditional_added_information"].reason == "missing_answer"
    assert values["conflict"].choice == "yes"
    # The module returns signals; a conflict does not silently rewrite another decision.
    both = envelope(req)
    both["response"]["answers"]["conflict"]["choice"] = "yes"
    assert all(item.choice == "yes" for item in decode_response(req, both).decisions)


@pytest.mark.parametrize("mutation", ["model", "request", "extra_id", "answers_list", "missing_model"])
def test_model_cache_or_answer_scope_mismatch_invalidates_whole_response(mutation):
    req = request()
    response = envelope(req)
    if mutation == "model":
        response["response"]["model"] = "synthetic-request"
    elif mutation == "request":
        response["request_key"] = "another-request"
    elif mutation == "extra_id":
        response["response"]["answers"]["unrequested"] = {"type": "choice", "choice": "yes"}
    elif mutation == "answers_list":
        response["response"]["answers"] = []
    else:
        del response["response"]["model"]
    decoded = decode_response(req, response)
    assert not decoded.envelope_valid and all(item.choice == "unknown" for item in decoded.decisions)
    with pytest.raises(ValueError, match="cannot enter"):
        freeze_response(req, response)


def test_duplicate_json_fields_and_nonstandard_json_are_rejected():
    req = request()
    valid = json.dumps(envelope(req))
    duplicate = valid.replace('"choice": "yes"', '"choice": "yes", "choice": "no"')
    assert not decode_response(req, duplicate).envelope_valid
    nonstandard = valid.replace('"choice": "yes"', '"choice": NaN')
    assert not decode_response(req, nonstandard).envelope_valid
    assert not decode_response(req, b"[]").envelope_valid
    assert not decode_response(req, valid, max_response_bytes=1).envelope_valid


def test_cache_replay_binds_arm_current_pack_model_and_internal_envelope():
    req = request()
    cached = freeze_response(req, envelope(req))
    assert decode_cached(req, cached) == decode_response(req, envelope(req))
    for alternative in (request(arm="standalone"), request(arm="plain_conditional"),
                        request(expected_response_model="another-model")):
        assert not decode_cached(alternative, cached).envelope_valid
        # Rewriting only the outer cache key cannot relabel an old response.
        forged = CachedResponse(alternative.cache_key, cached.response_bytes)
        assert not decode_cached(alternative, forged).envelope_valid


def test_no_default_execution_and_explicit_fake_transport_does_not_read_supervision():
    req = request()
    with pytest.raises(RuntimeError, match="disabled"):
        execute_offline(req)
    calls = []

    def fake(wire):
        calls.append(wire)
        wrapped = json.loads(wire)
        assert set(wrapped) == {"request_key", "payload"}
        # Fixed fake values, independent of fixture semantics and expected labels.
        return {"request_key": wrapped["request_key"], "response": {
            "model": "synthetic-fixed", "answers": {
                name: {"type": "choice", "choice": "unknown"}
                for name in wrapped["payload"]["questions"]}}}

    result = execute_offline(req, transport=fake)
    assert len(calls) == 1
    assert all(item.choice == "unknown" and item.reason == "model_unknown" for item in result.decisions)
