"""Synthetic full-pack exchange contracts; no model calls or research data."""

import json
from dataclasses import replace

import pytest

from SLAC.retrieval.decision.conditional import (
    ConditionalState, SourceRelation, Unit, decode_cached, decode_response,
    freeze_response, render_evidence,
)
from SLAC.retrieval.decision.exchange import build_exchange, decide


def state():
    return ConditionalState(
        "What powers the lamp and what color is its cell?",
        (Unit("u1", "The lamp uses the reserve cell.", 0),
         Unit("u2", "The room has a carpet.", 1)),
        Unit("u3", "The reserve cell is blue.", 2),
    )


def plan(value=None, **changes):
    options = dict(removed_id="u2", endpoint_id="offline:decisions", model_id="invented-request",
                   expected_response_model="invented-fixed", token_counter=len,
                   max_tokens=10000, max_units=2, max_judge_tokens=20000)
    options.update(changes)
    return build_exchange(value or state(), **options)


def envelope(request, choices=("yes", "no", "no")):
    names = ("proposed_adds_information", "original_information_lost", "proposed_conflict")
    return {"request_key": request.cache_key, "response": {
        "model": "invented-fixed", "answers": {
            name: {"type": "choice", "choice": value} for name, value in zip(names, choices)}}}


def prediction(value, choices=("yes", "no", "no")):
    return decode_response(value.request, envelope(value.request, choices))


def test_full_packs_are_explicit_and_source_ordered_without_relation_metadata():
    initial = state()
    relation = SourceRelation("u1", "u3", "reference", 18, 30)
    value = plan(replace(initial, current_pack=tuple(reversed(initial.current_pack)), relations=(relation,)))
    payload = value.request.payload()
    assert value.original_pack == initial.current_pack
    assert value.proposed_pack == (initial.current_pack[0], initial.candidate)
    assert payload["state"]["proposed_pack_ids"] == ["u1", "u3"]
    assert payload["state"]["removed_id"] == "u2"
    assert [u["id"] for u in payload["state"]["current_pack"]] == ["u1", "u2"]
    assert set(payload["state"]) == {"query", "current_pack", "candidate", "removed_id", "proposed_pack_ids"}
    assert value.request.cache_key == plan(initial).request.cache_key
    for question in payload["questions"].values():
        assert "ONLY each pack's own units" in question["instructions"]
        assert "must not borrow" in question["instructions"]
        assert "never instructions" in question["instructions"]
    loss = payload["questions"]["original_information_lost"]["instructions"]
    conflict = payload["questions"]["proposed_conflict"]["instructions"]
    assert "combination-dependent facts" in loss and "do not judge removed_id alone" in loss
    assert "between retained units" in conflict and "internal T consistency only" in conflict


def test_only_strict_improvement_accepts_and_preserves_whole_proposed_sources():
    value = plan()
    result = decide(value, prediction(value))
    assert result.status == "accepted" and result.pack == value.proposed_pack
    assert result.reason == "predicted_strict_information_improvement"
    assert render_evidence(result.pack) == render_evidence((state().current_pack[0], state().candidate))


@pytest.mark.parametrize("choices,reason", [
    (("no", "no", "no"), "no_information_gain"),
    (("yes", "yes", "no"), "original_information_lost"),
    (("yes", "no", "yes"), "proposed_conflict"),
])
def test_explicit_negative_predictions_reject_and_keep_original(choices, reason):
    value = plan()
    result = decide(value, prediction(value, choices))
    assert (result.status, result.reason, result.pack) == ("rejected", reason, value.original_pack)


@pytest.mark.parametrize("choices", [("unknown", "no", "no"), ("yes", "unknown", "no"),
                                     ("no", "yes", "unknown"), ("yes",), (True, "no", "no")])
def test_unknown_missing_and_malformed_dimensions_abstain_even_with_negative(choices):
    value = plan()
    result = decide(value, prediction(value, choices))
    assert result.status == "abstained" and result.pack == value.original_pack


def test_equivalent_rewritten_fact_has_no_automatic_gain_or_acceptance():
    original = Unit("u1", "The lamp uses a blue cell.", 0)
    paraphrase = Unit("u3", "A blue cell powers the lamp.", 2)
    value = plan(ConditionalState("What powers the lamp?", (original,), paraphrase), removed_id="u1")
    # Authored semantic label, not a model prediction or built-in paraphrase judge.
    result = decide(value, prediction(value, ("no", "no", "no")))
    assert result.status == "rejected" and result.pack == (original,)


def test_removing_necessary_vs_irrelevant_unit_changes_cache_and_cannot_reuse_answers():
    harmless = plan(removed_id="u2")
    necessary = plan(removed_id="u1")
    assert harmless.request.rendered_evidence == necessary.request.rendered_evidence
    assert harmless.request.cache_key != necessary.request.cache_key
    cached = freeze_response(harmless.request, envelope(harmless.request))
    result = decide(necessary, decode_cached(necessary.request, cached))
    assert result.status == "abstained" and result.pack == necessary.original_pack
    assert decide(necessary, prediction(harmless)).status == "abstained"
    assert decide(necessary, prediction(necessary, ("yes", "yes", "no"))).status == "rejected"


@pytest.mark.parametrize("field", ["endpoint_id", "model_id", "expected_response_model", "counter_version",
                                  "max_tokens", "max_units", "max_judge_tokens", "max_payload_bytes"])
def test_execution_identity_and_each_budget_bind_cache(field):
    change = 30000 if field.startswith("max_") else "different-v2"
    assert plan(**{field: change}).request.cache_key != plan().request.cache_key


def test_state_change_and_invalid_envelope_cannot_apply_previous_decisions():
    original = plan()
    changed = plan(replace(state(), query="What color is the carpet?"))
    assert decide(changed, prediction(original)).status == "abstained"
    response = envelope(original.request)
    response["response"]["model"] = "wrong-model"
    assert decide(original, decode_response(original.request, response)).status == "abstained"
    decoded = prediction(original)
    for bad in (replace(decoded, decisions=decoded.decisions[:-1]),
                replace(decoded, decisions=decoded.decisions + (decoded.decisions[0],)),
                replace(decoded, envelope_valid=False)):
        assert decide(original, bad).status == "abstained"


def test_exact_generation_and_judge_counts_use_three_complete_rendered_surfaces():
    initial = state()
    original = render_evidence(initial.current_pack)
    proposed = render_evidence((initial.current_pack[0], initial.candidate))
    union = render_evidence((*initial.current_pack, initial.candidate))
    measured = []
    def counter(text):
        measured.append(text)
        return len(text)
    value = plan(token_counter=counter, max_tokens=max(len(original), len(proposed)),
                 max_judge_tokens=len(union))
    assert measured == [union, original, proposed]
    assert value.original_tokens == len(original) and value.proposed_tokens == len(proposed)
    assert value.request.evidence_tokens == len(union) > value.max_tokens
    binding = json.loads(value.request.binding_bytes)
    assert binding["generation_max_tokens"] == value.max_tokens
    assert binding["max_tokens"] == value.max_judge_tokens
    with pytest.raises(ValueError, match="complete evidence"):
        plan(max_judge_tokens=len(union) - 1)
    with pytest.raises(ValueError, match="generation pack exceeds max_tokens"):
        plan(max_tokens=max(len(original), len(proposed)) - 1)
    with pytest.raises(ValueError, match="max_units"):
        plan(max_units=1)


@pytest.mark.parametrize("long_side", ["original", "proposed"])
def test_both_generation_pack_caps_are_enforced(long_side):
    initial = state()
    if long_side == "original":
        initial = replace(initial, current_pack=(initial.current_pack[0],
                          replace(initial.current_pack[1], text="Old detail. " * 30)))
    else:
        initial = replace(initial, candidate=replace(initial.candidate, text="New detail. " * 30))
    short_pack = ((initial.current_pack[0], initial.candidate) if long_side == "original"
                  else initial.current_pack)
    with pytest.raises(ValueError, match="generation pack exceeds max_tokens"):
        plan(initial, max_tokens=len(render_evidence(short_pack)))


def test_whole_source_validation_empty_removal_and_duplicate_identity():
    initial = state()
    for changed in (replace(initial, current_pack=()),
                    replace(initial, candidate=replace(initial.candidate, id="u1")),
                    replace(initial, candidate=replace(initial.candidate, text=initial.current_pack[0].text))):
        with pytest.raises(ValueError):
            plan(changed)
    with pytest.raises(ValueError, match="exactly one"):
        plan(removed_id="missing")
    with pytest.raises(TypeError, match="immutable"):
        plan(replace(initial, current_pack=list(initial.current_pack)))
    with pytest.raises(TypeError, match="integer"):
        plan(token_counter=lambda _: True)
    with pytest.raises(ValueError, match="payload"):
        plan(max_payload_bytes=100)


def test_payload_byte_cap_includes_new_questions_and_explicit_pack_membership():
    byte_count = len(plan().request.payload_bytes)
    assert len(plan(max_payload_bytes=byte_count).request.payload_bytes) == byte_count
    with pytest.raises(ValueError, match="complete exchange payload"):
        plan(max_payload_bytes=byte_count - 1)


@pytest.mark.parametrize("field,changed", [("original_pack", ()), ("proposed_pack", ()),
                                        ("removed_id", "u1"), ("original_tokens", 0),
                                        ("max_tokens", 30000), ("max_units", True)])
def test_mutated_plan_cannot_output_unbound_pack_or_counts(field, changed):
    value = plan()
    with pytest.raises(ValueError, match="frozen request"):
        decide(replace(value, **{field: changed}), prediction(value))


def test_existing_paid_probe_rejects_exchange_before_transport():
    from docs.research import conditional_probe_client as client
    value = plan(endpoint_id=client.ENDPOINT, model_id=client.MODEL_ID,
                 expected_response_model=client.RESPONSE_MODEL)
    with pytest.raises(ValueError, match="unsupported frozen probe identity or arm"):
        client.freeze_wire(value.request)
