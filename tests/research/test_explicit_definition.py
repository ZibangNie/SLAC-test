"""Authored-only v1 acronym rule tests; no API, credentials or dataset reads."""

from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from SLAC.retrieval.decision.conditional import SourceRelation, Unit
from SLAC.retrieval.decision.explicit_definition import extract_explicit_definitions


def sample(text="We use natural language processing (NLP)."):
    return [Unit("definition", text, 2, "paper"),
            Unit("mention", "The NLP system improves.", 5, "paper")]


def test_native_offsets_source_order_and_immutable_projection():
    units = sample("Préface. We use natural-language processing (NLP).")
    result = extract_explicit_definitions(units[::-1])
    assert len(result.definitions) == len(result.links) == 1
    definition, link = result.definitions[0], result.links[0]
    assert definition.anchor.text == "natural-language processing (NLP)"
    assert units[0].text[definition.anchor.start:definition.anchor.end] == definition.anchor.text
    assert link.mention.text == "NLP"
    assert link.verify({u.id: u for u in units}) == SourceRelation("mention", "definition", "definition", 4, 7)
    with pytest.raises(FrozenInstanceError):
        definition.acronym = "ABC"


@pytest.mark.parametrize("text", [
    "natural language processing (NLP)", "new lattice prediction (NLP)",
])
def test_repeated_definition_is_ambiguous_even_identical(text):
    units = sample() + [Unit("repeat", text, 3, "paper")]
    result = extract_explicit_definitions(units)
    assert len(result.definitions) == 2
    assert len(result.ambiguities) == 1
    assert len(result.ambiguities[0].occurrences) == 2
    assert not result.links


def test_repetition_in_same_unit_is_ambiguous():
    result = extract_explicit_definitions(sample("natural language processing (NLP); natural language processing (NLP)"))
    assert len(result.ambiguities) == 1
    assert not result.links


@pytest.mark.parametrize("replacement", [
    {"text": "We use natural language processing (NLP)!"},
    {"doc_id": "other"}, {"order": 1}, {"id": "spoof"},
])
@pytest.mark.parametrize("endpoint", [0, 1])
def test_modified_native_endpoint_rejected(replacement, endpoint):
    units = sample()
    link = extract_explicit_definitions(units).links[0]
    mapping = {u.id: u for u in units}
    mapping[units[endpoint].id] = replace(units[endpoint], **replacement)
    with pytest.raises(ValueError):
        link.verify(mapping)


def test_missing_endpoint_and_spoofed_anchor_rejected():
    units = sample()
    link = extract_explicit_definitions(units).links[0]
    mapping = {u.id: u for u in units}
    with pytest.raises(KeyError):
        link.verify({units[0].id: units[0]})
    with pytest.raises(ValueError):
        replace(link, mention=replace(link.mention, start=3)).verify(mapping)
    with pytest.raises(ValueError):
        replace(link, definition=replace(link.definition, acronym="ABC")).verify(mapping)


def test_forged_mention_ending_inside_word_rejected():
    units = sample()
    link = extract_explicit_definitions(units).links[0]
    units[1] = replace(units[1], text=units[1].text.replace("NLP", "NLPS"))
    forged = replace(link, mention=replace(link.mention,
        text_sha256=hashlib.sha256(units[1].text.encode("utf-8")).hexdigest()))
    with pytest.raises(ValueError):
        forged.verify({u.id: u for u in units})


def test_case_ascii_boundaries_prior_unit_and_foreign_document():
    units = sample() + [Unit("prior", "NLP is earlier.", 0, "paper"),
                        Unit("foreign", "NLP is foreign.", 8, "other")]
    units[1] = replace(units[1], text="nlp NLPS _NLP NLP2 2NLP xNLP NLP NLP-NLP éNLP")
    result = extract_explicit_definitions(units)
    assert [link.mention.start for link in result.links] == [29, 33, 37, 42]


def test_no_same_unit_links_and_mentions_inside_other_definition():
    units = [Unit("a", "alpha beta (AB); AB here.", 0, "d"),
             Unit("b", "AB classifier (AC). AB here.", 1, "d")]
    result = extract_explicit_definitions(units)
    assert len(result.links) == 1
    assert result.links[0].mention.start == 20


@pytest.mark.parametrize("text", [
    "natural language\nprocessing (NLP)", "natural, language processing (NLP)",
    "natural language processing (nlp)", "natural language processing (ＮＬＰ)",
    "naïve language processing (NLP)", "énatural language processing (NLP)",
    "natural language processing—(NLP)", "natural language processing ( NL P )",
    "natural language processing (NP)", "natural language\u2028processing (NLP)",
])
def test_conservative_rule_rejects_nonmatching_native_text(text):
    assert not extract_explicit_definitions(sample(text)).definitions


def test_no_normalization_of_native_whitespace():
    units = sample("natural\t language\u00a0processing (NLP)")
    result = extract_explicit_definitions(units)
    assert result.definitions[0].anchor.text == units[0].text
    changed = {u.id: u for u in units}
    changed[units[0].id] = replace(units[0], text=units[0].text.replace("\u00a0", " "))
    with pytest.raises(ValueError):
        result.links[0].verify(changed)


@pytest.mark.parametrize("extra", [
    Unit("definition", "other text", 8, "paper"),
    Unit("other", "other text", 2, "paper"),
    Unit("other", "other text", True, "paper"),
    Unit("other", "other text", -1, "paper"),
    Unit("", "other text", 8, "paper"),
])
def test_invalid_id_order_or_duplicate_rejected(extra):
    with pytest.raises(ValueError):
        extract_explicit_definitions(sample() + [extra])
