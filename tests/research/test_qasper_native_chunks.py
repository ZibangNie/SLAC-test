"""Raw-field identity and old-cache compatibility, without model or data access."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import build_qasper_native_chunks as adapter
from test_qasper_relation_replay import CharacterTokenizer


def fixture(doc="doc", source="source", *, paragraphs=None, offset=0):
    paper = {"title": " Title \t", "abstract": " Abstract\r\n text ", "full_text": [
        {"section_name": "  Section  ", "paragraphs": paragraphs or [" first \t paragraph\r\n", "same", "same", " \t\n", ""]},
        {"section_name": "", "paragraphs": ["another section"]}]}
    units, blocks, parts, cursor = [], [], [], 0
    for index, (kind, pointer, canonical_pointer, group, path, raw) in enumerate(adapter.native_fields(paper, source)):
        if not raw.strip():
            continue
        # Simulate the historical canonical representation; the adapter itself
        # must keep this cached input and the exact raw source independently.
        retrieval = raw.strip().replace("\r\n", "\n")
        if parts: parts.append("\n\n"); cursor += 2
        start, end = cursor, cursor + len(retrieval)
        parts.append(retrieval); cursor = end
        uid = f"b{index}"
        units.append(adapter.Unit(uid, len(units), kind, start, end, retrieval, raw))
        blocks.append({"block_id": uid, "kind": "heading" if kind == "title" else kind,
                       "source_locator": {"json_pointer": canonical_pointer}, "char_span": [start, end]})
    canonical = {"doc_id": doc, "source_id": source, "original_split": "validation",
                 "canonical_text": "".join(parts), "blocks": blocks}
    cache = {(doc, unit.unit_id): index + offset for index, unit in enumerate(units)}
    return canonical, paper, units, cache


def build(*, chunk_budget=512, **kwargs):
    canonical, paper, units, cache = fixture(**kwargs)
    result = adapter.build_document(canonical, paper, units, CharacterTokenizer(), cache, chunk_budget=chunk_budget)
    return result, units


def validate(result):
    doc, blocks, leaves, chunks, mappings = result
    adapter.validate_roundtrip([doc], blocks, leaves, chunks, mappings)


def test_raw_whitespace_empty_fields_and_duplicate_text_are_preserved_separately_from_cache():
    result, units = build()
    document, blocks, leaves, chunks, mappings = result
    validate(result)
    assert [leaf.text for leaf in leaves] == [unit.text for unit in units]
    assert len(leaves) == len(units) == 7
    assert len(blocks) == 10 and sum(not row["retrievable"] for row in blocks) == 3
    assert len([leaf for leaf in leaves if leaf.text == "same"]) == 2
    assert " first \t paragraph\r\n" in document["native_document_text"]
    assert " \t\n" in document["native_document_text"]
    assert document["native_document_text"] != document["canonical_text"]
    assert [leaf.meta["cache_embedding_row"] for leaf in leaves] == list(range(len(units)))
    assert any(not leaf.meta["raw_equals_retrieval_text"] for leaf in leaves)
    assert all(mapping["native_unit_id"] == units[index].unit_id for index, mapping in enumerate(mappings))


def test_native_field_pointer_scope_excludes_questions_figures_and_unrelated_fields():
    canonical, paper, _, _ = fixture()
    paper.update(qas=[{"question": "DO_NOT_EXPORT"}], figures_and_tables=[{"caption": "DO_NOT_EXPORT"}])
    fields = list(adapter.native_fields(paper, "source"))
    assert [field[1] for field in fields[:4]] == ["/source/title", "/source/abstract", "/source/full_text/0/section_name", "/source/full_text/0/paragraphs/0"]
    assert "DO_NOT_EXPORT" not in json.dumps(fields)


def test_chunks_never_cross_title_abstract_or_section_groups():
    result, _ = build()
    validate(result)
    assert [chunk.meta["section_group"] for chunk in result[3]] == ["title", "abstract", "section:0", "section:1"]
    assert result[3][2].num_atoms == 4
    assert result == build()[0]


def test_actual_complete_text_budget_is_respected_and_oversize_is_singleton():
    result, units = build(paragraphs=["short", "x" * 100, "tail"], chunk_budget=30)
    validate(result)
    oversize = [chunk for chunk in result[3] if chunk.meta["oversize_singleton"]]
    assert len(oversize) == 1 and oversize[0].text == "x" * 100 and oversize[0].num_atoms == 1
    assert oversize[0].token_est == 102
    assert all(chunk.token_est <= 30 or chunk.meta["oversize_singleton"] for chunk in result[3])
    assert all(chunk.token_est == len(chunk.text) + 2 for chunk in result[3])
    assert len(result[2]) == len(units)


def test_chunk_cut_happens_between_native_units_at_inclusive_limit():
    result, _ = build(paragraphs=["abcd", "efgh"], chunk_budget=12)
    validate(result)
    combined = [chunk for chunk in result[3] if chunk.text == "abcd\n\nefgh"]
    assert len(combined) == 1 and combined[0].token_est == 12
    lower, _ = build(paragraphs=["abcd", "efgh"], chunk_budget=11)
    assert not any(chunk.text == "abcd\n\nefgh" for chunk in lower[3])


def test_cross_document_ids_are_unique_even_when_native_unit_ids_repeat():
    first, units = build(doc="d1", source="s1")
    second, _ = build(doc="d2", source="s2", offset=len(units))
    combined = ([first[0], second[0]], first[1] + second[1], first[2] + second[2], first[3] + second[3], first[4] + second[4])
    adapter.validate_roundtrip(*combined)
    assert len({leaf.leaf_id for leaf in combined[2]}) == len(combined[2])
    assert len({chunk.chunk_id for chunk in combined[3]}) == len(combined[3])
    assert first[2][0].meta["native_unit_id"] == second[2][0].meta["native_unit_id"]
    assert first[2][0].leaf_id != second[2][0].leaf_id


@pytest.mark.parametrize("mutation", ["duplicate_unit", "overlap", "offset", "native_text", "cached_text", "missing_cache", "nonvalidation"])
def test_corrupted_identity_and_dual_representation_fail_closed(mutation):
    canonical, paper, units, cache = fixture()
    if mutation == "duplicate_unit": units[1] = replace(units[1], unit_id=units[0].unit_id)
    elif mutation == "overlap": units[1] = replace(units[1], start=units[0].start)
    elif mutation == "offset": units[1] = replace(units[1], end=units[1].end - 1)
    elif mutation == "native_text": units[1] = replace(units[1], native_text="altered")
    elif mutation == "cached_text": units[1] = replace(units[1], text="altered")
    elif mutation == "missing_cache": cache.pop(next(iter(cache)))
    else: canonical["original_split"] = "test"
    with pytest.raises(ValueError): adapter.build_document(canonical, paper, units, CharacterTokenizer(), cache)


@pytest.mark.parametrize("mutation", ["leaf_text", "raw_offset", "duplicate_block", "chunk_overlap", "mapping_cache", "duplicate_doc"])
def test_roundtrip_validator_rejects_lost_duplicated_or_mislocated_content(mutation):
    result, _ = build(chunk_budget=25)
    document, blocks, leaves, chunks, mappings = deepcopy(result)
    documents = [document]
    if mutation == "leaf_text": leaves[0].text += "changed"
    elif mutation == "raw_offset": blocks[0]["native_char_span"][0] += 1
    elif mutation == "duplicate_block": blocks.append(deepcopy(blocks[0]))
    elif mutation == "chunk_overlap": chunks[1].atom_start = 0
    elif mutation == "mapping_cache": mappings[0]["cache_embedding_row"] = 999
    else: documents.append(deepcopy(document))
    with pytest.raises(ValueError): adapter.validate_roundtrip(documents, blocks, leaves, chunks, mappings)


def test_offline_run_manifest_output_and_loading(tmp_path, monkeypatch):
    canonical, paper, units, cache = fixture()
    source = tmp_path / "frozen.txt"
    source.write_text("fixed", encoding="utf-8")
    dense = {"model_revision": "frozen-revision", "length_audit": {"candidate_units": {"p95": 272, "max": 787}}}
    monkeypatch.setattr(adapter, "load_sources", lambda *_: ({"doc": units}, {"doc": canonical}, {"doc": paper},
        {str(source): adapter.digest(source)}, dense, tmp_path, cache, tmp_path / "cached.safetensors"))
    monkeypatch.setattr(adapter.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: CharacterTokenizer())
    args = SimpleNamespace(pilot_prepared="unused", extended_prepared="unused", output=tmp_path / "out")
    manifest = adapter.run(args)
    loaded, documents, leaves, chunks = adapter.load_artifacts(args.output)
    assert len(leaves) == len(units) and loaded == manifest
    assert loaded["model_loaded"] is loaded["embedding_vectors_loaded"] is loaded["index_built"] is False
    assert loaded["api_calls"] == 0 and loaded["test_payload_read"] is False
    assert set(loaded["output_files_sha256"]) == set((*adapter.FILES, "public_summary.json"))
    assert (args.output / "public_summary.json").is_file()
    with pytest.raises(FileExistsError): adapter.run(args)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"): adapter.load_artifacts(args.output)
