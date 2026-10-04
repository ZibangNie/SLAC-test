"""Synthetic-only provenance checks for model text and exact source partitions."""
import pytest

from SLAC.refiner.pipeline.assemble import build_refiner_input as builder
from SLAC.refiner.pipeline.assemble.source_atomizer import (
    SourceTraceError, trace_atomize_unit_text,
)
from SLAC.refiner.slac_refiner.decoding.projector import (
    ProjectorConfig, rebuild_chunks_from_boundary_vector,
)


def config(**changes):
    values = dict(atom_min_chars=0, atom_min_tokens=0,
                  atom_max_chars=100, atom_max_tokens=100)
    values.update(changes)
    return builder.RefinerInputBuildConfig(**values)


def assert_trace_contract(source, trace, cfg):
    assert trace.source_text == source
    assert trace.model_atoms == builder.atomize_unit_text(source, cfg)
    assert trace.model_atoms == [a["model_text"] for a in trace.atoms]
    assert isinstance(trace.operations, list)
    assert all(isinstance(op, dict) for op in trace.operations)
    if not trace.atoms:
        assert trace.trivia_only and not source.strip()
        return
    assert trace.trivia_only is False
    assert "".join(a["source_text"] for a in trace.atoms) == source
    boundary = previous_origin_end = 0
    covered = set()
    for atom in trace.atoms:
        start, end = atom["source_span"]
        assert start == boundary and start < end <= len(source)
        assert atom["source_text"] == source[start:end]
        assert len(atom["model_char_origins"]) == len(atom["model_text"])
        for char, origin in zip(atom["model_text"], atom["model_char_origins"], strict=True):
            if origin is None:
                assert char == " "
                continue
            left, right = origin
            assert type(left) is type(right) is int
            assert start <= left < right <= end
            assert previous_origin_end <= left
            covered.update(range(left, right))
            previous_origin_end = right
        boundary = end
    assert boundary == len(source)
    assert {i for i, char in enumerate(source) if not char.isspace()} <= covered


@pytest.mark.parametrize("source,cfg,expected", [
    ('（“a”—【b】）', config(), ['("a"-[b])']),
    ("\t alpha\u00a0 \tbeta\r\n\r\ngamma  \n", config(), ["alpha beta", "gamma"]),
    ("🙂e\u0301汉。Ａ", config(), ["🙂e\u0301汉。", "Ａ"]),
    ("same. same.", config(), ["same.", "same."]),
    ("a.\r\nb.", None, ["a. b."]),
    ("甲。\n乙。", None, ["甲。乙。"]),
    ("abcdefghij.\t x.", config(atom_min_chars=5), ["abcdefghij. x."]),
    ("abcdefgh", config(atom_max_chars=4), ["abcd", "efgh"]),
])
def test_dual_views_keep_production_model_text_and_exact_source(source, cfg, expected):
    trace = trace_atomize_unit_text(source, cfg)
    assert trace.model_atoms == expected
    assert_trace_contract(source, trace, cfg)


def test_replaced_punctuation_keeps_original_character_origins():
    source = '（“a”—【b】）'
    trace = trace_atomize_unit_text(source, config())
    atom = trace.atoms[0]
    assert atom["model_text"] == '("a"-[b])'
    assert atom["model_char_origins"] == [(i, i + 1) for i in range(9)]
    assert atom["source_text"] == source and atom["source_text"] != atom["model_text"]


def test_collapsed_whitespace_and_crlf_retain_full_origin_intervals():
    source = "\t alpha\u00a0 \tbeta\r\n\r\ngamma  \n"
    trace = trace_atomize_unit_text(source, config())
    assert trace.atoms[0]["model_char_origins"][5] == (7, 10)
    assert source[7:10] == "\u00a0 \t"
    assert [a["source_span"] for a in trace.atoms] == [(0, 18), (18, 26)]
    # This newline survives sentence splitting, so its final model origin is visible.
    trace = trace_atomize_unit_text("a\r\nb. c.", config())
    assert trace.model_atoms == ["a\nb.", "c."]
    assert trace.atoms[0]["model_char_origins"][1] == (1, 3)
    assert_trace_contract("a\r\nb. c.", trace, config())


def test_unicode_uses_character_offsets_without_unicode_normalization():
    source = "🙂e\u0301汉。Ａ"
    trace = trace_atomize_unit_text(source, config())
    assert len(source) == 6 and len(source.encode("utf-8")) > 6
    assert trace.model_atoms == [source[:5], source[5:]]
    assert trace.atoms[0]["model_char_origins"] == [(i, i + 1) for i in range(5)]
    assert trace.atoms[1]["model_char_origins"] == [(5, 6)]


def test_repeated_text_keeps_each_original_occurrence():
    trace = trace_atomize_unit_text("same. same.", config())
    assert trace.model_atoms == ["same.", "same."]
    assert trace.atoms[0]["model_char_origins"] == [(i, i + 1) for i in range(5)]
    assert trace.atoms[1]["model_char_origins"] == [(i, i + 1) for i in range(6, 11)]
    assert [a["source_span"] for a in trace.atoms] == [(0, 6), (6, 11)]


@pytest.mark.parametrize("source,cfg,expected_origins", [
    ("a.\r\nb.", None, [(0, 1), (1, 2), None, (4, 5), (5, 6)]),
    ("abcdefghij.\t x.", config(atom_min_chars=5),
     [*( (i, i + 1) for i in range(11)), None, (13, 14), (14, 15)]),
])
def test_short_merge_marks_only_new_space_as_originless(source, cfg, expected_origins):
    trace = trace_atomize_unit_text(source, cfg)
    assert len(trace.atoms) == 1
    assert trace.atoms[0]["model_char_origins"] == expected_origins
    assert trace.atoms[0]["source_text"] == source


def test_cjk_merge_removes_model_joinspace_but_retains_source_newline():
    trace = trace_atomize_unit_text("甲。\n乙。")
    assert trace.model_atoms == ["甲。乙。"]
    assert trace.atoms[0]["model_char_origins"] == [(0, 1), (1, 2), (3, 4), (4, 5)]
    assert trace.atoms[0]["source_text"] == "甲。\n乙。"


@pytest.mark.parametrize("source", ["", "\r\n \t"])
def test_empty_or_whitespace_source_keeps_trivia_without_fake_atom(source):
    trace = trace_atomize_unit_text(source, config())
    assert trace.source_text == source
    assert trace.model_atoms == [] and trace.atoms == []
    assert trace.trivia_only is True


def test_true_projector_requires_render_atoms_and_reconstructs_all_raw_characters():
    source = "\t（a）.\r\n\r\nsame. same.\t "
    trace = trace_atomize_unit_text(source, config())
    render_atoms = [atom["source_text"] for atom in trace.atoms]
    spans = [atom["source_span"] for atom in trace.atoms]
    boundaries = [0] * (len(trace.atoms) - 1)
    cfg = ProjectorConfig(max_chunk_atoms=1, min_chunk_atoms=0,
        max_chunk_chars=100, min_chunk_chars=0, max_chunk_tokens=100, min_chunk_tokens=0)
    result = rebuild_chunks_from_boundary_vector(render_atoms, boundaries, cfg,
        source_text=source, atom_char_spans=spans, token_counter=len, strict=True)
    assert result["text_mode"] == "source_spans"
    assert "".join(unit["text"] for unit in result["projected_units"]) == source
    assert len(result["projected_units"]) == len(trace.atoms)
    for unit in result["projected_units"]:
        assert unit["text"] == source[unit["start_char"]:unit["end_char"]]
    # Normalized model text must not be passed off as matching raw source slices.
    with pytest.raises(ValueError, match="exactly match"):
        rebuild_chunks_from_boundary_vector(trace.model_atoms, boundaries, cfg,
            source_text=source, atom_char_spans=spans, token_counter=len, strict=True)


def test_production_parity_guard_rejects_changed_model_atom_output(monkeypatch):
    monkeypatch.setattr(builder, "atomize_unit_text", lambda *args, **kwargs: ["different model atom"])
    with pytest.raises(SourceTraceError) as caught:
        trace_atomize_unit_text("alpha", config())
    assert caught.value.code == "builder_parity_mismatch"


@pytest.mark.parametrize("source", [None, 123, b"alpha"])
def test_invalid_source_fails_closed(source):
    with pytest.raises(SourceTraceError) as caught:
        trace_atomize_unit_text(source, config())
    assert (caught.value.code, caught.value.stage) == ("invalid_source", "input")


@pytest.mark.parametrize("cfg", ["not a config", config(atom_max_chars=0), config(atom_max_tokens=0),
                                  config(atom_min_chars=-1), config(atom_min_tokens=-1)])
def test_invalid_config_fails_before_processing(cfg):
    with pytest.raises(SourceTraceError) as caught:
        trace_atomize_unit_text("alpha", cfg)
    assert (caught.value.code, caught.value.stage) == ("invalid_config", "input")
