from slac_refiner.atomize.normalize import normalize_text
from slac_refiner.atomize.splitter import split_text_to_atoms
from slac_refiner.atomize.mapping import build_atoms_b0_from_units


def mapped(texts):
    return build_atoms_b0_from_units(
        normalized_units=[{"unit_id": i, "text": normalize_text(text)} for i, text in enumerate(texts)],
        splitter_fn=split_text_to_atoms,
    )


def test_normal_units_keep_original_boundary():
    result = mapped(["第一段。第二句。", "Third unit. Another sentence."])
    assert len(result["atoms"]) == 4
    assert [(x.start_atom, x.end_atom) for x in result["unit2atom_span"]] == [(0, 2), (2, 4)]
    assert result["b0"] == [0, 1, 0]


def test_empty_middle_unit_preserves_coverage_and_records_projection_fix():
    result = mapped(["第一段。第二句。", "   ", "Third unit. Another sentence."])
    assert len(result["atoms"]) == 4
    assert [(x.start_atom, x.end_atom) for x in result["unit2atom_span"]] == [(0, 2), (2, 2), (2, 4)]
    assert result["b0"] == [0, 0, 0]
    assert result["meta"]["projection_fix"] == 2


def test_all_empty_units_have_no_atoms_or_gaps():
    result = mapped(["   ", "\n\n"])
    assert result["atoms"] == result["b0"] == []
    assert result["meta"]["all_units_empty"] is True
    assert [(x.start_atom, x.end_atom) for x in result["unit2atom_span"]] == [(0, 0), (0, 0)]
