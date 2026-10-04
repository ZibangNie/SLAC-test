"""Synthetic source-order regressions; no corpus, model or API access."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SLAC.refiner.pipeline.assemble.build_refiner_input import (
    RefinerInputBuildConfig,
    _split_long_text_once,
    _split_long_text_recursive,
    atomize_unit_text,
)


def without_whitespace(text):
    return "".join(char for char in text if not char.isspace())


@pytest.mark.parametrize("source,expected", [
    ("ABCDEFGH: IJ", ["ABCD", "EFGH:", "IJ"]),
    ("AB: CDEFGHIJ", ["AB:", "CDEF", "GHIJ"]),
    ("ABCDEFGHIJKLM", ["ABC", "DEF", "GHI", "JKLM"]),
])
@pytest.mark.parametrize("split", [_split_long_text_recursive, atomize_unit_text])
def test_split_preserves_source_order(source, expected, split):
    cfg = RefinerInputBuildConfig(
        atom_max_chars=5, atom_max_tokens=120,
        atom_min_chars=0, atom_min_tokens=0,
    )
    actual = split(source, cfg)
    assert actual == expected
    assert without_whitespace("".join(actual)) == without_whitespace(source)
    assert all(len(atom) <= cfg.atom_max_chars for atom in actual)


def test_default_limits_preserve_order_through_public_merging():
    # Before the fix, BFS emitted END first. Public short-atom merging then
    # produced ["END " + "A" * 240, "A" * 240 + ":"], still out of order.
    source = "A" * 480 + ": END"
    cfg = RefinerInputBuildConfig()
    assert cfg.atom_max_chars == 480
    assert _split_long_text_once(source, cfg) == ["A" * 480 + ":", "END"]

    recursive = _split_long_text_recursive(source, cfg)
    assert recursive == ["A" * 240, "A" * 240 + ":", "END"]
    actual = atomize_unit_text(source, cfg)
    assert actual == ["A" * 240, "A" * 240 + ": END"]
    for atoms in (recursive, actual):
        assert without_whitespace("".join(atoms)) == without_whitespace(source)
        assert all(len(atom) <= cfg.atom_max_chars for atom in atoms)
