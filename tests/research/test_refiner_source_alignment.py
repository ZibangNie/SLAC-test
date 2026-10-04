"""Invented-string exact-alignment checks; no corpus, model or API access."""
from itertools import product
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
from probe_refiner_source_alignment import align_exact_atoms, summarize_gaps


def exhaustive_chains(source, atoms):
    """Independent Cartesian oracle: no find(), DP or production helper calls."""
    occurrences = [
        tuple((start, start + len(atom)) for start in range(len(source) + 1)
              if source[start:start + len(atom)] == atom)
        for atom in atoms
    ]
    return tuple(chain for chain in product(*occurrences)
                 if all(left[1] <= right[0] for left, right in zip(chain, chain[1:])))


@pytest.mark.parametrize("source,atoms,status,occurrences,spans", [
    # Both atoms have three overlapping occurrences, but only one complete chain.
    ("aaaa", ["aa", "aa"], "unique", [3, 3], [(0, 2), (2, 4)]),
    # Three complete chains: choosing the first one would invent certainty.
    ("aaaaa", ["aa", "aa"], "ambiguous", [4, 4], None),
    ("ababa", ["aba", "ba"], "unique", [2, 2], [(0, 3), (3, 5)]),
    ("ba", ["a", "b"], "missing", [1, 1], None),
    ("abc", ["ab", "bc"], "missing", [1, 1], None),
    ("aba", ["a"], "ambiguous", [2], None),
    ("abc", ["z"], "missing", [0], None),
])
def test_chain_counterexamples(source, atoms, status, occurrences, spans):
    result = align_exact_atoms(source, atoms)
    assert result["status"] == status
    assert result["chain_count_capped"] == {"missing": 0, "unique": 1, "ambiguous": 2}[status]
    assert result["occurrence_counts"] == occurrences
    assert result["spans"] == spans


@pytest.mark.parametrize("source,atom", [
    ("alpha, beta", "alpha beta"),
    ("alpha\t beta", "alpha beta"),
    ("alpha\r\nbeta", "alpha\nbeta"),
    ("Ａ", "A"),  # NFKC would collapse this distinction.
    ("e\u0301", "é"),  # Canonically equivalent is still not exact source text.
    ("alpha", " alpha"),
])
def test_no_punctuation_whitespace_or_unicode_normalization(source, atom):
    result = align_exact_atoms(source, [atom])
    assert result["status"] == "missing"
    assert result["chain_count_capped"] == 0
    assert result["occurrence_counts"] == [0]
    assert result["spans"] is None


@pytest.mark.parametrize("source,expected", [
    (" \tA\n B\r\n", [("leading", 0, 2, 2, 0), ("inter", 3, 5, 2, 0), ("trailing", 6, 8, 2, 0)]),
    ("xA yBz", [("leading", 0, 1, 1, 1), ("inter", 2, 4, 2, 1), ("trailing", 5, 6, 1, 1)]),
])
def test_leading_intermediate_trailing_gap_classification(source, expected):
    aligned = align_exact_atoms(source, ["A", "B"])
    assert aligned["status"] == "unique"
    gaps = summarize_gaps(source, aligned["spans"])
    nonempty = [g for g in gaps if g["start"] != g["end"]]
    assert [(g["kind"], g["start"], g["end"], g["chars"], g["nonwhitespace_chars"])
            for g in nonempty] == expected
    assert sum(g["chars"] for g in gaps) + 2 == len(source)


def test_offsets_are_python_unicode_characters_not_utf8_bytes():
    source = "🙂汉a🙂"
    result = align_exact_atoms(source, ("汉a", "🙂"))
    assert len(source) == 4 and len(source.encode("utf-8")) == 12
    assert result["status"] == "unique"
    assert result["occurrence_counts"] == [1, 2]
    assert result["spans"] == [(1, 3), (3, 4)]
    assert [source[start:end] for start, end in result["spans"]] == ["汉a", "🙂"]
    leading = next(g for g in summarize_gaps(source, result["spans"]) if g["kind"] == "leading")
    assert (leading["start"], leading["end"], leading["chars"], leading["nonwhitespace_chars"]) == (0, 1, 1, 1)


@pytest.mark.parametrize("source,atoms", [
    (None, ["a"]), (123, ["a"]), (b"a", ["a"]),
    ("abc", []), ("abc", None), ("abc", [""]),
    ("abc", [None]), ("abc", [123]), ("abc", [b"a"]),
    ("abc", ["a", ""]),
])
def test_empty_or_invalid_inputs_fail_closed(source, atoms):
    with pytest.raises(ValueError):
        align_exact_atoms(source, atoms)


def test_small_exhaustive_cartesian_oracle():
    """All 7,740 combinations; enumeration is independent of production DP."""
    words = ["".join(chars) for length in (1, 2) for chars in product("ab", repeat=length)]
    seen = {"missing": 0, "unique": 0, "ambiguous": 0}
    comparisons = 0
    for source_length in range(1, 5):
        for chars in product("ab", repeat=source_length):
            source = "".join(chars)
            for atom_count in range(1, 4):
                for atoms in product(words, repeat=atom_count):
                    expected = exhaustive_chains(source, atoms)
                    status = "missing" if not expected else "unique" if len(expected) == 1 else "ambiguous"
                    result = align_exact_atoms(source, atoms)
                    assert result["status"] == status, (source, atoms, expected, result)
                    assert result["chain_count_capped"] == min(len(expected), 2)
                    assert result["spans"] == (list(expected[0]) if len(expected) == 1 else None)
                    assert result["occurrence_counts"] == [sum(
                        source[start:start + len(atom)] == atom for start in range(len(source) + 1)
                    ) for atom in atoms]
                    seen[status] += 1
                    comparisons += 1
    assert comparisons == 7740
    assert all(count > 0 for count in seen.values())
