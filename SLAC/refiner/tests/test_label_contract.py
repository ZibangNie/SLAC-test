"""Executable contract tests, including an independent dense optimum oracle."""
from __future__ import annotations

import itertools
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slac_refiner.label_contract import derive_canonical_labels, replay_labels, validate_boundary_vector


def vector(length, gaps):
    return [int(g in gaps) for g in range(length)]


def objective(labels):
    cost = 4 * sum(labels["insert"])
    keeps = matches = 0
    for item in labels["edit"]:
        if item["y"] == "DEL":
            cost += 4
        else:
            matches += 1
            if item["y"] == "KEEP":
                keeps += 1
            else:
                cost += abs(int(item["y"].split(":")[1]))
    return cost, -keeps, -matches


def dense_optimum(b0, target, radius):
    sources = [g for g, bit in enumerate(b0) if bit]
    targets = [g for g, bit in enumerate(target) if bit]
    table = [[None] * (len(targets) + 1) for _ in range(len(sources) + 1)]
    table[0][0] = (0, 0, 0)
    for i in range(len(sources) + 1):
        for j in range(len(targets) + 1):
            if i == j == 0:
                continue
            options = []
            if i:
                cost, keeps, matches = table[i - 1][j]
                options.append((cost + 4, keeps, matches))
            if j:
                cost, keeps, matches = table[i][j - 1]
                options.append((cost + 4, keeps, matches))
            if i and j and abs(sources[i - 1] - targets[j - 1]) <= radius:
                distance = abs(sources[i - 1] - targets[j - 1])
                cost, keeps, matches = table[i - 1][j - 1]
                options.append((cost + distance, keeps - int(distance == 0), matches - 1))
            table[i][j] = min(options)
    return table[-1][-1]


@pytest.mark.parametrize("radius", [0, 1, 2, 6, 10])
def test_exhaustive_small_alignments_are_optimal_and_replay(radius):
    vectors = list(itertools.product((0, 1), repeat=5))
    for b0 in vectors:
        for target in vectors:
            labels = derive_canonical_labels(b0, target, K=radius)
            assert replay_labels(b0, labels, K=radius) == list(target)
            assert objective(labels) == dense_optimum(b0, target, radius)


def test_far_move_becomes_delete_and_insert():
    labels = derive_canonical_labels(vector(10, {0}), vector(10, {9}))
    assert labels["edit"] == [{"g": 0, "y": "DEL"}]
    assert labels["insert"][9] == 1


def test_crossing_ancestry_is_reassigned_monotonically():
    labels = derive_canonical_labels(vector(18, {12, 13}), vector(18, {14, 16}))
    assert labels["edit"] == [{"g": 12, "y": "SHIFT:2"}, {"g": 13, "y": "SHIFT:3"}]


def test_del_does_not_reset_monotonicity():
    labels = {"edit": [{"g": 4, "y": "SHIFT:6"}, {"g": 5, "y": "DEL"}, {"g": 6, "y": "SHIFT:-4"}], "insert": [0] * 12}
    with pytest.raises(ValueError, match="across DEL"):
        replay_labels(vector(12, {4, 5, 6}), labels)


def test_adjacent_inserts_are_legal_and_duplicate_targets_are_rejected():
    b0 = vector(3, {0})
    labels = derive_canonical_labels(b0, vector(3, {0, 1, 2}))
    assert labels == {"edit": [{"g": 0, "y": "KEEP"}], "insert": [0, 1, 1]}
    labels["insert"][0] = 1
    with pytest.raises(ValueError, match="duplicate"):
        replay_labels(b0, labels)


def test_stable_ties_prefer_earlier_source_path():
    expected = {"edit": [{"g": 1, "y": "SHIFT:1"}, {"g": 3, "y": "DEL"}], "insert": [0] * 5}
    for _ in range(10):
        assert derive_canonical_labels(vector(5, {1, 3}), vector(5, {2})) == expected


@pytest.mark.parametrize("bad", [[0, 2], [True, 0], [0.0, 1], "01"])
def test_nonbinary_vectors_rejected(bad):
    with pytest.raises(ValueError):
        derive_canonical_labels(bad, [0, 1])


def test_exact_t_minus_one_domain_including_empty_and_single_atom():
    assert validate_boundary_vector([], num_atoms=0) == []
    assert validate_boundary_vector([], num_atoms=1) == []
    assert derive_canonical_labels([], []) == {"edit": [], "insert": []}
    with pytest.raises(ValueError, match="length"):
        validate_boundary_vector([0, 1], num_atoms=2)
    with pytest.raises(ValueError, match="domain"):
        replay_labels([0, 1], {"edit": [{"g": 1, "y": "SHIFT:1"}], "insert": [0, 0]})


def test_shift_zero_only_accepted_in_legacy_audit_mode():
    labels = {"edit": [{"g": 0, "y": "SHIFT:0"}], "insert": [0]}
    with pytest.raises(ValueError, match="KEEP"):
        replay_labels([1], labels)
    assert replay_labels([1], labels, require_canonical_spelling=False) == [1]


@pytest.mark.parametrize("labels", [
    {"edit": [], "insert": [0, 0]},
    {"edit": [{"g": 1, "y": "KEEP"}], "insert": [0, 0]},
    {"edit": [{"g": 0, "y": "SHIFT:7"}], "insert": [0, 0]},
])
def test_invalid_source_coverage_and_radius_rejected(labels):
    with pytest.raises(ValueError):
        replay_labels([1, 0], labels)
