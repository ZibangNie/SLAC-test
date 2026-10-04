"""Synthetic coordinate arithmetic only; no natural data, tokenizer, or model."""

import copy
import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / "docs/research/partition_coverage.py"
_SPEC = importlib.util.spec_from_file_location("partition_coverage", _PATH)
coverage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coverage)
minimal_cover = coverage.minimal_cover
assess_cover = coverage.assess_cover


def test_touching_endpoint_does_not_require_neighbor():
    assert minimal_cover([(0, 1), (1, 2)], [(1, 2)], 2) == {
        "required_indices": [1], "reference_union": [[1, 2]],
        "reference_chars": 1, "required_source_chars": 1,
    }


def test_duplicate_overlapping_adjacent_references_are_unioned_without_mutation():
    partition = [[0, 2], [2, 5], [5, 6]]
    references = [[4, 5], [1, 3], [2, 4], [1, 3]]
    before = copy.deepcopy((partition, references))
    assert minimal_cover(partition, references, 6) == {
        "required_indices": [0, 1], "reference_union": [[1, 5]],
        "reference_chars": 4, "required_source_chars": 5,
    }
    assert (partition, references) == before


@pytest.mark.parametrize("partition,length", [([], 0), ([(0, 2)], 2)])
def test_empty_reference(partition, length):
    assert minimal_cover(partition, [], length) == {
        "required_indices": [], "reference_union": [],
        "reference_chars": 0, "required_source_chars": 0,
    }


@pytest.mark.parametrize("partition,references,length", [
    ([], [], -1), ([], [], True), ([], [], 1.0),
    ([], [], 1), ([(0, 0)], [], 0), ([(0, 1)], [], 0),
    ([(1, 2)], [], 2), ([(0, 1), (2, 3)], [], 3),
    ([(0, 2), (1, 3)], [], 3), ([(1, 2), (0, 1)], [], 2),
    ([(0, 1)], [], 2), ([(False, 1)], [], 1), ([(0, 1.0)], [], 1),
    ([(0, 1, 2)], [], 2), ("01", [], 1), ([(0, 1)], None, 1),
    ([(0, 2)], [(1, 1)], 2), ([(0, 2)], [(-1, 1)], 2),
    ([(0, 2)], [(0, 3)], 2), ([(0, 2)], [(0, True)], 2),
    ([(0, 2)], ["01"], 2), ([(0, 2)], [(0, 1.5)], 2),
])
def test_invalid_coordinates_rejected(partition, references, length):
    with pytest.raises(ValueError):
        minimal_cover(partition, references, length)


def test_exhaustive_small_partitions_references_and_selected_subsets():
    # Independent oracle: source characters are finite sets, with no interval
    # intersection or union algorithm borrowed from the implementation.
    for length in range(6):
        for boundary_mask in range(1 << max(length - 1, 0)):
            endpoints = [0] + [i for i in range(1, length)
                               if boundary_mask & (1 << (i - 1))] + [length]
            partition = list(zip(endpoints, endpoints[1:])) if length else []
            chunk_characters = [set(range(start, end)) for start, end in partition]
            for reference_mask in range(1 << length):
                reference_characters = {i for i in range(length) if reference_mask & (1 << i)}
                references = [(i, i + 1) for i in sorted(reference_characters)]
                result = minimal_cover(partition, references, length)
                required = set(result["required_indices"])
                union_characters = {i for start, end in result["reference_union"]
                                    for i in range(start, end)}
                assert union_characters == reference_characters
                assert result["reference_chars"] == len(reference_characters)
                assert result["required_source_chars"] == sum(len(chunk_characters[i]) for i in required)
                assert result["required_indices"] == sorted(required)
                for selected_mask in range(1 << len(partition)):
                    selected = {i for i in range(len(partition)) if selected_mask & (1 << i)}
                    covered = {character for i in selected for character in chunk_characters[i]}
                    assert (reference_characters <= covered) == (required <= selected)


@pytest.mark.parametrize("required,total,tokens,expected", [
    (0, 0, None, "empty_reference"), (0, 4, 2000, "empty_reference"),
    (4, 4, None, "infeasible_chunk_count"), (4, 5, 1, "infeasible_chunk_count"),
    (1, 2, None, "budget_unmeasured"), (3, 3, None, "budget_unmeasured"),
    (1, 2, 0, "feasible_required_set"), (3, 5, 1024, "feasible_required_set"),
    (3, 5, 1025, "infeasible_budget_no_superset"),
    (1, 1, 1025, "infeasible_budget_no_superset"),
    (1, 2, 1025, "unresolved_budget_nonmonotone"),
    (2, 5, 1025, "unresolved_budget_nonmonotone"),
])
def test_assessment_states(required, total, tokens, expected):
    assert assess_cover(required, total, tokens) == expected


def test_nonmonotone_cost_does_not_exclude_feasible_superset():
    # Abstract synthetic costs, not measured BGE token counts.
    costs = {(0,): 1025, (0, 1): 1024}
    assert assess_cover(1, 2, costs[(0,)]) == "unresolved_budget_nonmonotone"
    assert assess_cover(2, 2, costs[(0, 1)]) == "feasible_required_set"
    assert assess_cover(1, 2, 6, max_chunks=1, max_tokens=5) == "infeasible_budget_no_superset"


@pytest.mark.parametrize("args,kwargs", [
    ((True, 2, 1), {}), ((1, False, 1), {}), ((1, 2, True), {}),
    ((-1, 2, 1), {}), ((0, -1, None), {}), ((1, 2, -1), {}),
    ((1.0, 2, 1), {}), ((1, 2.0, 1), {}), ((1, 2, 1.0), {}),
    ((2, 1, 1), {}), ((1, 2, 1), {"max_chunks": 0}),
    ((1, 2, 1), {"max_chunks": True}), ((1, 2, 1), {"max_chunks": 3.0}),
    ((1, 2, 1), {"max_tokens": 0}), ((1, 2, 1), {"max_tokens": False}),
    ((1, 2, 1), {"max_tokens": 1024.0}), ((0, 0, -1), {}),
])
def test_invalid_assessment_inputs_rejected_before_status(args, kwargs):
    with pytest.raises(ValueError):
        assess_cover(*args, **kwargs)
