"""Small synthetic parent-max baseline tests; no data, tokenizer, or model."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / "docs/research/granularity_lift_control.py"
_SPEC = importlib.util.spec_from_file_location("granularity_lift_control", _PATH)
control = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(control)
prepare = control.prepare_parent_layout
lift = control.lift_parent_max_scores


def candidate(identifier, task, owner, span, rank=0):
    return {"candidate_id": identifier, "task_id": task, "parent_native_unit_id": owner,
            "dense_rank": rank, "span": list(span)}


def fixture():
    units = [candidate("u1", "unit-one", "p1", (0, 4)),
             candidate("u2", "unit-two", "p2", (6, 8), 1)]
    atoms = [candidate("a1", "child-one", "p1", (0, 2)),
             candidate("a2", "child-two", "p1", (2, 4)),
             candidate("a3", "child-three", "p2", (6, 8), 1)]
    return units, atoms


def test_maximum_uses_only_children_and_outputs_are_independent_copies():
    units, atoms = fixture()
    scores = {"unit-one": 900, "unit-two": 800,
              "child-one": 0.2, "child-two": 0.8, "child-three": 0.5}
    before = deepcopy((units, atoms, scores))
    layout = prepare(units, atoms)
    saved_layout = deepcopy(layout)
    lifted, internal, audit = lift(layout, scores)
    assert internal == {"parent-max:u1": 0.8, "parent-max:u2": 0.5}
    assert [row["max_contributors"][0]["candidate_id"] for row in audit] == ["a2", "a3"]
    assert lifted == [dict(units[0], task_id="parent-max:u1"),
                      dict(units[1], task_id="parent-max:u2")]
    lifted[0]["span"][0] = 99
    audit[0]["max_contributors"][0]["span"][0] = 99
    assert (units, atoms, scores) == before
    assert layout == saved_layout


def test_shared_model_tasks_across_parents_do_not_collide_in_lifted_map():
    units, atoms = fixture()
    units[0]["task_id"] = units[1]["task_id"] = "same-whole-text"
    atoms[0]["task_id"] = atoms[2]["task_id"] = "same-child-text"
    layout = prepare(units, atoms)
    lifted, scores, audit = lift(layout, {"same-whole-text": 999,
                                        "same-child-text": 0.3, "child-two": 0.7})
    assert scores == {"parent-max:u1": 0.7, "parent-max:u2": 0.3}
    assert len({row["task_id"] for row in lifted}) == 2
    assert [row["original_unit_task_id"] for row in audit] == ["same-whole-text"] * 2


def test_all_maximum_witness_ties_are_preserved_in_source_order():
    units, atoms = fixture()
    atoms[1]["task_id"] = atoms[0]["task_id"]
    _, _, audit = lift(prepare(units, atoms), {"child-one": 0.4, "child-three": 0.4})
    assert [row["candidate_id"] for row in audit[0]["max_contributors"]] == ["a1", "a2"]
    assert [row["span"] for row in audit[0]["max_contributors"]] == [[0, 2], [2, 4]]


def test_input_shuffle_does_not_change_layout_or_lift():
    units, atoms = fixture()
    scores = {"child-one": 1, "child-two": 2, "child-three": 3}
    forward = prepare(units, atoms)
    shuffled = prepare(list(reversed(units)), [atoms[2], atoms[0], atoms[1]])
    assert forward == shuffled
    assert lift(forward, scores) == lift(shuffled, scores)


def test_invalid_parent_and_child_geometry_is_rejected():
    edits = [
        lambda u, a: a[0].update(parent_native_unit_id="unknown"),
        lambda u, a: a[0].update(dense_rank=9),
        lambda u, a: a[1].update(span=[3, 4]),  # child gap
        lambda u, a: a[1].update(span=[1, 4]),  # child overlap
        lambda u, a: a[1].update(span=[2, 5]),  # outside parent
        lambda u, a: a.pop(),  # missing all children of the second parent
        lambda u, a: u[1].update(span=[3, 8]),  # parent overlap
        lambda u, a: u[1].update(parent_native_unit_id="p1"),
    ]
    for edit in edits:
        units, atoms = fixture()
        edit(units, atoms)
        with pytest.raises(ValueError):
            prepare(units, atoms)


def test_candidate_identity_and_integer_domains_are_strict():
    edits = [
        lambda u, a: a[0].update(candidate_id="u1"),
        lambda u, a: u[1].update(candidate_id="u1"),
        lambda u, a: a[0].update(task_id=""),
        lambda u, a: a[0].update(parent_native_unit_id=True),
        lambda u, a: a[0].update(span=[False, 2]),
        lambda u, a: a[0].update(span=[0, 0]),
        lambda u, a: a[0].update(span=[0, 2.0]),
        lambda u, a: u[0].update(dense_rank=True),
        lambda u, a: u[0].update(dense_rank=-1),
        lambda u, a: a[0].update(text="unsupported payload"),
    ]
    for edit in edits:
        units, atoms = fixture()
        edit(units, atoms)
        with pytest.raises(ValueError):
            prepare(units, atoms)


def test_missing_and_nonfinite_or_nonreal_used_child_scores_are_rejected():
    units, atoms = fixture()
    layout = prepare(units, atoms)
    good = {"child-one": 1, "child-two": 2, "child-three": 3}
    with pytest.raises(ValueError, match="missing"):
        lift(layout, {"child-one": 1, "child-two": 2})
    for invalid in (True, None, "1", 1j, float("nan"), float("inf"), -float("inf")):
        scores = dict(good, **{"child-one": invalid})
        with pytest.raises(ValueError):
            lift(layout, scores)


def test_negative_scores_are_not_thresholded_and_unused_parent_scores_are_ignored():
    units, atoms = fixture()
    scores = {"child-one": -4, "child-two": -2.5, "child-three": 0,
              "unit-one": float("nan"), "unit-two": True}
    _, lifted_scores, _ = lift(prepare(units, atoms), scores)
    assert lifted_scores == {"parent-max:u1": -2.5, "parent-max:u2": 0}
    assert lift(prepare([], []), {}) == ([], {}, [])
