"""Pure parent-max comparison baseline; no semantic labels or inherited scores."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from numbers import Real


_FIELDS = {"candidate_id", "task_id", "parent_native_unit_id", "dense_rank", "span"}


def _sequence(value, name):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")


def _candidate(value):
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("candidate must contain exactly the five metadata fields")
    for field in ("candidate_id", "task_id", "parent_native_unit_id"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    if type(value["dense_rank"]) is not int or value["dense_rank"] < 0:
        raise ValueError("dense_rank must be a nonnegative integer, not bool")
    span = value["span"]
    _sequence(span, "span")
    if (len(span) != 2 or any(type(x) is not int for x in span)
            or not 0 <= span[0] < span[1]):
        raise ValueError("span must be a positive half-open nonnegative integer interval")
    copied = dict(value)
    copied["span"] = list(span)
    return copied


def prepare_parent_layout(unit_candidates, atom_candidates):
    """Validate exact child coverage and return copied metadata in source order.

    Different parents may share model task IDs. Candidate IDs remain unique
    across both representations. Gaps between distinct parents are allowed.
    """
    _sequence(unit_candidates, "unit_candidates")
    _sequence(atom_candidates, "atom_candidates")
    units = [_candidate(value) for value in unit_candidates]
    atoms = [_candidate(value) for value in atom_candidates]
    identifiers = [value["candidate_id"] for value in units + atoms]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate IDs must be unique across units and atoms")
    units.sort(key=lambda value: tuple(value["span"]))
    owners = {}
    previous_end = 0
    for unit in units:
        owner = unit["parent_native_unit_id"]
        if owner in owners:
            raise ValueError("each parent must have exactly one whole-unit candidate")
        if unit["span"][0] < previous_end:
            raise ValueError("whole-unit spans overlap")
        previous_end = unit["span"][1]
        owners[owner] = {"unit": unit, "atoms": []}
    for atom in atoms:
        owner = atom["parent_native_unit_id"]
        if owner not in owners:
            raise ValueError("atom has an unknown parent")
        parent = owners[owner]
        unit = parent["unit"]
        if atom["dense_rank"] != unit["dense_rank"]:
            raise ValueError("atom dense_rank must equal its parent dense_rank")
        if not unit["span"][0] <= atom["span"][0] < atom["span"][1] <= unit["span"][1]:
            raise ValueError("atom lies outside its parent span")
        parent["atoms"].append(atom)
    for parent in owners.values():
        parent["atoms"].sort(key=lambda value: tuple(value["span"]))
        cursor = parent["unit"]["span"][0]
        for atom in parent["atoms"]:
            if atom["span"][0] != cursor:
                raise ValueError("child atoms must cover their parent without gaps or overlap")
            cursor = atom["span"][1]
        if cursor != parent["unit"]["span"][1]:
            raise ValueError("child atoms must cover the complete parent span")
    return {"parents": list(owners.values())}


def lift_parent_max_scores(layout, fresh_scores):
    """Return (rank-ready units, separate scores, max-contributor audit rows).

    Only fresh child task scores are used. Unit task scores and JEV labels are
    neither consulted nor emitted. The caller retains its original score map.
    """
    if not isinstance(layout, Mapping) or set(layout) != {"parents"}:
        raise ValueError("invalid parent layout")
    _sequence(layout["parents"], "parents")
    units, atoms = [], []
    for parent in layout["parents"]:
        if not isinstance(parent, Mapping) or set(parent) != {"unit", "atoms"}:
            raise ValueError("invalid parent layout entry")
        _sequence(parent["atoms"], "parent atoms")
        units.append(parent["unit"])
        atoms.extend(parent["atoms"])
    validated = prepare_parent_layout(units, atoms)
    if validated != layout:
        raise ValueError("layout must have canonical source ordering and parent grouping")
    if not isinstance(fresh_scores, Mapping):
        raise ValueError("fresh_scores must be a mapping")

    child_scores = {}
    for atom in atoms:
        task = atom["task_id"]
        if task not in fresh_scores:
            raise ValueError(f"missing fresh child score: {task}")
        value = fresh_scores[task]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("fresh child score must be a finite real number, not bool")
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError("fresh child score must be finite")
        child_scores[task] = value

    lifted, scores, audit = [], {}, []
    for parent in validated["parents"]:
        unit = parent["unit"]
        maximum = max(child_scores[atom["task_id"]] for atom in parent["atoms"])
        task = "parent-max:" + unit["candidate_id"]
        candidate = deepcopy(unit)
        candidate["task_id"] = task
        lifted.append(candidate)
        scores[task] = maximum
        audit.append({
            "parent_native_unit_id": unit["parent_native_unit_id"],
            "unit_candidate_id": unit["candidate_id"],
            "original_unit_task_id": unit["task_id"],
            "lifted_task_id": task,
            "max_score": maximum,
            "max_contributors": [deepcopy(atom) for atom in parent["atoms"]
                                 if child_scores[atom["task_id"]] == maximum],
        })
    return lifted, scores, audit
