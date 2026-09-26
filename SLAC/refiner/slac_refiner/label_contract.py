"""Versioned, deterministic boundary-edit supervision with no ML dependencies.

There are T - 1 gaps for T atoms: valid gap indices are 0 through T - 2.
Each initial boundary gets exactly one KEEP, DEL or nonzero SHIFT in [-K, K].
Surviving edit targets are strictly increasing, including across DEL actions.
INSERT supplies unmatched final gaps; neighboring final boundaries are legal.

The canonical alignment minimizes DEL + INSERT + 0.25 * abs(SHIFT), with
KEEP costing zero. Ties maximize KEEP count, then matched boundary count,
then prefer the first path discovered in ascending source/target order.
This is an edit-cost objective, not an estimate of semantic gold quality.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from typing import Any

CONTRACT_VERSION = "slac-boundary-edit-v1"
DEFAULT_K = 6
ALIGNMENT_OBJECTIVE = {
    "delete_cost": 1,
    "insert_cost": 1,
    "shift_cost_per_gap": 0.25,
    "tie_break": ["max_keep", "max_matches", "first_ascending_source_target_path"],
}


def validate_boundary_vector(
    vector: Sequence[int], *, name: str = "boundary", num_atoms: int | None = None
) -> list[int]:
    """Reject malformed vectors rather than silently clipping/coercing them."""
    if not isinstance(vector, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    if num_atoms is not None:
        if type(num_atoms) is not int or num_atoms < 0:
            raise ValueError("num_atoms must be a nonnegative integer")
        if len(vector) != max(num_atoms - 1, 0):
            raise ValueError(f"{name} length must equal max(num_atoms - 1, 0)")
    if any(type(value) is not int or value not in (0, 1) for value in vector):
        raise ValueError(f"{name} must contain only integer 0 and 1")
    return list(vector)


def _validate_k(K: int) -> None:
    if type(K) is not int or K < 0:
        raise ValueError("K must be a nonnegative integer")


def replay_labels(
    b0: Sequence[int],
    labels: Mapping[str, Any],
    K: int = DEFAULT_K,
    *,
    require_monotone: bool = True,
    require_canonical_spelling: bool = True,
) -> list[int]:
    """Validate and replay edits; DEL never resets the previous emitted target.

    The two opt-outs support auditing legacy traces only. Even in legacy mode,
    source coverage, domains, label radius, and duplicate targets are checked.
    """
    initial = validate_boundary_vector(b0, name="b0")
    _validate_k(K)
    if not isinstance(labels, Mapping):
        raise ValueError("labels must be an object")
    edits = labels.get("edit")
    inserts = validate_boundary_vector(labels.get("insert"), name="insert")
    if len(inserts) != len(initial):
        raise ValueError("insert length must equal b0 length")
    if not isinstance(edits, list):
        raise ValueError("edit must be a list")
    source_gaps = [g for g, value in enumerate(initial) if value]
    if len(edits) != len(source_gaps):
        raise ValueError("edit must cover each initial boundary exactly once")
    emitted: set[int] = set()
    previous = -1
    for source, item in zip(source_gaps, edits):
        if not isinstance(item, Mapping) or type(item.get("g")) is not int:
            raise ValueError("edit items require an integer source g")
        if item["g"] != source:
            raise ValueError("edit sources must exactly match sorted b0 boundaries")
        label = item.get("y")
        if label == "DEL":
            continue
        if label == "KEEP":
            offset = 0
        elif isinstance(label, str) and label.startswith("SHIFT:"):
            try:
                offset = int(label[6:])
            except ValueError as exc:
                raise ValueError("invalid SHIFT offset") from exc
            if require_canonical_spelling and (offset == 0 or label != f"SHIFT:{offset}"):
                raise ValueError("use KEEP for zero shifts and canonical integer SHIFT spelling")
            if abs(offset) > K:
                raise ValueError("SHIFT offset exceeds K")
        else:
            raise ValueError("unknown edit label")
        target = source + offset
        if not 0 <= target < len(initial):
            raise ValueError("edit target is outside the valid gap domain")
        if target in emitted:
            raise ValueError("duplicate edit target")
        if require_monotone and target <= previous:
            raise ValueError("edit targets must be strictly increasing across DEL")
        emitted.add(target)
        previous = target
    inserted = {g for g, value in enumerate(inserts) if value}
    if emitted & inserted:
        raise ValueError("INSERT must not duplicate an emitted edit target")
    return [int(g in emitted or g in inserted) for g in range(len(initial))]


def derive_canonical_labels(
    b0: Sequence[int], b_final: Sequence[int], K: int = DEFAULT_K
) -> dict[str, list]:
    """Return the minimum-cost legal edit decomposition from b0 to b_final.

    Sparse monotone matching is equivalent to edit-distance alignment: matching
    source s with target t saves 8 - abs(t-s) quarter-cost units against DEL+INS.
    A Fenwick prefix maximum finds the best earlier target; updates are delayed
    until all candidates for one source are evaluated, so no source is reused.
    Time is O((n + m + E) log(m + 1)), space O(n + m + E), where E is the
    number of source/target pairs within K. Far moves remain possible via DEL+INS.
    """
    initial = validate_boundary_vector(b0, name="b0")
    final = validate_boundary_vector(b_final, name="b_final")
    _validate_k(K)
    if len(initial) != len(final):
        raise ValueError("b0 and b_final lengths must match")
    sources = [g for g, value in enumerate(initial) if value]
    targets = [g for g, value in enumerate(final) if value]
    # A node stores (objective tuple, previous node, source index, target index).
    nodes: list[tuple[tuple[int, int, int], int, int, int]] = [((0, 0, 0), -1, -1, -1)]
    tree = [0] * (len(targets) + 1)

    def better(left: int, right: int) -> int:
        if nodes[left][0] != nodes[right][0]:
            return left if nodes[left][0] > nodes[right][0] else right
        return min(left, right)

    def query(end: int) -> int:
        best = 0
        while end:
            best = better(best, tree[end])
            end -= end & -end
        return best

    def update(index: int, node_id: int) -> None:
        index += 1
        while index < len(tree):
            tree[index] = better(tree[index], node_id)
            index += index & -index

    for source_index, source in enumerate(sources):
        pending = []
        for target_index in range(bisect_left(targets, source - K), bisect_right(targets, source + K)):
            offset = targets[target_index] - source
            previous = query(target_index)  # strictly smaller target indices only
            saved, keeps, matches = nodes[previous][0]
            score = (saved + 8 - abs(offset), keeps + int(offset == 0), matches + 1)
            nodes.append((score, previous, source_index, target_index))
            pending.append((target_index, len(nodes) - 1))
        for target_index, node_id in pending:
            update(target_index, node_id)

    matched: dict[int, int] = {}
    last = query(len(targets))
    while last:
        _, previous, source_index, target_index = nodes[last]
        matched[source_index] = target_index
        last = previous
    edits = []
    inserts = final.copy()
    for source_index, source in enumerate(sources):
        if source_index not in matched:
            label = "DEL"
        else:
            target = targets[matched[source_index]]
            offset = target - source
            label = "KEEP" if offset == 0 else f"SHIFT:{offset}"
            inserts[target] = 0
        edits.append({"g": source, "y": label})
    labels = {"edit": edits, "insert": inserts}
    if replay_labels(initial, labels, K) != final:
        raise AssertionError("internal canonical alignment replay failure")
    return labels
