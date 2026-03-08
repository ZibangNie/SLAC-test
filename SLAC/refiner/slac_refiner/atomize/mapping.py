from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slac_refiner.datasets.schemas import UnitAtomSpan


def build_atoms_b0_from_units(
    normalized_units: List[Dict[str, Any]],
    splitter_fn: Callable[[str], List[str]],
) -> Dict[str, Any]:
    """
    Build:
      - atoms
      - unit2atom_span
      - b0

    Required engineering behaviors from the design:
    1) Empty-unit fix:
       if a unit atomizes to empty, merge it to the nearest non-empty unit,
       preferring the right side. If all units are empty, mark all_units_empty.
    2) Safe b0 projection:
       only project boundary j | j+1 when both adjacent original units are non-empty.
       Otherwise skip and record projection_fix in meta.
    3) Invariant checks:
       spans must be monotonic and cover the whole atoms sequence.

    Notes:
    - We preserve original unit order in unit2atom_span.
    - Empty units receive zero-length spans [cursor, cursor).
    - Non-empty target units may be re-atomized after merged text is attached.
    """
    _validate_input_units(normalized_units)

    unit_ids = [int(u["unit_id"]) for u in normalized_units]
    texts = [str(u.get("text", "")) for u in normalized_units]
    n = len(normalized_units)

    if n == 0:
        return {
            "atoms": [],
            "unit2atom_span": [],
            "b0": [],
            "meta": {
                "empty_unit_count": 0,
                "empty_unit_merged": 0,
                "projection_fix": 0,
                "all_units_empty": True,
            },
        }

    # First atomization pass on original normalized units
    first_pass_atoms: List[List[str]] = [splitter_fn(t) for t in texts]
    is_nonempty: List[bool] = [len(a) > 0 for a in first_pass_atoms]

    meta = {
        "empty_unit_count": 0,
        "empty_unit_merged": 0,
        "projection_fix": 0,
        "all_units_empty": False,
    }

    if not any(is_nonempty):
        meta["empty_unit_count"] = n
        meta["all_units_empty"] = True
        return {
            "atoms": [],
            "unit2atom_span": [
                UnitAtomSpan(unit_id=uid, start_atom=0, end_atom=0) for uid in unit_ids
            ],
            "b0": [],
            "meta": meta,
        }

    # Prepare merged texts for non-empty target units
    merged_texts = texts[:]

    # Empty unit -> nearest non-empty target (prefer right, else left)
    for i in range(n):
        if is_nonempty[i]:
            continue

        meta["empty_unit_count"] += 1
        tgt = _find_nearest_nonempty_prefer_right(is_nonempty, i)
        if tgt is None:
            # Should not happen because all-empty case has been handled
            continue

        meta["empty_unit_merged"] += 1

        empty_text = texts[i].strip()
        if not empty_text:
            continue

        # Preserve original text order:
        # if empty unit is before target, prepend it to target text;
        # if after target, append it after target text.
        if i < tgt:
            merged_texts[tgt] = _concat_texts(empty_text, merged_texts[tgt])
        else:
            merged_texts[tgt] = _concat_texts(merged_texts[tgt], empty_text)

    # Re-atomize non-empty original units after possible merges
    final_atoms_for_nonempty_unit: Dict[int, List[str]] = {}
    for i in range(n):
        if is_nonempty[i]:
            final_atoms_for_nonempty_unit[i] = splitter_fn(merged_texts[i])

    # Build atoms + spans in original unit order
    atoms: List[str] = []
    unit2atom_span: List[UnitAtomSpan] = []
    cursor = 0

    for i in range(n):
        uid = unit_ids[i]

        if is_nonempty[i]:
            local_atoms = final_atoms_for_nonempty_unit[i]
            start_atom = cursor
            atoms.extend(local_atoms)
            cursor = len(atoms)
            end_atom = cursor
        else:
            # Empty units keep zero-length spans at current cursor
            start_atom = cursor
            end_atom = cursor

        unit2atom_span.append(
            UnitAtomSpan(
                unit_id=uid,
                start_atom=start_atom,
                end_atom=end_atom,
            )
        )

    # b0 projection: only when BOTH adjacent original units are non-empty
    b0 = [0] * max(0, len(atoms) - 1)
    for i in range(n - 1):
        if is_nonempty[i] and is_nonempty[i + 1]:
            gap_idx = unit2atom_span[i].end_atom - 1  # g_j = e_j - 1
            if not (0 <= gap_idx < len(b0)):
                raise ValueError(
                    f"Invalid gap index during b0 projection: "
                    f"unit_id={unit2atom_span[i].unit_id}, gap_idx={gap_idx}, len(b0)={len(b0)}"
                )
            b0[gap_idx] = 1
        else:
            meta["projection_fix"] += 1

    _check_span_invariants(unit2atom_span, len(atoms))

    return {
        "atoms": atoms,
        "unit2atom_span": unit2atom_span,
        "b0": b0,
        "meta": meta,
    }


def _validate_input_units(normalized_units: List[Dict[str, Any]]) -> None:
    if not isinstance(normalized_units, list):
        raise ValueError("normalized_units must be a list")

    for i, u in enumerate(normalized_units):
        if not isinstance(u, dict):
            raise ValueError(f"normalized_units[{i}] must be a dict")
        if "unit_id" not in u:
            raise ValueError(f"normalized_units[{i}] missing unit_id")
        if "text" not in u:
            raise ValueError(f"normalized_units[{i}] missing text")


def _find_nearest_nonempty_prefer_right(flags: List[bool], idx: int) -> Optional[int]:
    """
    Prefer the nearest non-empty unit on the right.
    If none exists on the right, search left.
    """
    n = len(flags)

    # Right side first
    for j in range(idx + 1, n):
        if flags[j]:
            return j

    # Then left side
    for j in range(idx - 1, -1, -1):
        if flags[j]:
            return j

    return None


def _concat_texts(left: str, right: str) -> str:
    left = (left or "").strip()
    right = (right or "").strip()

    if left and right:
        return f"{left}\n{right}"
    if left:
        return left
    return right


def _check_span_invariants(unit2atom_span: List[UnitAtomSpan], total_atoms: int) -> None:
    """
    Strong invariant:
    - spans are monotonic
    - spans cover atoms without gaps
    - zero-length spans are allowed
    """
    cursor = 0
    for i, span in enumerate(unit2atom_span):
        if span.start_atom != cursor:
            raise ValueError(
                f"Span coverage gap/overlap at index={i}, unit_id={span.unit_id}, "
                f"expected start={cursor}, got start={span.start_atom}"
            )
        if span.end_atom < span.start_atom:
            raise ValueError(
                f"Invalid span order at index={i}, unit_id={span.unit_id}, "
                f"start={span.start_atom}, end={span.end_atom}"
            )
        cursor = span.end_atom

    if cursor != total_atoms:
        raise ValueError(
            f"Final span coverage mismatch: last_end={cursor}, total_atoms={total_atoms}"
        )