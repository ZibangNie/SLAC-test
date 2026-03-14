"""
Chunk0 adapter.

Main responsibility:
- convert rule segmenter outputs into stable chunk0_units
- standardize field names and ids for refiner input assembly
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _build_parent_map(units: List[Dict[str, Any]]) -> Dict[int, Optional[int]]:
    out: Dict[int, Optional[int]] = {}
    for u in units:
        uid = int(u["unit_id"])
        out[uid] = u.get("parent_id")
    return out


def _build_unit_lookup(units: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(u["unit_id"]): u for u in units}


def _sanitize_path_text(text: str) -> str:
    x = (text or "").strip()
    x = " ".join(x.split())
    return x


def _compute_path_for_unit(
    unit_id: int,
    unit_lookup: Dict[int, Dict[str, Any]],
    parent_map: Dict[int, Optional[int]],
    *,
    include_root: bool = False,
    include_non_heading_parent: bool = False,
) -> List[str]:
    path: List[str] = []
    cur = unit_id

    visited = set()
    while cur is not None and cur in unit_lookup and cur not in visited:
        visited.add(cur)
        u = unit_lookup[cur]
        pid = parent_map.get(cur)

        keep = False
        if cur == 0:
            keep = include_root
        else:
            utype = u.get("type")
            if utype in {"title", "heading"}:
                keep = True
            elif include_non_heading_parent and utype == "paragraph":
                keep = True

        if keep:
            txt = _sanitize_path_text(u.get("text", ""))
            if txt:
                path.append(txt)

        cur = pid

    path.reverse()
    return path


def build_chunk0_units_from_structure_doc(
    structure_doc: Dict[str, Any],
    *,
    include_title: bool = True,
    include_other_nonroot: bool = False,
    include_root_in_path: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert structure_doc["units"] to refiner-facing chunk0_units.

    Output unit_id is re-indexed to 0..N-1 for chunk0 stability.
    Original structure unit id is preserved as source_unit_id.
    """
    units = structure_doc.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("structure_doc missing non-empty 'units'")

    unit_lookup = _build_unit_lookup(units)
    parent_map = _build_parent_map(units)

    chunk0_units: List[Dict[str, Any]] = []
    for u in units:
        src_uid = int(u["unit_id"])
        if src_uid == 0:
            continue

        utype = u.get("type", "other")
        text = (u.get("text") or "").strip()
        if not text:
            continue

        if utype == "title" and not include_title:
            continue
        if utype == "other" and not include_other_nonroot:
            continue

        path = _compute_path_for_unit(
            unit_id=src_uid,
            unit_lookup=unit_lookup,
            parent_map=parent_map,
            include_root=include_root_in_path,
            include_non_heading_parent=False,
        )

        item: Dict[str, Any] = {
            "unit_id": len(chunk0_units),
            "source_unit_id": src_uid,
            "text": text,
            "type": utype,
            "level": u.get("level"),
            "parent_id": u.get("parent_id"),
            "path": path,
            "depth": int(u.get("level", 0)),
        }

        if u.get("num_prefix") is not None:
            item["num_prefix"] = u["num_prefix"]
        if u.get("marker_type") is not None:
            item["marker_type"] = u["marker_type"]
        if u.get("unit_hash") is not None:
            item["unit_hash"] = u["unit_hash"]

        chunk0_units.append(item)

    return chunk0_units


def attach_chunk0_units(
    structure_doc: Dict[str, Any],
    *,
    include_title: bool = True,
    include_other_nonroot: bool = False,
    include_root_in_path: bool = False,
) -> Dict[str, Any]:
    """
    Return a copied structure_doc-like dict with chunk0_units added.
    """
    out = dict(structure_doc)
    out["chunk0_units"] = build_chunk0_units_from_structure_doc(
        structure_doc,
        include_title=include_title,
        include_other_nonroot=include_other_nonroot,
        include_root_in_path=include_root_in_path,
    )
    return out