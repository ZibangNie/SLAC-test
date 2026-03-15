from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from SLAC.retrieval.schemas.records import (
    ChunkRecord,
    DocCatalogRecord,
    LeafRecord,
)


def read_jsonl(path: str | Path) -> Generator[Dict[str, Any], None, None]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid json at {path}:{line_no}: {e}") from e


def _pick(obj: Dict[str, Any], *keys: str, default=None, required: bool = False):
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    if required:
        raise KeyError(f"missing required keys {keys}; available keys={sorted(obj.keys())}")
    return default


def _as_int(x, default: Optional[int] = None) -> Optional[int]:
    if x is None:
        return default
    return int(x)


def _as_list(x, default=None):
    if x is None:
        return [] if default is None else default
    if isinstance(x, list):
        return x
    return [x]


def _normalize_parent_id(x):
    # 当前 Refiner stage2 真实输出里 parent_id=0 常作为根占位，不应视为真实父 chunk id
    if x in [None, "", 0, "0"]:
        return None
    return x


def load_chunk_records(path: str | Path) -> List[ChunkRecord]:
    items: List[ChunkRecord] = []
    for obj in read_jsonl(path):
        atom_start = _pick(obj, "atom_start", "start_atom", "span_start", "chunk_start", required=True)
        atom_end = _pick(obj, "atom_end", "end_atom", "span_end", "chunk_end", required=True)

        path_list = _as_list(_pick(obj, "path", "header_path", "path_titles", default=[]))
        depth = _pick(obj, "depth", "level", default=max(len(path_list), 0))

        num_atoms = _pick(
            obj,
            "num_atoms",
            default=(int(atom_end) - int(atom_start)),
        )

        items.append(
            ChunkRecord(
                doc_id=_pick(obj, "doc_id", required=True),
                chunk_id=_pick(obj, "chunk_id", required=True),
                chunk_index=_as_int(_pick(obj, "chunk_index", "index", "chunk_idx", required=True), 0),
                atom_start=_as_int(atom_start, 0),
                atom_end=_as_int(atom_end, 0),
                text=_pick(obj, "text", "content", required=True),
                num_atoms=_as_int(num_atoms, 0),
                path=path_list,
                depth=_as_int(depth, 0),
                parent_id=_normalize_parent_id(_pick(obj, "parent_id", "parent_chunk_id")),
                prev_chunk_id=_pick(obj, "prev_chunk_id", "prev_id"),
                next_chunk_id=_pick(obj, "next_chunk_id", "next_id"),
                token_est=_pick(obj, "token_est", "tokens"),
                path_text=_pick(obj, "path_text"),
                number_signature=_pick(obj, "number_signature"),
                anchor_text=_pick(obj, "anchor_text"),
                is_title_like=_pick(obj, "is_title_like"),
                indent_level=_pick(obj, "indent_level"),
                domain=_pick(obj, "domain"),
                meta={
                    **obj.get("meta", {}),
                    "source": obj.get("source"),
                    "boundary_meta": obj.get("boundary_meta"),
                },
            )
        )
    return items


def load_leaf_records(path: str | Path) -> List[LeafRecord]:
    items: List[LeafRecord] = []
    for obj in read_jsonl(path):
        # 当前真实 schema 是 atom_index，而不是 atom_start/atom_end
        atom_index = _pick(obj, "atom_index", "atom_start", "start_atom", "span_start", "leaf_start", "start", required=True)

        atom_start = int(atom_index)
        atom_end = atom_start + 1

        path_list = _as_list(_pick(obj, "path", "header_path", "path_titles", default=[]))
        depth = _pick(obj, "depth", "level", default=max(len(path_list), 0))

        items.append(
            LeafRecord(
                doc_id=_pick(obj, "doc_id", required=True),
                leaf_id=_pick(obj, "leaf_id", required=True),
                owner_chunk_id=_pick(obj, "owner_chunk_id", "chunk_id", "parent_chunk_id", required=True),
                leaf_index=_as_int(_pick(obj, "leaf_index", "index", "leaf_idx", required=True), 0),
                atom_start=atom_start,
                atom_end=atom_end,
                text=_pick(obj, "text", "content", required=True),
                path=path_list,
                depth=_as_int(depth, 0),
                prev_leaf_id=_pick(obj, "prev_leaf_id", "prev_id"),
                next_leaf_id=_pick(obj, "next_leaf_id", "next_id"),
                token_est=_pick(obj, "token_est", "tokens"),
                path_text=_pick(obj, "path_text"),
                owner_chunk_anchor=_pick(obj, "owner_chunk_anchor"),
                number_signature=_pick(obj, "number_signature"),
                anchor_text=_pick(obj, "anchor_text"),
                is_title_like=_pick(obj, "is_title_like"),
                indent_level=_pick(obj, "indent_level"),
                domain=_pick(obj, "domain"),
                meta={
                    **obj.get("meta", {}),
                    "source": obj.get("source"),
                    "chunk_index": obj.get("chunk_index"),
                    "parent_id": obj.get("parent_id"),
                    "atom_index": obj.get("atom_index"),
                },
            )
        )
    return items


def load_doc_catalog(path: str | Path) -> List[DocCatalogRecord]:
    items: List[DocCatalogRecord] = []
    for obj in read_jsonl(path):
        items.append(
            DocCatalogRecord(
                doc_id=_pick(obj, "doc_id", required=True),
                doc_title=_pick(obj, "doc_title", "doc_name", "title", default=_pick(obj, "doc_id")),
                source_path=_pick(obj, "source_path", "src_path", default="unknown"),
                domain=_pick(obj, "domain"),
                num_chunks=_pick(obj, "num_chunks", "num_refined_chunks"),
                num_leaves=_pick(obj, "num_leaves", "num_leaf_records"),
                meta={
                    **obj.get("meta", {}),
                    "source_type": obj.get("source_type"),
                    "max_depth": obj.get("max_depth"),
                    "num_atoms": obj.get("num_atoms"),
                    "num_chunk0_units": obj.get("num_chunk0_units"),
                    "num_seed_boundaries": obj.get("num_seed_boundaries"),
                    "num_refined_boundaries": obj.get("num_refined_boundaries"),
                    "selected_candidate": obj.get("selected_candidate"),
                },
            )
        )
    return items