from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List

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


def load_chunk_records(path: str | Path) -> List[ChunkRecord]:
    items: List[ChunkRecord] = []
    for obj in read_jsonl(path):
        items.append(
            ChunkRecord(
                doc_id=obj["doc_id"],
                chunk_id=obj["chunk_id"],
                chunk_index=int(obj["chunk_index"]),
                atom_start=int(obj["atom_start"]),
                atom_end=int(obj["atom_end"]),
                text=obj["text"],
                num_atoms=int(obj["num_atoms"]),
                path=list(obj.get("path", [])),
                depth=int(obj.get("depth", 0)),
                parent_id=obj.get("parent_id"),
                prev_chunk_id=obj.get("prev_chunk_id"),
                next_chunk_id=obj.get("next_chunk_id"),
                token_est=obj.get("token_est"),
                path_text=obj.get("path_text"),
                number_signature=obj.get("number_signature"),
                anchor_text=obj.get("anchor_text"),
                is_title_like=obj.get("is_title_like"),
                indent_level=obj.get("indent_level"),
                domain=obj.get("domain"),
                meta=obj.get("meta", {}),
            )
        )
    return items


def load_leaf_records(path: str | Path) -> List[LeafRecord]:
    items: List[LeafRecord] = []
    for obj in read_jsonl(path):
        items.append(
            LeafRecord(
                doc_id=obj["doc_id"],
                leaf_id=obj["leaf_id"],
                owner_chunk_id=obj["owner_chunk_id"],
                leaf_index=int(obj["leaf_index"]),
                atom_start=int(obj["atom_start"]),
                atom_end=int(obj["atom_end"]),
                text=obj["text"],
                path=list(obj.get("path", [])),
                depth=int(obj.get("depth", 0)),
                prev_leaf_id=obj.get("prev_leaf_id"),
                next_leaf_id=obj.get("next_leaf_id"),
                token_est=obj.get("token_est"),
                path_text=obj.get("path_text"),
                owner_chunk_anchor=obj.get("owner_chunk_anchor"),
                number_signature=obj.get("number_signature"),
                anchor_text=obj.get("anchor_text"),
                is_title_like=obj.get("is_title_like"),
                indent_level=obj.get("indent_level"),
                domain=obj.get("domain"),
                meta=obj.get("meta", {}),
            )
        )
    return items


def load_doc_catalog(path: str | Path) -> List[DocCatalogRecord]:
    items: List[DocCatalogRecord] = []
    for obj in read_jsonl(path):
        items.append(
            DocCatalogRecord(
                doc_id=obj["doc_id"],
                doc_title=obj["doc_title"],
                source_path=obj["source_path"],
                domain=obj.get("domain"),
                num_chunks=obj.get("num_chunks"),
                num_leaves=obj.get("num_leaves"),
                meta=obj.get("meta", {}),
            )
        )
    return items