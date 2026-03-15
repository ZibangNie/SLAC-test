from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

from .records import ChunkRecord, LeafRecord, DocCatalogRecord


class RetrievalSchemaError(ValueError):
    pass


def validate_chunk_records(chunks: List[ChunkRecord]) -> None:
    if not chunks:
        raise RetrievalSchemaError("chunk records are empty")

    seen_chunk_ids: Set[str] = set()

    for i, c in enumerate(chunks):
        _require_nonempty(c.doc_id, f"chunks[{i}].doc_id")
        _require_nonempty(c.chunk_id, f"chunks[{i}].chunk_id")

        if c.chunk_id in seen_chunk_ids:
            raise RetrievalSchemaError(f"duplicate chunk_id: {c.chunk_id}")
        seen_chunk_ids.add(c.chunk_id)

        if c.chunk_index < 0:
            raise RetrievalSchemaError(f"chunk_index must be >= 0: {c.chunk_id}")

        _check_span(
            c.atom_start,
            c.atom_end,
            f"chunk span invalid: {c.chunk_id}",
        )

        if c.num_atoms != (c.atom_end - c.atom_start):
            raise RetrievalSchemaError(
                f"num_atoms mismatch for {c.chunk_id}: "
                f"num_atoms={c.num_atoms}, span={c.atom_end - c.atom_start}"
            )

        if not isinstance(c.path, list):
            raise RetrievalSchemaError(f"path must be list for {c.chunk_id}")

        if c.depth < 0:
            raise RetrievalSchemaError(f"depth must be >=0 for {c.chunk_id}")

        _require_nonempty(c.text, f"chunk text empty: {c.chunk_id}")


def validate_leaf_records(
    leaves: List[LeafRecord],
    chunk_ids: Iterable[str],
) -> None:
    if not leaves:
        raise RetrievalSchemaError("leaf records are empty")

    chunk_id_set = set(chunk_ids)
    seen_leaf_ids: Set[str] = set()

    for i, l in enumerate(leaves):
        _require_nonempty(l.doc_id, f"leaves[{i}].doc_id")
        _require_nonempty(l.leaf_id, f"leaves[{i}].leaf_id")
        _require_nonempty(l.owner_chunk_id, f"leaves[{i}].owner_chunk_id")

        if l.leaf_id in seen_leaf_ids:
            raise RetrievalSchemaError(f"duplicate leaf_id: {l.leaf_id}")
        seen_leaf_ids.add(l.leaf_id)

        if l.owner_chunk_id not in chunk_id_set:
            raise RetrievalSchemaError(
                f"owner_chunk_id not found for leaf {l.leaf_id}: {l.owner_chunk_id}"
            )

        if l.leaf_index < 0:
            raise RetrievalSchemaError(f"leaf_index must be >= 0: {l.leaf_id}")

        _check_span(
            l.atom_start,
            l.atom_end,
            f"leaf span invalid: {l.leaf_id}",
        )

        if l.depth < 0:
            raise RetrievalSchemaError(f"depth must be >=0 for {l.leaf_id}")

        if not isinstance(l.path, list):
            raise RetrievalSchemaError(f"path must be list for {l.leaf_id}")

        _require_nonempty(l.text, f"leaf text empty: {l.leaf_id}")


def validate_doc_catalog_records(docs: List[DocCatalogRecord]) -> None:
    if not docs:
        raise RetrievalSchemaError("doc catalog is empty")

    seen_doc_ids: Set[str] = set()

    for i, d in enumerate(docs):
        _require_nonempty(d.doc_id, f"docs[{i}].doc_id")
        _require_nonempty(d.doc_title, f"docs[{i}].doc_title")
        _require_nonempty(d.source_path, f"docs[{i}].source_path")

        if d.doc_id in seen_doc_ids:
            raise RetrievalSchemaError(f"duplicate doc_id in doc catalog: {d.doc_id}")
        seen_doc_ids.add(d.doc_id)

        if d.num_chunks is not None and d.num_chunks < 0:
            raise RetrievalSchemaError(f"num_chunks must be >=0 for {d.doc_id}")
        if d.num_leaves is not None and d.num_leaves < 0:
            raise RetrievalSchemaError(f"num_leaves must be >=0 for {d.doc_id}")


def validate_chunk_doc_consistency(
    chunks: List[ChunkRecord],
    docs: List[DocCatalogRecord],
) -> None:
    doc_ids = {d.doc_id for d in docs}
    missing = sorted({c.doc_id for c in chunks if c.doc_id not in doc_ids})
    if missing:
        raise RetrievalSchemaError(
            f"chunk doc_ids missing in doc_catalog: {missing[:10]}"
        )


def validate_leaf_doc_consistency(
    leaves: List[LeafRecord],
    docs: List[DocCatalogRecord],
) -> None:
    doc_ids = {d.doc_id for d in docs}
    missing = sorted({l.doc_id for l in leaves if l.doc_id not in doc_ids})
    if missing:
        raise RetrievalSchemaError(
            f"leaf doc_ids missing in doc_catalog: {missing[:10]}"
        )


def summarize_tree_quality(chunks: List[ChunkRecord]) -> Dict[str, float]:
    total = max(len(chunks), 1)
    missing_parent = 0
    missing_prev = 0
    missing_next = 0

    for c in chunks:
        if c.depth > 0 and not c.parent_id:
            missing_parent += 1
        if c.prev_chunk_id is None:
            missing_prev += 1
        if c.next_chunk_id is None:
            missing_next += 1

    return {
        "num_chunks": len(chunks),
        "missing_parent_ratio": missing_parent / total,
        "missing_prev_ratio": missing_prev / total,
        "missing_next_ratio": missing_next / total,
    }


def _require_nonempty(x: str, name: str) -> None:
    if x is None or not str(x).strip():
        raise RetrievalSchemaError(f"{name} is empty")


def _check_span(start: int, end: int, msg: str) -> None:
    if start < 0 or end <= start:
        raise RetrievalSchemaError(f"{msg}: [{start}, {end})")