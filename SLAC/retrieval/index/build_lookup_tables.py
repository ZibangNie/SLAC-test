from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from SLAC.retrieval.preprocess.anchor_fields import (
    enrich_chunk_record,
    enrich_leaf_record,
    make_anchor_record_from_chunk,
    make_anchor_record_from_leaf,
)
from SLAC.retrieval.schemas.records import (
    AnchorRecord,
    ChunkRecord,
    LeafRecord,
    TreeAdjacencyRecord,
)
from SLAC.retrieval.schemas.validation import summarize_tree_quality


def enrich_all_records(
    chunks: List[ChunkRecord],
    leaves: List[LeafRecord],
    path_joiner: str = " > ",
) -> Tuple[List[ChunkRecord], List[LeafRecord]]:
    chunk_lookup: Dict[str, ChunkRecord] = {}

    enriched_chunks: List[ChunkRecord] = []
    for chunk in chunks:
        c = enrich_chunk_record(chunk, joiner=path_joiner)
        enriched_chunks.append(c)
        chunk_lookup[c.chunk_id] = c

    enriched_leaves: List[LeafRecord] = []
    for leaf in leaves:
        owner = chunk_lookup.get(leaf.owner_chunk_id)
        enriched_leaves.append(enrich_leaf_record(leaf, owner_chunk=owner, joiner=path_joiner))

    return enriched_chunks, enriched_leaves


def build_chunk_lookup(chunks: List[ChunkRecord]) -> Dict[str, ChunkRecord]:
    return {c.chunk_id: c for c in chunks}


def build_leaf_lookup(leaves: List[LeafRecord]) -> Dict[str, LeafRecord]:
    return {l.leaf_id: l for l in leaves}


def build_doc_to_chunks(chunks: List[ChunkRecord]) -> Dict[str, List[ChunkRecord]]:
    grouped: Dict[str, List[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.doc_id].append(chunk)
    for doc_id in grouped:
        grouped[doc_id].sort(key=lambda x: x.chunk_index)
    return dict(grouped)


def build_doc_to_leaves(leaves: List[LeafRecord]) -> Dict[str, List[LeafRecord]]:
    grouped: Dict[str, List[LeafRecord]] = defaultdict(list)
    for leaf in leaves:
        grouped[leaf.doc_id].append(leaf)
    for doc_id in grouped:
        grouped[doc_id].sort(key=lambda x: x.leaf_index)
    return dict(grouped)


def build_children_map(chunks: List[ChunkRecord]) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = defaultdict(list)
    for chunk in chunks:
        if chunk.parent_id:
            children[chunk.parent_id].append(chunk.chunk_id)

    # 保持 children 顺序稳定：按 chunk_index 排
    chunk_map = {c.chunk_id: c for c in chunks}
    for parent_id, child_ids in children.items():
        child_ids.sort(key=lambda cid: chunk_map[cid].chunk_index)
    return dict(children)


def build_tree_adjacency(chunks: List[ChunkRecord]) -> List[TreeAdjacencyRecord]:
    children_map = build_children_map(chunks)
    records: List[TreeAdjacencyRecord] = []

    for chunk in chunks:
        rec = TreeAdjacencyRecord(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            depth=chunk.depth,
            path=chunk.path,
            parent_id=chunk.parent_id,
            children_ids=children_map.get(chunk.chunk_id, []),
            prev_chunk_id=chunk.prev_chunk_id,
            next_chunk_id=chunk.next_chunk_id,
        )
        records.append(rec)
    return records


def build_anchor_lookup(
    chunks: List[ChunkRecord],
    leaves: List[LeafRecord],
) -> List[AnchorRecord]:
    out: List[AnchorRecord] = []
    for chunk in chunks:
        out.append(make_anchor_record_from_chunk(chunk))
    for leaf in leaves:
        out.append(make_anchor_record_from_leaf(leaf))
    return out


def build_quality_gates(chunks: List[ChunkRecord]) -> Dict[str, float]:
    tree_stats = summarize_tree_quality(chunks)

    out = dict(tree_stats)
    out["weak_tree_suspected"] = (
        tree_stats["missing_parent_ratio"] > 0.20
        or max(tree_stats["missing_prev_ratio"], tree_stats["missing_next_ratio"]) > 0.20
    )
    return out


def serialize_chunk_lookup_rows(chunks: Iterable[ChunkRecord]) -> List[dict]:
    return [c.to_dict() for c in chunks]


def serialize_leaf_lookup_rows(leaves: Iterable[LeafRecord]) -> List[dict]:
    return [l.to_dict() for l in leaves]


def serialize_tree_adjacency_rows(rows: Iterable[TreeAdjacencyRecord]) -> List[dict]:
    return [r.to_dict() for r in rows]


def serialize_anchor_lookup_rows(rows: Iterable[AnchorRecord]) -> List[dict]:
    return [r.to_dict() for r in rows]