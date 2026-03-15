from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    chunk_index: int
    atom_start: int
    atom_end: int
    text: str
    num_atoms: int
    path: List[str]
    depth: int
    parent_id: Optional[str] = None
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    # retrieval-enhanced fields
    token_est: Optional[int] = None
    path_text: Optional[str] = None
    number_signature: Optional[str] = None
    anchor_text: Optional[str] = None
    is_title_like: Optional[bool] = None
    indent_level: Optional[int] = None

    # passthrough/meta
    domain: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LeafRecord:
    doc_id: str
    leaf_id: str
    owner_chunk_id: str
    leaf_index: int
    atom_start: int
    atom_end: int
    text: str
    path: List[str]
    depth: int
    prev_leaf_id: Optional[str] = None
    next_leaf_id: Optional[str] = None

    # retrieval-enhanced fields
    token_est: Optional[int] = None
    path_text: Optional[str] = None
    owner_chunk_anchor: Optional[str] = None
    number_signature: Optional[str] = None
    anchor_text: Optional[str] = None
    is_title_like: Optional[bool] = None
    indent_level: Optional[int] = None

    domain: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocCatalogRecord:
    doc_id: str
    doc_title: str
    source_path: str
    domain: Optional[str] = None
    num_chunks: Optional[int] = None
    num_leaves: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnchorRecord:
    object_id: str
    object_type: str  # "chunk" | "leaf"
    doc_id: str
    path: List[str]
    path_text: str
    number_signature: Optional[str]
    anchor_text: str
    is_title_like: bool
    indent_level: Optional[int] = None
    text_norm: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TreeAdjacencyRecord:
    chunk_id: str
    doc_id: str
    depth: int
    path: List[str]
    parent_id: Optional[str]
    children_ids: List[str] = field(default_factory=list)
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalCandidate:
    chunk_id: str
    doc_id: str
    text: str
    path: List[str]
    depth: int

    retrieve_score_raw: Dict[str, float] = field(default_factory=dict)
    retrieve_rank_fused: Optional[int] = None
    source_views: List[str] = field(default_factory=list)

    best_leaf_score: Optional[float] = None
    best_chunk_score: Optional[float] = None
    best_anchor_rank: Optional[int] = None

    support_leaf_ids: List[str] = field(default_factory=list)
    hit_count: int = 0

    hit_type: str = "leaf_direct"
    expansion_from: Optional[str] = None
    expansion_type: Optional[str] = None
    expansion_depth: int = 0

    token_est: Optional[int] = None
    path_text: Optional[str] = None
    number_signature: Optional[str] = None
    anchor_text: Optional[str] = None
    is_title_like: Optional[bool] = None

    query_id: Optional[str] = None
    query_type: Optional[str] = None
    tree_mode: Optional[str] = None

    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PackedEvidenceItem:
    order: int
    chunk_id: str
    doc_id: str
    role: str  # direct | parent_context | sibling_support | neighbor_support | local_branch_support
    hit_type: str
    path: List[str]
    token_est: int
    text: str
    expansion_from: Optional[str] = None
    retrieve_rank_fused: Optional[int] = None
    source_views: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndexMeta:
    encoder_name: str
    tokenizer_name: str
    normalized: bool
    similarity_metric: str
    query_planner_version: str
    corpus_version: str
    build_time: str
    refined_chunks_path: str
    leaf_records_path: str
    doc_catalog_path: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)