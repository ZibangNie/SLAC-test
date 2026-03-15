from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from SLAC.retrieval.index.embedder import HFTextEmbedder
from SLAC.retrieval.index.faiss_utils import build_flat_ip_index, save_faiss_index, save_id_map
from SLAC.retrieval.schemas.records import ChunkRecord, LeafRecord


def compose_leaf_retrieval_text(leaf: LeafRecord, owner_chunk: ChunkRecord | None) -> str:
    parts: List[str] = []

    if leaf.path_text:
        parts.append(leaf.path_text)

    owner_anchor = None
    if owner_chunk is not None:
        owner_anchor = owner_chunk.anchor_text or owner_chunk.number_signature
    if not owner_anchor:
        owner_anchor = leaf.owner_chunk_anchor

    if owner_anchor:
        parts.append(str(owner_anchor))

    if leaf.text:
        parts.append(leaf.text)

    return " [SEP] ".join([p.strip() for p in parts if str(p).strip()])


def build_leaf_dense_index(
    leaves: List[LeafRecord],
    chunk_lookup: Dict[str, ChunkRecord],
    embedder: HFTextEmbedder,
    output_dir: str | Path,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_map: List[str] = []
    texts: List[str] = []

    for leaf in leaves:
        owner_chunk = chunk_lookup.get(leaf.owner_chunk_id)
        text = compose_leaf_retrieval_text(leaf, owner_chunk)
        id_map.append(leaf.leaf_id)
        texts.append(text)

    vectors = embedder.encode_texts(texts)
    index = build_flat_ip_index(vectors)

    save_faiss_index(index, output_dir / "faiss.index")
    save_id_map(id_map, output_dir / "id_map.npy")

    return {
        "num_items": len(id_map),
        "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 and len(vectors) else 0,
        "output_dir": str(output_dir),
    }