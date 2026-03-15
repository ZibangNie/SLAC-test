from __future__ import annotations

from pathlib import Path
from typing import List

from SLAC.retrieval.index.embedder import HFTextEmbedder
from SLAC.retrieval.index.faiss_utils import build_flat_ip_index, save_faiss_index, save_id_map
from SLAC.retrieval.schemas.records import ChunkRecord


def compose_chunk_retrieval_text(chunk: ChunkRecord) -> str:
    parts: List[str] = []

    if chunk.path_text:
        parts.append(chunk.path_text)
    if chunk.number_signature:
        parts.append(chunk.number_signature)
    if chunk.anchor_text:
        parts.append(chunk.anchor_text)
    if chunk.text:
        parts.append(chunk.text)

    return " [SEP] ".join([p.strip() for p in parts if str(p).strip()])


def build_chunk_dense_index(
    chunks: List[ChunkRecord],
    embedder: HFTextEmbedder,
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_map = [c.chunk_id for c in chunks]
    texts = [compose_chunk_retrieval_text(c) for c in chunks]

    vectors = embedder.encode_texts(texts)
    index = build_flat_ip_index(vectors)

    save_faiss_index(index, output_dir / "faiss.index")
    save_id_map(id_map, output_dir / "id_map.npy")

    return {
        "num_items": len(id_map),
        "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 and len(vectors) else 0,
        "output_dir": str(output_dir),
    }