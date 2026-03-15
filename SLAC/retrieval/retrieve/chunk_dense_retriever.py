from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from SLAC.retrieval.index.embedder import HFTextEmbedder
from SLAC.retrieval.index.faiss_utils import load_faiss_index, load_id_map


@dataclass
class ChunkDenseHit:
    object_id: str
    score: float
    rank: int
    source_view: str
    query_variant: str


class ChunkDenseRetriever:
    def __init__(
        self,
        index_dir: str | Path,
        embedder: HFTextEmbedder,
    ):
        index_dir = Path(index_dir)
        self.index = load_faiss_index(index_dir / "faiss.index")
        self.id_map = load_id_map(index_dir / "id_map.npy")
        self.embedder = embedder

    def search_many_variants(
        self,
        query_variants: Sequence[str],
        topk: int = 20,
        score_floor: float = 0.34,
    ) -> List[ChunkDenseHit]:
        merged: Dict[str, ChunkDenseHit] = {}

        for qv in [q for q in query_variants if str(q).strip()]:
            hits = self.search_one(qv, topk=topk)
            for h in hits:
                if h.score < score_floor and h.rank > 10:
                    continue
                prev = merged.get(h.object_id)
                if prev is None or h.score > prev.score or h.rank < prev.rank:
                    merged[h.object_id] = h

        return sorted(merged.values(), key=lambda x: (-x.score, x.rank))

    def search_one(self, query: str, topk: int = 20) -> List[ChunkDenseHit]:
        qvec = self.embedder.encode_texts([query])
        scores, ids = self.index.search(qvec.astype(np.float32), topk)

        out: List[ChunkDenseHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            if idx < 0 or idx >= len(self.id_map):
                continue
            out.append(
                ChunkDenseHit(
                    object_id=self.id_map[idx],
                    score=float(score),
                    rank=rank,
                    source_view="chunk_dense",
                    query_variant=query,
                )
            )
        return out