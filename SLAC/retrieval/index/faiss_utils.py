from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np


def build_flat_ip_index(vectors: np.ndarray) -> faiss.Index:
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape={vectors.shape}")
    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    return index


def save_faiss_index(index: faiss.Index, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: str | Path) -> faiss.Index:
    return faiss.read_index(str(path))


def save_id_map(id_map: list[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.array(id_map, dtype=object), allow_pickle=True)


def load_id_map(path: str | Path) -> list[str]:
    arr = np.load(str(path), allow_pickle=True)
    return [str(x) for x in arr.tolist()]