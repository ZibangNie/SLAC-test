from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from SLAC.retrieval.dataio.writers import write_jsonl
from SLAC.retrieval.schemas.records import RetrievalCandidate


def export_candidates_jsonl(
    query_id: str,
    query: str,
    query_type: str,
    tree_mode: str,
    candidates: List[RetrievalCandidate],
    output_path: str | Path,
) -> None:
    rows = []
    for cand in candidates:
        obj = cand.to_dict()
        obj["query_id"] = query_id
        obj["query"] = query
        obj["query_type"] = query_type
        obj["tree_mode"] = tree_mode
        rows.append(obj)
    write_jsonl(output_path, rows)