from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from SLAC.retrieval.dataio.writers import write_jsonl
from SLAC.retrieval.schemas.records import PackedEvidenceItem


def export_packed_evidence_jsonl(
    query_id: str,
    query: str,
    query_type: str,
    tree_mode: str,
    packed_items: List[PackedEvidenceItem],
    pack_summary: Dict,
    output_path: str | Path,
) -> None:
    rows = []
    for item in packed_items:
        obj = item.to_dict()
        obj["query_id"] = query_id
        obj["query"] = query
        obj["query_type"] = query_type
        obj["tree_mode"] = tree_mode
        obj["pack_summary"] = pack_summary
        rows.append(obj)
    write_jsonl(output_path, rows)