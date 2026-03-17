from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                obj.setdefault("_line_no", line_no)
                obj.setdefault("_source_file", str(path))
                records.append(obj)
    return records


def maybe_read_jsonl(path: str | Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        return []
    return read_jsonl(path)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_query_key(query_id: Optional[str], request_id: str) -> str:
    text = (query_id or "").strip()
    return text if text else request_id.strip()


def _pick_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def resolve_retrieval_paths(run_dir: str | Path, query_key: str) -> Dict[str, Optional[Path]]:
    run_dir = Path(run_dir)

    candidates = _pick_existing(
        [
            run_dir / f"{query_key}.candidates.jsonl",
            run_dir / "queries" / f"{query_key}.candidates.jsonl",
            run_dir / "candidates" / f"{query_key}.candidates.jsonl",
        ]
    )
    reranker_input = _pick_existing(
        [
            run_dir / f"{query_key}.reranker_input.jsonl",
            run_dir / "queries" / f"{query_key}.reranker_input.jsonl",
            run_dir / "reranker_input" / f"{query_key}.reranker_input.jsonl",
        ]
    )
    packed_evidence = _pick_existing(
        [
            run_dir / f"{query_key}.packed_evidence.jsonl",
            run_dir / "queries" / f"{query_key}.packed_evidence.jsonl",
            run_dir / "packed_evidence" / f"{query_key}.packed_evidence.jsonl",
        ]
    )

    return {
        "candidates": candidates,
        "reranker_input": reranker_input,
        "packed_evidence": packed_evidence,
    }


def resolve_reranker_paths(run_dir: str | Path, query_key: str) -> Dict[str, Optional[Path]]:
    run_dir = Path(run_dir)

    reranked_candidates = _pick_existing(
        [
            run_dir / "queries" / f"{query_key}.reranked_candidates.jsonl",
            run_dir / f"{query_key}.reranked_candidates.jsonl",
        ]
    )
    pack_bridge = _pick_existing(
        [
            run_dir / "pack_bridge" / f"{query_key}.for_packing.jsonl",
            run_dir / f"{query_key}.for_packing.jsonl",
        ]
    )

    return {
        "reranked_candidates": reranked_candidates,
        "pack_bridge": pack_bridge,
    }