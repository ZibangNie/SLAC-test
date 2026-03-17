from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from SLAC.integration.io.schemas import IntegrationRequest, RetrievalArtifacts


def _read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid jsonl at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(
                    f"expected JSON object in {path}:{line_no}, got {type(obj).__name__}"
                )
            rows.append(obj)
    return rows


class RetrievalRunDirAdapter:
    """
    Read retrieval artifacts from an existing retrieval run directory.

    Expected files under retrieval_run_dir:
      - {query_id}.candidates.jsonl
      - {query_id}.reranker_input.jsonl
      - {query_id}.packed_evidence.jsonl
    """

    def run_or_read(self, req: IntegrationRequest) -> RetrievalArtifacts:
        run_dir = str(req.context.retrieval_run_dir or "").strip()
        query_id = str(req.query_id or "").strip()

        if not run_dir:
            raise RuntimeError(
                "context.retrieval_run_dir is required for RetrievalRunDirAdapter."
            )
        if not query_id:
            raise RuntimeError("query_id is required for RetrievalRunDirAdapter.")

        base = Path(run_dir)
        if not base.exists():
            raise RuntimeError(f"retrieval_run_dir not found: {base}")
        if not base.is_dir():
            raise RuntimeError(f"retrieval_run_dir is not a directory: {base}")

        candidates_path = base / f"{query_id}.candidates.jsonl"
        reranker_input_path = base / f"{query_id}.reranker_input.jsonl"
        packed_evidence_path = base / f"{query_id}.packed_evidence.jsonl"

        candidates = _read_jsonl_file(candidates_path)
        reranker_input = _read_jsonl_file(reranker_input_path)
        packed_evidence = _read_jsonl_file(packed_evidence_path)

        if not candidates and not reranker_input and not packed_evidence:
            raise RuntimeError(
                "no retrieval artifacts found under "
                f"{base} for query_id={query_id} "
                f"(expected one or more of: "
                f"{candidates_path.name}, {reranker_input_path.name}, {packed_evidence_path.name})"
            )

        return RetrievalArtifacts(
            candidates=candidates,
            reranker_input=reranker_input,
            packed_evidence=packed_evidence,
            meta={
                "source": "retrieval_run_dir_adapter",
                "retrieval_run_dir": str(base),
                "query_id": query_id,
                "resolved_files": {
                    "candidates": str(candidates_path),
                    "reranker_input": str(reranker_input_path),
                    "packed_evidence": str(packed_evidence_path),
                },
                "counts": {
                    "candidates": len(candidates),
                    "reranker_input": len(reranker_input),
                    "packed_evidence": len(packed_evidence),
                },
            },
        )