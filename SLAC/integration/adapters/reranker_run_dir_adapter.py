from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from SLAC.integration.io.schemas import (
    IntegrationRequest,
    RetrievalArtifacts,
    RerankerArtifacts,
)


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


class RerankerRunDirAdapter:
    """
    Read reranker artifacts from an existing reranker run directory.

    Expected files under reranker_run_dir:
      - pack_bridge/{query_id}.for_packing.jsonl
      - queries/{query_id}.reranked_candidates.jsonl
    """

    def run_or_read(
        self,
        req: IntegrationRequest,
        retrieval_artifacts: RetrievalArtifacts,
    ) -> RerankerArtifacts:
        # retrieval_artifacts is accepted to match FinalIntegrator's adapter contract.
        _ = retrieval_artifacts

        run_dir = str(req.context.reranker_run_dir or "").strip()
        query_id = str(req.query_id or "").strip()

        if not run_dir:
            raise RuntimeError(
                "context.reranker_run_dir is required for RerankerRunDirAdapter."
            )
        if not query_id:
            raise RuntimeError("query_id is required for RerankerRunDirAdapter.")

        base = Path(run_dir)
        if not base.exists():
            raise RuntimeError(f"reranker_run_dir not found: {base}")
        if not base.is_dir():
            raise RuntimeError(f"reranker_run_dir is not a directory: {base}")

        pack_bridge_path = base / "pack_bridge" / f"{query_id}.for_packing.jsonl"
        reranked_candidates_path = base / "queries" / f"{query_id}.reranked_candidates.jsonl"

        pack_bridge = _read_jsonl_file(pack_bridge_path)
        reranked_candidates = _read_jsonl_file(reranked_candidates_path)

        if not pack_bridge and not reranked_candidates:
            raise RuntimeError(
                "no reranker artifacts found under "
                f"{base} for query_id={query_id} "
                f"(expected one or more of: "
                f"{pack_bridge_path}, {reranked_candidates_path})"
            )

        return RerankerArtifacts(
            pack_bridge=pack_bridge,
            reranked_candidates=reranked_candidates,
            meta={
                "source": "reranker_run_dir_adapter",
                "reranker_run_dir": str(base),
                "query_id": query_id,
                "resolved_files": {
                    "pack_bridge": str(pack_bridge_path),
                    "reranked_candidates": str(reranked_candidates_path),
                },
                "counts": {
                    "pack_bridge": len(pack_bridge),
                    "reranked_candidates": len(reranked_candidates),
                },
            },
        )