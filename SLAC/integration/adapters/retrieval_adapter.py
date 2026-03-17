from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from SLAC.integration.io.readers import (
    maybe_read_jsonl,
    resolve_query_key,
    resolve_retrieval_paths,
)
from SLAC.integration.io.schemas import IntegrationRequest, RetrievalArtifacts


class RetrievalAdapter:
    """
    当前实现优先“读已有 retrieval 产物”。
    如需自动执行 retrieval，可在 context.meta 中提供 retrieval_runner_command：
      - 字符串：支持 format 占位符
      - list[str]：每个元素都支持 format 占位符
    可用占位符：
      {query_id} {request_id} {query_text} {retrieval_run_dir} {working_dir}
    """

    def _format_command(self, template: Any, req: IntegrationRequest) -> list[str]:
        ctx = {
            "query_id": req.query_id or "",
            "request_id": req.request_id,
            "query_text": req.query_text,
            "retrieval_run_dir": req.context.retrieval_run_dir or "",
            "working_dir": req.context.working_dir or "",
        }
        if isinstance(template, str):
            return shlex.split(template.format(**ctx))
        if isinstance(template, list):
            return [str(x).format(**ctx) for x in template]
        raise ValueError("retrieval_runner_command must be str or list[str].")

    def _maybe_run_retrieval(self, req: IntegrationRequest) -> None:
        cmd_tpl = req.context.meta.get("retrieval_runner_command")
        if not cmd_tpl:
            return

        cmd = self._format_command(cmd_tpl, req)
        subprocess.run(
            cmd,
            cwd=req.context.working_dir or None,
            check=True,
            text=True,
        )

    def run_or_read(self, req: IntegrationRequest) -> RetrievalArtifacts:
        if req.context.retrieval_artifacts:
            data = req.context.retrieval_artifacts
            return RetrievalArtifacts(
                candidates=list(data.get("candidates", []) or []),
                reranker_input=list(data.get("reranker_input", []) or []),
                packed_evidence=list(data.get("packed_evidence", []) or []),
                meta=dict(data.get("meta", {}) or {}),
            )

        if not req.context.retrieval_run_dir:
            raise RuntimeError(
                "context.retrieval_run_dir is required when retrieval_artifacts are not provided."
            )

        query_key = resolve_query_key(req.query_id, req.request_id)
        paths = resolve_retrieval_paths(req.context.retrieval_run_dir, query_key)

        if not any(paths.values()):
            self._maybe_run_retrieval(req)
            paths = resolve_retrieval_paths(req.context.retrieval_run_dir, query_key)

        candidates = maybe_read_jsonl(paths["candidates"])
        reranker_input = maybe_read_jsonl(paths["reranker_input"])
        packed_evidence = maybe_read_jsonl(paths["packed_evidence"])

        top_k = req.pipeline_config.retrieval_top_k
        if top_k > 0:
            candidates = candidates[:top_k]
            reranker_input = reranker_input[:top_k]

        return RetrievalArtifacts(
            candidates=candidates,
            reranker_input=reranker_input,
            packed_evidence=packed_evidence,
            meta={
                "retrieval_run_dir": req.context.retrieval_run_dir,
                "resolved_paths": {k: (str(v) if v else None) for k, v in paths.items()},
            },
        )