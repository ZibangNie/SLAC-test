from __future__ import annotations

import shlex
import subprocess
from typing import Any

from SLAC.integration.io.readers import (
    maybe_read_jsonl,
    resolve_query_key,
    resolve_reranker_paths,
)
from SLAC.integration.io.schemas import IntegrationRequest, RetrievalArtifacts, RerankerArtifacts


class RerankerAdapter:
    """
    当前实现优先“读已有 reranker 产物”。
    如需自动执行 reranker，可在 context.meta 中提供 reranker_runner_command：
      - 字符串：支持 format 占位符
      - list[str]：每个元素都支持 format 占位符
    可用占位符：
      {query_id} {request_id} {query_text}
      {retrieval_run_dir} {reranker_run_dir} {working_dir}
    """

    def _format_command(self, template: Any, req: IntegrationRequest) -> list[str]:
        ctx = {
            "query_id": req.query_id or "",
            "request_id": req.request_id,
            "query_text": req.query_text,
            "retrieval_run_dir": req.context.retrieval_run_dir or "",
            "reranker_run_dir": req.context.reranker_run_dir or "",
            "working_dir": req.context.working_dir or "",
        }
        if isinstance(template, str):
            return shlex.split(template.format(**ctx))
        if isinstance(template, list):
            return [str(x).format(**ctx) for x in template]
        raise ValueError("reranker_runner_command must be str or list[str].")

    def _maybe_run_reranker(
        self,
        req: IntegrationRequest,
        retrieval_artifacts: RetrievalArtifacts,
    ) -> None:
        cmd_tpl = req.context.meta.get("reranker_runner_command")
        if not cmd_tpl:
            return

        cmd = self._format_command(cmd_tpl, req)
        subprocess.run(
            cmd,
            cwd=req.context.working_dir or None,
            check=True,
            text=True,
        )

    def run_or_read(
        self,
        req: IntegrationRequest,
        retrieval_artifacts: RetrievalArtifacts,
    ) -> RerankerArtifacts:
        if req.context.reranker_artifacts:
            data = req.context.reranker_artifacts
            return RerankerArtifacts(
                pack_bridge=list(data.get("pack_bridge", []) or []),
                reranked_candidates=list(data.get("reranked_candidates", []) or []),
                meta=dict(data.get("meta", {}) or {}),
            )

        if not req.context.reranker_run_dir:
            raise RuntimeError(
                "context.reranker_run_dir is required when reranker_artifacts are not provided."
            )

        query_key = resolve_query_key(req.query_id, req.request_id)
        paths = resolve_reranker_paths(req.context.reranker_run_dir, query_key)

        if not any(paths.values()):
            self._maybe_run_reranker(req, retrieval_artifacts)
            paths = resolve_reranker_paths(req.context.reranker_run_dir, query_key)

        pack_bridge = maybe_read_jsonl(paths["pack_bridge"])
        reranked_candidates = maybe_read_jsonl(paths["reranked_candidates"])

        top_k = req.pipeline_config.reranker_top_k
        if top_k > 0:
            pack_bridge = pack_bridge[:top_k]
            reranked_candidates = reranked_candidates[:top_k]

        return RerankerArtifacts(
            pack_bridge=pack_bridge,
            reranked_candidates=reranked_candidates,
            meta={
                "reranker_run_dir": req.context.reranker_run_dir,
                "resolved_paths": {k: (str(v) if v else None) for k, v in paths.items()},
            },
        )