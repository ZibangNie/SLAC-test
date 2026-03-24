from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


class IntegrationAdapter:
    """Call the existing Integration layer.

    Default strategy: invoke the already validated CLI runner via subprocess.
    This is the safest choice for the current "先跑通" phase.
    """

    def __init__(
        self,
        *,
        integration_runner: str,
        python_bin: Optional[str] = None,
        retrieval_run_dir: Optional[str] = None,
        reranker_run_dir: Optional[str] = None,
        working_dir: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        self.integration_runner = integration_runner
        self.python_bin = python_bin or sys.executable
        self.retrieval_run_dir = retrieval_run_dir
        self.reranker_run_dir = reranker_run_dir
        self.working_dir = working_dir
        self.debug = debug

    async def invoke(self, integration_request: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self._invoke_sync, integration_request)

    def _invoke_sync(self, integration_request: Dict[str, Any]) -> Dict[str, Any]:
        workdir = self.working_dir or os.getcwd()
        runner_path = Path(self.integration_runner)
        if not runner_path.is_absolute():
            runner_path = Path(workdir) / runner_path

        if not runner_path.exists():
            raise FileNotFoundError(f"Integration runner 不存在: {runner_path}")

        with tempfile.TemporaryDirectory(prefix="slac_openwebui_bridge_") as tmpdir:
            tmp = Path(tmpdir)
            requests_path = tmp / "requests.jsonl"
            out_dir = tmp / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            with requests_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(integration_request, ensure_ascii=False) + "\n")

            cmd = [
                self.python_bin,
                str(runner_path),
                "--requests",
                str(requests_path),
                "--out_dir",
                str(out_dir),
            ]
            if self.retrieval_run_dir:
                cmd += ["--retrieval_run_dir", self.retrieval_run_dir]
            if self.reranker_run_dir:
                cmd += ["--reranker_run_dir", self.reranker_run_dir]
            if workdir:
                cmd += ["--working_dir", workdir]

            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    "Integration runner 执行失败\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )

            responses_dir = out_dir / "responses"
            if not responses_dir.exists():
                raise FileNotFoundError(f"Integration 输出目录缺失: {responses_dir}")

            json_files = sorted(responses_dir.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"Integration 未生成 response json: {responses_dir}")

            response_path = json_files[0]
            with response_path.open("r", encoding="utf-8") as f:
                response = json.load(f)

            if self.debug:
                response.setdefault("meta", {})["bridge_debug"] = {
                    "runner": str(runner_path),
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            return response
