"""
title: SLAC OpenWebUI Pipe
version: 0.1.0
author: OpenAI
required_open_webui_version: 0.6.0
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from SLAC.openwebui_bridge.service.openwebui_bridge_service import OpenWebUIBridgeService


class Pipe:
    class Valves(BaseModel):
        PIPE_ID: str = Field(default="slac_pipe")
        PIPE_NAME: str = Field(default="SLAC")
        WORKING_DIR: str = Field(default="/root/autodl-tmp/SLAC-test")
        INTEGRATION_RUNNER: str = Field(default="SLAC/integration/run/run_integration_pipeline.py")
        PYTHON_BIN: str = Field(default="python")
        RETRIEVAL_RUN_DIR: str = Field(default="")
        RERANKER_RUN_DIR: str = Field(default="")

        USE_RETRIEVAL: bool = Field(default=True)
        USE_RERANKER: bool = Field(default=True)
        ALLOW_RETRIEVAL_FALLBACK: bool = Field(default=True)
        RETRIEVAL_TOP_K: int = Field(default=40)
        RERANKER_TOP_K: int = Field(default=15)
        MAX_EVIDENCE_ITEMS: int = Field(default=6)
        MAX_EVIDENCE_TOKENS: int = Field(default=1800)

        LLM_PROVIDER: str = Field(default="openai_compatible")
        LLM_MODEL_NAME: str = Field(default="deepseek-chat")
        LLM_API_BASE: str = Field(default="https://api.deepseek.com/v1")
        LLM_API_KEY_ENV: str = Field(default="DEEPSEEK_API_KEY")
        LLM_TEMPERATURE: float = Field(default=0.2)
        LLM_TOP_P: float = Field(default=0.95)
        LLM_MAX_TOKENS: int = Field(default=1024)
        LLM_TIMEOUT_S: int = Field(default=120)

        ANSWER_LANGUAGE: str = Field(default="zh")
        ANSWER_STYLE: str = Field(default="standard")
        REQUIRE_GROUNDING: bool = Field(default=True)
        INSUFFICIENT_EVIDENCE_POLICY: str = Field(default="explicit_uncertainty")

        INCLUDE_INTEGRATION_RESPONSE: bool = Field(default=False)
        INCLUDE_TRACE_SUMMARY: bool = Field(default=True)
        RETURN_FULL_UI_RESPONSE: bool = Field(default=False)
        DEBUG: bool = Field(default=False)

    def __init__(self):
        self.valves = self.Valves()
        self._service: Optional[OpenWebUIBridgeService] = None

    def pipes(self):
        return [{"id": self.valves.PIPE_ID, "name": self.valves.PIPE_NAME}]

    def _build_service(self) -> OpenWebUIBridgeService:
        default_pipeline_config = {
            "use_retrieval": self.valves.USE_RETRIEVAL,
            "use_reranker": self.valves.USE_RERANKER,
            "allow_retrieval_fallback": self.valves.ALLOW_RETRIEVAL_FALLBACK,
            "retrieval_top_k": self.valves.RETRIEVAL_TOP_K,
            "reranker_top_k": self.valves.RERANKER_TOP_K,
            "max_evidence_items": self.valves.MAX_EVIDENCE_ITEMS,
            "max_evidence_tokens": self.valves.MAX_EVIDENCE_TOKENS,
            "llm": {
                "provider": self.valves.LLM_PROVIDER,
                "model_name": self.valves.LLM_MODEL_NAME,
                "api_base": self.valves.LLM_API_BASE,
                "api_key_env": self.valves.LLM_API_KEY_ENV,
                "generation_config": {
                    "temperature": self.valves.LLM_TEMPERATURE,
                    "top_p": self.valves.LLM_TOP_P,
                    "max_tokens": self.valves.LLM_MAX_TOKENS,
                    "timeout_s": self.valves.LLM_TIMEOUT_S,
                },
            },
        }
        default_prompt_hints = {
            "answer_language": self.valves.ANSWER_LANGUAGE,
            "answer_style": self.valves.ANSWER_STYLE,
            "require_grounding": self.valves.REQUIRE_GROUNDING,
            "insufficient_evidence_policy": self.valves.INSUFFICIENT_EVIDENCE_POLICY,
        }
        response_options = {
            "include_integration_response": self.valves.INCLUDE_INTEGRATION_RESPONSE,
            "include_trace_summary": self.valves.INCLUDE_TRACE_SUMMARY,
        }
        return OpenWebUIBridgeService(
            integration_runner=self.valves.INTEGRATION_RUNNER,
            python_bin=self.valves.PYTHON_BIN,
            retrieval_run_dir=self.valves.RETRIEVAL_RUN_DIR or None,
            reranker_run_dir=self.valves.RERANKER_RUN_DIR or None,
            working_dir=self.valves.WORKING_DIR or os.getcwd(),
            default_pipeline_config=default_pipeline_config,
            default_prompt_hints=default_prompt_hints,
            response_options=response_options,
            debug=self.valves.DEBUG,
        )

    @property
    def service(self) -> OpenWebUIBridgeService:
        if self._service is None:
            self._service = self._build_service()
        return self._service

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[dict] = None,
        __request__=None,
        __metadata__: Optional[dict] = None,
        __event_emitter__=None,
        **kwargs,
    ):
        if __event_emitter__ is not None:
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "SLAC bridge 正在处理请求...", "done": False},
                    }
                )
            except Exception:
                pass

        raw_payload = dict(body or {})
        raw_payload["__user__"] = __user__
        raw_payload["__metadata__"] = __metadata__ or {}
        raw_payload.setdefault("openwebui_meta", {})
        if __metadata__:
            raw_payload["openwebui_meta"].update(__metadata__)
        if __user__:
            raw_payload.setdefault("meta", {})
            raw_payload["meta"]["openwebui_user"] = __user__

        ui_response = await self.service.handle(raw_payload)

        if __event_emitter__ is not None:
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"SLAC bridge 处理完成: {ui_response.get('status', 'unknown')}",
                            "done": True,
                        },
                    }
                )
            except Exception:
                pass

        if self.valves.RETURN_FULL_UI_RESPONSE:
            return ui_response
        return ui_response.get("answer_text", "")
