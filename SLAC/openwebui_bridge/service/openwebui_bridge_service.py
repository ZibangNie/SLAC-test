from __future__ import annotations

from typing import Any, Dict, Optional

from ..adapters.integration_adapter import IntegrationAdapter
from ..builders.integration_request_builder import (
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_PROMPT_HINTS,
    build_integration_request,
)
from ..builders.response_builder import build_openwebui_response
from ..io.validators import validate_integration_request, validate_openwebui_request
from ..parsers.pipe_request_parser import parse_pipe_payload


class OpenWebUIBridgeService:
    def __init__(
        self,
        *,
        integration_runner: str,
        python_bin: Optional[str] = None,
        retrieval_run_dir: Optional[str] = None,
        reranker_run_dir: Optional[str] = None,
        working_dir: Optional[str] = None,
        default_pipeline_config: Optional[Dict[str, Any]] = None,
        default_prompt_hints: Optional[Dict[str, Any]] = None,
        response_options: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> None:
        self.default_pipeline_config = default_pipeline_config or DEFAULT_PIPELINE_CONFIG
        self.default_prompt_hints = default_prompt_hints or DEFAULT_PROMPT_HINTS
        self.response_options = response_options or {}
        self.adapter = IntegrationAdapter(
            integration_runner=integration_runner,
            python_bin=python_bin,
            retrieval_run_dir=retrieval_run_dir,
            reranker_run_dir=reranker_run_dir,
            working_dir=working_dir,
            debug=debug,
        )
        self.debug = debug

    async def handle(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            openwebui_req = validate_openwebui_request(parse_pipe_payload(raw_payload))
            integration_req = validate_integration_request(
                build_integration_request(
                    openwebui_req,
                    default_pipeline_config=self.default_pipeline_config,
                    default_prompt_hints=self.default_prompt_hints,
                ).dict(exclude_none=True)
            )

            if openwebui_req.runtime_config.retrieval_run_dir:
                self.adapter.retrieval_run_dir = openwebui_req.runtime_config.retrieval_run_dir
            if openwebui_req.runtime_config.reranker_run_dir:
                self.adapter.reranker_run_dir = openwebui_req.runtime_config.reranker_run_dir
            if openwebui_req.runtime_config.working_dir:
                self.adapter.working_dir = openwebui_req.runtime_config.working_dir

            integration_response = await self.adapter.invoke(integration_req.dict(exclude_none=True))

            return build_openwebui_response(
                integration_response,
                include_integration_response=bool(
                    openwebui_req.runtime_config.response_options.get("include_integration_response")
                    if openwebui_req.runtime_config.response_options
                    else self.response_options.get("include_integration_response", False)
                ),
                include_trace_summary=bool(
                    openwebui_req.runtime_config.response_options.get("include_trace_summary", True)
                    if openwebui_req.runtime_config.response_options
                    else self.response_options.get("include_trace_summary", True)
                ),
                meta={
                    "chat_id": openwebui_req.openwebui_meta.get("metadata", {}).get("chat_id")
                    or openwebui_req.openwebui_meta.get("chat_id"),
                    "selected_model": openwebui_req.openwebui_meta.get("selected_model"),
                },
            )
        except Exception as e:
            return {
                "schema_version": "slac_openwebui_response_v1",
                "record_type": "openwebui_response",
                "status": "error",
                "request_id": raw_payload.get("request_id", "unknown_request"),
                "session_id": (raw_payload.get("__metadata__") or {}).get("chat_id")
                or raw_payload.get("session_id"),
                "query_id": raw_payload.get("query_id"),
                "answer_text": "SLAC OpenWebUI bridge 执行失败，请检查后端日志。",
                "display_messages": [
                    {"role": "assistant", "content": "SLAC OpenWebUI bridge 执行失败，请检查后端日志。"}
                ],
                "trace_summary": {},
                "error": {"type": e.__class__.__name__, "message": str(e)},
                "meta": {},
            }
