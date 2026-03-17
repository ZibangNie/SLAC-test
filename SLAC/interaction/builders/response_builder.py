from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from SLAC.interaction.io.schemas import OpenWebUIMessage, OpenWebUIRequest, OpenWebUIResponse


def _to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj


class ResponseBuilder:
    def build(
        self,
        req: OpenWebUIRequest,
        integration_response: Any,
    ) -> OpenWebUIResponse:
        response_options = dict(req.runtime_config.response_options or {})
        int_resp = _to_dict(integration_response) or {}

        status = str(int_resp.get("status", "error") or "error")
        answer_text = str(int_resp.get("answer_text", "") or "")

        trace = dict(int_resp.get("trace", {}) or {})
        meta = dict(int_resp.get("meta", {}) or {})

        return_integration_response = bool(
            response_options.get("return_integration_response", True)
        )
        return_trace_summary = bool(
            response_options.get("return_trace_summary", True)
        )
        return_display_messages = bool(
            response_options.get("return_display_messages", True)
        )

        display_messages = []
        if return_display_messages:
            display_messages = [
                OpenWebUIMessage(
                    role="assistant",
                    content=answer_text,
                    meta={
                        "request_id": req.request_id,
                        "query_id": req.query_id,
                    },
                )
            ]

        integration_response_obj = int_resp if return_integration_response else None

        trace_summary = None
        if return_trace_summary:
            trace_summary = {
                "candidate_source": trace.get("candidate_source"),
                "num_evidence_selected": trace.get("num_evidence_selected"),
                "retrieval_used": trace.get("retrieval_used"),
                "reranker_used": trace.get("reranker_used"),
                "degraded_to_retrieval": trace.get("degraded_to_retrieval"),
                "llm_response_id": trace.get("llm_response_id"),
                "llm_status": meta.get("llm_status"),
            }

        error = None
        if status == "error":
            error = {
                "stage": meta.get("error_stage"),
                "message": meta.get("error_message") or answer_text or "Unknown error",
            }

        return OpenWebUIResponse(
            schema_version="slac_openwebui_response_v1",
            record_type="openwebui_response",
            status=status,
            request_id=req.request_id,
            session_id=req.session_id,
            query_id=req.query_id,
            answer_text=answer_text,
            display_messages=display_messages,
            integration_response=integration_response_obj,
            trace_summary=trace_summary,
            error=error,
            meta={
                "chat_id": req.openwebui_meta.get("chat_id"),
                "selected_model": req.openwebui_meta.get("selected_model"),
                "stream": req.openwebui_meta.get("stream"),
            },
        )