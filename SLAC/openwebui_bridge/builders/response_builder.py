from __future__ import annotations

from typing import Any, Dict

from ..io.schemas import BridgeMessage, OpenWebUIResponse



def _build_trace_summary(integration_response: Dict[str, Any]) -> Dict[str, Any]:
    trace = dict(integration_response.get("trace") or {})
    return {
        "retrieval_used": trace.get("retrieval_used"),
        "reranker_used": trace.get("reranker_used"),
        "degraded_to_retrieval": trace.get("degraded_to_retrieval"),
        "num_candidates_read": trace.get("num_candidates_read"),
        "num_evidence_selected": trace.get("num_evidence_selected"),
        "llm_request_id": trace.get("llm_request_id"),
        "llm_response_id": trace.get("llm_response_id"),
        "llm_status": (integration_response.get("answer_result") or {}).get("status"),
    }



def build_openwebui_response(
    integration_response: Dict[str, Any],
    *,
    include_integration_response: bool = False,
    include_trace_summary: bool = True,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    status = integration_response.get("status", "error")
    answer_text = str(integration_response.get("answer_text") or "")
    request_id = integration_response.get("request_id")
    session_id = integration_response.get("session_id")
    query_id = integration_response.get("query_id")

    resp = OpenWebUIResponse(
        status=status,
        request_id=request_id,
        session_id=session_id,
        query_id=query_id,
        answer_text=answer_text,
        display_messages=[BridgeMessage(role="assistant", content=answer_text)],
        integration_response=integration_response if include_integration_response else None,
        trace_summary=_build_trace_summary(integration_response) if include_trace_summary else {},
        error=integration_response.get("error"),
        meta=meta or {},
    )
    return resp.dict(exclude_none=True)
