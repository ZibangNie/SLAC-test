from __future__ import annotations

from typing import Any, Dict

from .schemas import IntegrationRequest, OpenWebUIRequest


def validate_openwebui_request(data: Dict[str, Any]) -> OpenWebUIRequest:
    req = OpenWebUIRequest(**data)
    if not req.query_text or not req.query_text.strip():
        raise ValueError("query_text 不能为空")
    return req



def validate_integration_request(data: Dict[str, Any]) -> IntegrationRequest:
    req = IntegrationRequest(**data)
    if not req.query_text or not req.query_text.strip():
        raise ValueError("integration request 的 query_text 不能为空")
    if not req.pipeline_config:
        raise ValueError("integration request 缺少 pipeline_config")
    return req
