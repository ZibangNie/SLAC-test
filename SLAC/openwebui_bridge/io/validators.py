from __future__ import annotations

from typing import Any, Dict, Union

from .schemas import IntegrationRequest, OpenWebUIRequest


def validate_openwebui_request(
    data: Union[Dict[str, Any], OpenWebUIRequest]
) -> OpenWebUIRequest:
    if isinstance(data, OpenWebUIRequest):
        req = data
    else:
        req = OpenWebUIRequest(**data)

    if not req.query_text or not req.query_text.strip():
        raise ValueError("query_text 不能为空")
    return req


def validate_integration_request(
    data: Union[Dict[str, Any], IntegrationRequest]
) -> IntegrationRequest:
    if isinstance(data, IntegrationRequest):
        req = data
    else:
        req = IntegrationRequest(**data)

    if not req.query_text or not req.query_text.strip():
        raise ValueError("integration request 的 query_text 不能为空")
    if not req.pipeline_config:
        raise ValueError("integration request 缺少 pipeline_config")
    return req