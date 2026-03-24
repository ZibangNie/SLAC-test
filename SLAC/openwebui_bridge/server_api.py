from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from SLAC.openwebui_bridge.openwebui_pipe import Pipe as LocalPipe

logger = logging.getLogger("slac.openwebui_bridge.server_api")
logging.basicConfig(
    level=os.environ.get("SLAC_BRIDGE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_env_value(raw: str, current_value: Any) -> Any:
    """
    按当前字段类型做尽量稳妥的环境变量类型转换。
    """
    if isinstance(current_value, bool):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw)
    if isinstance(current_value, float):
        return float(raw)
    if isinstance(current_value, (dict, list)):
        return json.loads(raw)
    return raw


def _build_local_pipe_from_env() -> LocalPipe:
    """
    复用现有 openwebui_pipe.py 的配置结构。
    这样 HTTP 入口和 Pipe 入口共用同一套默认参数与 service 构造逻辑。
    """
    pipe = LocalPipe()

    model_fields = getattr(pipe.valves.__class__, "model_fields", None)
    if model_fields is None:
        model_fields = getattr(pipe.valves.__class__, "__fields__", {})

    for field_name in model_fields.keys():
        if field_name in os.environ:
            current_value = getattr(pipe.valves, field_name)
            try:
                setattr(
                    pipe.valves,
                    field_name,
                    _coerce_env_value(os.environ[field_name], current_value),
                )
                logger.info("Applied env override for valve: %s", field_name)
            except Exception as e:
                logger.warning(
                    "Failed to apply env override for valve %s: %s",
                    field_name,
                    e,
                )

    return pipe


LOCAL_PIPE = _build_local_pipe_from_env()
SERVICE = LOCAL_PIPE.service

APP_TITLE = os.environ.get("SLAC_BRIDGE_APP_TITLE", "SLAC OpenWebUI Bridge API")
APP_VERSION = os.environ.get("SLAC_BRIDGE_APP_VERSION", "1.0.0")
API_TOKEN = os.environ.get("SLAC_BRIDGE_API_TOKEN", "").strip()

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

allow_origins_raw = os.environ.get("SLAC_BRIDGE_ALLOW_ORIGINS", "*").strip()
allow_origins = ["*"] if allow_origins_raw == "*" else [x.strip() for x in allow_origins_raw.split(",") if x.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_auth(authorization: Optional[str]) -> None:
    """
    若设置了 SLAC_BRIDGE_API_TOKEN，则要求 Bearer Token。
    未设置则默认不开鉴权，方便你先联调。
    """
    if not API_TOKEN:
        return

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
        )

    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must use Bearer token.",
        )

    token = authorization[len(prefix):].strip()
    if token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token.",
        )


def _safe_config_summary() -> Dict[str, Any]:
    """
    返回不含敏感信息的配置摘要，便于排查。
    """
    valves = LOCAL_PIPE.valves
    api_key_env_name = getattr(valves, "LLM_API_KEY_ENV", "")
    api_key_present = bool(api_key_env_name and os.environ.get(api_key_env_name, "").strip())

    return {
        "pipe_id": getattr(valves, "PIPE_ID", ""),
        "pipe_name": getattr(valves, "PIPE_NAME", ""),
        "working_dir": getattr(valves, "WORKING_DIR", ""),
        "integration_runner": getattr(valves, "INTEGRATION_RUNNER", ""),
        "retrieval_run_dir": getattr(valves, "RETRIEVAL_RUN_DIR", ""),
        "reranker_run_dir": getattr(valves, "RERANKER_RUN_DIR", ""),
        "use_retrieval": getattr(valves, "USE_RETRIEVAL", None),
        "use_reranker": getattr(valves, "USE_RERANKER", None),
        "llm_provider": getattr(valves, "LLM_PROVIDER", ""),
        "llm_model_name": getattr(valves, "LLM_MODEL_NAME", ""),
        "llm_api_base": getattr(valves, "LLM_API_BASE", ""),
        "llm_api_key_env": api_key_env_name,
        "llm_api_key_present": api_key_present,
        "answer_language": getattr(valves, "ANSWER_LANGUAGE", ""),
        "answer_style": getattr(valves, "ANSWER_STYLE", ""),
        "debug": getattr(valves, "DEBUG", False),
    }


def _extract_answer_text(ui_response: Dict[str, Any]) -> str:
    """
    统一兜底提取答案文本，兼容不同返回结构。
    """
    if not isinstance(ui_response, dict):
        return str(ui_response)

    candidates = [
        ui_response.get("answer_text"),
        ui_response.get("text"),
        ui_response.get("content"),
    ]

    answer_obj = ui_response.get("answer")
    if isinstance(answer_obj, dict):
        candidates.extend(
            [
                answer_obj.get("text"),
                answer_obj.get("answer_text"),
                answer_obj.get("content"),
            ]
        )

    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()

    return ""


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "slac_openwebui_bridge",
        "version": APP_VERSION,
        "time": _utc_now_iso(),
    }


@app.get("/info")
async def info(
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _check_auth(authorization)
    return {
        "ok": True,
        "service": "slac_openwebui_bridge",
        "version": APP_VERSION,
        "time": _utc_now_iso(),
        "config": _safe_config_summary(),
    }


@app.post("/chat")
async def chat(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    _check_auth(authorization)

    request_id = request.headers.get("x-request-id") or f"slac-{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()

    try:
        raw_payload = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON body: {e}",
        )

    if not isinstance(raw_payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request body must be a JSON object.",
        )

    raw_payload.setdefault("meta", {})
    raw_payload["meta"]["http_request_id"] = request_id
    raw_payload["meta"]["http_received_at"] = _utc_now_iso()
    raw_payload["meta"]["http_client_host"] = request.client.host if request.client else None

    try:
        ui_response = await SERVICE.handle(raw_payload)

        if not isinstance(ui_response, dict):
            ui_response = {
                "status": "ok",
                "answer_text": str(ui_response),
            }

        ui_response.setdefault("status", "ok")
        ui_response.setdefault("answer_text", _extract_answer_text(ui_response))
        ui_response.setdefault("request_id", request_id)
        ui_response.setdefault("server_time", _utc_now_iso())
        ui_response.setdefault("latency_ms", int((time.perf_counter() - t0) * 1000))

        return JSONResponse(content=ui_response, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.exception("Chat request failed. request_id=%s", request_id)

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "request_id": request_id,
                "server_time": _utc_now_iso(),
                "latency_ms": latency_ms,
                "answer_text": "",
                "error": {
                    "type": e.__class__.__name__,
                    "message": str(e),
                },
            },
        )