from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from SLAC.openwebui_bridge.openwebui_pipe import Pipe as LocalPipe
from SLAC.openwebui_bridge.upload_ingest.upload_ingest_service import UploadIngestService

logger = logging.getLogger("slac.openwebui_bridge.server_api")
logging.basicConfig(
    level=os.environ.get("SLAC_BRIDGE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_env_value(raw: str, current_value: Any) -> Any:
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
    pipe = LocalPipe()
    model_fields = getattr(pipe.valves.__class__, "model_fields", None)
    if model_fields is None:
        model_fields = getattr(pipe.valves.__class__, "__fields__", {})
    for field_name in model_fields.keys():
        if field_name in os.environ:
            current_value = getattr(pipe.valves, field_name)
            try:
                setattr(pipe.valves, field_name, _coerce_env_value(os.environ[field_name], current_value))
                logger.info("Applied env override for valve: %s", field_name)
            except Exception as e:
                logger.warning("Failed to apply env override for valve %s: %s", field_name, e)
    return pipe


def _extract_query_text_from_payload(raw_payload: Dict[str, Any]) -> str:
    messages = raw_payload.get("messages") or []
    for item in reversed(messages):
        if item.get("role") != "user":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
            return "\n".join([x for x in texts if x]).strip()
    return str(raw_payload.get("query_text", "")).strip()


LOCAL_PIPE = _build_local_pipe_from_env()
SERVICE = LOCAL_PIPE.service

UPLOAD_INGEST = UploadIngestService(
    project_root=os.environ.get("WORKING_DIR", "/root/autodl-tmp/SLAC-test"),
    work_root=os.environ.get("UPLOAD_WORK_ROOT", "/root/autodl-tmp/data/openwebui_upload_workspace"),
    python_bin=os.environ.get("PYTHON_BIN", "python"),
    refiner_config=os.environ.get(
        "REFINER_CONFIG",
        "/root/autodl-tmp/SLAC-test/SLAC/refiner/pipeline/configs/pipeline_config.yaml",
    ),
    retrieval_config=os.environ.get(
        "RETRIEVAL_CONFIG",
        "/root/autodl-tmp/SLAC-test/SLAC/retrieval/configs/retrieval_config.yaml",
    ),
    reranker_config=os.environ.get(
        "RERANKER_CONFIG",
        "/root/autodl-tmp/SLAC-test/SLAC/reranker/configs/reranker_config.yaml",
    ),
    bilingual_terms_path=os.environ.get(
        "BILINGUAL_TERMS_PATH",
        "/root/autodl-tmp/SLAC-test/SLAC/retrieval/configs/bilingual_terms.yaml",
    ),
    default_domain=os.environ.get("DEFAULT_DOMAIN", "rail"),
    debug=os.environ.get("DEBUG", "false").lower() == "true",
)

APP_TITLE = os.environ.get("SLAC_BRIDGE_APP_TITLE", "SLAC OpenWebUI Bridge API")
APP_VERSION = os.environ.get("SLAC_BRIDGE_APP_VERSION", "1.1.0")
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
    if not API_TOKEN:
        return
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header.")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header must use Bearer token.")
    token = authorization[len(prefix):].strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token.")


def _extract_answer_text(ui_response: Dict[str, Any]) -> str:
    if not isinstance(ui_response, dict):
        return str(ui_response)
    candidates = [
        ui_response.get("answer_text"),
        ui_response.get("text"),
        ui_response.get("content"),
    ]
    answer_obj = ui_response.get("answer")
    if isinstance(answer_obj, dict):
        candidates.extend([answer_obj.get("text"), answer_obj.get("answer_text"), answer_obj.get("content")])
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


@app.post("/sessions/{session_id}/files")
async def upload_session_files(
    session_id: str,
    files: List[UploadFile] = File(...),
    domain: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
):
    _check_auth(authorization)

    prepared: List[Dict[str, Any]] = []
    for f in files:
        prepared.append(
            {
                "name": f.filename or "uploaded_file",
                "content_type": f.content_type,
                "content": await f.read(),
            }
        )

    result = UPLOAD_INGEST.save_uploaded_files(session_id, prepared)
    result["domain"] = domain
    result["time"] = _utc_now_iso()
    return result


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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid JSON body: {e}")

    if not isinstance(raw_payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body must be a JSON object.")

    session_id = (
        raw_payload.get("session_id")
        or raw_payload.get("chat_id")
        or (raw_payload.get("openwebui_meta") or {}).get("chat_id")
        or f"sess_{uuid.uuid4().hex[:12]}"
    )
    raw_payload["session_id"] = session_id

    raw_payload.setdefault("meta", {})
    raw_payload["meta"]["http_request_id"] = request_id
    raw_payload["meta"]["http_received_at"] = _utc_now_iso()
    raw_payload["meta"]["http_client_host"] = request.client.host if request.client else None

    query_text = _extract_query_text_from_payload(raw_payload)
    query_id = raw_payload.get("query_id") or f"qt_{uuid.uuid4().hex[:12]}"
    raw_payload["query_id"] = query_id

    # 若当前会话已经上传过文件，则先为当前 query 准备 retrieval/reranker 产物
    try:
        prep = None
        try:
            prep = UPLOAD_INGEST.prepare_query_runs(
                session_id=session_id,
                query_id=query_id,
                query_text=query_text,
                domain=((raw_payload.get("context") or {}).get("domain") or os.environ.get("DEFAULT_DOMAIN", "rail")),
            )
        except RuntimeError as e:
            # 没有上传文件时，不强行报错；继续走默认 retrieval/reranker run
            logger.info("No session-specific uploaded assets used. session_id=%s reason=%s", session_id, e)

        if prep:
            raw_payload.setdefault("runtime_config", {})
            raw_payload["runtime_config"]["retrieval_run_dir"] = prep["retrieval_run_dir"]
            raw_payload["runtime_config"]["reranker_run_dir"] = prep["reranker_run_dir"]
            raw_payload["runtime_config"]["working_dir"] = str(LOCAL_PIPE.valves.WORKING_DIR)

            raw_payload.setdefault("context", {})
            if prep.get("doc_scope"):
                raw_payload["context"]["doc_scope"] = prep["doc_scope"]

            raw_payload.setdefault("meta", {})
            raw_payload["meta"]["session_asset_version"] = prep.get("asset_version")
            raw_payload["meta"]["session_bound_retrieval_run_dir"] = prep["retrieval_run_dir"]
            raw_payload["meta"]["session_bound_reranker_run_dir"] = prep["reranker_run_dir"]

        ui_response = await SERVICE.handle(raw_payload)

        if not isinstance(ui_response, dict):
            ui_response = {"status": "ok", "answer_text": str(ui_response)}

        if isinstance(ui_response, dict):
            logger.info(
                "Bridge ui_response request_id=%s body=%s",
                request_id,
                json.dumps(ui_response, ensure_ascii=False, default=str),
            )

        ui_response.setdefault("status", "ok")
        ui_response.setdefault("answer_text", _extract_answer_text(ui_response))
        ui_response.setdefault("request_id", request_id)
        ui_response.setdefault("session_id", session_id)
        ui_response.setdefault("query_id", query_id)
        ui_response.setdefault("server_time", _utc_now_iso())
        ui_response.setdefault("latency_ms", int((time.perf_counter() - t0) * 1000))

        return JSONResponse(content=ui_response, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat request failed. request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "request_id": request_id,
                "session_id": session_id,
                "query_id": query_id,
                "server_time": _utc_now_iso(),
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "answer_text": "",
                "error": {
                    "type": e.__class__.__name__,
                    "message": str(e),
                },
            },
        )