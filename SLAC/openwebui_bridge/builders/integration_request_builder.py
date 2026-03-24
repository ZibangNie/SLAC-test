from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from ..io.schemas import IntegrationRequest, OpenWebUIRequest
from ..parsers.memory_extractor import extract_current_messages_and_memory


DEFAULT_PIPELINE_CONFIG: Dict[str, Any] = {
    "use_retrieval": True,
    "use_reranker": True,
    "allow_retrieval_fallback": True,
    "retrieval_top_k": 40,
    "reranker_top_k": 15,
    "max_evidence_items": 6,
    "max_evidence_tokens": 1800,
    "llm": {
        "provider": "openai_compatible",
        "model_name": "deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "generation_config": {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 1024,
            "timeout_s": 120,
        },
    },
}

DEFAULT_PROMPT_HINTS: Dict[str, Any] = {
    "answer_language": "zh",
    "answer_style": "standard",
    "require_grounding": True,
    "insufficient_evidence_policy": "explicit_uncertainty",
}



def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result



def build_integration_request(
    req: OpenWebUIRequest,
    *,
    default_pipeline_config: Dict[str, Any] | None = None,
    default_prompt_hints: Dict[str, Any] | None = None,
) -> IntegrationRequest:
    current_messages, memory = extract_current_messages_and_memory(req)

    pipeline_config = _deep_merge(default_pipeline_config or DEFAULT_PIPELINE_CONFIG, req.runtime_config.pipeline_overrides)
    prompt_hints = _deep_merge(default_prompt_hints or DEFAULT_PROMPT_HINTS, req.runtime_config.prompt_hints_overrides)

    context = {
        "doc_scope": req.context.doc_scope,
        "expected_doc_hint": req.context.expected_doc_hint,
        "domain": req.context.domain,
        "source_type": req.context.source_type,
        "uploaded_file_refs": [str(f.get("id") or f.get("name") or f) for f in req.context.files],
    }

    meta = {
        **req.meta,
        "openwebui_meta": req.openwebui_meta,
        "runtime_config": {
            "retrieval_run_dir": req.runtime_config.retrieval_run_dir,
            "reranker_run_dir": req.runtime_config.reranker_run_dir,
            "working_dir": req.runtime_config.working_dir,
        },
    }

    return IntegrationRequest(
        request_id=req.request_id,
        session_id=req.session_id,
        query_id=req.query_id,
        query_text=req.query_text,
        messages=current_messages,
        memory=memory,
        context=context,
        pipeline_config=pipeline_config,
        prompt_hints=prompt_hints,
        meta=meta,
    )
