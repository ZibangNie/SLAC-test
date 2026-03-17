from __future__ import annotations

from typing import Any, Dict, Optional

from SLAC.llm.clients.registry import get_client
from SLAC.llm.io.schemas import (
    ANSWER_RECORD_TYPE,
    ANSWER_SCHEMA_VERSION,
    LLMAnswerResult,
    LLMRequest,
    UsageInfo,
    build_error_result,
)
from SLAC.llm.io.validators import ValidationError, validate_llm_request
from SLAC.llm.service.request_compiler import compile_provider_payload


class LLMService:
    def invoke(self, req: LLMRequest) -> LLMAnswerResult:
        try:
            validate_llm_request(req)
            payload = compile_provider_payload(req)
            client = get_client(
                req.provider,
                api_base=req.api_base,
                api_key_env=req.api_key_env,
                timeout_s=req.generation_config.timeout_s,
            )
            raw = client.chat(payload)
            return self._normalize_provider_response(raw, req)
        except ValidationError:
            raise
        except Exception as exc:
            return build_error_result(
                req,
                error_message=str(exc),
                error_type=exc.__class__.__name__,
            )

    def _normalize_provider_response(
        self,
        raw: Dict[str, Any],
        req: LLMRequest,
    ) -> LLMAnswerResult:
        choices = raw.get("choices") or []
        if not choices:
            return build_error_result(
                req,
                error_message="provider response does not contain choices",
                response_id=raw.get("id"),
                raw_response=raw,
                error_type="ProviderResponseFormatError",
            )

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        answer_text = _extract_answer_text(message)

        usage = UsageInfo.from_dict(raw.get("usage"))
        finish_reason = first_choice.get("finish_reason")

        answer = LLMAnswerResult(
            schema_version=ANSWER_SCHEMA_VERSION,
            record_type=ANSWER_RECORD_TYPE,
            status="ok",
            request_id=req.request_id,
            session_id=req.session_id,
            query_id=req.query_id,
            provider=req.provider,
            model_name=req.model_name,
            response_id=raw.get("id"),
            answer_text=answer_text,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            usage=usage,
            memory_used=bool(req.memory and (req.memory.messages or req.memory.summary_text)),
            memory_message_count=len(req.memory.messages) if req.memory else 0,
            evidence_count=len(req.evidence),
            evidence_refs=[
                {
                    "chunk_id": ev.chunk_id,
                    "doc_id": ev.doc_id,
                    "rerank_rank": ev.rerank_rank,
                }
                for ev in req.evidence
            ],
            raw_response=raw if req.options.get("return_raw_response") else None,
            meta={},
        )
        return answer


def _extract_answer_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
        return "\n".join(p for p in parts if p).strip()
    return str(content)
