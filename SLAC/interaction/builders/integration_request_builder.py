from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from SLAC.integration.io.schemas import (
    ChatMessage,
    ConversationMemory,
    IntegrationContext,
    IntegrationRequest,
    PipelineConfig,
    PromptHints,
)
from SLAC.interaction.io.schemas import ExtractedConversation, OpenWebUIRequest


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for k, v in override.items():
        if (
            k in merged
            and isinstance(merged[k], dict)
            and isinstance(v, dict)
        ):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = deepcopy(v)
    return merged


def _default_pipeline_config() -> Dict[str, Any]:
    return {
        "use_retrieval": True,
        "use_reranker": True,
        "allow_retrieval_fallback": True,
        "retrieval_top_k": 40,
        "reranker_top_k": 15,
        "max_evidence_items": 6,
        "max_evidence_tokens": 1800,
        "prefer_direct_first": True,
        "min_direct_evidence": 1,
        "llm": {
            "provider": "openai_compatible",
            "model_name": "deepseek-chat",
            "api_base": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 1024,
            "timeout_s": 120,
        },
    }


def _default_prompt_hints() -> Dict[str, Any]:
    return {
        "answer_language": "zh",
        "require_grounding": True,
        "answer_style": "concise",
        "insufficient_evidence_policy": "state_insufficiency",
    }


class IntegrationRequestBuilder:
    def __init__(
        self,
        *,
        default_pipeline_config: Optional[Dict[str, Any]] = None,
        default_prompt_hints: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.default_pipeline_config = default_pipeline_config or _default_pipeline_config()
        self.default_prompt_hints = default_prompt_hints or _default_prompt_hints()

    def build(
        self,
        req: OpenWebUIRequest,
        extracted: ExtractedConversation,
    ) -> IntegrationRequest:
        runtime = req.runtime_config

        pipeline_config_dict = _deep_merge(
            self.default_pipeline_config,
            runtime.pipeline_overrides,
        )
        prompt_hints_dict = _deep_merge(
            self.default_prompt_hints,
            runtime.prompt_hints_overrides,
        )

        integration_context = IntegrationContext(
            retrieval_run_dir=runtime.retrieval_run_dir,
            reranker_run_dir=runtime.reranker_run_dir,
            working_dir=runtime.working_dir,
            prefer_pack_bridge=bool(runtime.extra.get("prefer_pack_bridge", True)),
            retrieval_artifacts=dict(runtime.extra.get("retrieval_artifacts", {}) or {}),
            reranker_artifacts=dict(runtime.extra.get("reranker_artifacts", {}) or {}),
            meta={
                "openwebui_context": {
                    "doc_scope": req.context.doc_scope[:],
                    "expected_doc_hint": req.context.expected_doc_hint,
                    "domain": req.context.domain,
                    "source_type": req.context.source_type,
                    "files": [
                        {
                            "file_id": x.file_id,
                            "doc_id": x.doc_id,
                            "file_name": x.file_name,
                            "source_type": x.source_type,
                            "meta": dict(x.meta),
                        }
                        for x in req.context.files
                    ],
                    "meta": dict(req.context.meta),
                },
                "openwebui_meta": dict(req.openwebui_meta),
            },
        )

        current_messages = [
            ChatMessage(role=msg.role, content=msg.content)
            for msg in extracted.current_messages
        ]

        memory = None
        if extracted.memory is not None:
            memory = ConversationMemory(
                source=extracted.memory.source,
                messages=[
                    ChatMessage(role=msg.role, content=msg.content)
                    for msg in extracted.memory.messages
                ],
                summary_text=extracted.memory.summary_text,
            )

        integration_request = IntegrationRequest(
            schema_version="slac_integration_request_v1",
            record_type="integration_request",
            request_id=req.request_id,
            session_id=req.session_id,
            query_id=req.query_id,
            query_text=extracted.query_text,
            messages=current_messages,
            memory=memory,
            context=integration_context,
            pipeline_config=PipelineConfig.from_dict(pipeline_config_dict),
            prompt_hints=PromptHints.from_dict(prompt_hints_dict),
            meta={
                "source_module": "SLAC.interaction",
                "openwebui_request_meta": dict(req.meta),
                "extracted_conversation_meta": dict(extracted.meta),
            },
        )
        return integration_request