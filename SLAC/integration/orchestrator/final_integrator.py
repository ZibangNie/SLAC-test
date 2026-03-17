from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from SLAC.integration.evidence.normalizers import normalize_candidate_list
from SLAC.integration.evidence.selectors import select_evidence
from SLAC.integration.io.schemas import (
    IntegrationRequest,
    IntegrationResponse,
    IntegrationTrace,
    PromptBundle,
    RetrievalArtifacts,
    RerankerArtifacts,
    SelectedEvidence,
)
from SLAC.integration.io.validators import validate_integration_request
from SLAC.integration.prompt.builders import build_prompt_bundle


def _obj_to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj


@dataclass
class IntegrationExecutionArtifacts:
    retrieval_artifacts: Optional[RetrievalArtifacts] = None
    reranker_artifacts: Optional[RerankerArtifacts] = None
    selected_evidence: List[SelectedEvidence] = None
    prompt_bundle: Optional[PromptBundle] = None
    llm_request: Any = None

    def __post_init__(self) -> None:
        if self.selected_evidence is None:
            self.selected_evidence = []


class FinalIntegrator:
    def __init__(
        self,
        *,
        retrieval_adapter: Any | None = None,
        reranker_adapter: Any | None = None,
        llm_adapter: Any | None = None,
    ) -> None:
        self.retrieval_adapter = retrieval_adapter
        self.reranker_adapter = reranker_adapter
        self.llm_adapter = llm_adapter

    def _read_or_run_retrieval(self, req: IntegrationRequest) -> RetrievalArtifacts:
        if req.context.retrieval_artifacts:
            data = req.context.retrieval_artifacts
            return RetrievalArtifacts(
                candidates=list(data.get("candidates", []) or []),
                reranker_input=list(data.get("reranker_input", []) or []),
                packed_evidence=list(data.get("packed_evidence", []) or []),
                meta=dict(data.get("meta", {}) or {}),
            )

        if self.retrieval_adapter is None:
            raise RuntimeError("retrieval_adapter is required when no context.retrieval_artifacts are provided.")

        result = self.retrieval_adapter.run_or_read(req)
        if isinstance(result, RetrievalArtifacts):
            return result

        return RetrievalArtifacts(
            candidates=list(result.get("candidates", []) or []),
            reranker_input=list(result.get("reranker_input", []) or []),
            packed_evidence=list(result.get("packed_evidence", []) or []),
            meta=dict(result.get("meta", {}) or {}),
        )

    def _read_or_run_reranker(
        self,
        req: IntegrationRequest,
        retrieval_artifacts: RetrievalArtifacts,
    ) -> RerankerArtifacts:
        if req.context.reranker_artifacts:
            data = req.context.reranker_artifacts
            return RerankerArtifacts(
                pack_bridge=list(data.get("pack_bridge", []) or []),
                reranked_candidates=list(data.get("reranked_candidates", []) or []),
                meta=dict(data.get("meta", {}) or {}),
            )

        if self.reranker_adapter is None:
            raise RuntimeError("reranker_adapter is required when no context.reranker_artifacts are provided.")

        result = self.reranker_adapter.run_or_read(req, retrieval_artifacts)
        if isinstance(result, RerankerArtifacts):
            return result

        return RerankerArtifacts(
            pack_bridge=list(result.get("pack_bridge", []) or []),
            reranked_candidates=list(result.get("reranked_candidates", []) or []),
            meta=dict(result.get("meta", {}) or {}),
        )

    def _choose_candidate_source(
        self,
        req: IntegrationRequest,
        retrieval_artifacts: Optional[RetrievalArtifacts],
        reranker_artifacts: Optional[RerankerArtifacts],
    ) -> Tuple[List[Dict[str, Any]], str]:
        if reranker_artifacts is not None:
            if req.context.prefer_pack_bridge and reranker_artifacts.pack_bridge:
                return reranker_artifacts.pack_bridge, "reranker_pack_bridge"
            if reranker_artifacts.reranked_candidates:
                return reranker_artifacts.reranked_candidates, "reranker_reranked_candidates"

        if retrieval_artifacts is not None:
            if retrieval_artifacts.packed_evidence:
                return retrieval_artifacts.packed_evidence, "retrieval_packed_evidence"
            if retrieval_artifacts.candidates:
                return retrieval_artifacts.candidates, "retrieval_candidates"
            if retrieval_artifacts.reranker_input:
                return retrieval_artifacts.reranker_input, "retrieval_reranker_input"

        return [], "empty"

    def build_llm_request(
        self,
        req: IntegrationRequest,
        selected_evidence: List[SelectedEvidence],
        prompt_bundle: PromptBundle,
    ) -> Any:
        from SLAC.llm.io.schemas import (
            ChatMessage as LLMChatMessage,
            ConversationMemory as LLMConversationMemory,
            EvidenceItem,
            GenerationConfig,
            LLMRequest,
        )

        llm_messages = [
            LLMChatMessage(role=msg.role, content=msg.content)
            for msg in prompt_bundle.current_messages
        ]

        llm_memory = None
        if req.memory is not None:
            llm_memory = LLMConversationMemory(
                source=req.memory.source,
                messages=[
                    LLMChatMessage(role=msg.role, content=msg.content)
                    for msg in req.memory.messages
                ],
            )

        evidence = [
            EvidenceItem(
                chunk_id=ev.chunk_id,
                doc_id=ev.doc_id,
                passage_text=ev.passage_text,
                path_text=ev.path_text,
                query_id=ev.query_id,
                query_text=ev.query_text,
                rerank_rank=ev.rerank_rank,
                rerank_score=ev.rerank_score,
                retrieve_rank_fused=ev.retrieve_rank_fused,
                role=ev.role,
                hit_type=ev.hit_type,
                source_views=ev.source_views[:],
                token_est=ev.token_est,
                expansion_depth=ev.expansion_depth,
                meta=dict(ev.meta),
            )
            for ev in selected_evidence
        ]

        pc = req.pipeline_config
        llm_cfg = pc.llm
        assert llm_cfg is not None

        meta = {
            "integration_module": "SLAC.integration",
            "prompt_bundle": {
                "num_current_messages": len(prompt_bundle.current_messages),
                "num_selected_evidence": len(selected_evidence),
            },
        }
        if req.memory and req.memory.summary_text:
            meta["memory_summary_text"] = req.memory.summary_text

        return LLMRequest(
            schema_version="slac_llm_request_v1",
            record_type="answer_request",
            request_id=req.request_id,
            session_id=req.session_id,
            query_id=req.query_id,
            query_text=req.query_text,
            provider=llm_cfg.provider,
            model_name=llm_cfg.model_name,
            api_base=llm_cfg.api_base,
            api_key_env=llm_cfg.api_key_env,
            generation_config=GenerationConfig(
                temperature=llm_cfg.temperature,
                top_p=llm_cfg.top_p,
                max_tokens=llm_cfg.max_tokens,
                timeout_s=llm_cfg.timeout_s,
            ),
            system_prompt=prompt_bundle.system_prompt,
            messages=llm_messages,
            prompt=None,
            memory=llm_memory,
            evidence=evidence,
            options={
                "memory_merge_policy": "prepend",
                "evidence_render_policy": "append_as_context_block",
                "return_raw_response": False,
            },
            meta=meta,
        )

    def build_integration_response(
        self,
        *,
        req: IntegrationRequest,
        answer_result: Any | None,
        selected_evidence: List[SelectedEvidence],
        trace: IntegrationTrace,
        status: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> IntegrationResponse:
        answer_dict = _obj_to_dict(answer_result)
        answer_text = ""
        if answer_dict and isinstance(answer_dict, dict):
            answer_text = str(answer_dict.get("answer_text", "") or "")

        return IntegrationResponse(
            schema_version="slac_integration_response_v1",
            record_type="integration_response",
            status=status,
            request_id=req.request_id,
            session_id=req.session_id,
            query_id=req.query_id,
            query_text=req.query_text,
            answer_text=answer_text,
            answer_result=answer_dict,
            evidence=selected_evidence,
            trace=trace,
            meta=meta or {},
        )

    def run_with_artifacts(
        self,
        data: Dict[str, Any] | IntegrationRequest,
    ) -> Tuple[IntegrationResponse, IntegrationExecutionArtifacts]:
        req = validate_integration_request(data)

        trace = IntegrationTrace(
            retrieval_used=False,
            reranker_used=False,
            degraded_to_retrieval=False,
            evidence_budget_tokens=req.pipeline_config.max_evidence_tokens,
        )
        artifacts = IntegrationExecutionArtifacts()

        retrieval_artifacts: Optional[RetrievalArtifacts] = None
        reranker_artifacts: Optional[RerankerArtifacts] = None
        answer_result: Any | None = None

        if req.pipeline_config.use_retrieval or req.context.retrieval_artifacts:
            try:
                retrieval_artifacts = self._read_or_run_retrieval(req)
                artifacts.retrieval_artifacts = retrieval_artifacts
                trace.retrieval_used = True
            except Exception as exc:
                trace.errors.append(f"retrieval failed: {exc}")
                response = self.build_integration_response(
                    req=req,
                    answer_result=None,
                    selected_evidence=[],
                    trace=trace,
                    status="error",
                    meta={"error_stage": "retrieval", "error_message": str(exc)},
                )
                return response, artifacts

        if req.pipeline_config.use_reranker:
            try:
                reranker_artifacts = self._read_or_run_reranker(
                    req,
                    retrieval_artifacts or RetrievalArtifacts(),
                )
                artifacts.reranker_artifacts = reranker_artifacts
                trace.reranker_used = True
            except Exception as exc:
                if req.pipeline_config.allow_retrieval_fallback:
                    trace.degraded_to_retrieval = True
                    trace.warnings.append(f"reranker failed, fallback to retrieval: {exc}")
                else:
                    trace.errors.append(f"reranker failed: {exc}")
                    response = self.build_integration_response(
                        req=req,
                        answer_result=None,
                        selected_evidence=[],
                        trace=trace,
                        status="error",
                        meta={"error_stage": "reranker", "error_message": str(exc)},
                    )
                    return response, artifacts

        candidate_records, candidate_source = self._choose_candidate_source(
            req=req,
            retrieval_artifacts=retrieval_artifacts,
            reranker_artifacts=reranker_artifacts,
        )
        trace.candidate_source = candidate_source
        trace.num_candidates_read = len(candidate_records)

        normalized_candidates = normalize_candidate_list(
            candidate_records,
            query_id=req.query_id,
            query_text=req.query_text,
            source_name=candidate_source,
        )

        selected_evidence = select_evidence(
            normalized_candidates,
            max_items=req.pipeline_config.max_evidence_items,
            max_tokens=req.pipeline_config.max_evidence_tokens,
            prefer_direct_first=req.pipeline_config.prefer_direct_first,
            min_direct_evidence=req.pipeline_config.min_direct_evidence,
        )
        artifacts.selected_evidence = selected_evidence[:]
        trace.num_evidence_selected = len(selected_evidence)

        prompt_bundle = build_prompt_bundle(req, selected_evidence)
        artifacts.prompt_bundle = prompt_bundle

        llm_request = self.build_llm_request(req, selected_evidence, prompt_bundle)
        artifacts.llm_request = llm_request
        trace.llm_request_id = getattr(llm_request, "request_id", None)

        if self.llm_adapter is None:
            from SLAC.integration.adapters.llm_adapter import LLMAdapter
            self.llm_adapter = LLMAdapter()

        try:
            answer_result = self.llm_adapter.invoke(llm_request)
            answer_dict = _obj_to_dict(answer_result) or {}
            trace.llm_response_id = answer_dict.get("response_id")
        except Exception as exc:
            trace.errors.append(f"llm invoke failed: {exc}")
            response = self.build_integration_response(
                req=req,
                answer_result=None,
                selected_evidence=selected_evidence,
                trace=trace,
                status="error",
                meta={"error_stage": "llm_invoke", "error_message": str(exc)},
            )
            return response, artifacts

        answer_dict = _obj_to_dict(answer_result) or {}
        llm_status = str(answer_dict.get("status", "ok"))

        if llm_status == "error":
            final_status = "error"
        elif trace.degraded_to_retrieval:
            final_status = "degraded"
        else:
            final_status = "ok"

        response = self.build_integration_response(
            req=req,
            answer_result=answer_result,
            selected_evidence=selected_evidence,
            trace=trace,
            status=final_status,
            meta={
                "candidate_source": candidate_source,
                "prompt_bundle_meta": dict(prompt_bundle.meta),
                "llm_status": llm_status,
            },
        )
        return response, artifacts

    def run(self, data: Dict[str, Any] | IntegrationRequest) -> IntegrationResponse:
        response, _ = self.run_with_artifacts(data)
        return response