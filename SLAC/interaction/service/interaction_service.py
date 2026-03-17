from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from SLAC.interaction.adapters.integration_adapter import IntegrationAdapter
from SLAC.interaction.builders.integration_request_builder import IntegrationRequestBuilder
from SLAC.interaction.builders.response_builder import ResponseBuilder
from SLAC.interaction.io.schemas import OpenWebUIRequest, OpenWebUIResponse
from SLAC.interaction.io.validators import validate_openwebui_request
from SLAC.interaction.parsers.memory_extractor import MemoryExtractor
from SLAC.interaction.parsers.pipe_request_parser import PipeRequestParser


class InteractionService:
    """
    模块 1 统一入口：
      raw payload
        -> parse / normalize
        -> validate
        -> memory extract
        -> build slac_integration_request_v1
        -> invoke integration
        -> build slac_openwebui_response_v1
    """

    def __init__(
        self,
        *,
        parser: PipeRequestParser | None = None,
        memory_extractor: MemoryExtractor | None = None,
        integration_request_builder: IntegrationRequestBuilder | None = None,
        integration_adapter: IntegrationAdapter | None = None,
        response_builder: ResponseBuilder | None = None,
        default_pipeline_config: Optional[Dict[str, Any]] = None,
        default_prompt_hints: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.parser = parser or PipeRequestParser()
        self.memory_extractor = memory_extractor or MemoryExtractor()
        self.integration_request_builder = integration_request_builder or IntegrationRequestBuilder(
            default_pipeline_config=default_pipeline_config,
            default_prompt_hints=default_prompt_hints,
        )
        self.integration_adapter = integration_adapter or IntegrationAdapter()
        self.response_builder = response_builder or ResponseBuilder()

    def handle(self, raw_payload: Dict[str, Any]) -> OpenWebUIResponse:
        parsed_request = self.parser.parse(raw_payload)
        req = validate_openwebui_request(parsed_request)

        extracted = self.memory_extractor.extract(req)
        integration_request = self.integration_request_builder.build(req, extracted)
        integration_response = self.integration_adapter.invoke(integration_request)

        return self.response_builder.build(req, integration_response)

    def handle_with_artifacts(
        self,
        raw_payload: Dict[str, Any],
    ) -> Tuple[OpenWebUIResponse, Dict[str, Any]]:
        parsed_request = self.parser.parse(raw_payload)
        req = validate_openwebui_request(parsed_request)

        extracted = self.memory_extractor.extract(req)
        integration_request = self.integration_request_builder.build(req, extracted)

        invoke_result = self.integration_adapter.invoke_with_artifacts(integration_request)
        if isinstance(invoke_result, tuple) and len(invoke_result) == 2:
            integration_response, integration_artifacts = invoke_result
        else:
            integration_response, integration_artifacts = invoke_result, None

        openwebui_response = self.response_builder.build(req, integration_response)

        artifacts = {
            "parsed_request": parsed_request,
            "extracted_conversation": extracted,
            "integration_request": integration_request,
            "integration_response": integration_response,
            "integration_artifacts": integration_artifacts,
        }
        return openwebui_response, artifacts