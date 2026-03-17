from __future__ import annotations

from typing import Any


class IntegrationAdapter:
    def __init__(self, integrator: Any | None = None) -> None:
        if integrator is None:
            from SLAC.integration.adapters.reranker_run_dir_adapter import (
                RerankerRunDirAdapter,
            )
            from SLAC.integration.adapters.retrieval_run_dir_adapter import (
                RetrievalRunDirAdapter,
            )
            from SLAC.integration.orchestrator.final_integrator import FinalIntegrator

            integrator = FinalIntegrator(
                retrieval_adapter=RetrievalRunDirAdapter(),
                reranker_adapter=RerankerRunDirAdapter(),
            )

        self.integrator = integrator

    def invoke(self, integration_request: Any) -> Any:
        return self.integrator.run(integration_request)

    def invoke_with_artifacts(self, integration_request: Any) -> Any:
        if hasattr(self.integrator, "run_with_artifacts"):
            return self.integrator.run_with_artifacts(integration_request)
        response = self.integrator.run(integration_request)
        return response, None