from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from SLAC.integration.adapters.retrieval_adapter import RetrievalAdapter
from SLAC.integration.adapters.reranker_adapter import RerankerAdapter
from SLAC.integration.io.readers import ensure_dir, read_jsonl
from SLAC.integration.io.validators import validate_integration_request
from SLAC.integration.io.writers import (
    write_integration_response,
    write_llm_request,
    write_prompt_bundle,
    write_run_summary,
    write_selected_evidence,
)
from SLAC.integration.orchestrator.final_integrator import FinalIntegrator
from SLAC.integration.utils.time_utils import now_iso


def _to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SLAC integration pipeline.")
    parser.add_argument("--requests", required=True, help="Path to requests.jsonl")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--retrieval_run_dir", default=None, help="Optional default retrieval run dir")
    parser.add_argument("--reranker_run_dir", default=None, help="Optional default reranker run dir")
    parser.add_argument("--working_dir", default=None, help="Optional working dir for adapter subprocess calls")
    parser.add_argument("--no_save_debug", action="store_true", help="Disable saving llm_request/prompt/evidence debug files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "responses")
    ensure_dir(out_dir / "llm_requests")
    ensure_dir(out_dir / "debug")
    ensure_dir(out_dir / "summaries")

    request_records = read_jsonl(args.requests)

    integrator = FinalIntegrator(
        retrieval_adapter=RetrievalAdapter(),
        reranker_adapter=RerankerAdapter(),
    )

    status_counts: Dict[str, int] = {"ok": 0, "degraded": 0, "error": 0}
    per_request: List[Dict[str, Any]] = []

    for raw in request_records:
        req = validate_integration_request(raw)

        if args.retrieval_run_dir and not req.context.retrieval_run_dir:
            req.context.retrieval_run_dir = args.retrieval_run_dir
        if args.reranker_run_dir and not req.context.reranker_run_dir:
            req.context.reranker_run_dir = args.reranker_run_dir
        if args.working_dir and not req.context.working_dir:
            req.context.working_dir = args.working_dir

        response, artifacts = integrator.run_with_artifacts(req)

        query_key = (req.query_id or req.request_id).strip()
        write_integration_response(out_dir, query_key, response)

        if not args.no_save_debug:
            if artifacts.llm_request is not None:
                write_llm_request(out_dir, query_key, artifacts.llm_request)
            if artifacts.prompt_bundle is not None:
                write_prompt_bundle(out_dir, query_key, artifacts.prompt_bundle)
            if artifacts.selected_evidence is not None:
                write_selected_evidence(out_dir, query_key, artifacts.selected_evidence)

        status = response.status
        status_counts[status] = status_counts.get(status, 0) + 1

        per_request.append(
            {
                "request_id": req.request_id,
                "query_id": req.query_id,
                "query_text": req.query_text,
                "status": response.status,
                "answer_text_len": len(response.answer_text or ""),
                "trace": _to_dict(response.trace),
                "meta": _to_dict(response.meta),
            }
        )

    summary = {
        "schema_version": "slac_integration_run_summary_v1",
        "created_at": now_iso(),
        "requests_path": str(args.requests),
        "out_dir": str(out_dir),
        "num_requests": len(request_records),
        "status_counts": status_counts,
        "per_request": per_request,
    }
    write_run_summary(out_dir, summary)


if __name__ == "__main__":
    main()