from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from SLAC.interaction.service.interaction_service import InteractionService
from SLAC.interaction.utils.ids import safe_stem
from SLAC.interaction.utils.io import ensure_dir, read_jsonl, write_json
from SLAC.interaction.utils.time_utils import now_iso


def _to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SLAC interaction smoke test.")
    parser.add_argument("--requests", required=True, help="Path to requests.jsonl")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--retrieval_run_dir", default=None, help="Default retrieval run dir")
    parser.add_argument("--reranker_run_dir", default=None, help="Default reranker run dir")
    parser.add_argument("--working_dir", default=None, help="Default working dir")
    parser.add_argument(
        "--no_save_debug",
        action="store_true",
        help="Disable saving parsed/extracted/integration debug files",
    )
    return parser.parse_args()


def _inject_runtime_defaults(
    raw: Dict[str, Any],
    *,
    retrieval_run_dir: str | None,
    reranker_run_dir: str | None,
    working_dir: str | None,
) -> Dict[str, Any]:
    payload = dict(raw)
    runtime = dict(payload.get("runtime_config", {}) or {})

    if retrieval_run_dir and not runtime.get("retrieval_run_dir"):
        runtime["retrieval_run_dir"] = retrieval_run_dir
    if reranker_run_dir and not runtime.get("reranker_run_dir"):
        runtime["reranker_run_dir"] = reranker_run_dir
    if working_dir and not runtime.get("working_dir"):
        runtime["working_dir"] = working_dir

    payload["runtime_config"] = runtime
    return payload


def _resolve_query_key(response_obj: Dict[str, Any]) -> str:
    query_id = str(response_obj.get("query_id", "") or "").strip()
    request_id = str(response_obj.get("request_id", "") or "").strip()
    return safe_stem(query_id or request_id or "unknown")


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "responses")
    ensure_dir(out_dir / "debug")
    ensure_dir(out_dir / "summaries")

    request_records = read_jsonl(args.requests)

    service = InteractionService()

    status_counts: Dict[str, int] = {"ok": 0, "degraded": 0, "error": 0}
    per_request: List[Dict[str, Any]] = []

    for raw in request_records:
        payload = _inject_runtime_defaults(
            raw,
            retrieval_run_dir=args.retrieval_run_dir,
            reranker_run_dir=args.reranker_run_dir,
            working_dir=args.working_dir,
        )

        response_obj, artifacts = service.handle_with_artifacts(payload)
        response_dict = _to_dict(response_obj) or {}

        query_key = _resolve_query_key(response_dict)
        write_json(out_dir / "responses" / f"{query_key}.openwebui_response.json", response_obj)

        if not args.no_save_debug:
            parsed_request = artifacts.get("parsed_request")
            extracted_conversation = artifacts.get("extracted_conversation")
            integration_request = artifacts.get("integration_request")
            integration_response = artifacts.get("integration_response")
            integration_artifacts = artifacts.get("integration_artifacts")

            if parsed_request is not None:
                write_json(
                    out_dir / "debug" / f"{query_key}.parsed_request.json",
                    parsed_request,
                )
            if extracted_conversation is not None:
                write_json(
                    out_dir / "debug" / f"{query_key}.extracted_conversation.json",
                    extracted_conversation,
                )
            if integration_request is not None:
                write_json(
                    out_dir / "debug" / f"{query_key}.integration_request.json",
                    integration_request,
                )
            if integration_response is not None:
                write_json(
                    out_dir / "debug" / f"{query_key}.integration_response.json",
                    integration_response,
                )
            if integration_artifacts is not None:
                write_json(
                    out_dir / "debug" / f"{query_key}.integration_artifacts.json",
                    integration_artifacts,
                )

        status = str(response_dict.get("status", "error") or "error")
        status_counts[status] = status_counts.get(status, 0) + 1

        trace_summary = dict(response_dict.get("trace_summary", {}) or {})
        error_obj = dict(response_dict.get("error", {}) or {})

        per_request.append(
            {
                "request_id": response_dict.get("request_id"),
                "query_id": response_dict.get("query_id"),
                "status": status,
                "answer_text_len": len(str(response_dict.get("answer_text", "") or "")),
                "trace_summary": trace_summary,
                "error": error_obj if error_obj else None,
            }
        )

    summary = {
        "schema_version": "slac_openwebui_run_summary_v1",
        "created_at": now_iso(),
        "requests_path": str(args.requests),
        "out_dir": str(out_dir),
        "num_requests": len(request_records),
        "status_counts": status_counts,
        "per_request": per_request,
    }
    write_json(out_dir / "summaries" / "interaction_run_summary.json", summary)


if __name__ == "__main__":
    main()