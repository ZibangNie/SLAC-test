from __future__ import annotations

import argparse
import sys
from pathlib import Path

from SLAC.llm.io.readers import iter_llm_requests_jsonl
from SLAC.llm.io.validators import ValidationError
from SLAC.llm.io.writers import append_llm_answer_jsonl, write_summary_json
from SLAC.llm.service.llm_service import LLMService
from SLAC.llm.utils.io import ensure_dir
from SLAC.llm.utils.time_utils import now_iso, seconds_between


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SLAC LLM infer-only pipeline.")
    parser.add_argument("--requests", required=True, help="Path to requests.jsonl")
    parser.add_argument("--out_dir", required=True, help="Output run directory")
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately on the first validation or runtime error.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_path = Path(args.requests)
    out_dir = Path(args.out_dir)
    summaries_dir = out_dir / "summaries"
    ensure_dir(out_dir)
    ensure_dir(summaries_dir)

    answers_path = out_dir / "answers.jsonl"
    errors_path = out_dir / "errors.jsonl"
    summary_path = summaries_dir / "run_llm_infer_summary.json"

    service = LLMService()
    start_ts = now_iso()
    success = 0
    failed = 0
    total = 0

    for req in iter_llm_requests_jsonl(request_path):
        total += 1
        try:
            answer = service.invoke(req)
            if answer.status == "ok":
                success += 1
                append_llm_answer_jsonl(answers_path, answer)
            else:
                failed += 1
                append_llm_answer_jsonl(errors_path, answer)
                if args.stop_on_error:
                    break
        except ValidationError as exc:
            failed += 1
            err_payload = {
                "schema_version": "slac_llm_answer_v1",
                "record_type": "answer_result",
                "status": "error",
                "request_id": req.request_id,
                "session_id": req.session_id,
                "query_id": req.query_id,
                "provider": req.provider,
                "model_name": req.model_name,
                "response_id": None,
                "answer_text": "",
                "finish_reason": None,
                "usage": {},
                "memory_used": bool(req.memory and (req.memory.messages or req.memory.summary_text)),
                "memory_message_count": len(req.memory.messages) if req.memory else 0,
                "evidence_count": len(req.evidence),
                "evidence_refs": [
                    {
                        "chunk_id": ev.chunk_id,
                        "doc_id": ev.doc_id,
                        "rerank_rank": ev.rerank_rank,
                    }
                    for ev in req.evidence
                ],
                "meta": {
                    "error_message": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            }
            from SLAC.llm.utils.io import append_jsonl
            append_jsonl(errors_path, err_payload)
            if args.stop_on_error:
                break

    end_ts = now_iso()
    summary = {
        "module": "SLAC.llm.run.run_llm_infer_only",
        "request_path": str(request_path),
        "out_dir": str(out_dir),
        "started_at": start_ts,
        "finished_at": end_ts,
        "total_requests": total,
        "success_count": success,
        "error_count": failed,
    }
    write_summary_json(summary_path, summary)

    print("=== SLAC LLM infer-only finished ===")
    print(f"requests: {total}")
    print(f"success:  {success}")
    print(f"error:    {failed}")
    print(f"summary:  {summary_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
