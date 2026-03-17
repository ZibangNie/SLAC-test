from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from SLAC.integration.io.readers import ensure_dir


def _to_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    return obj


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_to_serializable(record), ensure_ascii=False))
            f.write("\n")


def write_integration_response(out_dir: str | Path, query_key: str, response: Any) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "responses" / f"{query_key}.integration_response.json"
    write_json(path, response)
    return path


def write_llm_request(out_dir: str | Path, query_key: str, llm_request: Any) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "llm_requests" / f"{query_key}.llm_request.json"
    write_json(path, llm_request)
    return path


def write_selected_evidence(out_dir: str | Path, query_key: str, evidence: List[Any]) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "debug" / f"{query_key}.selected_evidence.json"
    write_json(path, evidence)
    return path


def write_prompt_bundle(out_dir: str | Path, query_key: str, prompt_bundle: Any) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "debug" / f"{query_key}.prompt_bundle.json"
    write_json(path, prompt_bundle)
    return path


def write_run_summary(out_dir: str | Path, summary: Dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "summaries" / "integration_run_summary.json"
    write_json(path, summary)
    return path