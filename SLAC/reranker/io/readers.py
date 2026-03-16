from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .validators import RerankerInputValidationError, normalize_reranker_input_record


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"jsonl line must be object at {path}:{line_no}")
            obj["_line_no"] = line_no
            obj["_source_file"] = str(path)
            yield obj


def read_jsonl(path: str | Path) -> List[dict]:
    return list(iter_jsonl(path))


def load_reranker_input_file(
    path: str | Path,
    *,
    strict: bool = False,
) -> List[dict]:
    path = Path(path)
    records: List[dict] = []
    for raw in iter_jsonl(path):
        line_no = raw.pop("_line_no", None)
        source_file = raw.pop("_source_file", str(path))
        try:
            rec = normalize_reranker_input_record(raw, strict=strict)
        except RerankerInputValidationError as exc:
            raise RerankerInputValidationError(
                f"{path}:{line_no}: {exc}"
            ) from exc
        rec["_line_no"] = line_no
        rec["_source_file"] = source_file
        records.append(rec)
    return records


def discover_reranker_input_files(
    input_path: str | Path,
    *,
    recursive: bool = False,
) -> List[Path]:
    """
    Accepts:
      - a single *.reranker_input.jsonl file
      - a directory containing query-level reranker input files
      - a run root; we will look under <run_root>/queries first, then recursively
    """
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    candidates: List[Path] = []

    queries_dir = input_path / "queries"
    if queries_dir.exists() and queries_dir.is_dir():
        candidates.extend(sorted(queries_dir.glob("*.reranker_input.jsonl")))

    if not candidates:
        candidates.extend(sorted(input_path.glob("*.reranker_input.jsonl")))

    if recursive and not candidates:
        candidates.extend(sorted(input_path.rglob("*.reranker_input.jsonl")))

    # De-duplicate while preserving order.
    seen = set()
    unique: List[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)

    if not unique:
        raise FileNotFoundError(
            f"no *.reranker_input.jsonl files found under: {input_path}"
        )
    return unique


def maybe_candidate_sidecar_path(reranker_input_path: str | Path) -> Optional[Path]:
    p = Path(reranker_input_path)
    name = p.name
    if name.endswith(".reranker_input.jsonl"):
        candidate_name = name.replace(".reranker_input.jsonl", ".candidates.jsonl")
        candidate_path = p.with_name(candidate_name)
        if candidate_path.exists():
            return candidate_path
    return None


def load_candidate_sidecar(
    path: str | Path,
) -> Dict[Tuple[str, str], dict]:
    """
    Keyed by (query_id, chunk_id). Falls back to ('__any__', chunk_id) when query_id is absent.
    """
    path = Path(path)
    out: Dict[Tuple[str, str], dict] = {}
    for raw in iter_jsonl(path):
        query_id = raw.get("query_id")
        chunk_id = raw.get("chunk_id")
        if chunk_id is None:
            continue

        chunk_id = str(chunk_id).strip()
        if not chunk_id:
            continue

        if query_id is None or str(query_id).strip() == "":
            key = ("__any__", chunk_id)
        else:
            key = (str(query_id).strip(), chunk_id)

        raw.pop("_line_no", None)
        raw.pop("_source_file", None)
        out[key] = raw
    return out


def attach_candidate_sidecar_metadata(
    reranker_records: List[dict],
    candidate_sidecar: Dict[Tuple[str, str], dict],
) -> List[dict]:
    if not candidate_sidecar:
        return reranker_records

    merged: List[dict] = []
    for rec in reranker_records:
        query_id = rec["query_id"]
        chunk_id = rec["chunk_id"]
        sidecar = candidate_sidecar.get((query_id, chunk_id))
        if sidecar is None:
            sidecar = candidate_sidecar.get(("__any__", chunk_id))

        if not sidecar:
            merged.append(rec)
            continue

        new_rec = dict(rec)
        for key, value in sidecar.items():
            if key in new_rec:
                continue
            new_rec[key] = value
        merged.append(new_rec)
    return merged