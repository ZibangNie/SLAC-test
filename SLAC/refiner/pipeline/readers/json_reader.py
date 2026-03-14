"""
JSON reader for refiner pipeline.

Main responsibility:
- load pre-structured intermediate JSON
- support debugging and partial pipeline reruns
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from ..utils.ids import stable_doc_id
from ..utils.io import iter_input_files, load_json


PathLike = Union[str, Path]


_TEXT_KEYS = [
    "raw_text",
    "normalized_text",
    "cleaned_text",
    "text",
    "content",
    "body",
]


def _join_unit_texts(units: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for u in units:
        if not isinstance(u, dict):
            continue
        txt = (u.get("text") or "").strip()
        if txt:
            parts.append(txt)
    return "\n\n".join(parts).strip()


def _extract_text_from_json_obj(obj: Any) -> tuple[str, str]:
    """
    Returns:
        (raw_text, mode)
    """
    if isinstance(obj, dict):
        for k in _TEXT_KEYS:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v, f"field:{k}"

        # structure_doc-like
        units = obj.get("units")
        if isinstance(units, list) and units:
            txt = _join_unit_texts(units)
            if txt:
                return txt, "joined_units"

        chunk0_units = obj.get("chunk0_units")
        if isinstance(chunk0_units, list) and chunk0_units:
            txt = _join_unit_texts(chunk0_units)
            if txt:
                return txt, "joined_chunk0_units"

        structure_doc = obj.get("structure_doc")
        if isinstance(structure_doc, dict):
            units2 = structure_doc.get("units")
            if isinstance(units2, list) and units2:
                txt = _join_unit_texts(units2)
                if txt:
                    return txt, "field:structure_doc.units"

    raise ValueError(
        "Could not extract text from JSON. Expected one of "
        f"{_TEXT_KEYS}, or units / chunk0_units / structure_doc.units"
    )


def read_json_document(
    path: PathLike,
    *,
    doc_id: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    json_path = Path(path).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")
    if json_path.suffix.lower() != ".json":
        raise ValueError(f"Not a JSON file: {json_path}")

    obj = load_json(json_path)
    raw_text, mode = _extract_text_from_json_obj(obj)

    title_hint = None
    if isinstance(obj, dict):
        for k in ("doc_name", "title", "name"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                title_hint = v.strip()
                break

    final_doc_id = stable_doc_id(
        explicit_doc_id=doc_id or (obj.get("doc_id") if isinstance(obj, dict) else None),
        source_path=json_path,
        title_hint=title_hint,
    )

    meta: Dict[str, Any] = {
        "reader": "json_reader",
        "json_extract_mode": mode,
        "num_chars": len(raw_text),
        "title_hint": title_hint,
    }
    if extra_meta:
        meta.update(extra_meta)

    return {
        "doc_id": final_doc_id,
        "source_path": json_path.as_posix(),
        "source_type": "json",
        "raw_text": raw_text,
        "meta": meta,
    }


def iter_json_documents(
    input_paths: Sequence[PathLike],
    *,
    recursive: bool = True,
) -> Iterator[Dict[str, Any]]:
    files = iter_input_files(
        input_paths,
        suffixes=[".json"],
        recursive=recursive,
        sort_result=True,
        allow_hidden=False,
    )
    for fp in files:
        yield read_json_document(fp)