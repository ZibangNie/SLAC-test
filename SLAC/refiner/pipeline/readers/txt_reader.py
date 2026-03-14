"""
TXT reader for refiner pipeline.

Main responsibility:
- read txt
- normalize encoding/newlines at load time
- provide unified document object
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from ..utils.ids import stable_doc_id
from ..utils.io import iter_input_files


PathLike = Union[str, Path]


_DEFAULT_ENCODINGS: Tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "gb18030",
    "gbk",
    "big5",
    "latin-1",
)


def _read_text_with_fallback(
    path: PathLike,
    *,
    encoding: Optional[str] = None,
    errors: str = "strict",
    fallback_encodings: Sequence[str] = _DEFAULT_ENCODINGS,
) -> Tuple[str, str]:
    """
    Read text file with explicit encoding or a robust fallback list.

    Returns:
        (text, used_encoding)
    """
    p = Path(path)

    if encoding:
        text = p.read_text(encoding=encoding, errors=errors)
        return text, encoding

    last_error: Optional[Exception] = None
    for enc in fallback_encodings:
        try:
            text = p.read_text(encoding=enc, errors=errors)
            return text, enc
        except Exception as e:
            last_error = e

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Failed to decode text file with fallback encodings: {fallback_encodings}. "
        f"Last error: {last_error}",
    )


def _count_nonempty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _infer_title_hint_from_text(text: str, max_scan_lines: int = 20) -> Optional[str]:
    """
    Try to infer a lightweight title hint from the first few non-empty lines.
    This is only a fallback and should not be treated as a formal title extractor.
    """
    scanned = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        scanned += 1
        if 1 <= len(s) <= 120:
            return s
        if scanned >= max_scan_lines:
            break
    return None


def read_txt_document(
    path: PathLike,
    *,
    doc_id: Optional[str] = None,
    root: Optional[PathLike] = None,
    encoding: Optional[str] = None,
    errors: str = "strict",
    fallback_encodings: Sequence[str] = _DEFAULT_ENCODINGS,
    infer_title_hint: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Read a txt-like document into the pipeline's unified document record.

    Output schema:
    {
      "doc_id": "...",
      "source_path": "...",
      "source_type": "txt",
      "raw_text": "...",
      "meta": {...}
    }
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"TXT file does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"TXT path is not a file: {p}")

    raw_text, used_encoding = _read_text_with_fallback(
        p,
        encoding=encoding,
        errors=errors,
        fallback_encodings=fallback_encodings,
    )

    # Reader-level normalization kept intentionally minimal.
    # Heavy normalization/cleaning should happen in preprocess modules.
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    title_hint = _infer_title_hint_from_text(raw_text) if infer_title_hint else None
    final_doc_id = stable_doc_id(
        explicit_doc_id=doc_id,
        source_path=p,
        title_hint=title_hint,
    )

    meta: Dict[str, Any] = {
        "reader": "txt_reader",
        "encoding": used_encoding,
        "num_chars": len(raw_text),
        "num_lines": len(raw_text.splitlines()),
        "num_nonempty_lines": _count_nonempty_lines(raw_text),
        "title_hint": title_hint,
    }
    if root is not None:
        try:
            meta["relative_path"] = p.relative_to(Path(root).expanduser().resolve()).as_posix()
        except Exception:
            meta["relative_path"] = None

    if extra_meta:
        meta.update(extra_meta)

    return {
        "doc_id": final_doc_id,
        "source_path": p.as_posix(),
        "source_type": "txt",
        "raw_text": raw_text,
        "meta": meta,
    }


def iter_txt_documents(
    input_paths: Sequence[PathLike],
    *,
    root: Optional[PathLike] = None,
    encoding: Optional[str] = None,
    errors: str = "strict",
    fallback_encodings: Sequence[str] = _DEFAULT_ENCODINGS,
    suffixes: Sequence[str] = (".txt", ".text", ".md"),
    recursive: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Iterate txt-like files from one or more paths and yield unified document records.
    """
    files = iter_input_files(
        input_paths,
        suffixes=suffixes,
        recursive=recursive,
        sort_result=True,
        allow_hidden=False,
    )
    for fp in files:
        yield read_txt_document(
            fp,
            root=root,
            encoding=encoding,
            errors=errors,
            fallback_encodings=fallback_encodings,
        )


def load_txt_documents(
    input_paths: Sequence[PathLike],
    *,
    root: Optional[PathLike] = None,
    encoding: Optional[str] = None,
    errors: str = "strict",
    fallback_encodings: Sequence[str] = _DEFAULT_ENCODINGS,
    suffixes: Sequence[str] = (".txt", ".text", ".md"),
    recursive: bool = True,
) -> List[Dict[str, Any]]:
    return list(
        iter_txt_documents(
            input_paths,
            root=root,
            encoding=encoding,
            errors=errors,
            fallback_encodings=fallback_encodings,
            suffixes=suffixes,
            recursive=recursive,
        )
    )