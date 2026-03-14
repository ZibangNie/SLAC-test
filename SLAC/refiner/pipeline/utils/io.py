"""
I/O utilities.

Main responsibility:
- read/write JSON and JSONL
- batch file traversal
- common path helpers
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """
    Ensure a directory exists and return it as Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent_dir(path: PathLike) -> Path:
    """
    Ensure the parent directory of a file path exists.
    Return the file path as Path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def path_exists(path: PathLike) -> bool:
    return Path(path).exists()


def is_file(path: PathLike) -> bool:
    return Path(path).is_file()


def is_dir(path: PathLike) -> bool:
    return Path(path).is_dir()


def resolve_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def normalize_path_str(path: PathLike) -> str:
    """
    Normalize path to a stable POSIX-like string for logging / hashing.
    """
    p = Path(path)
    return p.as_posix()


def atomic_write_text(path: PathLike, text: str, encoding: str = "utf-8") -> Path:
    """
    Atomically write text to a file.
    """
    out_path = ensure_parent_dir(path)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".txt", dir=str(out_path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return out_path


def atomic_write_json(
    path: PathLike,
    obj: Any,
    *,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> Path:
    """
    Atomically write JSON to a file.
    """
    text = json.dumps(
        obj,
        ensure_ascii=ensure_ascii,
        indent=indent,
        sort_keys=sort_keys,
    )
    if not text.endswith("\n"):
        text += "\n"
    return atomic_write_text(path, text, encoding="utf-8")


def load_text(path: PathLike, encoding: str = "utf-8", errors: str = "ignore") -> str:
    return Path(path).read_text(encoding=encoding, errors=errors)


def dump_text(path: PathLike, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_text(path, text, encoding=encoding)


def load_json(path: PathLike, encoding: str = "utf-8") -> Any:
    with Path(path).open("r", encoding=encoding) as f:
        return json.load(f)


def dump_json(
    path: PathLike,
    obj: Any,
    *,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> Path:
    return atomic_write_json(
        path,
        obj,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
    )


def _validate_jsonl_record(record: Any, *, line_no: Optional[int] = None) -> None:
    if not isinstance(record, dict):
        loc = f" at line {line_no}" if line_no is not None else ""
        raise TypeError(f"JSONL record must be a dict{loc}, got {type(record).__name__}")


def iter_jsonl(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    skip_blank: bool = True,
    validate_dict: bool = False,
) -> Iterator[Any]:
    """
    Stream JSONL records.
    """
    p = Path(path)
    with p.open("r", encoding=encoding) as f:
        for i, line in enumerate(f, start=1):
            if skip_blank and not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL in {p} line {i}: {e}") from e
            if validate_dict:
                _validate_jsonl_record(obj, line_no=i)
            yield obj


def load_jsonl(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    skip_blank: bool = True,
    validate_dict: bool = False,
) -> List[Any]:
    return list(
        iter_jsonl(
            path,
            encoding=encoding,
            skip_blank=skip_blank,
            validate_dict=validate_dict,
        )
    )


def dump_jsonl(
    path: PathLike,
    records: Iterable[Any],
    *,
    ensure_ascii: bool = False,
    validate_dict: bool = False,
) -> Path:
    """
    Atomically write JSONL.
    """
    out_path = ensure_parent_dir(path)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".jsonl", dir=str(out_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for idx, rec in enumerate(records, start=1):
                if validate_dict:
                    _validate_jsonl_record(rec, line_no=idx)
                f.write(json.dumps(rec, ensure_ascii=ensure_ascii))
                f.write("\n")
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return out_path


def append_jsonl(
    path: PathLike,
    records: Iterable[Any],
    *,
    ensure_ascii: bool = False,
    validate_dict: bool = False,
) -> Path:
    """
    Append records to JSONL. Non-atomic, but useful for logs / incremental dumps.
    """
    out_path = ensure_parent_dir(path)
    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        for idx, rec in enumerate(records, start=1):
            if validate_dict:
                _validate_jsonl_record(rec, line_no=idx)
            f.write(json.dumps(rec, ensure_ascii=ensure_ascii))
            f.write("\n")
    return out_path


def read_first_jsonl_record(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    validate_dict: bool = False,
) -> Optional[Any]:
    for obj in iter_jsonl(path, encoding=encoding, validate_dict=validate_dict):
        return obj
    return None


def count_jsonl(path: PathLike, *, encoding: str = "utf-8", skip_blank: bool = True) -> int:
    cnt = 0
    with Path(path).open("r", encoding=encoding) as f:
        for line in f:
            if skip_blank and not line.strip():
                continue
            cnt += 1
    return cnt


def iter_input_files(
    input_paths: Sequence[PathLike],
    *,
    suffixes: Optional[Sequence[str]] = None,
    recursive: bool = True,
    sort_result: bool = True,
    allow_hidden: bool = False,
) -> Iterator[Path]:
    """
    Iterate files from a mix of files / directories.

    suffixes:
        e.g. [".pdf", ".txt", ".docx"]
        comparison is case-insensitive.
    """
    normalized_suffixes = None
    if suffixes:
        normalized_suffixes = {s.lower() if s.startswith(".") else f".{s.lower()}" for s in suffixes}

    seen: set[str] = set()
    collected: List[Path] = []

    for item in input_paths:
        p = Path(item).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Input path does not exist: {p}")

        candidates: List[Path] = []
        if p.is_file():
            candidates = [p]
        elif p.is_dir():
            if recursive:
                candidates = [x for x in p.rglob("*") if x.is_file()]
            else:
                candidates = [x for x in p.iterdir() if x.is_file()]
        else:
            continue

        for fp in candidates:
            if not allow_hidden and any(part.startswith(".") for part in fp.parts):
                continue
            if normalized_suffixes and fp.suffix.lower() not in normalized_suffixes:
                continue
            key = str(fp.resolve())
            if key in seen:
                continue
            seen.add(key)
            collected.append(fp)

    if sort_result:
        collected.sort(key=lambda x: x.as_posix().lower())

    yield from collected


def relative_to_or_self(path: PathLike, root: PathLike) -> Path:
    """
    Return relative path if possible; otherwise return original resolved path.
    """
    p = Path(path).resolve()
    r = Path(root).resolve()
    try:
        return p.relative_to(r)
    except Exception:
        return p


def safe_stem(path: PathLike) -> str:
    return Path(path).stem.strip()


def maybe_unlink(path: PathLike) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()


def list_files(
    input_paths: Sequence[PathLike],
    *,
    suffixes: Optional[Sequence[str]] = None,
    recursive: bool = True,
    sort_result: bool = True,
    allow_hidden: bool = False,
) -> List[Path]:
    return list(
        iter_input_files(
            input_paths,
            suffixes=suffixes,
            recursive=recursive,
            sort_result=sort_result,
            allow_hidden=allow_hidden,
        )
    )


def validate_required_keys(obj: Dict[str, Any], required_keys: Sequence[str], *, obj_name: str = "object") -> None:
    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise KeyError(f"{obj_name} missing required keys: {missing}")


def deep_get(obj: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def json_size_hint(obj: Any) -> int:
    """
    Rough serialized size in bytes.
    """
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))