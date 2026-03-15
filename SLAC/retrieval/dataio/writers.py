from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, obj: Dict[str, Any], indent: int = 2) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")