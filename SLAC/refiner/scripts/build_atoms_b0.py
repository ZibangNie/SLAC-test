from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

# 允许从项目根目录直接运行
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slac_refiner.datasets.schemas import Chunk0Doc, AtomsB0Doc  # noqa: E402

# 先占位，后面你再实现具体逻辑
from slac_refiner.atomize.normalize import normalize_text  # noqa: E402
from slac_refiner.atomize.splitter import split_text_to_atoms  # noqa: E402
from slac_refiner.atomize.mapping import build_atoms_b0_from_units  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build atoms, unit2atom_span, and b0 from chunk0_units."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a single input JSON or JSONL file.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to a directory containing JSON/JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.jsonl",
        help="Glob pattern for --input_dir mode. Default: *.jsonl",
    )
    return parser.parse_args()


def iter_input_files(input_path: str | None, input_dir: str | None, pattern: str) -> List[Path]:
    files: List[Path] = []

    if input_path:
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        files.append(p)

    if input_dir:
        d = Path(input_dir)
        if not d.exists():
            raise FileNotFoundError(f"Input directory not found: {d}")
        files.extend(sorted(d.rglob(pattern)))

    if not files:
        raise ValueError("You must provide either --input or --input_dir")

    return files


def load_docs_from_file(path: Path) -> Iterator[Chunk0Doc]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                yield Chunk0Doc.from_dict(item)
        elif isinstance(data, dict):
            yield Chunk0Doc.from_dict(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
        return

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    yield Chunk0Doc.from_dict(item)
                except Exception as e:
                    raise ValueError(f"Failed parsing {path} line {line_no}: {e}") from e
        return

    raise ValueError(f"Unsupported file type: {path}")


def process_doc(doc: Chunk0Doc) -> AtomsB0Doc | None:
    normalized_units = []
    for unit in doc.chunk0_units:
        normalized_units.append(
            {
                "unit_id": unit.unit_id,
                "text": normalize_text(unit.text),
            }
        )

    result = build_atoms_b0_from_units(
        normalized_units=normalized_units,
        splitter_fn=split_text_to_atoms,
    )

    if result["meta"].get("all_units_empty", False):
        return None

    atoms_doc = AtomsB0Doc(
        doc_id=doc.doc_id,
        domain=doc.domain,
        chunk0_units=doc.chunk0_units,
        atoms=result["atoms"],
        unit2atom_span=result["unit2atom_span"],
        b0=result["b0"],
        meta={
            **doc.meta,
            **result["meta"],
            "num_units": len(doc.chunk0_units),
            "num_atoms": len(result["atoms"]),
        },
    )
    atoms_doc.validate()
    return atoms_doc


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_files = iter_input_files(args.input, args.input_dir, args.glob)

    output_path = Path(args.output)
    ensure_parent_dir(output_path)

    total_docs = 0
    total_atoms = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for file_path in input_files:
            for doc in load_docs_from_file(file_path):
                atoms_doc = process_doc(doc)
                if atoms_doc is None:
                    continue
                fout.write(json.dumps(atoms_doc.to_dict(), ensure_ascii=False) + "\n")
                total_docs += 1
                total_atoms += len(atoms_doc.atoms)

    print(
        f"Done. Wrote {total_docs} docs to {output_path}. "
        f"Total atoms: {total_atoms}."
    )


if __name__ == "__main__":
    main()