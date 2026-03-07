from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slac_refiner.atomize.normalize import normalize_text
from slac_refiner.atomize.splitter import split_text_to_atoms
from slac_refiner.atomize.mapping import build_atoms_b0_from_units
from slac_refiner.noise.apply_noise import (
    apply_boundary_noise,
    spans_from_boundary_vector,
    units_from_atoms_and_spans,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build denoising training pairs for Boundary Refiner.")
    parser.add_argument("--input", type=str, default=None, help="Single JSON/JSONL file.")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of JSON/JSONL files.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--glob", type=str, default="*.jsonl", help="Glob pattern for input_dir mode.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--p_shift", type=float, default=0.45)
    parser.add_argument("--p_insert", type=float, default=0.30)
    parser.add_argument("--p_delete", type=float, default=0.25)
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
            raise FileNotFoundError(f"Input dir not found: {d}")
        files.extend(sorted(d.rglob(pattern)))

    if not files:
        raise ValueError("You must provide either --input or --input_dir")

    return files


def load_docs_from_file(path: Path) -> Iterator[Dict]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for x in data:
                yield x
        elif isinstance(data, dict):
            yield data
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
                    yield json.loads(line)
                except Exception as e:
                    raise ValueError(f"Failed parsing {path} line {line_no}: {e}") from e
        return

    raise ValueError(f"Unsupported file type: {path}")


def extract_gold_units(raw_doc: Dict) -> List[Dict]:
    """
    Accept multiple raw schemas.

    Priority:
    1) gold_units
    2) chunk0_units
    3) units   <- for your real LLM-annotated structured / semi-structured files

    We treat these units as teacher chunks for denoising pretraining.
    """
    units = raw_doc.get("gold_units")
    if units is None:
        units = raw_doc.get("chunk0_units")
    if units is None:
        units = raw_doc.get("units")

    if units is None:
        raise ValueError("Raw doc must contain one of: gold_units / chunk0_units / units")

    out = []
    for i, u in enumerate(units):
        if not isinstance(u, dict):
            raise ValueError(f"units[{i}] must be a dict")

        text = str(u.get("text", "")).strip()
        if not text:
            continue

        unit_id = int(u.get("unit_id", i))
        out.append({"unit_id": unit_id, "text": text})

    if len(out) == 0:
        raise ValueError("No non-empty units found in raw doc")

    return out


def build_gold_atoms_and_boundaries(gold_units: List[Dict]) -> Dict:
    normalized_units = [
        {"unit_id": int(u["unit_id"]), "text": normalize_text(u["text"])}
        for u in gold_units
    ]

    result = build_atoms_b0_from_units(
        normalized_units=normalized_units,
        splitter_fn=split_text_to_atoms,
    )
    return result


def convert_noisy_spans_to_json(spans):
    return [{"unit_id": i, "s": s, "e": e} for i, (s, e) in enumerate(spans)]


def process_one_doc(raw_doc: Dict, args: argparse.Namespace, sample_idx: int) -> Dict | None:
    doc_id = str(raw_doc.get("doc_id", f"doc_{sample_idx:06d}"))
    domain = "llm_gold"

    language = raw_doc.get("lang")
    if language is None:
        language = raw_doc.get("language", "other")
    language = str(language)

    gold_units = extract_gold_units(raw_doc)
    gold_result = build_gold_atoms_and_boundaries(gold_units)

    if gold_result["meta"].get("all_units_empty", False):
        return None

    atoms = gold_result["atoms"]
    b_gold = gold_result["b0"]

    noise = apply_boundary_noise(
        b_gold=b_gold,
        K=args.K,
        p_shift=args.p_shift,
        p_insert=args.p_insert,
        p_delete=args.p_delete,
        seed=args.seed + sample_idx,
    )

    b_noisy = noise["b_noisy"]
    labels = noise["labels"]

    # Important:
    # training input chunk0_units should correspond to b_noisy (training-time b0)
    noisy_spans = spans_from_boundary_vector(len(atoms), b_noisy)
    chunk0_units = units_from_atoms_and_spans(atoms, noisy_spans)
    unit2atom_span = convert_noisy_spans_to_json(noisy_spans)

    sample = {
        "sample_id": f"{doc_id}::dn::{sample_idx}",
        "doc_id": doc_id,
        "domain": domain,
        "chunk0_units": chunk0_units,
        "atoms": [{"aid": i, "text": t} for i, t in enumerate(atoms)],
        "unit2atom_span": unit2atom_span,
        "b0": b_noisy,
        "labels": labels,
        "meta": {
            "K": args.K,
            "tokenizer": "bge-m3",
            "min_chunk_tokens": 48,
            "max_chunk_tokens": 384,
            "gold_boundary_count": int(sum(b_gold)),
            "noisy_boundary_count": int(sum(b_noisy)),
            "atom_count": len(atoms),
            "noise_stats": noise["stats"],
            "language": language,
        },
        # debug field; later you can remove if you want a cleaner training JSONL
        "b_gold": b_gold,
    }
    return sample


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    files = iter_input_files(args.input, args.input_dir, args.glob)

    output_path = Path(args.output)
    ensure_parent_dir(output_path)

    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        sample_idx = 0
        for file_path in files:
            for raw_doc in load_docs_from_file(file_path):
                sample = process_one_doc(raw_doc, args, sample_idx)
                sample_idx += 1

                if sample is None:
                    skipped += 1
                    continue

                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1

    print(f"Done. wrote={written}, skipped={skipped}, output={output_path}")


if __name__ == "__main__":
    main()