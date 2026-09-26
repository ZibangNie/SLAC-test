"""Create a non-destructive, diagnostics-only canonical export of named train/dev files.

This command does not discover inputs, read a test split, train, or call an API.
An existing output directory is refused. A failed export retains a failed manifest
and partial output for inspection; only status=complete marks a valid export.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slac_refiner.label_contract import (
    ALIGNMENT_OBJECTIVE,
    CONTRACT_VERSION,
    derive_canonical_labels,
    replay_labels,
    validate_boundary_vector,
)

EXPORT_SCHEMA = "slac-legacy-label-diagnostic-v1"
MAX_ROW_BYTES = 32 * 1024 * 1024
LINEAGE_FLAGS = {
    "legacy_mechanical_repair": True,
    "semantic_gold_verified": False,
    "independent_evaluation": False,
    "cleared_for_training": False,
    "source_split_lineage_unresolved": True,
}


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_input(value: str | Path, split: str) -> Path:
    value = str(value)
    if any(character in value for character in "*?[]"):
        raise ValueError("input paths must be explicit files; glob expressions are forbidden")
    path = Path(value).resolve(strict=True)
    tokens = re.split(r"[^a-z0-9]+", path.stem.lower())
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        raise ValueError("inputs must be existing JSONL files")
    if split not in tokens or "test" in tokens:
        raise ValueError(f"the {split} input filename must identify {split} and must not identify test")
    return path


def repair_row(row: Any, *, split: str, line_number: int, input_sha256: str) -> tuple[dict, dict]:
    if not isinstance(row, dict):
        raise ValueError("row must be a JSON object")
    atoms = row.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("atoms must be a nonempty list")
    if any(not isinstance(atom, (str, dict)) or (isinstance(atom, dict) and not isinstance(atom.get("text"), str)) for atom in atoms):
        raise ValueError("each atom must be text or an object with text")
    for key in ("sample_id", "doc_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"{key} must be a nonempty string")
    meta = row.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("meta must be an object")
    if meta.get("split", split) != split:
        raise ValueError("meta.split disagrees with the explicitly named input split")
    b0 = validate_boundary_vector(row.get("b0"), name="b0", num_atoms=len(atoms))
    target = validate_boundary_vector(row.get("b_gold"), name="b_gold", num_atoms=len(atoms))
    old_labels = row.get("labels")
    old_K = meta.get("K", 6)
    # Legacy SHIFT:0 is accepted only for validating original replay.
    if replay_labels(b0, old_labels, old_K, require_monotone=False, require_canonical_spelling=False) != target:
        raise ValueError("original labels do not replay to b_gold")
    old_targets = []
    for item in old_labels["edit"]:
        if item["y"] != "DEL":
            old_targets.append(item["g"] + (int(item["y"][6:]) if item["y"].startswith("SHIFT:") else 0))
    labels = derive_canonical_labels(b0, target, K=6)
    if replay_labels(b0, labels, K=6) != target:
        raise ValueError("repaired labels do not replay to b_gold")
    normalized_old = [{"g": item["g"], "y": "KEEP" if item["y"] == "SHIFT:0" else item["y"]} for item in old_labels["edit"]]
    updated = dict(row)
    updated["labels"] = labels
    updated["meta"] = {
        **meta,
        "K": 6,
        "num_insert_labels": sum(labels["insert"]),
        "label_repair": {
            "contract_version": CONTRACT_VERSION,
            "export_schema": EXPORT_SCHEMA,
            "input_sha256": input_sha256,
            "input_line": line_number,
            "input_split": split,
            "original_labels_sha256": hashlib.sha256(json_bytes(old_labels)).hexdigest(),
            "original_K": old_K,
            "original_num_insert_labels": meta.get("num_insert_labels"),
            "original_noise_metadata_is_historical_only": True,
            "original_split": row.get("orig_split", "unknown"),
            "flags": dict(LINEAGE_FLAGS),
        },
    }
    emitted = set(g for g, value in enumerate(target) if value and not labels["insert"][g])
    stats = {
        "rows": 1,
        "atoms": len(atoms),
        "original_nonmonotone_rows": int(any(left >= right for left, right in zip(old_targets, old_targets[1:]))),
        "changed_action_assignment_rows": int(normalized_old != labels["edit"] or old_labels["insert"] != labels["insert"]),
        "rows_with_legacy_shift_zero": int(any(item["y"] == "SHIFT:0" for item in old_labels["edit"])),
        "repaired_insert_positives": sum(labels["insert"]),
        "repaired_insert_next_to_edit": sum(int(bool({g - 1, g + 1} & emitted)) for g, value in enumerate(labels["insert"]) if value),
        "validated_replay_rows": 1,
        "validated_monotone_rows": 1,
    }
    for item in labels["edit"]:
        action = "SHIFT" if item["y"].startswith("SHIFT:") else item["y"]
        key = f"repaired_{action.lower()}_actions"
        stats[key] = stats.get(key, 0) + 1
    return updated, stats


def export_split(source: Path, output: Path, split: str) -> tuple[dict, dict[str, set[str]]]:
    source_hash = sha256_file(source)
    counts: Counter = Counter()
    lineage: Counter = Counter()
    families: Counter = Counter()
    identities = {name: set() for name in ("doc_id", "sample_id", "exact_atom_texts")}
    output_hash = hashlib.sha256()
    max_atoms = 0
    with source.open("rb") as incoming, output.open("xb") as outgoing:
        line_number = 0
        while raw := incoming.readline(MAX_ROW_BYTES + 1):
            line_number += 1
            if len(raw) > MAX_ROW_BYTES:
                raise ValueError(f"{split} line {line_number}: row exceeds size limit")
            if not raw.strip():
                raise ValueError(f"{split} line {line_number}: blank JSONL rows are forbidden")
            try:
                row = json.loads(raw)
                repaired, row_stats = repair_row(row, split=split, line_number=line_number, input_sha256=source_hash)
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                # Error messages identify the row and failure class without corpus text.
                raise ValueError(f"{split} line {line_number}: invalid row ({type(exc).__name__})") from None
            counts.update(row_stats)
            max_atoms = max(max_atoms, len(row["atoms"]))
            lineage[str(row.get("orig_split", "unknown"))] += 1
            families[str(row.get("source_family", "unknown"))] += 1
            for name in ("doc_id", "sample_id"):
                identities[name].add(hashlib.sha256(json_bytes(row[name])).hexdigest())
            texts = [atom["text"] if isinstance(atom, dict) else atom for atom in row["atoms"]]
            identities["exact_atom_texts"].add(hashlib.sha256(json_bytes(texts)).hexdigest())
            payload = json_bytes(repaired) + b"\n"
            outgoing.write(payload)
            output_hash.update(payload)
    if not counts["rows"]:
        raise ValueError(f"{split} input is empty")
    if sha256_file(source) != source_hash:
        raise ValueError(f"{split} input changed during export")
    return {
        "input_path": str(source),
        "input_sha256": source_hash,
        "input_unchanged_verified": True,
        "output_file": output.name,
        "output_sha256": output_hash.hexdigest(),
        "counts": dict(counts),
        "max_atoms": max_atoms,
        "original_split_counts": dict(lineage),
        "source_family_counts": dict(families),
        "unique_identities": {name: len(values) for name, values in identities.items()},
    }, identities


def run_export(train_input: str | Path, dev_input: str | Path, output_dir: str | Path) -> dict:
    inputs = {"train": resolve_input(train_input, "train"), "dev": resolve_input(dev_input, "dev")}
    if inputs["train"] == inputs["dev"]:
        raise ValueError("train and dev inputs must be distinct files")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = destination / "manifest.json"
    manifest: dict[str, Any] = {
        "schema": EXPORT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "status": "in_progress",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "K": 6,
        "alignment_objective": ALIGNMENT_OBJECTIVE,
        "usage": "legacy development diagnostics only",
        "flags": LINEAGE_FLAGS,
        "test_split_payload_read": False,
        "limits": [
            "Repair proves mechanical replay and legal action geometry, not semantic gold quality.",
            "Input split names do not establish source independence; orig_split is retained and counted.",
            "Exact atom text hashes do not detect near duplicates or related source documents.",
            "Historical noise ancestry is retained as history and does not define repaired action labels.",
        ],
        "splits": {},
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    try:
        identities = {}
        for split, source in inputs.items():
            report, identities[split] = export_split(source, destination / f"refiner_{split}_diagnostic.jsonl", split)
            manifest["splits"][split] = report
        manifest["train_dev_identity_overlap"] = {
            name: len(identities["train"][name] & identities["dev"][name])
            for name in identities["train"]
        }
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_type"] = type(exc).__name__
        raise
    finally:
        # This manifest is owned by this run; preexisting directories were refused.
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--dev-input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        manifest = run_export(args.train_input, args.dev_input, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(json.dumps({"status": manifest["status"], "output_dir": str(Path(args.output_dir).resolve()), "rows": {split: data["counts"]["rows"] for split, data in manifest["splits"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
