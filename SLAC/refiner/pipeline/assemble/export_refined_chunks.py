"""
Refined chunk exporter.

Main responsibility:
- read refiner predictions
- recover final chunk spans from b_pred_sparse
- export refined_chunks.jsonl for retrieval
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RefinedChunkExportConfig:
    candidate_select_policy: str = "greedy_first"  # greedy_first | best_teacher_stats
    export_leaf_records: bool = True
    strict_validate: bool = True


def _normalize_candidate_type(x: Optional[str]) -> str:
    return (x or "").strip().lower()


def _candidate_priority_key_greedy_first(rec: Dict[str, Any]) -> Tuple[int, float, float, int]:
    """
    Smaller tuple is better.
    Priority:
      1) greedy candidate first
      2) higher mean_edit_prob
      3) higher mean_insert_prob
      4) fewer chunk count inflation (mild preference)
    """
    ctype = _normalize_candidate_type(rec.get("candidate_type"))
    is_not_greedy = 0 if ctype == "greedy" else 1

    teacher_stats = rec.get("teacher_stats", {}) or {}
    mean_edit_prob = float(teacher_stats.get("mean_edit_prob", 0.0) or 0.0)
    mean_insert_prob = float(teacher_stats.get("mean_insert_prob", 0.0) or 0.0)
    num_chunks = int((rec.get("chunk_stats", {}) or {}).get("num_chunks", 10**9))

    return (
        is_not_greedy,
        -mean_edit_prob,
        -mean_insert_prob,
        num_chunks,
    )


def _candidate_priority_key_best_teacher(rec: Dict[str, Any]) -> Tuple[float, float, int, int]:
    """
    Smaller tuple is better.
    Priority:
      1) higher mean_edit_prob
      2) higher mean_insert_prob
      3) prefer greedy on tie
      4) fewer chunks
    """
    ctype = _normalize_candidate_type(rec.get("candidate_type"))
    is_not_greedy = 0 if ctype == "greedy" else 1

    teacher_stats = rec.get("teacher_stats", {}) or {}
    mean_edit_prob = float(teacher_stats.get("mean_edit_prob", 0.0) or 0.0)
    mean_insert_prob = float(teacher_stats.get("mean_insert_prob", 0.0) or 0.0)
    num_chunks = int((rec.get("chunk_stats", {}) or {}).get("num_chunks", 10**9))

    return (
        -mean_edit_prob,
        -mean_insert_prob,
        is_not_greedy,
        num_chunks,
    )


def select_best_candidate(
    candidate_records: List[Dict[str, Any]],
    cfg: Optional[RefinedChunkExportConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or RefinedChunkExportConfig()

    if not candidate_records:
        raise ValueError("candidate_records is empty")

    if cfg.candidate_select_policy == "greedy_first":
        ranked = sorted(candidate_records, key=_candidate_priority_key_greedy_first)
    elif cfg.candidate_select_policy == "best_teacher_stats":
        ranked = sorted(candidate_records, key=_candidate_priority_key_best_teacher)
    else:
        raise ValueError(f"Unsupported candidate_select_policy: {cfg.candidate_select_policy}")

    return ranked[0]


def _validate_b_pred_sparse(b_pred_sparse: List[int], num_atoms: int) -> List[int]:
    if num_atoms < 1:
        raise ValueError("num_atoms must be >= 1")

    out: List[int] = []
    for g in b_pred_sparse:
        gi = int(g)
        if 0 <= gi < num_atoms - 1:
            out.append(gi)

    out = sorted(set(out))
    return out


def recover_chunk_spans_from_boundaries(
    num_atoms: int,
    b_pred_sparse: List[int],
) -> List[Tuple[int, int]]:
    """
    Boundaries are gaps g between atom g and atom g+1.
    Recover chunk spans in [start_atom, end_atom).
    """
    if num_atoms < 1:
        raise ValueError("num_atoms must be >= 1")

    gaps = _validate_b_pred_sparse(b_pred_sparse, num_atoms)
    spans: List[Tuple[int, int]] = []

    start = 0
    for g in gaps:
        end = g + 1
        if not (0 <= start < end <= num_atoms):
            raise ValueError(f"Invalid recovered span [{start}, {end}) from gap {g}")
        spans.append((start, end))
        start = end

    if not (0 <= start < num_atoms + 1):
        raise ValueError(f"Invalid final start={start} with num_atoms={num_atoms}")
    if start < num_atoms:
        spans.append((start, num_atoms))

    if not spans:
        spans = [(0, num_atoms)]

    return spans


def _join_atoms(atom_slice: List[str]) -> str:
    if not atom_slice:
        return ""
    text = " ".join(x.strip() for x in atom_slice if x and x.strip())
    return " ".join(text.split()).strip()


def _find_covering_seed_unit(
    chunk0_units: List[Dict[str, Any]],
    unit2atom_span: List[Dict[str, int]],
    atom_start: int,
    atom_end: int,
) -> Optional[Dict[str, Any]]:
    """
    Choose the seed unit with max overlap against recovered refined chunk span.
    """
    best_idx: Optional[int] = None
    best_overlap = -1

    for idx, sp in enumerate(unit2atom_span):
        s = int(sp["start_atom"])
        e = int(sp["end_atom"])
        ov = max(0, min(atom_end, e) - max(atom_start, s))
        if ov > best_overlap:
            best_overlap = ov
            best_idx = idx

    if best_idx is None or best_idx >= len(chunk0_units):
        return None
    return chunk0_units[best_idx]


def _extract_path_depth_parent(seed_unit: Optional[Dict[str, Any]]) -> Tuple[List[str], Optional[int], Optional[int]]:
    if not seed_unit:
        return [], None, None
    path = seed_unit.get("path") or []
    if not isinstance(path, list):
        path = []
    depth = seed_unit.get("depth")
    parent_id = seed_unit.get("parent_id")
    try:
        depth = int(depth) if depth is not None else None
    except Exception:
        depth = None
    try:
        parent_id = int(parent_id) if parent_id is not None else None
    except Exception:
        parent_id = None
    return path, depth, parent_id


def export_refined_chunks_from_candidate(
    refiner_input_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
    cfg: Optional[RefinedChunkExportConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or RefinedChunkExportConfig()

    doc_id = refiner_input_record["doc_id"]
    atoms = refiner_input_record["atoms"]
    chunk0_units = refiner_input_record.get("chunk0_units", [])
    unit2atom_span = refiner_input_record.get("unit2atom_span", [])
    num_atoms = len(atoms)

    prediction = candidate_record.get("prediction", {}) or {}
    b_pred_sparse = prediction.get("b_pred_sparse", [])
    b_pred_sparse = _validate_b_pred_sparse(b_pred_sparse, num_atoms)
    chunk_spans = recover_chunk_spans_from_boundaries(num_atoms, b_pred_sparse)

    refined_chunks: List[Dict[str, Any]] = []
    leaf_records: List[Dict[str, Any]] = []

    for idx, (atom_start, atom_end) in enumerate(chunk_spans):
        atom_slice = atoms[atom_start:atom_end]
        text = _join_atoms(atom_slice)
        seed_unit = _find_covering_seed_unit(
            chunk0_units=chunk0_units,
            unit2atom_span=unit2atom_span,
            atom_start=atom_start,
            atom_end=atom_end,
        )
        path, depth, parent_id = _extract_path_depth_parent(seed_unit)

        chunk_id = f"{doc_id}::chunk_{idx:05d}"
        prev_chunk_id = f"{doc_id}::chunk_{idx-1:05d}" if idx > 0 else None
        next_chunk_id = f"{doc_id}::chunk_{idx+1:05d}" if idx + 1 < len(chunk_spans) else None

        refined_chunk = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "atom_start": atom_start,
            "atom_end": atom_end,
            "text": text,
            "num_atoms": atom_end - atom_start,
            "path": path,
            "depth": depth,
            "parent_id": parent_id,
            "prev_chunk_id": prev_chunk_id,
            "next_chunk_id": next_chunk_id,
            "source": "refiner_epoch8",
            "boundary_meta": {
                "seed_type": "chunk0",
                "candidate_type": candidate_record.get("candidate_type"),
                "candidate_id": candidate_record.get("candidate_id"),
                "teacher_ckpt": candidate_record.get("teacher_ckpt"),
                "decode_cfg": candidate_record.get("decode_cfg"),
                "teacher_stats": candidate_record.get("teacher_stats"),
                "input_stats": candidate_record.get("input_stats"),
            },
        }
        refined_chunks.append(refined_chunk)

        if cfg.export_leaf_records:
            for local_i, atom_text in enumerate(atom_slice):
                global_atom_idx = atom_start + local_i
                leaf_records.append(
                    {
                        "doc_id": doc_id,
                        "leaf_id": f"{doc_id}::leaf_{global_atom_idx:05d}",
                        "owner_chunk_id": chunk_id,
                        "leaf_index": global_atom_idx,
                        "chunk_index": idx,
                        "atom_index": global_atom_idx,
                        "text": atom_text,
                        "path": path,
                        "depth": depth,
                        "parent_id": parent_id,
                        "source": "refiner_epoch8",
                    }
                )

    export_record = {
        "doc_id": doc_id,
        "candidate_selected": {
            "candidate_id": candidate_record.get("candidate_id"),
            "candidate_type": candidate_record.get("candidate_type"),
            "teacher_ckpt": candidate_record.get("teacher_ckpt"),
            "decode_cfg": candidate_record.get("decode_cfg"),
            "teacher_stats": candidate_record.get("teacher_stats"),
            "chunk_stats": candidate_record.get("chunk_stats"),
        },
        "prediction": {
            "b_pred_sparse": b_pred_sparse,
        },
        "refined_chunks": refined_chunks,
    }

    if cfg.export_leaf_records:
        export_record["leaf_records"] = leaf_records

    if cfg.strict_validate:
        validate_refined_export_record(export_record)

    return export_record


def export_refined_chunks_from_candidates(
    refiner_input_record: Dict[str, Any],
    candidate_records: List[Dict[str, Any]],
    cfg: Optional[RefinedChunkExportConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or RefinedChunkExportConfig()
    best = select_best_candidate(candidate_records, cfg=cfg)
    return export_refined_chunks_from_candidate(
        refiner_input_record=refiner_input_record,
        candidate_record=best,
        cfg=cfg,
    )


def flatten_refined_chunks(export_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(export_record.get("refined_chunks", []) or [])


def flatten_leaf_records(export_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(export_record.get("leaf_records", []) or [])


def validate_refined_export_record(export_record: Dict[str, Any]) -> None:
    if "doc_id" not in export_record:
        raise KeyError("export_record missing doc_id")
    if "refined_chunks" not in export_record:
        raise KeyError("export_record missing refined_chunks")

    doc_id = export_record["doc_id"]
    chunks = export_record["refined_chunks"]

    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be non-empty string")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("refined_chunks must be a non-empty list")

    prev_end = None
    for i, ch in enumerate(chunks):
        for k in (
            "doc_id",
            "chunk_id",
            "chunk_index",
            "atom_start",
            "atom_end",
            "text",
            "num_atoms",
            "path",
            "depth",
            "parent_id",
            "prev_chunk_id",
            "next_chunk_id",
            "source",
            "boundary_meta",
        ):
            if k not in ch:
                raise ValueError(f"refined_chunks[{i}] missing key: {k}")

        if ch["doc_id"] != doc_id:
            raise ValueError(f"refined_chunks[{i}].doc_id mismatch: {ch['doc_id']} vs {doc_id}")
        if ch["chunk_index"] != i:
            raise ValueError(f"refined_chunks[{i}].chunk_index must equal its position, got {ch['chunk_index']}")

        s = int(ch["atom_start"])
        e = int(ch["atom_end"])
        if not (0 <= s < e):
            raise ValueError(f"Invalid chunk span [{s}, {e}) at chunk {i}")

        if prev_end is not None and s != prev_end:
            raise ValueError(f"Chunks must be contiguous: chunk {i} starts at {s}, prev_end={prev_end}")
        prev_end = e

        if int(ch["num_atoms"]) != e - s:
            raise ValueError(
                f"Chunk num_atoms mismatch at chunk {i}: {ch['num_atoms']} vs expected {e-s}"
            )

    leaf_records = export_record.get("leaf_records")
    if leaf_records is not None:
        if not isinstance(leaf_records, list):
            raise ValueError("leaf_records must be a list when provided")
        for i, lr in enumerate(leaf_records):
            for k in (
                "doc_id",
                "leaf_id",
                "owner_chunk_id",
                "leaf_index",
                "chunk_index",
                "atom_index",
                "text",
                "path",
                "depth",
                "parent_id",
                "source",
            ):
                if k not in lr:
                    raise ValueError(f"leaf_records[{i}] missing key: {k}")
            if lr["doc_id"] != doc_id:
                raise ValueError(f"leaf_records[{i}].doc_id mismatch")