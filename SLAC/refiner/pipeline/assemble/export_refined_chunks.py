from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RefinedChunkExportConfig:
    candidate_select_policy: str = "greedy_first"   # greedy_first | best_teacher_stats
    export_leaf_records: bool = True
    export_doc_catalog: bool = True
    strict_validate: bool = True


# --------------------------------------------------
# Candidate selection
# --------------------------------------------------


def _normalize_candidate_type(x: Optional[str]) -> str:
    return (x or "").strip().lower()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _candidate_priority_key_greedy_first(rec: Dict[str, Any]) -> Tuple[int, float, float, int]:
    """
    Smaller tuple is better.

    Priority:
      1) greedy candidate first
      2) higher mean_edit_prob
      3) higher mean_insert_prob
      4) fewer chunks
    """
    ctype = _normalize_candidate_type(rec.get("candidate_type"))
    is_not_greedy = 0 if ctype == "greedy" else 1

    teacher_stats = rec.get("teacher_stats", {}) or {}
    mean_edit_prob = _safe_float(teacher_stats.get("mean_edit_prob"), 0.0)
    mean_insert_prob = _safe_float(teacher_stats.get("mean_insert_prob"), 0.0)
    num_chunks = _safe_int((rec.get("chunk_stats", {}) or {}).get("num_chunks"), 10**9)

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
    mean_edit_prob = _safe_float(teacher_stats.get("mean_edit_prob"), 0.0)
    mean_insert_prob = _safe_float(teacher_stats.get("mean_insert_prob"), 0.0)
    num_chunks = _safe_int((rec.get("chunk_stats", {}) or {}).get("num_chunks"), 10**9)

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


# --------------------------------------------------
# Boundary / span recovery
# --------------------------------------------------


def _validate_sparse_gaps(gaps: List[int], num_atoms: int) -> List[int]:
    if num_atoms < 1:
        raise ValueError("num_atoms must be >= 1")

    out: List[int] = []
    for g in gaps:
        gi = _safe_int(g, -1)
        if 0 <= gi < num_atoms - 1:
            out.append(gi)
    return sorted(set(out))


def _derive_sparse_from_dense_b0(b0: List[int]) -> List[int]:
    out: List[int] = []
    for i, x in enumerate(b0):
        if _safe_int(x, 0) == 1:
            out.append(i)
    return out


def recover_chunk_spans_from_boundaries(num_atoms: int, b_pred_sparse: List[int]) -> List[Tuple[int, int]]:
    """
    Gap g means boundary between atom g and atom g+1.
    Recover chunk spans in [start_atom, end_atom).
    """
    if num_atoms < 1:
        raise ValueError("num_atoms must be >= 1")

    gaps = _validate_sparse_gaps(b_pred_sparse, num_atoms)
    spans: List[Tuple[int, int]] = []

    start = 0
    for g in gaps:
        end = g + 1
        if not (0 <= start < end <= num_atoms):
            raise ValueError(f"Invalid recovered span [{start}, {end}) from gap={g}")
        spans.append((start, end))
        start = end

    if start < num_atoms:
        spans.append((start, num_atoms))

    if not spans:
        spans = [(0, num_atoms)]

    return spans


# --------------------------------------------------
# Metadata helpers
# --------------------------------------------------


def _join_atoms(atom_slice: List[str]) -> str:
    if not atom_slice:
        return ""
    text = " ".join(x.strip() for x in atom_slice if isinstance(x, str) and x.strip())
    return " ".join(text.split()).strip()


def _find_covering_seed_unit(
    chunk0_units: List[Dict[str, Any]],
    unit2atom_span: List[Dict[str, int]],
    atom_start: int,
    atom_end: int,
) -> Optional[Dict[str, Any]]:
    """
    Choose seed unit with maximum overlap with refined chunk span.
    """
    best_idx: Optional[int] = None
    best_overlap = -1

    for idx, sp in enumerate(unit2atom_span):
        s = _safe_int(sp.get("start_atom"), -1)
        e = _safe_int(sp.get("end_atom"), -1)
        if not (0 <= s < e):
            continue
        ov = max(0, min(atom_end, e) - max(atom_start, s))
        if ov > best_overlap:
            best_overlap = ov
            best_idx = idx

    if best_idx is None:
        return None
    if best_idx >= len(chunk0_units):
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


def _extract_doc_name(chunk0_units: List[Dict[str, Any]]) -> str:
    for u in chunk0_units:
        if (u.get("type") or "").strip().lower() == "title":
            txt = (u.get("text") or "").strip()
            if txt:
                return txt
    if chunk0_units:
        txt = (chunk0_units[0].get("text") or "").strip()
        if txt:
            return txt
    return "unknown_title"


# --------------------------------------------------
# Builders
# --------------------------------------------------


def build_refined_boundary_record(
    refiner_input_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
) -> Dict[str, Any]:
    doc_id = refiner_input_record["doc_id"]
    atoms = refiner_input_record["atoms"]
    b0 = refiner_input_record["b0"]

    prediction = candidate_record.get("prediction", {}) or {}
    b0_sparse_pred = prediction.get("b0_sparse")
    b_pred_sparse = prediction.get("b_pred_sparse", [])

    if not isinstance(b0_sparse_pred, list) or not b0_sparse_pred:
        b0_sparse_pred = _derive_sparse_from_dense_b0(b0)

    b0_sparse = _validate_sparse_gaps(b0_sparse_pred, len(atoms))
    b_pred_sparse = _validate_sparse_gaps(b_pred_sparse, len(atoms))

    chunk_spans = recover_chunk_spans_from_boundaries(len(atoms), b_pred_sparse)

    return {
        "doc_id": doc_id,
        "num_atoms": len(atoms),
        "b0_sparse": b0_sparse,
        "b_pred_sparse": b_pred_sparse,
        "num_seed_boundaries": len(b0_sparse),
        "num_refined_boundaries": len(b_pred_sparse),
        "num_refined_chunks": len(chunk_spans),
        "selected_candidate": {
            "candidate_id": candidate_record.get("candidate_id"),
            "candidate_type": candidate_record.get("candidate_type"),
            "teacher_ckpt": candidate_record.get("teacher_ckpt"),
            "decode_cfg": candidate_record.get("decode_cfg"),
            "teacher_stats": candidate_record.get("teacher_stats"),
            "chunk_stats": candidate_record.get("chunk_stats"),
            "input_stats": candidate_record.get("input_stats"),
        },
    }


def build_refined_chunks(
    refiner_input_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    doc_id = refiner_input_record["doc_id"]
    atoms = refiner_input_record["atoms"]
    chunk0_units = refiner_input_record.get("chunk0_units", []) or []
    unit2atom_span = refiner_input_record.get("unit2atom_span", []) or []

    prediction = candidate_record.get("prediction", {}) or {}
    b_pred_sparse = _validate_sparse_gaps(prediction.get("b_pred_sparse", []), len(atoms))
    chunk_spans = recover_chunk_spans_from_boundaries(len(atoms), b_pred_sparse)

    refined_chunks: List[Dict[str, Any]] = []

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

        refined_chunks.append(
            {
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
        )

    return refined_chunks


def build_leaf_records(refined_chunks: List[Dict[str, Any]], atoms: List[str]) -> List[Dict[str, Any]]:
    leaf_records: List[Dict[str, Any]] = []

    for ch in refined_chunks:
        doc_id = ch["doc_id"]
        chunk_id = ch["chunk_id"]
        chunk_index = ch["chunk_index"]
        atom_start = _safe_int(ch["atom_start"], 0)
        atom_end = _safe_int(ch["atom_end"], 0)
        path = ch.get("path") or []
        depth = ch.get("depth")
        parent_id = ch.get("parent_id")

        for atom_idx in range(atom_start, atom_end):
            atom_text = atoms[atom_idx]
            leaf_records.append(
                {
                    "doc_id": doc_id,
                    "leaf_id": f"{doc_id}::leaf_{atom_idx:05d}",
                    "owner_chunk_id": chunk_id,
                    "leaf_index": atom_idx,
                    "chunk_index": chunk_index,
                    "atom_index": atom_idx,
                    "text": atom_text,
                    "path": path,
                    "depth": depth,
                    "parent_id": parent_id,
                    "source": "refiner_epoch8",
                }
            )

    return leaf_records


def build_doc_catalog_record(
    refiner_input_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
    refined_boundary_record: Dict[str, Any],
    refined_chunks: List[Dict[str, Any]],
    leaf_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    meta = refiner_input_record.get("meta", {}) or {}
    source_path = meta.get("source_path")
    source_type = meta.get("source_type")
    domain = refiner_input_record.get("domain")
    chunk0_units = refiner_input_record.get("chunk0_units", []) or []

    max_depth = None
    depth_values = [ch.get("depth") for ch in refined_chunks if ch.get("depth") is not None]
    if depth_values:
        try:
            max_depth = max(int(x) for x in depth_values)
        except Exception:
            max_depth = None

    return {
        "doc_id": refiner_input_record["doc_id"],
        "doc_name": _extract_doc_name(chunk0_units),
        "source_path": source_path,
        "source_type": source_type,
        "domain": domain,
        "num_chunk0_units": len(chunk0_units),
        "num_atoms": len(refiner_input_record.get("atoms", [])),
        "num_seed_boundaries": refined_boundary_record["num_seed_boundaries"],
        "num_refined_boundaries": refined_boundary_record["num_refined_boundaries"],
        "num_refined_chunks": len(refined_chunks),
        "num_leaf_records": len(leaf_records),
        "max_depth": max_depth,
        "selected_candidate": {
            "candidate_id": candidate_record.get("candidate_id"),
            "candidate_type": candidate_record.get("candidate_type"),
            "teacher_ckpt": candidate_record.get("teacher_ckpt"),
        },
    }


# --------------------------------------------------
# Main export APIs
# --------------------------------------------------


def export_refined_chunks_from_candidate(
    refiner_input_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
    cfg: Optional[RefinedChunkExportConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or RefinedChunkExportConfig()

    boundary_record = build_refined_boundary_record(refiner_input_record, candidate_record)
    refined_chunks = build_refined_chunks(refiner_input_record, candidate_record)

    leaf_records: List[Dict[str, Any]] = []
    if cfg.export_leaf_records:
        leaf_records = build_leaf_records(refined_chunks, refiner_input_record["atoms"])

    export_record: Dict[str, Any] = {
        "doc_id": refiner_input_record["doc_id"],
        "candidate_selected": boundary_record["selected_candidate"],
        "refined_boundary": boundary_record,
        "refined_chunks": refined_chunks,
    }

    if cfg.export_leaf_records:
        export_record["leaf_records"] = leaf_records

    if cfg.export_doc_catalog:
        export_record["doc_catalog"] = build_doc_catalog_record(
            refiner_input_record=refiner_input_record,
            candidate_record=candidate_record,
            refined_boundary_record=boundary_record,
            refined_chunks=refined_chunks,
            leaf_records=leaf_records,
        )

    if cfg.strict_validate:
        validate_export_record(export_record)

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


# --------------------------------------------------
# Flatten helpers
# --------------------------------------------------


def flatten_selected_candidate(export_record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_id": export_record["doc_id"],
        **(export_record.get("candidate_selected", {}) or {}),
    }


def flatten_refined_boundary(export_record: Dict[str, Any]) -> Dict[str, Any]:
    return dict(export_record.get("refined_boundary", {}) or {})


def flatten_refined_chunks(export_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(export_record.get("refined_chunks", []) or [])


def flatten_leaf_records(export_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(export_record.get("leaf_records", []) or [])


def flatten_doc_catalog(export_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    x = export_record.get("doc_catalog")
    if isinstance(x, dict):
        return x
    return None


# --------------------------------------------------
# Validation
# --------------------------------------------------


def validate_export_record(export_record: Dict[str, Any]) -> None:
    if "doc_id" not in export_record:
        raise KeyError("export_record missing doc_id")
    if "candidate_selected" not in export_record:
        raise KeyError("export_record missing candidate_selected")
    if "refined_boundary" not in export_record:
        raise KeyError("export_record missing refined_boundary")
    if "refined_chunks" not in export_record:
        raise KeyError("export_record missing refined_chunks")

    doc_id = export_record["doc_id"]
    boundary = export_record["refined_boundary"]
    chunks = export_record["refined_chunks"]

    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be non-empty string")

    if boundary.get("doc_id") != doc_id:
        raise ValueError("refined_boundary.doc_id mismatch")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("refined_chunks must be a non-empty list")

    prev_end = None
    for i, ch in enumerate(chunks):
        required_keys = [
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
        ]
        missing = [k for k in required_keys if k not in ch]
        if missing:
            raise ValueError(f"refined_chunks[{i}] missing keys: {missing}")

        if ch["doc_id"] != doc_id:
            raise ValueError(f"refined_chunks[{i}].doc_id mismatch")
        if _safe_int(ch["chunk_index"], -1) != i:
            raise ValueError(f"refined_chunks[{i}].chunk_index mismatch")

        s = _safe_int(ch["atom_start"], -1)
        e = _safe_int(ch["atom_end"], -1)
        if not (0 <= s < e):
            raise ValueError(f"Invalid chunk span [{s}, {e}) at index={i}")

        if prev_end is not None and s != prev_end:
            raise ValueError(
                f"Chunks must be contiguous: chunk {i} starts at {s}, prev_end={prev_end}"
            )
        prev_end = e

        if _safe_int(ch["num_atoms"], -1) != e - s:
            raise ValueError(f"Chunk num_atoms mismatch at index={i}")

    leaf_records = export_record.get("leaf_records")
    if leaf_records is not None:
        if not isinstance(leaf_records, list):
            raise ValueError("leaf_records must be a list")
        for i, lr in enumerate(leaf_records):
            required_keys = [
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
            ]
            missing = [k for k in required_keys if k not in lr]
            if missing:
                raise ValueError(f"leaf_records[{i}] missing keys: {missing}")
            if lr["doc_id"] != doc_id:
                raise ValueError(f"leaf_records[{i}].doc_id mismatch")

    doc_catalog = export_record.get("doc_catalog")
    if doc_catalog is not None:
        if not isinstance(doc_catalog, dict):
            raise ValueError("doc_catalog must be a dict when provided")
        if doc_catalog.get("doc_id") != doc_id:
            raise ValueError("doc_catalog.doc_id mismatch")