"""
Run only reading/cleaning/rule-based chunk0 generation.

Useful for debugging hard segmentation quality before refiner.
"""
from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..assemble.build_refiner_input import (
    RefinerInputBuildConfig,
    build_refiner_input_from_structure_doc,
)
from ..preprocess.clean_text import CleanTextConfig, clean_document_record
from ..preprocess.normalize_text import NormalizeConfig, normalize_document_record
from ..readers.txt_reader import iter_txt_documents
from ..segment.chunk0_adapter import attach_chunk0_units
from ..segment.rule_segmenter import RuleSegmenterConfig, segment_document_record
from ..utils.io import dump_json, dump_jsonl, ensure_dir
from ..utils.log_utils import log_doc_event, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run txt -> normalize -> clean -> rule segment -> chunk0 -> atoms_b0 pipeline."
    )

    p.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="Input txt/text/md files or directories.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for pipeline artifacts.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories for txt-like files.",
    )
    p.add_argument(
        "--encoding",
        default=None,
        help="Optional explicit text encoding. If omitted, fallback decoding is used.",
    )

    # Output controls
    p.add_argument(
        "--dump_structure_doc",
        action="store_true",
        help="Dump per-doc structure_doc json.",
    )
    p.add_argument(
        "--dump_intermediate_records",
        action="store_true",
        help="Dump per-doc normalized/cleaned/segment intermediate records.",
    )
    p.add_argument(
        "--dump_chunk0_json",
        action="store_true",
        help="Dump per-doc chunk0-enriched structure json.",
    )

    # Domain / metadata
    p.add_argument(
        "--domain",
        default=None,
        help="Optional domain tag to include in atoms_b0 output, e.g. rail.",
    )

    # Normalize config
    p.add_argument("--collapse_inner_whitespace", action="store_true")
    p.add_argument("--drop_blank_lines", action="store_true")
    p.add_argument("--max_consecutive_blank_lines", type=int, default=2)

    # Clean config
    p.add_argument("--disable_drop_decorative_lines", action="store_true")
    p.add_argument("--disable_drop_page_number_lines", action="store_true")
    p.add_argument("--disable_drop_repeated_short_lines", action="store_true")
    p.add_argument("--repeated_short_line_min_repeats", type=int, default=3)
    p.add_argument("--repeated_short_line_max_len", type=int, default=80)
    p.add_argument("--disable_drop_isolated_noise_lines", action="store_true")
    p.add_argument("--noise_line_max_len", type=int, default=3)

    # Segment config
    p.add_argument("--disable_toc_detection", action="store_true")
    p.add_argument("--min_strong_marker_ratio", type=float, default=0.02)
    p.add_argument("--max_weak_marker_ratio", type=float, default=0.45)
    p.add_argument("--disable_allow_single_number_heading", action="store_true")
    p.add_argument("--max_single_heading_number", type=int, default=199)
    p.add_argument("--disable_strict_validate", action="store_true")

    # Atomizer config
    p.add_argument("--atom_max_tokens", type=int, default=120)
    p.add_argument("--atom_max_chars", type=int, default=480)
    p.add_argument("--atom_min_tokens", type=int, default=8)
    p.add_argument("--atom_min_chars", type=int, default=20)
    p.add_argument("--disable_fix_empty_units", action="store_true")
    p.add_argument("--prefer_merge_empty_to_left", action="store_true")

    return p.parse_args()


def build_normalize_config(args: argparse.Namespace) -> NormalizeConfig:
    return NormalizeConfig(
        apply_nfkc=True,
        normalize_newlines=True,
        remove_invisible_chars=True,
        replace_common_typography=True,
        strip_trailing_spaces=True,
        collapse_inner_whitespace=args.collapse_inner_whitespace,
        keep_blank_lines=not args.drop_blank_lines,
        max_consecutive_blank_lines=args.max_consecutive_blank_lines,
        strip_text_edges=True,
    )


def build_clean_config(args: argparse.Namespace) -> CleanTextConfig:
    return CleanTextConfig(
        drop_decorative_lines=not args.disable_drop_decorative_lines,
        drop_page_number_lines=not args.disable_drop_page_number_lines,
        drop_repeated_short_lines=not args.disable_drop_repeated_short_lines,
        repeated_short_line_min_repeats=args.repeated_short_line_min_repeats,
        repeated_short_line_max_len=args.repeated_short_line_max_len,
        repeated_short_line_min_alpha_ratio=0.0,
        drop_isolated_noise_lines=not args.disable_drop_isolated_noise_lines,
        noise_line_max_len=args.noise_line_max_len,
        keep_blank_lines=True,
        max_consecutive_blank_lines=args.max_consecutive_blank_lines,
        strip_text_edges=True,
    )


def build_segment_config(args: argparse.Namespace) -> RuleSegmenterConfig:
    return RuleSegmenterConfig(
        page_break_char="\f",
        max_merge_line_len=220,
        min_unit_text_chars=3,
        min_strong_marker_ratio=args.min_strong_marker_ratio,
        max_weak_marker_ratio=args.max_weak_marker_ratio,
        allow_single_number_heading=not args.disable_allow_single_number_heading,
        max_single_heading_number=args.max_single_heading_number,
        title_scan_limit=30,
        enable_toc_detection=not args.disable_toc_detection,
        toc_early_line_window=200,
        toc_dot_leader_ratio=0.22,
        toc_trailing_page_ratio=0.22,
        table_like_min_separators=2,
        numeric_heavy_token_ratio=0.60,
        strict_validate=not args.disable_strict_validate,
    )


def build_refiner_input_config(args: argparse.Namespace) -> RefinerInputBuildConfig:
    return RefinerInputBuildConfig(
        atom_max_tokens=args.atom_max_tokens,
        atom_max_chars=args.atom_max_chars,
        atom_min_tokens=args.atom_min_tokens,
        atom_min_chars=args.atom_min_chars,
        split_cjk_sentences=True,
        split_en_sentences=True,
        dot_in_cjk=True,
        line_fallback=True,
        fix_empty_units=not args.disable_fix_empty_units,
        prefer_merge_empty_to_right=not args.prefer_merge_empty_to_left,
        require_nonempty_atoms=True,
        strict_validate=not args.disable_strict_validate,
    )


def ensure_output_subdirs(output_dir: Path) -> Dict[str, Path]:
    out = {
        "root": output_dir,
        "logs": output_dir / "logs",
        "atoms_b0": output_dir / "atoms_b0",
        "structure_doc": output_dir / "structure_doc",
        "chunk0_json": output_dir / "chunk0_json",
        "intermediate": output_dir / "intermediate",
        "summaries": output_dir / "summaries",
    }
    for p in out.values():
        ensure_dir(p)
    return out


def safe_doc_file_name(doc_id: str) -> str:
    return doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")


def build_run_summary_template(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "args": vars(args),
        "num_docs_total": 0,
        "num_docs_success": 0,
        "num_docs_failed": 0,
        "docs": [],
    }


def process_one_document(
    doc_record: Dict[str, Any],
    *,
    out_dirs: Dict[str, Path],
    logger: logging.Logger,
    normalize_cfg: NormalizeConfig,
    clean_cfg: CleanTextConfig,
    segment_cfg: RuleSegmenterConfig,
    refiner_input_cfg: RefinerInputBuildConfig,
    domain: Optional[str],
    dump_structure_doc: bool,
    dump_intermediate_records: bool,
    dump_chunk0_json: bool,
) -> Dict[str, Any]:
    doc_id = doc_record["doc_id"]
    source_path = doc_record.get("source_path")
    doc_file = safe_doc_file_name(doc_id)

    log_doc_event(
        logger,
        logging.INFO,
        f"Start processing doc: {doc_id}",
        doc_id=doc_id,
        source_path=source_path,
        event="doc_start",
    )

    # 1) normalize
    normalized_record = normalize_document_record(
        doc_record,
        cfg=normalize_cfg,
        src_field="raw_text",
        dst_field="normalized_text",
    )

    # 2) clean
    cleaned_record = clean_document_record(
        normalized_record,
        cfg=clean_cfg,
        src_field="normalized_text",
        dst_field="cleaned_text",
    )

    # 3) segment
    segment_record = segment_document_record(
        cleaned_record,
        src_field="cleaned_text",
        cfg=segment_cfg,
        logger=logger,
    )

    structure_doc = segment_record["structure_doc"]

    # 4) attach chunk0 units
    structure_doc_with_chunk0 = attach_chunk0_units(
        structure_doc,
        include_title=True,
        include_other_nonroot=False,
        include_root_in_path=False,
    )

    # 5) build refiner input
    refiner_input_record = build_refiner_input_from_structure_doc(
        structure_doc_with_chunk0,
        domain=domain,
        cfg=refiner_input_cfg,
        meta={
            "source_path": source_path,
            "source_type": doc_record.get("source_type"),
            "pipeline_stage": "run_rule_chunk0_only",
        },
    )

    # ---- optional dumps ----
    if dump_intermediate_records:
        dump_json(
            out_dirs["intermediate"] / f"{doc_file}.normalized.json",
            normalized_record,
            indent=2,
            ensure_ascii=False,
        )
        dump_json(
            out_dirs["intermediate"] / f"{doc_file}.cleaned.json",
            cleaned_record,
            indent=2,
            ensure_ascii=False,
        )
        dump_json(
            out_dirs["intermediate"] / f"{doc_file}.segment_record.json",
            segment_record,
            indent=2,
            ensure_ascii=False,
        )

    if dump_structure_doc:
        dump_json(
            out_dirs["structure_doc"] / f"{doc_file}.structure_doc.json",
            structure_doc,
            indent=2,
            ensure_ascii=False,
        )

    if dump_chunk0_json:
        dump_json(
            out_dirs["chunk0_json"] / f"{doc_file}.chunk0.json",
            structure_doc_with_chunk0,
            indent=2,
            ensure_ascii=False,
        )

    dump_json(
        out_dirs["atoms_b0"] / f"{doc_file}.atoms_b0.json",
        refiner_input_record,
        indent=2,
        ensure_ascii=False,
    )

    summary = {
        "doc_id": doc_id,
        "source_path": source_path,
        "status": "success",
        "language": structure_doc.get("language"),
        "doc_name": structure_doc.get("doc_name"),
        "num_units": len(structure_doc.get("units", [])),
        "num_chunk0_units": len(structure_doc_with_chunk0.get("chunk0_units", [])),
        "num_atoms": len(refiner_input_record.get("atoms", [])),
        "num_seed_boundaries": int(sum(refiner_input_record.get("b0", []))),
        "segment_diagnostics": segment_record.get("diagnostics", {}),
        "refiner_builder_stats": refiner_input_record.get("meta", {}).get("stats", {}),
        "atoms_b0_path": (out_dirs["atoms_b0"] / f"{doc_file}.atoms_b0.json").as_posix(),
    }

    log_doc_event(
        logger,
        logging.INFO,
        f"Finished doc: {doc_id} | atoms={summary['num_atoms']} | chunk0={summary['num_chunk0_units']}",
        doc_id=doc_id,
        source_path=source_path,
        event="doc_success",
        extra_json={
            "num_atoms": summary["num_atoms"],
            "num_chunk0_units": summary["num_chunk0_units"],
        },
    )

    return {
        "summary": summary,
        "refiner_input_record": refiner_input_record,
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    out_dirs = ensure_output_subdirs(output_dir)

    logger = setup_logger(
        "refiner.pipeline.run_rule_chunk0_only",
        log_file=out_dirs["logs"] / "run.log",
        level=logging.INFO,
        console=True,
        jsonl_file=False,
    )
    structured_logger = setup_logger(
        "refiner.pipeline.run_rule_chunk0_only.jsonl",
        log_file=out_dirs["logs"] / "run_events.jsonl",
        level=logging.INFO,
        console=False,
        jsonl_file=True,
    )

    normalize_cfg = build_normalize_config(args)
    clean_cfg = build_clean_config(args)
    segment_cfg = build_segment_config(args)
    refiner_input_cfg = build_refiner_input_config(args)

    run_summary = build_run_summary_template(args)
    atoms_b0_records: List[Dict[str, Any]] = []

    try:
        docs_iter = iter_txt_documents(
            input_paths=args.input_paths,
            root=None,
            encoding=args.encoding,
            errors="strict",
            recursive=args.recursive,
        )

        for doc_record in docs_iter:
            run_summary["num_docs_total"] += 1
            doc_id = doc_record.get("doc_id")
            source_path = doc_record.get("source_path")

            try:
                result = process_one_document(
                    doc_record,
                    out_dirs=out_dirs,
                    logger=logger,
                    normalize_cfg=normalize_cfg,
                    clean_cfg=clean_cfg,
                    segment_cfg=segment_cfg,
                    refiner_input_cfg=refiner_input_cfg,
                    domain=args.domain,
                    dump_structure_doc=args.dump_structure_doc,
                    dump_intermediate_records=args.dump_intermediate_records,
                    dump_chunk0_json=args.dump_chunk0_json,
                )
                run_summary["num_docs_success"] += 1
                run_summary["docs"].append(result["summary"])
                atoms_b0_records.append(result["refiner_input_record"])

                log_doc_event(
                    structured_logger,
                    logging.INFO,
                    f"doc success: {doc_id}",
                    doc_id=doc_id,
                    source_path=source_path,
                    event="doc_success",
                    extra_json=result["summary"],
                )

            except Exception as e:
                run_summary["num_docs_failed"] += 1
                err_summary = {
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=20),
                }
                run_summary["docs"].append(err_summary)

                logger.exception(f"Failed doc: {doc_id}")
                log_doc_event(
                    structured_logger,
                    logging.ERROR,
                    f"doc failed: {doc_id}",
                    doc_id=doc_id,
                    source_path=source_path,
                    event="doc_failed",
                    extra_json=err_summary,
                )

        # Aggregate outputs
        dump_jsonl(
            out_dirs["root"] / "atoms_b0.jsonl",
            atoms_b0_records,
            ensure_ascii=False,
            validate_dict=True,
        )
        dump_json(
            out_dirs["summaries"] / "run_summary.json",
            run_summary,
            indent=2,
            ensure_ascii=False,
        )

        logger.info(
            f"Done. total={run_summary['num_docs_total']} "
            f"success={run_summary['num_docs_success']} "
            f"failed={run_summary['num_docs_failed']}"
        )

    except Exception:
        logger.exception("Fatal pipeline failure")
        raise


if __name__ == "__main__":
    main()