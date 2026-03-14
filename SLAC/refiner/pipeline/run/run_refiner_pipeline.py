from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..configs.loader import (
    filter_allowed_keys,
    get_runner_config,
    load_pipeline_config,
    merge_resolved_args,
    namespace_from_dict,
    preparse_config_arg,
)
from ..utils.io import dump_json, ensure_dir
from ..utils.log_utils import setup_logger


DEFAULTS: Dict[str, Any] = {
    "input_paths": None,
    "output_dir": None,
    "recursive": False,
    "domain": None,

    "skip_stage1": False,
    "skip_stage2": False,
    "reuse_atoms_b0_jsonl": None,

    "dump_structure_doc": False,
    "dump_chunk0_json": False,
    "dump_intermediate_records": False,

    "pdf_max_pages": 0,
    "pdf_min_font_size": 5.0,
    "pdf_max_weird_ratio_span": 0.35,
    "pdf_drop_tiny_spans": True,
    "pdf_garbled_page_weird_ratio": 0.25,
    "pdf_garbled_min_len": 10,
    "pdf_ocr_on_images": True,
    "pdf_ocr_always_on_images": False,
    "pdf_ocr_dpi": 300,
    "pdf_ocr_lang": "chi_sim+eng",
    "pdf_ocr_backend": "tesseract",
    "pdf_enable_pdftotext": True,
    "pdf_ocr_when_text_short": 120,
    "pdf_image_area_ratio_for_ocr": 0.35,

    "collapse_inner_whitespace": False,
    "drop_blank_lines": False,
    "max_consecutive_blank_lines": 2,

    "drop_decorative_lines": True,
    "drop_page_number_lines": True,
    "drop_repeated_short_lines": True,
    "repeated_short_line_min_repeats": 3,
    "repeated_short_line_max_len": 80,
    "drop_isolated_noise_lines": True,
    "noise_line_max_len": 3,

    "toc_detection": True,
    "min_strong_marker_ratio": 0.02,
    "max_weak_marker_ratio": 0.45,
    "allow_single_number_heading": True,
    "max_single_heading_number": 199,
    "strict_validate": True,

    "atom_max_tokens": 120,
    "atom_max_chars": 480,
    "atom_min_tokens": 8,
    "atom_min_chars": 20,
    "fix_empty_units": True,
    "prefer_merge_empty_to_left": False,

    "ckpt": "/root/autodl-tmp/runs/refiner/week2_mixed_llmgold_x4_e3/checkpoints/epoch_8.pt",
    "doc_layers": 1,
    "window_size": 8,
    "k_shift": 6,
    "refine_passes": 2,
    "temperatures": [0.90, 1.00],
    "insert_thresholds": [0.45, 0.50],
    "seeds": [11, 22, 33],
    "include_identity": False,
    "include_greedy": False,

    "candidate_select_policy": "greedy_first",
    "export_leaf_records": True,
    "export_doc_catalog": True,

    "infer_script": None,
}



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    config_arg, remaining = preparse_config_arg(argv)
    raw_cfg, loaded_cfg_path = load_pipeline_config(config_arg)
    runner_cfg = get_runner_config(raw_cfg, "run_refiner_pipeline")
    config_values, unknown_cfg_keys = filter_allowed_keys(runner_cfg, set(DEFAULTS.keys()))

    p = argparse.ArgumentParser(
        description=(
            "End-to-end refiner pipeline runner: "
            "raw docs (pdf/docx/txt/json) -> rule chunk0 -> atoms_b0 -> "
            "frozen refiner infer -> refined outputs"
        ),
        argument_default=argparse.SUPPRESS,
    )

    p.add_argument("--config", help="Optional pipeline config YAML path.")
    p.add_argument("--input_paths", nargs="+")
    p.add_argument("--output_dir")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--domain")

    p.add_argument("--skip_stage1", action="store_true")
    p.add_argument("--skip_stage2", action="store_true")
    p.add_argument("--reuse_atoms_b0_jsonl")

    p.add_argument("--dump_structure_doc", action="store_true")
    p.add_argument("--dump_chunk0_json", action="store_true")
    p.add_argument("--dump_intermediate_records", action="store_true")

    # PDF
    p.add_argument("--pdf_max_pages", type=int)
    p.add_argument("--pdf_min_font_size", type=float)
    p.add_argument("--pdf_max_weird_ratio_span", type=float)
    p.add_argument("--disable_pdf_drop_tiny_spans", action="store_true")
    p.add_argument("--pdf_garbled_page_weird_ratio", type=float)
    p.add_argument("--pdf_garbled_min_len", type=int)
    p.add_argument("--disable_pdf_ocr_on_images", action="store_true")
    p.add_argument("--pdf_ocr_always_on_images", action="store_true")
    p.add_argument("--pdf_ocr_dpi", type=int)
    p.add_argument("--pdf_ocr_lang")
    p.add_argument("--pdf_ocr_backend")
    p.add_argument("--disable_pdf_enable_pdftotext", action="store_true")
    p.add_argument("--pdf_ocr_when_text_short", type=int)
    p.add_argument("--pdf_image_area_ratio_for_ocr", type=float)

    # normalize
    p.add_argument("--collapse_inner_whitespace", action="store_true")
    p.add_argument("--drop_blank_lines", action="store_true")
    p.add_argument("--max_consecutive_blank_lines", type=int)

    # clean
    p.add_argument("--disable_drop_decorative_lines", action="store_true")
    p.add_argument("--disable_drop_page_number_lines", action="store_true")
    p.add_argument("--disable_drop_repeated_short_lines", action="store_true")
    p.add_argument("--repeated_short_line_min_repeats", type=int)
    p.add_argument("--repeated_short_line_max_len", type=int)
    p.add_argument("--disable_drop_isolated_noise_lines", action="store_true")
    p.add_argument("--noise_line_max_len", type=int)

    # segment
    p.add_argument("--disable_toc_detection", action="store_true")
    p.add_argument("--min_strong_marker_ratio", type=float)
    p.add_argument("--max_weak_marker_ratio", type=float)
    p.add_argument("--disable_allow_single_number_heading", action="store_true")
    p.add_argument("--max_single_heading_number", type=int)
    p.add_argument("--disable_strict_validate", action="store_true")

    # atomizer
    p.add_argument("--atom_max_tokens", type=int)
    p.add_argument("--atom_max_chars", type=int)
    p.add_argument("--atom_min_tokens", type=int)
    p.add_argument("--atom_min_chars", type=int)
    p.add_argument("--disable_fix_empty_units", action="store_true")
    p.add_argument("--prefer_merge_empty_to_left", action="store_true")

    # stage2
    p.add_argument("--ckpt")
    p.add_argument("--doc_layers", type=int)
    p.add_argument("--window_size", type=int)
    p.add_argument("--k_shift", type=int)
    p.add_argument("--refine_passes", type=int)
    p.add_argument("--temperatures", nargs="+", type=float)
    p.add_argument("--insert_thresholds", nargs="+", type=float)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--include_identity", action="store_true")
    p.add_argument("--include_greedy", action="store_true")
    p.add_argument(
        "--candidate_select_policy",
        choices=["greedy_first", "best_teacher_stats"],
    )
    p.add_argument("--disable_export_leaf_records", action="store_true")
    p.add_argument("--disable_export_doc_catalog", action="store_true")
    p.add_argument("--infer_script")

    cli_values = vars(p.parse_args(remaining))
    resolved = merge_resolved_args(DEFAULTS, config_values, cli_values)

    if cli_values.get("disable_pdf_drop_tiny_spans"):
        resolved["pdf_drop_tiny_spans"] = False
    if cli_values.get("disable_pdf_ocr_on_images"):
        resolved["pdf_ocr_on_images"] = False
    if cli_values.get("disable_pdf_enable_pdftotext"):
        resolved["pdf_enable_pdftotext"] = False

    if cli_values.get("disable_drop_decorative_lines"):
        resolved["drop_decorative_lines"] = False
    if cli_values.get("disable_drop_page_number_lines"):
        resolved["drop_page_number_lines"] = False
    if cli_values.get("disable_drop_repeated_short_lines"):
        resolved["drop_repeated_short_lines"] = False
    if cli_values.get("disable_drop_isolated_noise_lines"):
        resolved["drop_isolated_noise_lines"] = False

    if cli_values.get("disable_toc_detection"):
        resolved["toc_detection"] = False
    if cli_values.get("disable_allow_single_number_heading"):
        resolved["allow_single_number_heading"] = False
    if cli_values.get("disable_strict_validate"):
        resolved["strict_validate"] = False
    if cli_values.get("disable_fix_empty_units"):
        resolved["fix_empty_units"] = False

    if cli_values.get("disable_export_leaf_records"):
        resolved["export_leaf_records"] = False
    if cli_values.get("disable_export_doc_catalog"):
        resolved["export_doc_catalog"] = False

    resolved["config"] = str(loaded_cfg_path) if loaded_cfg_path is not None else config_arg
    resolved["_unknown_config_keys"] = unknown_cfg_keys

    if not resolved.get("output_dir"):
        p.error("Missing required output: --output_dir (or set run_refiner_pipeline.output_dir in config).")

    if not resolved.get("skip_stage1"):
        if not resolved.get("input_paths") and not resolved.get("reuse_atoms_b0_jsonl"):
            p.error(
                "Stage 1 needs --input_paths unless you set --reuse_atoms_b0_jsonl "
                "or configure run_refiner_pipeline.input_paths."
            )

    return namespace_from_dict(resolved)


def ensure_output_subdirs(output_dir: Path) -> dict[str, Path]:
    out = {
        "root": output_dir,
        "logs": output_dir / "logs",
        "stage1": output_dir / "stage1_rule_chunk0",
        "stage2": output_dir / "stage2_refiner_infer",
        "summaries": output_dir / "summaries",
    }
    for p in out.values():
        ensure_dir(p)
    return out


def build_stage1_cmd(args: argparse.Namespace, stage1_output_dir: Path) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "SLAC.refiner.pipeline.run.run_rule_chunk0_only",
        "--output_dir", str(stage1_output_dir),
    ]

    if args.config:
        cmd.extend(["--config", args.config])

    if args.input_paths:
        cmd.extend(["--input_paths", *args.input_paths])
    if args.recursive:
        cmd.append("--recursive")
    if args.domain is not None:
        cmd.extend(["--domain", args.domain])

    if args.dump_structure_doc:
        cmd.append("--dump_structure_doc")
    if args.dump_chunk0_json:
        cmd.append("--dump_chunk0_json")
    if args.dump_intermediate_records:
        cmd.append("--dump_intermediate_records")

    # pdf config
    cmd.extend(["--pdf_max_pages", str(args.pdf_max_pages)])
    cmd.extend(["--pdf_min_font_size", str(args.pdf_min_font_size)])
    cmd.extend(["--pdf_max_weird_ratio_span", str(args.pdf_max_weird_ratio_span)])
    if not args.pdf_drop_tiny_spans:
        cmd.append("--disable_pdf_drop_tiny_spans")
    cmd.extend(["--pdf_garbled_page_weird_ratio", str(args.pdf_garbled_page_weird_ratio)])
    cmd.extend(["--pdf_garbled_min_len", str(args.pdf_garbled_min_len)])
    if not args.pdf_ocr_on_images:
        cmd.append("--disable_pdf_ocr_on_images")
    if args.pdf_ocr_always_on_images:
        cmd.append("--pdf_ocr_always_on_images")
    cmd.extend(["--pdf_ocr_dpi", str(args.pdf_ocr_dpi)])
    cmd.extend(["--pdf_ocr_lang", args.pdf_ocr_lang])
    cmd.extend(["--pdf_ocr_backend", args.pdf_ocr_backend])
    if not args.pdf_enable_pdftotext:
        cmd.append("--disable_pdf_enable_pdftotext")
    cmd.extend(["--pdf_ocr_when_text_short", str(args.pdf_ocr_when_text_short)])
    cmd.extend(["--pdf_image_area_ratio_for_ocr", str(args.pdf_image_area_ratio_for_ocr)])

    # normalize config
    if args.collapse_inner_whitespace:
        cmd.append("--collapse_inner_whitespace")
    if args.drop_blank_lines:
        cmd.append("--drop_blank_lines")
    cmd.extend(["--max_consecutive_blank_lines", str(args.max_consecutive_blank_lines)])

    # clean config
    if not args.drop_decorative_lines:
        cmd.append("--disable_drop_decorative_lines")
    if not args.drop_page_number_lines:
        cmd.append("--disable_drop_page_number_lines")
    if not args.drop_repeated_short_lines:
        cmd.append("--disable_drop_repeated_short_lines")
    cmd.extend(["--repeated_short_line_min_repeats", str(args.repeated_short_line_min_repeats)])
    cmd.extend(["--repeated_short_line_max_len", str(args.repeated_short_line_max_len)])
    if not args.drop_isolated_noise_lines:
        cmd.append("--disable_drop_isolated_noise_lines")
    cmd.extend(["--noise_line_max_len", str(args.noise_line_max_len)])

    # segment config
    if not args.toc_detection:
        cmd.append("--disable_toc_detection")
    cmd.extend(["--min_strong_marker_ratio", str(args.min_strong_marker_ratio)])
    cmd.extend(["--max_weak_marker_ratio", str(args.max_weak_marker_ratio)])
    if not args.allow_single_number_heading:
        cmd.append("--disable_allow_single_number_heading")
    cmd.extend(["--max_single_heading_number", str(args.max_single_heading_number)])
    if not args.strict_validate:
        cmd.append("--disable_strict_validate")

    # atomizer config
    cmd.extend(["--atom_max_tokens", str(args.atom_max_tokens)])
    cmd.extend(["--atom_max_chars", str(args.atom_max_chars)])
    cmd.extend(["--atom_min_tokens", str(args.atom_min_tokens)])
    cmd.extend(["--atom_min_chars", str(args.atom_min_chars)])
    if not args.fix_empty_units:
        cmd.append("--disable_fix_empty_units")
    if args.prefer_merge_empty_to_left:
        cmd.append("--prefer_merge_empty_to_left")

    return cmd


def build_stage2_cmd(args: argparse.Namespace, atoms_b0_jsonl: Path, stage2_output_dir: Path) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "SLAC.refiner.pipeline.run.run_refiner_infer_only",
        "--input_jsonl", str(atoms_b0_jsonl),
        "--ckpt", str(Path(args.ckpt).expanduser().resolve()),
        "--output_dir", str(stage2_output_dir),
        "--doc_layers", str(args.doc_layers),
        "--window_size", str(args.window_size),
        "--k_shift", str(args.k_shift),
        "--refine_passes", str(args.refine_passes),
        "--temperatures", *[str(x) for x in args.temperatures],
        "--insert_thresholds", *[str(x) for x in args.insert_thresholds],
        "--seeds", *[str(x) for x in args.seeds],
        "--candidate_select_policy", args.candidate_select_policy,
    ]

    if args.config:
        cmd.extend(["--config", args.config])

    if args.include_identity:
        cmd.append("--include_identity")
    if args.include_greedy:
        cmd.append("--include_greedy")
    if not args.export_leaf_records:
        cmd.append("--disable_export_leaf_records")
    if not args.export_doc_catalog:
        cmd.append("--disable_export_doc_catalog")
    if args.infer_script is not None:
        cmd.extend(["--infer_script", args.infer_script])

    return cmd


def run_subprocess(
    cmd: List[str],
    *,
    stage_name: str,
    logger: logging.Logger,
    log_json_path: Path,
) -> subprocess.CompletedProcess[str]:
    logger.info("[%s] Running command:", stage_name)
    logger.info("[%s] %s", stage_name, " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )
        dump_json(
            log_json_path,
            {
                "stage": stage_name,
                "status": "success",
                "cmd": cmd,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
            ensure_ascii=False,
            indent=2,
        )
        return proc
    except subprocess.CalledProcessError as e:
        dump_json(
            log_json_path,
            {
                "stage": stage_name,
                "status": "failed",
                "cmd": cmd,
                "returncode": e.returncode,
                "stdout": e.stdout,
                "stderr": e.stderr,
            },
            ensure_ascii=False,
            indent=2,
        )
        raise


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    out_dirs = ensure_output_subdirs(output_dir)

    logger = setup_logger(
        "refiner.pipeline.run_refiner_pipeline",
        log_file=out_dirs["logs"] / "run.log",
        level=logging.INFO,
        console=True,
        jsonl_file=False,
    )

    if getattr(args, "_unknown_config_keys", None):
        logger.warning(
            "Ignored unknown config keys for run_refiner_pipeline: %s",
            args._unknown_config_keys,
        )

    summary: dict = {
        "status": "running",
        "args": vars(args),
        "stage1": {
            "status": "not_run",
            "output_dir": str(out_dirs["stage1"]),
        },
        "stage2": {
            "status": "not_run",
            "output_dir": str(out_dirs["stage2"]),
        },
        "artifacts": {},
    }

    try:
        if args.reuse_atoms_b0_jsonl is not None:
            atoms_b0_jsonl = Path(args.reuse_atoms_b0_jsonl).expanduser().resolve()
            if not atoms_b0_jsonl.exists():
                raise FileNotFoundError(f"--reuse_atoms_b0_jsonl not found: {atoms_b0_jsonl}")
            summary["artifacts"]["atoms_b0_jsonl"] = str(atoms_b0_jsonl)
            logger.info("Using existing atoms_b0.jsonl: %s", atoms_b0_jsonl)
        else:
            atoms_b0_jsonl = out_dirs["stage1"] / "atoms_b0.jsonl"

        # -----------------------------
        # Stage 1
        # -----------------------------
        if not args.skip_stage1:
            if not args.input_paths:
                raise ValueError("Stage 1 requires --input_paths unless you skip it and reuse atoms_b0.jsonl.")

            stage1_cmd = build_stage1_cmd(args, out_dirs["stage1"])
            run_subprocess(
                stage1_cmd,
                stage_name="stage1_rule_chunk0",
                logger=logger,
                log_json_path=out_dirs["logs"] / "stage1_rule_chunk0.json",
            )

            if not atoms_b0_jsonl.exists():
                raise FileNotFoundError(
                    f"Stage 1 finished but expected atoms_b0.jsonl not found: {atoms_b0_jsonl}"
                )

            summary["stage1"]["status"] = "success"
            summary["artifacts"]["atoms_b0_jsonl"] = str(atoms_b0_jsonl)
            logger.info("Stage 1 success. atoms_b0.jsonl = %s", atoms_b0_jsonl)

        else:
            summary["stage1"]["status"] = "skipped"
            if not atoms_b0_jsonl.exists():
                raise FileNotFoundError(
                    f"Stage 1 skipped but atoms_b0.jsonl not found: {atoms_b0_jsonl}. "
                    f"Provide --reuse_atoms_b0_jsonl or do not skip stage 1."
                )
            summary["artifacts"]["atoms_b0_jsonl"] = str(atoms_b0_jsonl)
            logger.info("Stage 1 skipped.")

        # -----------------------------
        # Stage 2
        # -----------------------------
        if not args.skip_stage2:
            stage2_cmd = build_stage2_cmd(args, atoms_b0_jsonl, out_dirs["stage2"])
            run_subprocess(
                stage2_cmd,
                stage_name="stage2_refiner_infer",
                logger=logger,
                log_json_path=out_dirs["logs"] / "stage2_refiner_infer.json",
            )

            refined_chunks_jsonl = out_dirs["stage2"] / "refined_chunks.jsonl"
            selected_candidates_jsonl = out_dirs["stage2"] / "selected_candidates.jsonl"
            refined_boundaries_jsonl = out_dirs["stage2"] / "refined_boundaries.jsonl"
            leaf_records_jsonl = out_dirs["stage2"] / "leaf_records.jsonl"
            doc_catalog_jsonl = out_dirs["stage2"] / "doc_catalog.jsonl"

            if not refined_chunks_jsonl.exists():
                raise FileNotFoundError(
                    f"Stage 2 finished but expected refined_chunks.jsonl not found: {refined_chunks_jsonl}"
                )

            summary["stage2"]["status"] = "success"
            summary["artifacts"]["refined_chunks_jsonl"] = str(refined_chunks_jsonl)
            if selected_candidates_jsonl.exists():
                summary["artifacts"]["selected_candidates_jsonl"] = str(selected_candidates_jsonl)
            if refined_boundaries_jsonl.exists():
                summary["artifacts"]["refined_boundaries_jsonl"] = str(refined_boundaries_jsonl)
            if leaf_records_jsonl.exists():
                summary["artifacts"]["leaf_records_jsonl"] = str(leaf_records_jsonl)
            if doc_catalog_jsonl.exists():
                summary["artifacts"]["doc_catalog_jsonl"] = str(doc_catalog_jsonl)

            logger.info("Stage 2 success. refined_chunks.jsonl = %s", refined_chunks_jsonl)

        else:
            summary["stage2"]["status"] = "skipped"
            logger.info("Stage 2 skipped.")

        summary["status"] = "success"
        dump_json(
            out_dirs["summaries"] / "run_summary.json",
            summary,
            ensure_ascii=False,
            indent=2,
        )
        logger.info("Full pipeline finished successfully.")

    except Exception as e:
        summary["status"] = "failed"
        summary["error_type"] = type(e).__name__
        summary["error"] = str(e)
        summary["traceback"] = traceback.format_exc(limit=50)
        dump_json(
            out_dirs["summaries"] / "run_summary.json",
            summary,
            ensure_ascii=False,
            indent=2,
        )
        logger.exception("Full pipeline failed.")
        raise


if __name__ == "__main__":
    main()