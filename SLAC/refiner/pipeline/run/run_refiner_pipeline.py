"""
Main end-to-end refiner pipeline runner.

Pipeline:
raw file -> read -> normalize/clean -> rule chunk0 -> build refiner input
-> refiner inference -> export refined chunks
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from ..utils.io import dump_json, ensure_dir
from ..utils.log_utils import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end refiner pipeline runner: "
            "txt -> rule chunk0 -> atoms_b0 -> frozen refiner infer -> refined_chunks"
        )
    )

    # -----------------------------
    # Input / Output
    # -----------------------------
    p.add_argument(
        "--input_paths",
        nargs="+",
        default=None,
        help="Input txt/text/md files or directories. Required unless --reuse_atoms_b0_jsonl is provided.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Unified output directory for the full refiner pipeline.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories for txt-like files in stage 1.",
    )
    p.add_argument(
        "--encoding",
        default=None,
        help="Optional explicit text encoding for stage 1.",
    )
    p.add_argument(
        "--domain",
        default=None,
        help="Optional domain tag, e.g. rail.",
    )

    # -----------------------------
    # Stage control
    # -----------------------------
    p.add_argument(
        "--skip_stage1",
        action="store_true",
        help="Skip rule chunk0 / atoms_b0 building stage.",
    )
    p.add_argument(
        "--skip_stage2",
        action="store_true",
        help="Skip refiner inference/export stage.",
    )
    p.add_argument(
        "--reuse_atoms_b0_jsonl",
        default=None,
        help="Use an existing atoms_b0.jsonl instead of generating a new one.",
    )

    # -----------------------------
    # Stage 1 dump flags
    # -----------------------------
    p.add_argument("--dump_structure_doc", action="store_true")
    p.add_argument("--dump_chunk0_json", action="store_true")
    p.add_argument("--dump_intermediate_records", action="store_true")

    # -----------------------------
    # Normalize config (stage 1)
    # -----------------------------
    p.add_argument("--collapse_inner_whitespace", action="store_true")
    p.add_argument("--drop_blank_lines", action="store_true")
    p.add_argument("--max_consecutive_blank_lines", type=int, default=2)

    # -----------------------------
    # Clean config (stage 1)
    # -----------------------------
    p.add_argument("--disable_drop_decorative_lines", action="store_true")
    p.add_argument("--disable_drop_page_number_lines", action="store_true")
    p.add_argument("--disable_drop_repeated_short_lines", action="store_true")
    p.add_argument("--repeated_short_line_min_repeats", type=int, default=3)
    p.add_argument("--repeated_short_line_max_len", type=int, default=80)
    p.add_argument("--disable_drop_isolated_noise_lines", action="store_true")
    p.add_argument("--noise_line_max_len", type=int, default=3)

    # -----------------------------
    # Segment config (stage 1)
    # -----------------------------
    p.add_argument("--disable_toc_detection", action="store_true")
    p.add_argument("--min_strong_marker_ratio", type=float, default=0.02)
    p.add_argument("--max_weak_marker_ratio", type=float, default=0.45)
    p.add_argument("--disable_allow_single_number_heading", action="store_true")
    p.add_argument("--max_single_heading_number", type=int, default=199)
    p.add_argument("--disable_strict_validate", action="store_true")

    # -----------------------------
    # Atomizer config (stage 1)
    # -----------------------------
    p.add_argument("--atom_max_tokens", type=int, default=120)
    p.add_argument("--atom_max_chars", type=int, default=480)
    p.add_argument("--atom_min_tokens", type=int, default=8)
    p.add_argument("--atom_min_chars", type=int, default=20)
    p.add_argument("--disable_fix_empty_units", action="store_true")
    p.add_argument("--prefer_merge_empty_to_left", action="store_true")

    # -----------------------------
    # Stage 2 refiner infer config
    # -----------------------------
    p.add_argument(
        "--ckpt",
        default="/root/autodl-tmp/runs/refiner/week2_mixed_llmgold_x4_e3/checkpoints/epoch_8.pt",
        help="Frozen refiner checkpoint path.",
    )
    p.add_argument("--doc_layers", type=int, default=1)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--k_shift", type=int, default=6)
    p.add_argument("--refine_passes", type=int, default=2)
    p.add_argument("--temperatures", nargs="+", type=float, default=[0.90, 1.00])
    p.add_argument("--insert_thresholds", nargs="+", type=float, default=[0.45, 0.50])
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    p.add_argument("--include_identity", action="store_true")
    p.add_argument("--include_greedy", action="store_true")

    # -----------------------------
    # Export policy (stage 2)
    # -----------------------------
    p.add_argument(
        "--candidate_select_policy",
        default="greedy_first",
        choices=["greedy_first", "best_teacher_stats"],
    )
    p.add_argument("--disable_export_leaf_records", action="store_true")

    # -----------------------------
    # Script path override
    # -----------------------------
    p.add_argument(
        "--rule_runner_script",
        default=None,
        help="Optional override for module path execution is not needed; reserved for future use.",
    )
    p.add_argument(
        "--infer_runner_script",
        default=None,
        help="Optional override for module path execution is not needed; reserved for future use.",
    )
    p.add_argument(
        "--infer_script",
        default=None,
        help="Optional override for SLAC/refiner/scripts/infer_bestofn.py path, passed to stage 2.",
    )

    return p.parse_args()


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

    if args.input_paths:
        cmd.extend(["--input_paths", *args.input_paths])
    if args.recursive:
        cmd.append("--recursive")
    if args.encoding is not None:
        cmd.extend(["--encoding", args.encoding])
    if args.domain is not None:
        cmd.extend(["--domain", args.domain])

    if args.dump_structure_doc:
        cmd.append("--dump_structure_doc")
    if args.dump_chunk0_json:
        cmd.append("--dump_chunk0_json")
    if args.dump_intermediate_records:
        cmd.append("--dump_intermediate_records")

    if args.collapse_inner_whitespace:
        cmd.append("--collapse_inner_whitespace")
    if args.drop_blank_lines:
        cmd.append("--drop_blank_lines")
    cmd.extend(["--max_consecutive_blank_lines", str(args.max_consecutive_blank_lines)])

    if args.disable_drop_decorative_lines:
        cmd.append("--disable_drop_decorative_lines")
    if args.disable_drop_page_number_lines:
        cmd.append("--disable_drop_page_number_lines")
    if args.disable_drop_repeated_short_lines:
        cmd.append("--disable_drop_repeated_short_lines")
    cmd.extend(["--repeated_short_line_min_repeats", str(args.repeated_short_line_min_repeats)])
    cmd.extend(["--repeated_short_line_max_len", str(args.repeated_short_line_max_len)])
    if args.disable_drop_isolated_noise_lines:
        cmd.append("--disable_drop_isolated_noise_lines")
    cmd.extend(["--noise_line_max_len", str(args.noise_line_max_len)])

    if args.disable_toc_detection:
        cmd.append("--disable_toc_detection")
    cmd.extend(["--min_strong_marker_ratio", str(args.min_strong_marker_ratio)])
    cmd.extend(["--max_weak_marker_ratio", str(args.max_weak_marker_ratio)])
    if args.disable_allow_single_number_heading:
        cmd.append("--disable_allow_single_number_heading")
    cmd.extend(["--max_single_heading_number", str(args.max_single_heading_number)])
    if args.disable_strict_validate:
        cmd.append("--disable_strict_validate")

    cmd.extend(["--atom_max_tokens", str(args.atom_max_tokens)])
    cmd.extend(["--atom_max_chars", str(args.atom_max_chars)])
    cmd.extend(["--atom_min_tokens", str(args.atom_min_tokens)])
    cmd.extend(["--atom_min_chars", str(args.atom_min_chars)])
    if args.disable_fix_empty_units:
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

    if args.include_identity:
        cmd.append("--include_identity")
    if args.include_greedy:
        cmd.append("--include_greedy")
    if args.disable_export_leaf_records:
        cmd.append("--disable_export_leaf_records")
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
        # -----------------------------
        # Determine atoms_b0 source
        # -----------------------------
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
            leaf_records_jsonl = out_dirs["stage2"] / "leaf_records.jsonl"

            if not refined_chunks_jsonl.exists():
                raise FileNotFoundError(
                    f"Stage 2 finished but expected refined_chunks.jsonl not found: {refined_chunks_jsonl}"
                )

            summary["stage2"]["status"] = "success"
            summary["artifacts"]["refined_chunks_jsonl"] = str(refined_chunks_jsonl)
            if selected_candidates_jsonl.exists():
                summary["artifacts"]["selected_candidates_jsonl"] = str(selected_candidates_jsonl)
            if leaf_records_jsonl.exists():
                summary["artifacts"]["leaf_records_jsonl"] = str(leaf_records_jsonl)

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