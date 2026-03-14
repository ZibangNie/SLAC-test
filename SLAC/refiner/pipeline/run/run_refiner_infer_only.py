from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..assemble.export_refined_chunks import (
    RefinedChunkExportConfig,
    export_refined_chunks_from_candidates,
    flatten_doc_catalog,
    flatten_leaf_records,
    flatten_refined_boundary,
    flatten_refined_chunks,
    flatten_selected_candidate,
)
from ..configs.loader import (
    filter_allowed_keys,
    get_runner_config,
    load_pipeline_config,
    merge_resolved_args,
    namespace_from_dict,
    preparse_config_arg,
)
from ..utils.io import dump_json, dump_jsonl, ensure_dir, load_jsonl
from ..utils.log_utils import log_doc_event, setup_logger


DEFAULTS: Dict[str, Any] = {
    "input_jsonl": None,
    "ckpt": "/root/autodl-tmp/runs/refiner/week2_mixed_llmgold_x4_e3/checkpoints/epoch_8.pt",
    "output_dir": None,

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
    runner_cfg = get_runner_config(raw_cfg, "run_refiner_infer_only")
    config_values, unknown_cfg_keys = filter_allowed_keys(runner_cfg, set(DEFAULTS.keys()))

    p = argparse.ArgumentParser(
        description="Run frozen refiner inference on prepared atoms_b0 JSONL and export standard outputs.",
        argument_default=argparse.SUPPRESS,
    )

    p.add_argument("--config", help="Optional pipeline config YAML path.")
    p.add_argument("--input_jsonl")
    p.add_argument("--ckpt")
    p.add_argument("--output_dir")

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

    if cli_values.get("disable_export_leaf_records"):
        resolved["export_leaf_records"] = False
    if cli_values.get("disable_export_doc_catalog"):
        resolved["export_doc_catalog"] = False

    resolved["config"] = str(loaded_cfg_path) if loaded_cfg_path is not None else config_arg
    resolved["_unknown_config_keys"] = unknown_cfg_keys

    if not resolved.get("input_jsonl"):
        p.error("Missing required input: --input_jsonl (or set run_refiner_infer_only.input_jsonl in config).")
    if not resolved.get("output_dir"):
        p.error("Missing required output: --output_dir (or set run_refiner_infer_only.output_dir in config).")

    return namespace_from_dict(resolved)


def ensure_output_subdirs(output_dir: Path) -> Dict[str, Path]:
    out = {
        "root": output_dir,
        "logs": output_dir / "logs",
        "infer_raw": output_dir / "infer_raw",
        "selected": output_dir / "selected",
        "refined_boundaries_dir": output_dir / "refined_boundaries",
        "refined_chunks_dir": output_dir / "refined_chunks",
        "leaf_records_dir": output_dir / "leaf_records",
        "doc_catalog_dir": output_dir / "doc_catalog",
        "summaries": output_dir / "summaries",
    }
    for p in out.values():
        ensure_dir(p)
    return out


def resolve_infer_script(user_override: str | None) -> Path:
    if user_override:
        p = Path(user_override).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"infer_script does not exist: {p}")
        return p

    cur = Path(__file__).resolve()
    repo_root = cur.parents[4]
    infer_script = repo_root / "SLAC" / "refiner" / "scripts" / "infer_bestofn.py"
    if not infer_script.exists():
        raise FileNotFoundError(f"Auto-resolved infer script not found: {infer_script}")
    return infer_script


def build_infer_cmd(args: argparse.Namespace, infer_script: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(infer_script),
        "--input_jsonl", str(Path(args.input_jsonl).expanduser().resolve()),
        "--ckpt", str(Path(args.ckpt).expanduser().resolve()),
        "--output_dir", str((Path(args.output_dir).expanduser().resolve() / "infer_raw")),
        "--doc_layers", str(args.doc_layers),
        "--window_size", str(args.window_size),
        "--k_shift", str(args.k_shift),
        "--refine_passes", str(args.refine_passes),
        "--temperatures", *[str(x) for x in args.temperatures],
        "--insert_thresholds", *[str(x) for x in args.insert_thresholds],
        "--seeds", *[str(x) for x in args.seeds],
    ]
    if args.include_identity:
        cmd.append("--include_identity")
    if args.include_greedy:
        cmd.append("--include_greedy")
    return cmd


def group_candidates_by_doc_id(candidate_records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in candidate_records:
        doc_id = rec.get("doc_id")
        if not doc_id:
            continue
        grouped.setdefault(doc_id, []).append(rec)
    return grouped


def safe_doc_file_name(doc_id: str) -> str:
    return doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    out_dirs = ensure_output_subdirs(output_dir)

    logger = setup_logger(
        "refiner.pipeline.run_refiner_infer_only",
        log_file=out_dirs["logs"] / "run.log",
        level=logging.INFO,
        console=True,
        jsonl_file=False,
    )

    if getattr(args, "_unknown_config_keys", None):
        logger.warning(
            "Ignored unknown config keys for run_refiner_infer_only: %s",
            args._unknown_config_keys,
        )

    structured_logger = setup_logger(
        "refiner.pipeline.run_refiner_infer_only.jsonl",
        log_file=out_dirs["logs"] / "run_events.jsonl",
        level=logging.INFO,
        console=False,
        jsonl_file=True,
    )

    input_records = load_jsonl(args.input_jsonl, validate_dict=True)
    if not input_records:
        raise ValueError(f"No records found in input_jsonl: {args.input_jsonl}")

    infer_script = resolve_infer_script(args.infer_script)
    infer_cmd = build_infer_cmd(args, infer_script)

    logger.info("Running infer_bestofn.py ...")
    logger.info("Command: %s", " ".join(infer_cmd))

    try:
        proc = subprocess.run(
            infer_cmd,
            check=True,
            text=True,
            capture_output=True,
        )
        dump_json(
            out_dirs["logs"] / "infer_stdout.json",
            {"stdout": proc.stdout, "stderr": proc.stderr},
            ensure_ascii=False,
            indent=2,
        )
    except subprocess.CalledProcessError as e:
        dump_json(
            out_dirs["logs"] / "infer_failure.json",
            {
                "returncode": e.returncode,
                "stdout": e.stdout,
                "stderr": e.stderr,
                "cmd": infer_cmd,
            },
            ensure_ascii=False,
            indent=2,
        )
        logger.exception("infer_bestofn.py failed")
        raise

    infer_raw_dir = out_dirs["infer_raw"]
    all_candidates_path = infer_raw_dir / "all_candidates.jsonl"
    if not all_candidates_path.exists():
        raise FileNotFoundError(f"Expected infer output not found: {all_candidates_path}")

    candidate_records = load_jsonl(all_candidates_path, validate_dict=True)
    if not candidate_records:
        raise ValueError(f"No candidate records found in {all_candidates_path}")

    input_by_doc_id = {rec["doc_id"]: rec for rec in input_records}
    cand_by_doc_id = group_candidates_by_doc_id(candidate_records)

    export_cfg = RefinedChunkExportConfig(
        candidate_select_policy=args.candidate_select_policy,
        export_leaf_records=args.export_leaf_records,
        export_doc_catalog=args.export_doc_catalog,
        strict_validate=True,
    )

    selected_records: List[Dict[str, Any]] = []
    refined_boundary_records: List[Dict[str, Any]] = []
    refined_chunks_all: List[Dict[str, Any]] = []
    leaf_records_all: List[Dict[str, Any]] = []
    doc_catalog_all: List[Dict[str, Any]] = []

    run_summary: Dict[str, Any] = {
        "num_docs_total": len(input_records),
        "num_docs_success": 0,
        "num_docs_failed": 0,
        "docs": [],
        "input_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "ckpt": str(Path(args.ckpt).expanduser().resolve()),
        "infer_script": str(infer_script),
    }

    for doc_id, input_rec in input_by_doc_id.items():
        try:
            doc_cands = cand_by_doc_id.get(doc_id, [])
            if not doc_cands:
                raise ValueError(f"No candidate records found for doc_id={doc_id}")

            export_record = export_refined_chunks_from_candidates(
                refiner_input_record=input_rec,
                candidate_records=doc_cands,
                cfg=export_cfg,
            )

            selected_candidate = flatten_selected_candidate(export_record)
            refined_boundary = flatten_refined_boundary(export_record)
            refined_chunks = flatten_refined_chunks(export_record)
            leaf_records = flatten_leaf_records(export_record)
            doc_catalog = flatten_doc_catalog(export_record)

            selected_records.append(selected_candidate)
            refined_boundary_records.append(refined_boundary)
            refined_chunks_all.extend(refined_chunks)
            if args.export_leaf_records:
                leaf_records_all.extend(leaf_records)
            if args.export_doc_catalog and doc_catalog is not None:
                doc_catalog_all.append(doc_catalog)

            doc_file = safe_doc_file_name(doc_id)

            dump_json(
                out_dirs["selected"] / f"{doc_file}.selected.json",
                export_record,
                ensure_ascii=False,
                indent=2,
            )
            dump_json(
                out_dirs["refined_boundaries_dir"] / f"{doc_file}.refined_boundary.json",
                refined_boundary,
                ensure_ascii=False,
                indent=2,
            )
            dump_json(
                out_dirs["refined_chunks_dir"] / f"{doc_file}.refined_chunks.json",
                refined_chunks,
                ensure_ascii=False,
                indent=2,
            )

            if args.export_leaf_records:
                dump_json(
                    out_dirs["leaf_records_dir"] / f"{doc_file}.leaf_records.json",
                    leaf_records,
                    ensure_ascii=False,
                    indent=2,
                )

            if args.export_doc_catalog and doc_catalog is not None:
                dump_json(
                    out_dirs["doc_catalog_dir"] / f"{doc_file}.doc_catalog.json",
                    doc_catalog,
                    ensure_ascii=False,
                    indent=2,
                )

            summary_item = {
                "doc_id": doc_id,
                "status": "success",
                "num_candidates": len(doc_cands),
                "num_refined_chunks": len(refined_chunks),
                "num_leaf_records": len(leaf_records),
                "selected_candidate": selected_candidate,
            }
            run_summary["docs"].append(summary_item)
            run_summary["num_docs_success"] += 1

            log_doc_event(
                structured_logger,
                logging.INFO,
                f"doc success: {doc_id}",
                doc_id=doc_id,
                source_path=input_rec.get("meta", {}).get("source_path"),
                event="doc_success",
                extra_json=summary_item,
            )

        except Exception as e:
            err_summary = {
                "doc_id": doc_id,
                "status": "failed",
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(limit=20),
            }
            run_summary["docs"].append(err_summary)
            run_summary["num_docs_failed"] += 1

            logger.exception("Failed exporting refined outputs for doc_id=%s", doc_id)
            log_doc_event(
                structured_logger,
                logging.ERROR,
                f"doc failed: {doc_id}",
                doc_id=doc_id,
                source_path=input_rec.get("meta", {}).get("source_path"),
                event="doc_failed",
                extra_json=err_summary,
            )

    dump_jsonl(out_dirs["root"] / "selected_candidates.jsonl", selected_records, ensure_ascii=False, validate_dict=True)
    dump_jsonl(out_dirs["root"] / "refined_boundaries.jsonl", refined_boundary_records, ensure_ascii=False, validate_dict=True)
    dump_jsonl(out_dirs["root"] / "refined_chunks.jsonl", refined_chunks_all, ensure_ascii=False, validate_dict=True)

    if args.export_leaf_records:
        dump_jsonl(out_dirs["root"] / "leaf_records.jsonl", leaf_records_all, ensure_ascii=False, validate_dict=True)

    if args.export_doc_catalog:
        dump_jsonl(out_dirs["root"] / "doc_catalog.jsonl", doc_catalog_all, ensure_ascii=False, validate_dict=True)

    dump_json(
        out_dirs["summaries"] / "run_summary.json",
        run_summary,
        ensure_ascii=False,
        indent=2,
    )

    logger.info(
        "Done. total=%s success=%s failed=%s",
        run_summary["num_docs_total"],
        run_summary["num_docs_success"],
        run_summary["num_docs_failed"],
    )


if __name__ == "__main__":
    main()