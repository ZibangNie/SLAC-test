from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from SLAC.reranker.io.readers import (
    attach_candidate_sidecar_metadata,
    discover_reranker_input_files,
    load_candidate_sidecar,
    load_reranker_input_file,
    maybe_candidate_sidecar_path,
)
from SLAC.reranker.io.writers import append_jsonl, write_json, write_jsonl
from SLAC.reranker.models.bge_reranker_v2_m3 import (
    BGERerankerConfig,
    BGERerankerV2M3,
)
from SLAC.reranker.pipeline.build_pairs import group_records_by_query_id
from SLAC.reranker.pipeline.pack_bridge import build_pack_bridge_records
from SLAC.reranker.pipeline.rank_candidates import (
    build_scored_records,
    truncate_scored_records,
)
from SLAC.reranker.pipeline.score_pairs import score_query_records


DEFAULT_CONFIG: Dict[str, Any] = {
    "common": {
        "model_name": "BAAI/bge-reranker-v2-m3",
        "model_path": None,
        "device": "cuda",
        "torch_dtype": "float16",
        "batch_size": 8,
        "max_length": 1024,
        "normalize": True,
        "strict_schema": False,
    },
    "candidate_pool": {
        "direct_top_n": 30,
        "expanded_top_n": 20,
        "max_pairs_per_query": 50,
        "dedupe_key": "chunk_id",
    },
    "ranking": {
        "output_top_n": 50,
    },
    "pack_bridge": {
        "enabled": True,
        "pack_top_n": 12,
        "reserve_direct_top_n": 0,
        "reserve_expanded_top_n": 0,
        "max_token_budget": None,
    },
    "debug": {
        "dump_scored_pairs": True,
        "save_summary": True,
        "fail_fast": True,
        "recursive_scan": False,
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError("config yaml root must be a mapping")
    return deep_update(DEFAULT_CONFIG, user_cfg)


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default
        try:
            return int(value)
        except ValueError:
            return default
    return default


def sanitize_filename_component(text: str, default: str = "unknown") -> str:
    text = text.strip()
    if not text:
        return default
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or default


def input_file_tag(input_file: Path) -> str:
    name = input_file.name
    if name.endswith(".reranker_input.jsonl"):
        name = name[: -len(".reranker_input.jsonl")]
    else:
        name = input_file.stem
    return sanitize_filename_component(name, default="input")


def derive_output_stem(input_file: Path, query_id: str, total_query_groups: int) -> str:
    file_tag = input_file_tag(input_file)
    query_tag = sanitize_filename_component(query_id, default="query")
    if total_query_groups == 1:
        return file_tag
    return f"{file_tag}.{query_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full SLAC reranker pipeline: rerank + optional pack bridge."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)

    parser.add_argument("--direct_top_n", type=int, default=None)
    parser.add_argument("--expanded_top_n", type=int, default=None)
    parser.add_argument("--max_pairs_per_query", type=int, default=None)
    parser.add_argument("--output_top_n", type=int, default=None)

    parser.add_argument("--pack_top_n", type=int, default=None)
    parser.add_argument("--reserve_direct_top_n", type=int, default=None)
    parser.add_argument("--reserve_expanded_top_n", type=int, default=None)
    parser.add_argument("--max_token_budget", type=int, default=None)
    parser.add_argument("--disable_pack_bridge", action="store_true")

    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--strict_schema", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--no_dump_scored_pairs", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    def maybe_set(group: str, key: str, value: Any) -> None:
        if value is not None:
            out[group][key] = value

    maybe_set("common", "model_name", args.model_name)
    maybe_set("common", "model_path", args.model_path)
    maybe_set("common", "device", args.device)
    maybe_set("common", "torch_dtype", args.torch_dtype)
    maybe_set("common", "batch_size", args.batch_size)
    maybe_set("common", "max_length", args.max_length)

    maybe_set("candidate_pool", "direct_top_n", args.direct_top_n)
    maybe_set("candidate_pool", "expanded_top_n", args.expanded_top_n)
    maybe_set("candidate_pool", "max_pairs_per_query", args.max_pairs_per_query)
    maybe_set("ranking", "output_top_n", args.output_top_n)

    maybe_set("pack_bridge", "pack_top_n", args.pack_top_n)
    maybe_set("pack_bridge", "reserve_direct_top_n", args.reserve_direct_top_n)
    maybe_set("pack_bridge", "reserve_expanded_top_n", args.reserve_expanded_top_n)
    maybe_set("pack_bridge", "max_token_budget", args.max_token_budget)

    if args.disable_pack_bridge:
        out["pack_bridge"]["enabled"] = False
    if args.recursive:
        out["debug"]["recursive_scan"] = True
    if args.strict_schema:
        out["common"]["strict_schema"] = True
    if args.no_normalize:
        out["common"]["normalize"] = False
    if args.fail_fast:
        out["debug"]["fail_fast"] = True
    if args.no_dump_scored_pairs:
        out["debug"]["dump_scored_pairs"] = False

    return out


def make_event(level: str, message: str, **kwargs: Any) -> dict:
    event = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,
        "message": message,
    }
    event.update(kwargs)
    return event


def build_run_summary(
    *,
    started_at: float,
    input_files: List[Path],
    all_query_summaries: List[dict],
    cfg: Dict[str, Any],
    reranker: BGERerankerV2M3,
) -> dict:
    elapsed = time.time() - started_at
    query_latencies = [
        float(x["latency_sec"]) for x in all_query_summaries if x.get("latency_sec") is not None
    ]
    selected_counts = [int(x["num_selected_for_rerank"]) for x in all_query_summaries]
    output_counts = [int(x["num_output_candidates"]) for x in all_query_summaries]
    pack_counts = [
        int(x["num_selected_for_packing"])
        for x in all_query_summaries
        if x.get("num_selected_for_packing") is not None
    ]

    return {
        "schema_version": "slac_reranker_run_summary_v1",
        "record_type": "reranker_run_summary",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(elapsed, 4),
        "num_input_files": len(input_files),
        "num_queries": len(all_query_summaries),
        "runtime": reranker.runtime_info,
        "config_snapshot": cfg,
        "aggregate": {
            "num_selected_pairs_total": sum(selected_counts) if selected_counts else 0,
            "num_output_candidates_total": sum(output_counts) if output_counts else 0,
            "num_selected_for_packing_total": sum(pack_counts) if pack_counts else 0,
            "avg_selected_pairs_per_query": round(statistics.mean(selected_counts), 4) if selected_counts else 0.0,
            "avg_output_candidates_per_query": round(statistics.mean(output_counts), 4) if output_counts else 0.0,
            "avg_selected_for_packing_per_query": round(statistics.mean(pack_counts), 4) if pack_counts else 0.0,
            "avg_query_latency_sec": round(statistics.mean(query_latencies), 4) if query_latencies else 0.0,
            "max_query_latency_sec": round(max(query_latencies), 4) if query_latencies else 0.0,
        },
        "query_summaries": all_query_summaries,
        "input_files": [str(p) for p in input_files],
    }


def process_one_input_file(
    *,
    input_file: Path,
    output_dir: Path,
    reranker: BGERerankerV2M3,
    cfg: Dict[str, Any],
    log_path: Path,
) -> List[dict]:
    strict_schema = bool(cfg["common"]["strict_schema"])
    normalize = bool(cfg["common"]["normalize"])
    output_top_n = _safe_int(cfg["ranking"].get("output_top_n"), 50)
    dump_scored_pairs = bool(cfg["debug"]["dump_scored_pairs"])
    pack_bridge_enabled = bool(cfg["pack_bridge"]["enabled"])

    records = load_reranker_input_file(
        input_file,
        strict=strict_schema,
    )
    append_jsonl(
        log_path,
        make_event(
            "INFO",
            "loaded reranker input file",
            input_file=str(input_file),
            num_records=len(records),
        ),
    )

    candidate_sidecar_path = maybe_candidate_sidecar_path(input_file)
    if candidate_sidecar_path is not None:
        candidate_sidecar = load_candidate_sidecar(candidate_sidecar_path)
        records = attach_candidate_sidecar_metadata(records, candidate_sidecar)
        append_jsonl(
            log_path,
            make_event(
                "INFO",
                "attached candidate sidecar metadata",
                input_file=str(input_file),
                candidate_sidecar=str(candidate_sidecar_path),
                num_sidecar_records=len(candidate_sidecar),
            ),
        )

    query_groups = group_records_by_query_id(records)

    queries_dir = output_dir / "queries"
    pack_bridge_dir = output_dir / "pack_bridge"
    debug_dir = output_dir / "debug"

    queries_dir.mkdir(parents=True, exist_ok=True)
    if pack_bridge_enabled:
        pack_bridge_dir.mkdir(parents=True, exist_ok=True)
    if dump_scored_pairs:
        debug_dir.mkdir(parents=True, exist_ok=True)

    file_summaries: List[dict] = []

    for query_id, group in query_groups.items():
        query_t0 = time.time()

        scoring_bundle = score_query_records(
            query_records=group,
            reranker=reranker,
            candidate_pool_cfg=cfg["candidate_pool"],
        )

        scored_records_full = build_scored_records(
            selected_records=scoring_bundle["selected_records"],
            raw_scores=scoring_bundle["raw_scores"],
            runtime_info=scoring_bundle["runtime_info"],
            normalize=normalize,
        )
        scored_records = truncate_scored_records(
            scored_records=scored_records_full,
            output_top_n=output_top_n,
        )

        output_stem = derive_output_stem(
            input_file=input_file,
            query_id=query_id,
            total_query_groups=len(query_groups),
        )

        reranked_output_file = queries_dir / f"{output_stem}.reranked_candidates.jsonl"
        write_jsonl(reranked_output_file, scored_records)

        debug_scored_pairs_file = None
        if dump_scored_pairs:
            debug_scored_pairs_file = debug_dir / f"{output_stem}.scored_pairs.jsonl"
            write_jsonl(debug_scored_pairs_file, scoring_bundle["scored_pair_records"])

        pack_bridge_output_file = None
        pack_bridge_summary = None
        if pack_bridge_enabled:
            bridge_records, bridge_bundle = build_pack_bridge_records(
                reranked_records=scored_records,
                pack_bridge_cfg=cfg["pack_bridge"],
            )
            pack_bridge_output_file = pack_bridge_dir / f"{output_stem}.for_packing.jsonl"
            write_jsonl(pack_bridge_output_file, bridge_records)
            pack_bridge_summary = bridge_bundle["summary"]

        query_latency_sec = round(time.time() - query_t0, 4)
        top1 = scored_records[0] if scored_records else None

        summary = {
            "query_id": query_id,
            "input_file": str(input_file),
            "reranked_output_file": str(reranked_output_file),
            "debug_scored_pairs_file": str(debug_scored_pairs_file) if debug_scored_pairs_file else None,
            "pack_bridge_output_file": str(pack_bridge_output_file) if pack_bridge_output_file else None,
            "num_input_candidates": scoring_bundle["selection_stats"]["num_input_candidates"],
            "num_selected_for_rerank": scoring_bundle["selection_stats"]["num_selected_total"],
            "num_output_candidates": len(scored_records),
            "num_selected_for_packing": (
                pack_bridge_summary["num_selected_for_packing"]
                if pack_bridge_summary is not None else None
            ),
            "selected_total_token_est": (
                pack_bridge_summary["selected_total_token_est"]
                if pack_bridge_summary is not None else None
            ),
            "top1_chunk_id": top1.get("chunk_id") if top1 else None,
            "top1_doc_id": top1.get("doc_id") if top1 else None,
            "top1_rerank_score_norm": top1.get("rerank_score_norm") if top1 else None,
            "latency_sec": query_latency_sec,
            "selection_stats": scoring_bundle["selection_stats"],
            "score_stats": scoring_bundle["score_stats"],
            "pack_bridge_summary": pack_bridge_summary,
        }
        file_summaries.append(summary)

        append_jsonl(
            log_path,
            make_event(
                "INFO",
                "finished query rerank pipeline",
                query_id=query_id,
                input_file=str(input_file),
                reranked_output_file=str(reranked_output_file),
                pack_bridge_output_file=str(pack_bridge_output_file) if pack_bridge_output_file else None,
                num_input_candidates=summary["num_input_candidates"],
                num_selected_for_rerank=summary["num_selected_for_rerank"],
                num_output_candidates=summary["num_output_candidates"],
                num_selected_for_packing=summary["num_selected_for_packing"],
                latency_sec=summary["latency_sec"],
            ),
        )

    return file_summaries


def main() -> int:
    args = parse_args()
    cfg = apply_cli_overrides(load_yaml_config(args.config), args)

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "queries").mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "logs" / "reranker_pipeline_events.jsonl"
    started_at = time.time()

    append_jsonl(
        log_path,
        make_event(
            "INFO",
            "starting reranker pipeline run",
            input_path=str(input_path),
            output_dir=str(output_dir),
            config_path=args.config,
        ),
    )

    input_files = discover_reranker_input_files(
        input_path,
        recursive=bool(cfg["debug"]["recursive_scan"]),
    )

    reranker_cfg = BGERerankerConfig(
        model_name=str(cfg["common"]["model_name"]),
        model_path=cfg["common"].get("model_path"),
        device=str(cfg["common"]["device"]),
        torch_dtype=str(cfg["common"]["torch_dtype"]),
        batch_size=_safe_int(cfg["common"].get("batch_size"), 8),
        max_length=_safe_int(cfg["common"].get("max_length"), 1024),
        trust_remote_code=False,
    )
    reranker = BGERerankerV2M3(reranker_cfg)

    append_jsonl(
        log_path,
        make_event(
            "INFO",
            "initialized reranker model",
            runtime=reranker.runtime_info,
        ),
    )

    all_query_summaries: List[dict] = []
    fail_fast = bool(cfg["debug"]["fail_fast"])

    for input_file in input_files:
        try:
            file_summaries = process_one_input_file(
                input_file=input_file,
                output_dir=output_dir,
                reranker=reranker,
                cfg=cfg,
                log_path=log_path,
            )
            all_query_summaries.extend(file_summaries)
        except Exception as exc:
            append_jsonl(
                log_path,
                make_event(
                    "ERROR",
                    "failed to process reranker pipeline input file",
                    input_file=str(input_file),
                    error_type=type(exc).__name__,
                    error=str(exc),
                ),
            )
            if fail_fast:
                raise

    summary = build_run_summary(
        started_at=started_at,
        input_files=input_files,
        all_query_summaries=all_query_summaries,
        cfg=cfg,
        reranker=reranker,
    )

    summary_path = output_dir / "summaries" / "reranker_run_summary.json"
    if bool(cfg["debug"]["save_summary"]):
        write_json(summary_path, summary)

    append_jsonl(
        log_path,
        make_event(
            "INFO",
            "finished reranker pipeline run",
            summary_path=str(summary_path),
            num_queries=len(all_query_summaries),
            elapsed_sec=summary["elapsed_sec"],
        ),
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "num_input_files": len(input_files),
                "num_queries": len(all_query_summaries),
                "summary_path": str(summary_path),
                "runtime": reranker.runtime_info,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())