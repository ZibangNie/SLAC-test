from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "common": {
        "encoder_name": "BAAI/bge-m3",
        "tokenizer_name": "BAAI/bge-m3",
        "normalized": True,
        "similarity_metric": "inner_product",
        "query_planner_version": "v1_1",
        "batch_size": 32,
    },
    "retrieve": {
        "leaf_topk_per_variant": 40,
        "leaf_topk_merged_max": 120,
        "chunk_topk": 20,
        "anchor_topk": 30,
        "fused_candidate_topn": 40,
    },
    "gates": {
        "leaf_dense_floor": 0.38,
        "leaf_dense_strong": 0.52,
        "chunk_dense_floor": 0.34,
        "chunk_dense_strong": 0.48,
        "parent_expand_topr": 20,
        "parent_expand_min_score": 0.46,
        "second_ancestor_min_score": 0.55,
        "sibling_expand_min_score": 0.50,
        "neighbor_expand_min_score": 0.48,
        "local_branch_trigger_score": 0.54,
        "short_chunk_tokens": 120,
        "long_parent_tokens": 280,
    },
    "pack": {
        "prompt_total_budget_hard": 3400,
        "evidence_budget_tokens": 2500,
        "max_packed_items": 10,
        "dedup_jaccard_threshold": 0.85,
    },
    "modes": {
        "enable_full_tree_mode": True,
        "enable_weak_tree_mode": True,
        "enable_leaf_only_mode": True,
    },
}


def load_config(
    config_path: Optional[str | Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)

    if config_path:
        with Path(config_path).open("r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, file_cfg)

    if cli_overrides:
        cfg = deep_update(cfg, cli_overrides)

    return cfg


def deep_update(base: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in new_data.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out