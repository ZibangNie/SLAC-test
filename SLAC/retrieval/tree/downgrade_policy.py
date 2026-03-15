from __future__ import annotations

from typing import Dict


VALID_TREE_MODES = {"full_tree", "weak_tree", "leaf_only"}


def decide_tree_mode(
    quality_gates: Dict[str, float],
    config: Dict,
) -> str:
    """
    根据 build 阶段输出的质量统计决定 tree mode。
    设计原则：
    - 优先 full_tree
    - 树结构明显不稳则退到 weak_tree
    - 更差则 leaf_only
    """
    modes = config.get("modes", {})
    enable_full = bool(modes.get("enable_full_tree_mode", True))
    enable_weak = bool(modes.get("enable_weak_tree_mode", True))
    enable_leaf_only = bool(modes.get("enable_leaf_only_mode", True))

    missing_parent_ratio = float(quality_gates.get("missing_parent_ratio", 0.0))
    missing_prev_ratio = float(quality_gates.get("missing_prev_ratio", 0.0))
    missing_next_ratio = float(quality_gates.get("missing_next_ratio", 0.0))
    weak_tree_suspected = bool(quality_gates.get("weak_tree_suspected", False))

    neighbor_missing_ratio = max(missing_prev_ratio, missing_next_ratio)

    # 明显太差：直接 leaf_only
    if missing_parent_ratio > 0.50 or neighbor_missing_ratio > 0.55:
        if enable_leaf_only:
            return "leaf_only"
        if enable_weak:
            return "weak_tree"
        return "full_tree"

    # 中度不稳：weak_tree
    if weak_tree_suspected or missing_parent_ratio > 0.20 or neighbor_missing_ratio > 0.20:
        if enable_weak:
            return "weak_tree"
        if enable_leaf_only:
            return "leaf_only"
        return "full_tree"

    # 树质量稳定
    if enable_full:
        return "full_tree"
    if enable_weak:
        return "weak_tree"
    return "leaf_only"