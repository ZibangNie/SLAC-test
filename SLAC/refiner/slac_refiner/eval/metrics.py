from __future__ import annotations

from typing import Dict, List, Sequence


def _vector_to_set(b: Sequence[int]) -> set[int]:
    return {i for i, x in enumerate(b) if int(x) == 1}


def boundary_prf(pred_b: Sequence[int], gold_b: Sequence[int]) -> Dict[str, float]:
    pred = _vector_to_set(pred_b)
    gold = _vector_to_set(gold_b)

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def edit_action_accuracy(
    pred_edit_labels: List[Dict],
    gold_edit_labels: List[Dict],
) -> Dict[str, float]:
    pred_map = {int(x["g"]): str(x["y"]) for x in pred_edit_labels}
    gold_map = {int(x["g"]): str(x["y"]) for x in gold_edit_labels}

    if set(pred_map.keys()) != set(gold_map.keys()):
        return {
            "correct": 0.0,
            "total": float(len(gold_map)),
            "acc": 0.0,
        }

    total = len(gold_map)
    correct = sum(1 for g in gold_map if pred_map[g] == gold_map[g])
    acc = correct / total if total > 0 else 0.0

    return {
        "correct": float(correct),
        "total": float(total),
        "acc": float(acc),
    }


def insert_accuracy(
    pred_insert_labels: Sequence[int],
    gold_insert_labels: Sequence[int],
) -> Dict[str, float]:
    if len(pred_insert_labels) != len(gold_insert_labels):
        return {
            "correct": 0.0,
            "total": float(len(gold_insert_labels)),
            "acc": 0.0,
        }

    total = len(gold_insert_labels)
    correct = sum(int(p) == int(g) for p, g in zip(pred_insert_labels, gold_insert_labels))
    acc = correct / total if total > 0 else 0.0

    return {
        "correct": float(correct),
        "total": float(total),
        "acc": float(acc),
    }


def chunk_length_stats_from_spans(spans: Sequence[tuple[int, int]]) -> Dict[str, float]:
    lengths = [e - s for s, e in spans]
    if not lengths:
        return {
            "num_chunks": 0.0,
            "mean_atoms_per_chunk": 0.0,
            "min_atoms_per_chunk": 0.0,
            "max_atoms_per_chunk": 0.0,
        }

    return {
        "num_chunks": float(len(lengths)),
        "mean_atoms_per_chunk": float(sum(lengths) / len(lengths)),
        "min_atoms_per_chunk": float(min(lengths)),
        "max_atoms_per_chunk": float(max(lengths)),
    }


def aggregate_metric_dicts(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}

    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        out[k] = sum(m[k] for m in metric_list) / len(metric_list)
    return out


def _vector_to_set(b: Sequence[int]) -> set[int]:
    return {i for i, x in enumerate(b) if int(x) == 1}


def boundary_prf(pred_b: Sequence[int], gold_b: Sequence[int]) -> Dict[str, float]:
    pred = _vector_to_set(pred_b)
    gold = _vector_to_set(gold_b)

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def edit_action_accuracy(
    pred_edit_labels: List[Dict],
    gold_edit_labels: List[Dict],
) -> Dict[str, float]:
    pred_map = {int(x["g"]): str(x["y"]) for x in pred_edit_labels}
    gold_map = {int(x["g"]): str(x["y"]) for x in gold_edit_labels}

    if set(pred_map.keys()) != set(gold_map.keys()):
        return {
            "correct": 0.0,
            "total": float(len(gold_map)),
            "acc": 0.0,
        }

    total = len(gold_map)
    correct = sum(1 for g in gold_map if pred_map[g] == gold_map[g])
    acc = correct / total if total > 0 else 0.0

    return {
        "correct": float(correct),
        "total": float(total),
        "acc": float(acc),
    }


def insert_accuracy(
    pred_insert_labels: Sequence[int],
    gold_insert_labels: Sequence[int],
) -> Dict[str, float]:
    if len(pred_insert_labels) != len(gold_insert_labels):
        return {
            "correct": 0.0,
            "total": float(len(gold_insert_labels)),
            "acc": 0.0,
        }

    total = len(gold_insert_labels)
    correct = sum(int(p) == int(g) for p, g in zip(pred_insert_labels, gold_insert_labels))
    acc = correct / total if total > 0 else 0.0

    return {
        "correct": float(correct),
        "total": float(total),
        "acc": float(acc),
    }


def insert_prf(
    pred_insert_labels: Sequence[int],
    gold_insert_labels: Sequence[int],
) -> Dict[str, float]:
    if len(pred_insert_labels) != len(gold_insert_labels):
        return {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    pred = {i for i, x in enumerate(pred_insert_labels) if int(x) == 1}
    gold = {i for i, x in enumerate(gold_insert_labels) if int(x) == 1}

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _parse_shift_label(y: str) -> int | None:
    if not isinstance(y, str):
        return None
    if not y.startswith("SHIFT:"):
        return None
    try:
        return int(y.split(":", 1)[1])
    except Exception:
        return None


def shift_mae(
    pred_edit_labels: List[Dict],
    gold_edit_labels: List[Dict],
) -> Dict[str, float]:
    """
    只在 gold 为 SHIFT 的边界上统计 |k_pred - k_gold| 的 MAE。
    若 pred 不是 SHIFT，则按 k_pred = 0 处理。
    """
    pred_map = {int(x["g"]): str(x["y"]) for x in pred_edit_labels}
    gold_map = {int(x["g"]): str(x["y"]) for x in gold_edit_labels}

    errs = []
    count = 0

    for g, gold_y in gold_map.items():
        gold_k = _parse_shift_label(gold_y)
        if gold_k is None:
            continue

        pred_y = pred_map.get(g, "KEEP")
        pred_k = _parse_shift_label(pred_y)
        if pred_k is None:
            pred_k = 0

        errs.append(abs(pred_k - gold_k))
        count += 1

    mae = sum(errs) / count if count > 0 else 0.0
    return {
        "count": float(count),
        "mae": float(mae),
    }


def chunk_length_stats_from_spans(spans: Sequence[tuple[int, int]]) -> Dict[str, float]:
    lengths = [e - s for s, e in spans]
    if not lengths:
        return {
            "num_chunks": 0.0,
            "mean_atoms_per_chunk": 0.0,
            "min_atoms_per_chunk": 0.0,
            "max_atoms_per_chunk": 0.0,
        }

    return {
        "num_chunks": float(len(lengths)),
        "mean_atoms_per_chunk": float(sum(lengths) / len(lengths)),
        "min_atoms_per_chunk": float(min(lengths)),
        "max_atoms_per_chunk": float(max(lengths)),
    }


def aggregate_metric_dicts(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}

    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        out[k] = sum(m[k] for m in metric_list) / len(metric_list)
    return out