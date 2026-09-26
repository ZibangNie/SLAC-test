from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import torch


EDIT_KEEP = 0
EDIT_DEL = 1
EDIT_SHIFT = 2


@dataclass
class DecodeOutput:
    pred_b: List[List[int]]
    pred_edit_labels: List[List[Dict]]
    pred_insert_labels: List[List[int]]
    pred_gaps: List[List[int]]


def _vector_to_gaps(b: Sequence[int]) -> List[int]:
    return [i for i, x in enumerate(b) if int(x) == 1]


def _gaps_to_vector(gaps: Sequence[int], num_gaps: int) -> List[int]:
    out = [0] * num_gaps
    for g in gaps:
        if 0 <= g < num_gaps:
            out[g] = 1
    return out


def _safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(x, eps))


def _build_candidates_for_one_boundary(
    g: int,
    num_gaps: int,
    edit_choice_logit_row: torch.Tensor,   # [2K+2]
    K: int,
    lambda_del: float,
    lambda_shift: float,
    temperature: float = 1.0,
) -> List[Dict]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if K < 0 or edit_choice_logit_row.shape != (2 * K + 2,):
        raise ValueError("edit logits must have 2*K+2 classes")
    log_probs = torch.log_softmax(
        edit_choice_logit_row.detach().float() / temperature, dim=-1
    ).cpu().tolist()
    if any(math.isnan(p) for p in log_probs):
        raise ValueError("edit logits must define a valid probability distribution")

    candidates: List[Dict] = []

    # class 0 = DEL
    candidates.append(
        {
            "kind": "DEL",
            "pos": None,
            "score": float(log_probs[0] - lambda_del),
            "label": "DEL",
            "prob": math.exp(log_probs[0]),
        }
    )

    # classes 1..2K+1 correspond to k in [-K..K]
    for cls_idx in range(1, 2 * K + 2):
        k = cls_idx - 1 - K
        pos = g + k
        if not (0 <= pos < num_gaps):
            continue

        score = float(log_probs[cls_idx] - lambda_shift * abs(k))

        label = "KEEP" if k == 0 else f"SHIFT:{k}"
        kind = "KEEP" if k == 0 else "SHIFT"

        candidates.append(
            {
                "kind": kind,
                "pos": pos,
                "score": score,
                "label": label,
                "prob": math.exp(log_probs[cls_idx]),
                "shift": k,
            }
        )

    return candidates


def _select_monotonic_candidates(all_candidates: Sequence[Sequence[Dict]]) -> List[Dict]:
    """Maximize the sum of candidate scores with strictly increasing emissions.

    State is the last emitted gap, including -1 before any emission. DEL keeps
    that state, so intervening deletions cannot permit a crossing or duplicate.
    Prefix maxima avoid testing every previous state for each emitted candidate.
    The sampler also uses this solver after perturbing its candidate scores.
    """
    scores = {-1: 0.0}
    history: List[Dict[int, Tuple[int, int]]] = []
    for candidates in all_candidates:
        previous = sorted(scores)
        prefix_best: List[int] = []
        best = previous[0]
        for last in previous:
            if scores[last] > scores[best]:
                best = last
            prefix_best.append(best)

        next_scores: Dict[int, float] = {}
        back: Dict[int, Tuple[int, int]] = {}
        for index, candidate in enumerate(candidates):
            pos = candidate["pos"]
            if pos is None:
                transitions = ((last, last) for last in previous)
            else:
                # Gaps are integers; only states strictly below pos are legal.
                stop = bisect_left(previous, pos)
                transitions = ((prefix_best[stop - 1], int(pos)),) if stop else ()
            for last, emitted in transitions:
                value = scores[last] + float(candidate["score"])
                if value > next_scores.get(emitted, -math.inf):
                    next_scores[emitted] = value
                    back[emitted] = (last, index)
        if not next_scores:
            raise ValueError("no finite legal edit path")
        scores = next_scores
        history.append(back)

    if not history:
        return []
    last = max(scores, key=scores.get)
    chosen: List[Dict] = []
    for row in range(len(history) - 1, -1, -1):
        last, index = history[row][last]
        chosen.append(all_candidates[row][index])
    chosen.reverse()
    return chosen


def _valid_boundary_rows(
    b0: Sequence[int], g0_positions: Sequence[int], edit_choice_logits: torch.Tensor, K: int
) -> List[Tuple[int, int]]:
    if K < 0 or edit_choice_logits.ndim != 2 or edit_choice_logits.shape[1] != 2 * K + 2:
        raise ValueError("edit logits must have shape [B0, 2*K+2]")
    rows = [(row, int(g)) for row, g in enumerate(g0_positions) if int(g) >= 0]
    gaps = [g for _, g in rows]
    if any(g >= len(b0) for g in gaps) or gaps != sorted(set(gaps)):
        raise ValueError("initial gaps must be distinct, increasing, and inside b0")
    if rows and rows[-1][0] >= edit_choice_logits.shape[0]:
        raise ValueError("missing edit-logit rows for initial boundaries")
    if gaps != _vector_to_gaps(b0):
        raise ValueError("g0_positions must enumerate the boundaries in b0")
    return rows


def _dp_monotonic_edit_decode(
    b0: Sequence[int],
    g0_positions: Sequence[int],
    edit_choice_logits: torch.Tensor,
    K: int = 6,
    lambda_del: float = 1.0,
    lambda_shift: float = 0.25,
) -> Tuple[List[int], List[Dict]]:
    """Decode local edits; gap g is after atom g, within [0, T-2]."""
    num_gaps = len(b0)
    rows = _valid_boundary_rows(b0, g0_positions, edit_choice_logits, K)

    all_candidates: List[List[Dict]] = []
    for j, g in rows:
        cand_j = _build_candidates_for_one_boundary(
            g=g,
            num_gaps=num_gaps,
            edit_choice_logit_row=edit_choice_logits[j],
            K=K,
            lambda_del=lambda_del,
            lambda_shift=lambda_shift,
        )
        all_candidates.append(cand_j)

    chosen = _select_monotonic_candidates(all_candidates)

    pred_edit_labels: List[Dict] = []
    edit_boundary_positions: List[int] = []

    for (_, g), cand in zip(rows, chosen):
        pred_edit_labels.append({"g": g, "y": cand["label"]})
        if cand["pos"] is not None:
            edit_boundary_positions.append(int(cand["pos"]))

    return edit_boundary_positions, pred_edit_labels


def _decode_insert_with_suppression(
    insert_logits: torch.Tensor,          # [G]
    edit_boundary_positions: Sequence[int],
    insert_threshold: float = 0.5,
    min_sep: int = 0,
) -> Tuple[List[int], List[int]]:
    """
    Threshold + neighborhood suppression:
    if a predicted insert gap is too close to any edit boundary, skip it.
    """
    if min_sep < 0 or not 0 <= insert_threshold <= 1:
        raise ValueError("min_sep must be nonnegative and threshold must lie in [0, 1]")
    probs = torch.sigmoid(insert_logits).detach().cpu().tolist()

    pred_insert_labels = [0] * len(probs)
    accepted = []

    for g, p in enumerate(probs):
        if p < insert_threshold:
            continue

        too_close = any(abs(g - b) <= min_sep for b in edit_boundary_positions)
        if too_close:
            continue

        pred_insert_labels[g] = 1
        accepted.append(g)

    return pred_insert_labels, accepted


def decode_one(
    b0: Sequence[int],
    g0_positions: Sequence[int],
    edit_choice_logits: torch.Tensor,   # [B0, 2K+2]
    insert_logits: torch.Tensor,        # [G]
    K: int = 6,
    insert_threshold: float = 0.5,
    min_sep: int = 0,
    lambda_del: float = 1.0,
    lambda_ins: float = 1.0,
    lambda_shift: float = 0.25,
) -> Dict:
    num_gaps = len(b0)
    if insert_logits.ndim != 1 or insert_logits.shape[0] < num_gaps:
        raise ValueError("insert logits must cover every real gap")

    # A) Edit-DP
    edit_boundary_positions, pred_edit_labels = _dp_monotonic_edit_decode(
        b0=b0,
        g0_positions=g0_positions,
        edit_choice_logits=edit_choice_logits,
        K=K,
        lambda_del=lambda_del,
        lambda_shift=lambda_shift,
    )

    # B) Insert threshold + suppression
    pred_insert_labels, insert_boundary_positions = _decode_insert_with_suppression(
        insert_logits=insert_logits[:num_gaps],
        edit_boundary_positions=edit_boundary_positions,
        insert_threshold=insert_threshold,
        min_sep=min_sep,
    )

    # C) Union
    final_gaps = sorted(set(edit_boundary_positions) | set(insert_boundary_positions))
    pred_b = _gaps_to_vector(final_gaps, num_gaps)

    return {
        "pred_b": pred_b,
        "pred_edit_labels": pred_edit_labels,
        "pred_insert_labels": pred_insert_labels,
        "pred_gaps": final_gaps,
    }


def batch_decode(
    b0: torch.Tensor,
    g0_positions: torch.Tensor,
    edit_choice_logits: torch.Tensor,   # [B, B0, 2K+2]
    insert_logits: torch.Tensor,
    K: int = 6,
    insert_threshold: float = 0.5,
    min_sep: int = 0,
    lambda_del: float = 1.0,
    lambda_ins: float = 1.0,
    lambda_shift: float = 0.25,
    num_gaps: Optional[Sequence[int]] = None,
) -> DecodeOutput:
    B = b0.shape[0]
    # Omission retains the historical unpadded-batch API. Mixed-length callers
    # must supply their true lengths; trailing zero boundaries cannot reveal it.
    lengths = [b0.shape[1]] * B if num_gaps is None else [int(n) for n in num_gaps]
    if len(lengths) != B or any(n < 0 or n > b0.shape[1] for n in lengths):
        raise ValueError("num_gaps must contain one valid length per document")

    pred_b_all = []
    pred_edit_all = []
    pred_insert_all = []
    pred_gaps_all = []

    for i in range(B):
        b0_i = b0[i, :lengths[i]].detach().cpu().tolist()
        g0_i = g0_positions[i].detach().cpu().tolist()

        out = decode_one(
            b0=b0_i,
            g0_positions=g0_i,
            edit_choice_logits=edit_choice_logits[i],
            insert_logits=insert_logits[i],
            K=K,
            insert_threshold=insert_threshold,
            min_sep=min_sep,
            lambda_del=lambda_del,
            lambda_ins=lambda_ins,
            lambda_shift=lambda_shift,
        )
        pred_b_all.append(out["pred_b"])
        pred_edit_all.append(out["pred_edit_labels"])
        pred_insert_all.append(out["pred_insert_labels"])
        pred_gaps_all.append(out["pred_gaps"])

    return DecodeOutput(
        pred_b=pred_b_all,
        pred_edit_labels=pred_edit_all,
        pred_insert_labels=pred_insert_all,
        pred_gaps=pred_gaps_all,
    )
