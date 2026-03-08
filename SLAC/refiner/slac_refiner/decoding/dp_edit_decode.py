from __future__ import annotations

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
) -> List[Dict]:
    log_probs = torch.log_softmax(edit_choice_logit_row, dim=-1).detach().cpu().tolist()

    candidates: List[Dict] = []

    # class 0 = DEL
    candidates.append(
        {
            "kind": "DEL",
            "pos": None,
            "score": float(log_probs[0] - lambda_del),
            "label": "DEL",
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
            }
        )

    return candidates


def _dp_monotonic_edit_decode(
    b0: Sequence[int],
    g0_positions: Sequence[int],
    edit_choice_logits: torch.Tensor,
    K: int = 6,
    lambda_del: float = 1.0,
    lambda_shift: float = 0.25,
) -> Tuple[List[int], List[Dict]]:
    """
    DP over initial boundaries:
      candidates[j] = {DEL} U {g_j + k}
    constraint:
      chosen_pos(j) < chosen_pos(j+1)
      DEL is treated as "no emitted boundary" and does not constrain monotonicity directly.

    We keep a DP state over candidate index per boundary.
    Transition validity:
      - DEL -> any is allowed
      - any -> DEL is allowed
      - pos -> pos' requires pos < pos'
    """
    num_gaps = len(b0)
    g0_positions = [int(x) for x in g0_positions if int(x) >= 0]
    n = len(g0_positions)

    if n == 0:
        return [], []

    all_candidates: List[List[Dict]] = []
    for j, g in enumerate(g0_positions):
        cand_j = _build_candidates_for_one_boundary(
            g=g,
            num_gaps=num_gaps,
            edit_choice_logit_row=edit_choice_logits[j],
            K=K,
            lambda_del=lambda_del,
            lambda_shift=lambda_shift,
        )
        all_candidates.append(cand_j)

    NEG_INF = -1e18
    dp: List[List[float]] = []
    back: List[List[Optional[int]]] = []

    # init
    first = all_candidates[0]
    dp.append([c["score"] for c in first])
    back.append([None] * len(first))

    # transition
    for j in range(1, n):
        prev = all_candidates[j - 1]
        cur = all_candidates[j]

        dp_j = [NEG_INF] * len(cur)
        back_j: List[Optional[int]] = [None] * len(cur)

        for c_idx, c in enumerate(cur):
            c_pos = c["pos"]

            for p_idx, p in enumerate(prev):
                p_pos = p["pos"]

                ok = False
                if p_pos is None:
                    ok = True
                elif c_pos is None:
                    ok = True
                else:
                    ok = p_pos < c_pos

                if not ok:
                    continue

                cand_score = dp[j - 1][p_idx] + c["score"]
                if cand_score > dp_j[c_idx]:
                    dp_j[c_idx] = cand_score
                    back_j[c_idx] = p_idx

        dp.append(dp_j)
        back.append(back_j)

    # backtrack
    last_idx = max(range(len(dp[-1])), key=lambda i: dp[-1][i])
    chosen: List[Tuple[int, Dict]] = []

    cur_idx = last_idx
    for j in range(n - 1, -1, -1):
        chosen.append((j, all_candidates[j][cur_idx]))
        prev_idx = back[j][cur_idx]
        if prev_idx is None:
            break
        cur_idx = prev_idx

    chosen.reverse()

    pred_edit_labels: List[Dict] = []
    edit_boundary_positions: List[int] = []

    for j, cand in chosen:
        g = g0_positions[j]
        pred_edit_labels.append({"g": g, "y": cand["label"]})
        if cand["pos"] is not None:
            edit_boundary_positions.append(int(cand["pos"]))

    edit_boundary_positions = sorted(set(edit_boundary_positions))
    return edit_boundary_positions, pred_edit_labels


def _decode_insert_with_suppression(
    insert_logits: torch.Tensor,          # [G]
    edit_boundary_positions: Sequence[int],
    insert_threshold: float = 0.5,
    min_sep: int = 1,
) -> Tuple[List[int], List[int]]:
    """
    Threshold + neighborhood suppression:
    if a predicted insert gap is too close to any edit boundary, skip it.
    """
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
    min_sep: int = 1,
    lambda_del: float = 1.0,
    lambda_ins: float = 1.0,
    lambda_shift: float = 0.25,
) -> Dict:
    num_gaps = len(b0)

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
        insert_logits=insert_logits,
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
    min_sep: int = 1,
    lambda_del: float = 1.0,
    lambda_ins: float = 1.0,
    lambda_shift: float = 0.25,
) -> DecodeOutput:
    B = b0.shape[0]

    pred_b_all = []
    pred_edit_all = []
    pred_insert_all = []
    pred_gaps_all = []

    for i in range(B):
        b0_i = b0[i].detach().cpu().tolist()
        g0_i = [int(x) for x in g0_positions[i].detach().cpu().tolist() if int(x) >= 0]

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