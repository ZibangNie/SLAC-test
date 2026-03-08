from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple


def vector_to_gaps(b: Sequence[int]) -> List[int]:
    return [i for i, x in enumerate(b) if int(x) == 1]


def gaps_to_vector(gaps: Sequence[int], num_gaps: int) -> List[int]:
    out = [0] * num_gaps
    for g in gaps:
        if 0 <= g < num_gaps:
            out[g] = 1
    return out


def spans_from_boundary_vector(num_atoms: int, b: Sequence[int]) -> List[Tuple[int, int]]:
    """
    Convert boundary vector b (length M-1) into chunk spans over atoms [s, e).
    """
    if num_atoms == 0:
        return []

    gaps = vector_to_gaps(b)
    spans: List[Tuple[int, int]] = []
    start = 0
    for g in gaps:
        end = g + 1
        spans.append((start, end))
        start = end
    spans.append((start, num_atoms))
    return spans


def units_from_atoms_and_spans(atoms: Sequence[str], spans: Sequence[Tuple[int, int]]) -> List[Dict]:
    units = []
    for uid, (s, e) in enumerate(spans):
        text = "\n".join(atoms[s:e]).strip()
        units.append({"unit_id": uid, "text": text})
    return units


def sample_discrete_laplace_nonzero(K: int, scale: float, rng: random.Random) -> int:
    """
    Sample k in [-K, K] \\ {0} with probability proportional to exp(-|k| / scale).
    """
    candidates = [k for k in range(-K, K + 1) if k != 0]
    weights = [math.exp(-abs(k) / max(scale, 1e-6)) for k in candidates]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for k, w in zip(candidates, weights):
        acc += w
        if acc >= r:
            return k
    return candidates[-1]


def _choose_shift_target(
    g: int,
    occupied: set[int],
    num_gaps: int,
    K: int,
    scale: float,
    rng: random.Random,
) -> int:
    """
    Pick a shifted target within [-K, K], preferring small offsets.
    If no valid target exists, return original g.
    """
    candidates = []
    weights = []

    for k in range(-K, K + 1):
        if k == 0:
            continue
        t = g + k
        if 0 <= t < num_gaps and t not in occupied:
            candidates.append((t, k))
            weights.append(math.exp(-abs(k) / max(scale, 1e-6)))

    if not candidates:
        return g

    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for (t, _k), w in zip(candidates, weights):
        acc += w
        if acc >= r:
            return t
    return candidates[-1][0]


def derive_labels_from_noisy_and_gold(
    b_noisy: Sequence[int],
    b_gold: Sequence[int],
    K: int = 6,
    lambda_del: float = 1.0,
    lambda_ins: float = 1.0,
    lambda_shift: float = 0.25,
) -> Dict[str, List]:
    """
    Align noisy boundaries to gold boundaries with a monotonic DP.

    State meaning:
    - noisy boundaries are initial boundaries G0, which need edit labels
    - unmatched gold boundaries become insert labels

    DP transitions:
    - delete noisy_i
    - insert gold_j
    - match noisy_i -> gold_j if |diff| <= K, cost=lambda_shift*|diff|
    """
    noisy = vector_to_gaps(b_noisy)
    gold = vector_to_gaps(b_gold)

    n = len(noisy)
    m = len(gold)
    INF = 10**18

    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + lambda_del
        back[i][0] = ("DEL", i - 1, 0)

    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + lambda_ins
        back[0][j] = ("INS", 0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # delete noisy[i-1]
            cand = dp[i - 1][j] + lambda_del
            if cand < dp[i][j]:
                dp[i][j] = cand
                back[i][j] = ("DEL", i - 1, j)

            # insert gold[j-1]
            cand = dp[i][j - 1] + lambda_ins
            if cand < dp[i][j]:
                dp[i][j] = cand
                back[i][j] = ("INS", i, j - 1)

            # match noisy[i-1] -> gold[j-1]
            diff = gold[j - 1] - noisy[i - 1]
            if abs(diff) <= K:
                cand = dp[i - 1][j - 1] + lambda_shift * abs(diff)
                if cand < dp[i][j]:
                    dp[i][j] = cand
                    back[i][j] = ("MATCH", i - 1, j - 1)

    edit_labels_rev: List[Dict] = []
    insert_gaps: set[int] = set()

    i, j = n, m
    while i > 0 or j > 0:
        op, pi, pj = back[i][j]

        if op == "DEL":
            g = noisy[i - 1]
            edit_labels_rev.append({"g": g, "y": "DEL"})
            i, j = pi, pj

        elif op == "INS":
            g = gold[j - 1]
            insert_gaps.add(g)
            i, j = pi, pj

        elif op == "MATCH":
            g_noisy = noisy[i - 1]
            g_gold = gold[j - 1]
            diff = g_gold - g_noisy
            if diff == 0:
                y = "KEEP"
            else:
                y = f"SHIFT:{diff}"
            edit_labels_rev.append({"g": g_noisy, "y": y})
            i, j = pi, pj

        else:
            raise RuntimeError(f"Unknown back op: {op}")

    edit_labels = list(reversed(edit_labels_rev))
    insert_labels = [0] * len(b_gold)
    for g in insert_gaps:
        if 0 <= g < len(insert_labels):
            insert_labels[g] = 1

    return {
        "edit": edit_labels,
        "insert": insert_labels,
    }


def apply_boundary_noise(
    b_gold: Sequence[int],
    K: int = 6,
    p_shift: float = 0.45,
    p_insert: float = 0.30,
    p_delete: float = 0.25,
    shift_scale: float = 2.0,
    seed: int | None = None,
) -> Dict:
    """
    Engineering interpretation of the spec:
    1) For each gold boundary, sample keep/delete/shift.
       keep_prob = 1 - p_delete - p_shift
    2) Sample extra spurious inserts on non-boundary gaps.

    This produces b_noisy, then supervision is derived by DP alignment.
    """
    rng = random.Random(seed)
    num_gaps = len(b_gold)
    gold_gaps = vector_to_gaps(b_gold)

    keep_prob = max(0.0, 1.0 - p_delete - p_shift)

    noisy_set: set[int] = set()
    noise_ops: List[Dict] = []

    for g in gold_gaps:
        r = rng.random()

        if r < p_delete:
            noise_ops.append({"src": g, "op": "DELETE"})
            continue

        if r < p_delete + p_shift:
            t = _choose_shift_target(
                g=g,
                occupied=noisy_set,
                num_gaps=num_gaps,
                K=K,
                scale=shift_scale,
                rng=rng,
            )
            noisy_set.add(t)
            noise_ops.append({"src": g, "op": "SHIFT", "dst": t, "k": t - g})
            continue

        noisy_set.add(g)
        noise_ops.append({"src": g, "op": "KEEP"})

    # Extra spurious insertions
    num_extra = sum(1 for _ in gold_gaps if rng.random() < p_insert)
    candidates = [i for i in range(num_gaps) if i not in noisy_set and i not in gold_gaps]
    rng.shuffle(candidates)

    for g in candidates[:num_extra]:
        noisy_set.add(g)
        noise_ops.append({"src": None, "op": "INSERT", "dst": g})

    noisy_gaps = sorted(noisy_set)
    b_noisy = gaps_to_vector(noisy_gaps, num_gaps)

    labels = derive_labels_from_noisy_and_gold(
        b_noisy=b_noisy,
        b_gold=b_gold,
        K=K,
        lambda_del=1.0,
        lambda_ins=1.0,
        lambda_shift=0.25,
    )

    return {
        "b_noisy": b_noisy,
        "labels": labels,
        "noise_ops": noise_ops,
        "stats": {
            "gold_boundary_count": len(gold_gaps),
            "noisy_boundary_count": len(noisy_gaps),
            "insert_pos_count": int(sum(labels["insert"])),
            "edit_count": len(labels["edit"]),
        },
    }