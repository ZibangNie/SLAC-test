"""Finite fact-set audit only: no model, corpus, token counter or API.

The 512 combinations are the complete powerset of three artificial atoms,
not dataset sampling or measured decision accuracy. Conflicts, ambiguity and
cross-unit inference are deliberately outside this simple union model.
"""

from __future__ import annotations

import itertools
import json


def audit() -> dict:
    universe = frozenset("abd")
    subsets = tuple(frozenset(x for i, x in enumerate(sorted(universe)) if mask & (1 << i))
                    for mask in range(8))
    policies = {
        "original_relative_add_only": lambda a, r, c, s, t, u: bool(u - s),
        "retained_relative_gain_and_direct_no_loss": lambda a, r, c, s, t, u: bool(c - a) and not (s - t),
        "whole_pack_gain_and_direct_no_loss": lambda a, r, c, s, t, u: bool(t - s) and not (s - t),
        "full_closure_forward_gain_and_reverse_no_loss": lambda a, r, c, s, t, u: bool(u - s) and not (u - t),
    }
    counts = {name: dict(accepted=0, beneficial=0, information_lost=0, semantic_no_op=0,
                         missed_strict_improvement=0) for name in policies}
    strict_count = 0
    for a, r, c in itertools.product(subsets, repeat=3):
        s, t = a | r, a | c
        u = s | c
        strict = s < t
        strict_count += strict
        for name, policy in policies.items():
            accepted = policy(a, r, c, s, t, u)
            row = counts[name]
            row["accepted"] += accepted
            row["beneficial"] += accepted and strict
            row["information_lost"] += accepted and bool(s - t)
            row["semantic_no_op"] += accepted and s == t
            row["missed_strict_improvement"] += strict and not accepted
    for name in ("whole_pack_gain_and_direct_no_loss", "full_closure_forward_gain_and_reverse_no_loss"):
        assert counts[name] == dict(accepted=strict_count, beneficial=strict_count,
                                    information_lost=0, semantic_no_op=0, missed_strict_improvement=0)
    return {"kind": "exhaustive_toy_fact_union_audit", "atoms": sorted(universe),
            "combinations": len(subsets) ** 3, "strict_improvements": strict_count,
            "policies": counts, "api_calls": 0, "model_inferences": 0,
            "natural_data_read": False,
            "limitations": ["not model accuracy", "not answer quality", "not a novelty proof",
                            "union facts only; no conflict or inference dependencies"]}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, sort_keys=True, indent=2))
