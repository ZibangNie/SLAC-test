"""Reproduce the public support figure from the complete public aggregate only.

No private artifact, model, tokenizer, API, or QA input is required. Every method
and every primary pair is shown; output files must be new.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASES = ("dense", "I_jev", "I_general", "ordinal_then_p_yes", "p_yes_only")
LABELS = {"dense": "Dense", "I_jev": "JEV ordinal", "I_general": "General ordinal",
          "ordinal_then_p_yes": "JEV ordinal + raw score", "p_yes_only": "JEV raw score only"}
COLORS = ("#667085", "#2467b2", "#19856d", "#dd8b20", "#8b4a9e")


def render(aggregate, svg, png):
    data = json.loads(Path(aggregate).read_text(encoding="utf-8"))
    svg, png = Path(svg), Path(png)
    if svg.exists() or png.exists():
        raise FileExistsError("figure outputs must be new; keep existing artifacts immutable")
    expected = [f"{base}_k{k}" for k in (1, 2, 3) for base in BASES]
    means = data["support"]["method_means"]
    if ([m["method"] for m in means] != expected or data["question_count"] != 77
            or data["family_count"] != 24 or data["reported_interval_count"] != 240):
        raise ValueError("complete fixed aggregate required")
    by_method = {m["method"]: m["metrics"] for m in means}
    pairs = [p for p in data["support"]["paired_comparisons"] if p["plus"].endswith("_k3")]
    ordered = tuple(f"{base}_k3" for base in BASES)
    if [(p["plus"], p["minus"]) for p in pairs] != [(b, a) for a, b in combinations(ordered, 2)]:
        raise ValueError("all ten primary pairs in fixed orientation required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "slac-primary-support-20260927",
        "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig = plt.figure(figsize=(17, 8.5))
    grid = fig.add_gridspec(1, 2, width_ratios=(.95, 1.2), left=.055, right=.98,
                          bottom=.18, top=.83, wspace=.82)
    ax, forest = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    ax.axvspan(2.88, 3.12, color="#f0f1f4", zorder=0)
    for base, color in zip(BASES, COLORS):
        values = [by_method[f"{base}_k{k}"]["official_evidence_f1"]["question_weighted"] for k in (1, 2, 3)]
        ax.plot((1, 2, 3), values, color=color, marker="x" if base == "p_yes_only" else "o",
                linestyle=":" if base == "p_yes_only" else "-", linewidth=2.1,
                markersize=7 if base == "p_yes_only" else 5, label=LABELS[base])
    ax.set(xlim=(.86, 3.14), ylim=(0, .55), xticks=(1, 2, 3),
           xlabel="Maximum selected native units (k)", ylabel="Evidence F1 (question weighted)")
    ax.set_title("A. All 15 fixed settings", loc="left", fontweight="bold", pad=17)
    ax.grid(axis="y", color="#e7e9ed", linewidth=.7)
    ax.text(3, .525, "Primary", ha="center", fontsize=9, color="#414651")
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    ys = np.arange(len(pairs))
    short = lambda method: LABELS[method.removesuffix("_k3")]
    for i in range(len(pairs)):
        if i % 2 == 0:
            forest.axhspan(i - .45, i + .45, color="#f6f7f9", zorder=0)
    for weight, offset, color, marker, label in (
        ("question_weighted", -.13, "#2467b2", "o", "Question weighted"),
        ("family_balanced", .13, "#dd8b20", "D", "Family balanced")):
        points = np.array([p["metrics"]["official_evidence_f1"][weight]["delta"] for p in pairs])
        bounds = np.array([p["metrics"]["official_evidence_f1"][weight]["bootstrap_percentile_95"] for p in pairs])
        forest.errorbar(points, ys + offset, xerr=np.vstack((points - bounds[:, 0], bounds[:, 1] - points)),
                        fmt=marker, color=color, markersize=4.5, linewidth=1.4, capsize=3, label=label)
    forest.axvline(0, color="#656b76", linewidth=1, linestyle="--", zorder=1)
    forest.set(yticks=ys, yticklabels=[f"{short(p['plus'])} − {short(p['minus'])}" for p in pairs],
               xlabel="Evidence F1 difference and exploratory 95% interval", xlim=(-.13, .35),
               ylim=(len(pairs) - .45, -.55))
    forest.tick_params(axis="y", labelsize=8.7, length=0, pad=9)
    forest.set_title("B. Primary k=3: all ten pairs", loc="left", fontweight="bold", pad=17)
    forest.grid(axis="x", color="#e7e9ed", linewidth=.7)
    forest.legend(loc="lower right", frameon=False, fontsize=9)

    fig.suptitle("Recovered primary support: complete development results", x=.055, ha="left",
                 y=.962, fontsize=19, fontweight="bold", color="#182230")
    fig.text(.055, .905, "77 questions · 24 families · given-document fixed candidate pool · 1,024-token whole-pack budget",
             fontsize=11, color="#535862")
    fig.text(.055, .04,
             "The two raw-score rules coincide at every k. Family-balanced means are retained in the report.\n"
             "Intervals: 10,000 shared family bootstrap draws; no multiplicity correction. Actual evidence lengths differ.\n"
             "These results do not establish independent confirmation, calibrated probabilities, shared-relation gains, or final answer quality.",
             fontsize=9, color="#535862", linespacing=1.6)
    for path in (svg, png):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg, metadata={"Date": None, "Creator": "SLAC research; reproducible Matplotlib figure"})
    fig.savefig(png, dpi=180, metadata={"Software": "SLAC research; reproducible Matplotlib figure"})
    plt.close(fig)
    return {"methods": 15, "primary_pairs": 10, "displayed_primary_intervals": 20}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--svg", required=True)
    parser.add_argument("--png", required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.aggregate, args.svg, args.png)))
