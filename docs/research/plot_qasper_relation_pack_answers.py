"""Plot every frozen full-adjacency answer and length contrast from the public aggregate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ("I_jev_k3", "dense_k3", "reranker_k3", "p_yes_only_k3")
LABELS = ("vs. JEV ordinal (primary)", "vs. dense", "vs. BGE reranker", "vs. JEV raw score")


def render(aggregate, svg, png):
    data = json.loads(Path(aggregate).read_text(encoding="utf-8"))
    svg, png = Path(svg), Path(png)
    if svg.exists() or png.exists():
        raise FileExistsError("figure outputs must be new")
    pairs = data["paired_comparisons"]
    if (data.get("status") != "completed" or data.get("question_count") != 77
            or data.get("family_count") != 24 or data["specification"].get("paired_intervals") != 16
            or data["specification"].get("posthoc_exposed_development") is not True
            or [(p["plus"], p["minus"]) for p in pairs] != [("full_adjacency", m) for m in METHODS]):
        raise ValueError("complete fixed full-adjacency aggregate required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "slac-relation-pack-answers-20260927",
        "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), sharey=True)
    fig.subplots_adjust(left=.225, right=.98, top=.73, bottom=.26, wspace=.20)
    ys = np.arange(4)
    metrics = ("official_answer_f1", "actual_evidence_tokens")
    for ax, metric, title, xlabel in zip(axes, metrics,
            ("A. Answer F1", "B. Actual evidence length"),
            ("Answer F1 difference", "BGE-token difference")):
        for i in range(4):
            if i % 2 == 0: ax.axhspan(i - .45, i + .45, color="#f5f6f8", zorder=0)
        for weight, offset, color, marker, label in (
            ("question_weighted", -.13, "#2467b2", "o", "Question weighted"),
            ("family_balanced", .13, "#c77916", "D", "Family balanced")):
            points = np.array([p["metrics"][metric][weight]["delta"] for p in pairs])
            bounds = np.array([p["metrics"][metric][weight]["bootstrap_percentile_95"] for p in pairs])
            if not np.isfinite(points).all() or not np.isfinite(bounds).all():
                raise ValueError("nonfinite plot values")
            ax.errorbar(points, ys + offset, xerr=np.vstack((points - bounds[:, 0], bounds[:, 1] - points)),
                        fmt=marker, color=color, markersize=5, linewidth=1.7, capsize=3, label=label)
        ax.axvline(0, color="#656b76", linewidth=1, linestyle="--", zorder=1)
        ax.set(yticks=ys, yticklabels=LABELS, ylim=(3.5, -.5), xlabel=xlabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=17)
        ax.grid(axis="x", color="#e1e4e8", linewidth=.7)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.margins(x=.08)
    fig.suptitle("Full adjacency: complete reachable-pack answer study",
                 x=.025, ha="left", y=.96, fontsize=17, fontweight="bold", color="#182230")
    fig.text(.025, .885, "Every difference is full adjacency minus comparator. 77 questions · 24 families · k=3 · 1,024-token cap",
             fontsize=11, color="#535862")
    fig.text(.025, .835, "All four pairs and all 16 exploratory intervals are shown; negative length means a shorter evidence pack.",
             fontsize=10, color="#535862")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.62, .125), ncol=2, frameon=False)
    fig.text(.025, .038,
        "Exploratory development study. 10,000 shared family bootstrap draws; no multiplicity correction.\n"
        "All 96 reachable packs were covered before scoring; 14 new calls and exact historical response reuse.\n"
        "One saved generation per payload. Actual lengths differ; no learned-relation effect or held-out confirmation.",
        fontsize=9, color="#535862", linespacing=1.45)
    for path in (svg, png): path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg, metadata={"Date": None, "Creator": "SLAC research; reproducible Matplotlib figure"})
    fig.savefig(png, dpi=180, metadata={"Software": "SLAC research; reproducible Matplotlib figure"})
    plt.close(fig)
    return {"pairs": 4, "metrics": 2, "weightings": 2, "displayed_intervals": 16, "posthoc": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aggregate", "svg", "png"): parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.aggregate, args.svg, args.png)))
