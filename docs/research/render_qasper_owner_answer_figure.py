"""Render all five owner-order answer arms and all five frozen paired contrasts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

METHODS = ("dense_k3", "leaf_owner_k3", "dual_owner_k3", "leaf_owner_leaf_score_k3", "dual_owner_leaf_score_k3")
PAIRS = ((METHODS[3], METHODS[1]), (METHODS[4], METHODS[2]),
         (METHODS[3], METHODS[0]), (METHODS[4], METHODS[0]), (METHODS[4], METHODS[3]))
NAMES = dict(zip(METHODS, ("Dense", "Leaf original", "Dual original", "Leaf score", "Dual score")))


def render(source, output):
    raw = Path(source).read_bytes()
    data = json.loads(raw)
    if (data.get("schema") != "slac-qasper-owner-order-answer-publication-v1"
            or data.get("status") != "complete_audited_development_results"
            or data.get("question_count") != 77 or data.get("family_count") != 24
            or data.get("logical_predictions") != 385 or data.get("new_unique_requests") != 7
            or data.get("inherited_unique_payloads") != 155):
        raise ValueError("complete public answer aggregate required")
    methods, paired = data["metrics"], data["paired_comparisons"]
    if (tuple(row["method"] for row in methods) != METHODS
            or tuple((r["plus"], r["minus"]) for r in paired) != PAIRS):
        raise ValueError("all five methods and fixed comparisons required")
    values = [[row[metric] for row in methods] for metric in
              ("official_answer_f1_question_weighted", "answer_f1_family_balanced")]
    if not np.isfinite(values).all() or not np.all((np.asarray(values) >= 0) & (np.asarray(values) <= 1)):
        raise ValueError("invalid answer F1 means")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "svg.fonttype":"none", "axes.spines.top":False, "axes.spines.right":False,
        "savefig.facecolor":"#fbfbf8"})
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 6.6), gridspec_kw={"width_ratios":[1.1, 1]})
    figure.patch.set_facecolor("#fbfbf8")
    colors, labels = ("#245b7a", "#bf7047"), ("Question weighted", "Family balanced")
    for axis in axes:
        axis.set_facecolor("#fbfbf8")
    for offset, means, color, label in zip((-.18, .18), values, colors, labels, strict=True):
        bars = axes[0].bar(np.arange(5)+offset, means, .32, color=color, label=label, zorder=3)
        axes[0].bar_label(bars, labels=[f"{v:.3f}" for v in means], padding=4, fontsize=8)
    ticks = [NAMES[row["method"]]+f"\n{row['actual_evidence_tokens_question_weighted']:.0f} tok" for row in methods]
    axes[0].set_xticks(range(5), ticks, fontsize=9)
    axes[0].set_ylabel("Official Qasper Answer F1")
    axes[0].set_ylim(0, min(1.05, max(.1, max(max(v) for v in values)*1.25)))
    axes[0].grid(axis="y", color="#dce1e2", linewidth=.7, zorder=0)
    axes[0].set_title("All five frozen arms", loc="left", pad=18, fontsize=12)
    axes[1].axvline(0, color="#626d74", linewidth=1, linestyle="--")
    limits = [0]
    for i, row in enumerate(paired):
        for shift, weighting, color, marker in zip((.12, -.12),
                ("question_weighted", "family_balanced"), colors, ("o", "s"), strict=True):
            estimate = row[weighting]
            point, (low, high) = estimate["delta"], estimate["bootstrap_percentile_95"]
            axes[1].errorbar(point, 4-i+shift, xerr=[[point-low], [high-point]],
                fmt=marker, color=color, markersize=5.5, capsize=4, zorder=3)
            limits.extend((low, high))
    axes[1].set_yticks(range(4,-1,-1), [NAMES[r["plus"]]+" minus "+NAMES[r["minus"]] for r in paired], fontsize=9)
    axes[1].set_ylim(-.6, 4.6)
    span = max(max(limits)-min(limits), .02)
    axes[1].set_xlim(min(limits)-span*.12, max(limits)+span*.12)
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    axes[1].set_xlabel("Paired Answer F1 difference", labelpad=12)
    axes[1].grid(axis="x", color="#dce1e2", linewidth=.7)
    axes[1].set_title("All five fixed contrasts / descriptive 95% intervals", loc="left", pad=18, fontsize=12)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(.04, .87), ncol=2, frameon=False)
    figure.suptitle("Owner-order control: complete downstream answer comparison", x=.04, y=.98,
                   ha="left", fontsize=16, fontweight="bold")
    figure.text(.04, .917, "77 questions / 24 families  |  385 logical predictions / 155 inherited responses + 7 new requests  |  Same Qwen generator and prompt",
                fontsize=10, color="#444d52")
    figure.text(.04, .056,
        "Token labels are mean evidence lengths. Leaf/Dual score: fixed owner candidates, leaf-score priority, source-order rendering.\n"
        "Posthoc development results; 10,000 whole-family bootstrap draws, no multiplicity correction or repeated-generation uncertainty.\n"
        "75/77 and 70/77 control payloads equal Dense; reused answers are not independent replications. No JEV claim.",
        fontsize=9, color="#444d52", linespacing=1.6)
    figure.subplots_adjust(left=.065, right=.975, bottom=.24, top=.77, wspace=.57)
    paths = []
    for suffix in ("png", "svg"):
        path = output/("owner_order_answers."+suffix)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
        paths.append(path)
    plt.close(figure)
    manifest = {"status":"rendered", "public_source_sha256":hashlib.sha256(raw).hexdigest(),
        "output_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "all_five_methods_and_pairs":True, "raw_data_read":False}
    (output/"figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.input, args.output), indent=2))
