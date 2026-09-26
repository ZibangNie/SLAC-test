"""Plot all four order-only contrasts from the public audited aggregate."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(source, output):
    raw = Path(source).read_bytes()
    data = json.loads(raw)
    if (data.get("status") != "completed" or data.get("question_count") != 77
            or data.get("record_count") != 616 or data.get("posthoc") is not True
            or data.get("all_candidate_sets_and_recall_unchanged") is not True):
        raise ValueError("requires the complete audited order-control aggregate")
    rows = data["comparisons"]
    expected = [(scope, owner) for scope in ("given_document", "corpus_32")
                for owner in ("leaf_owner", "dual_owner")]
    if [(row["scope"], row["owner_method"]) for row in rows] != expected:
        raise ValueError("all four frozen comparisons required")
    for row in rows:
        comparison = row["comparison"]
        if (comparison["plus"] != "leaf_score" or comparison["minus"] != "original_owner"
                or not comparison["candidate_membership_and_recall_unchanged"]):
            raise ValueError("direction or candidate invariant differs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "svg.fonttype":"none", "axes.spines.top":False, "axes.spines.right":False,
        "savefig.facecolor":"#fbfbf8"})
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)
    figure.patch.set_facecolor("#fbfbf8")
    colors = ("#245b7a", "#bf7047")
    weights = ("question_weighted", "family_balanced")
    labels = ("Question weighted", "Family balanced")
    names = ["Given paper / leaf owner", "Given paper / dual owner",
             "Query-only 32 papers / leaf owner", "Query-only 32 papers / dual owner"]
    for axis, metric, title in zip(axes,
            ("source_qualified_evidence_f1", "actual_evidence_tokens"),
            ("Evidence F1 change", "Actual evidence-token change"), strict=True):
        axis.set_facecolor("#fbfbf8")
        axis.axvline(0, color="#626d74", linewidth=1, linestyle="--")
        axis.axhline(1.5, color="#ccd3d5", linewidth=1)
        for i, row in enumerate(rows):
            values = row["comparison"]["metrics"][metric]
            for shift, weight, color, label, marker in zip(
                    (.11, -.11), weights, colors, labels, ("o", "s"), strict=True):
                estimate = values[weight]
                point = estimate["delta"]
                lo, hi = estimate["bootstrap_percentile_95"]
                axis.errorbar(point, 3-i+shift, xerr=[[point-lo], [hi-point]],
                    fmt=marker, color=color, markersize=5.5, capsize=4,
                    label=label if i == 0 else None, zorder=3)
        axis.set_title(title, loc="left", fontsize=12, pad=15)
        axis.set_xlabel("Leaf-score order minus original owner order", labelpad=12)
        axis.set_ylim(-.6, 3.6)
        axis.grid(axis="x", color="#dce1e2", linewidth=.7)
    axes[0].set_yticks([3, 2, 1, 0], names)
    axes[0].set_xlim(-.09, .12)
    axes[0].set_xticks([-.05, 0, .05, .10], ["-0.05", "0", "+0.05", "+0.10"])
    axes[1].set_xlim(-155, 145)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(.035, .87),
                  ncol=2, frameon=False)
    figure.suptitle("Fixed candidates: packing order changes evidence quality and length",
                   x=.035, y=.98, ha="left", fontsize=15, fontweight="bold")
    figure.text(.035, .916,
        "77 questions / 24 families  |  Candidate membership and recall unchanged  |  At most 3 whole units / 1,024 BGE tokens",
        fontsize=10, color="#444d52")
    figure.text(.035, .063,
        "Given-paper F1 rises with longer packs; query-only cross-paper F1 falls with shorter packs. Both scopes are retained.\n"
        "Posthoc development diagnosis; 10,000 family bootstrap draws, unadjusted descriptive 95% intervals. No JEV or answer-quality result.",
        fontsize=9, color="#444d52", linespacing=1.6)
    figure.subplots_adjust(left=.28, right=.965, bottom=.23, top=.76, wspace=.20)
    paths = []
    for extension in ("png", "svg"):
        path = output / ("native_owner_order."+extension)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n",
                            encoding="utf-8")
        paths.append(path)
    plt.close(figure)
    manifest = {"status":"rendered", "public_source_sha256":hashlib.sha256(raw).hexdigest(),
        "output_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "all_four_pairs_and_two_weights":True, "raw_data_read":False, "new_model_calculation":False}
    (output/"figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.input, args.output), indent=2))
