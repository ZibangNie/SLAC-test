"""Render the six complete candidate-oracle aggregates without changing scores."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(source, output):
    source, output = Path(source), Path(output)
    data = json.loads(source.read_bytes())
    if (data["status"] != "completed" or data["record_count"] != 462
            or data["publication"]["status"] != "complete_audited_development_results"):
        raise ValueError("only complete audited public aggregates may be rendered")
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "slac-candidate-oracle-20260927"})
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1, 1.45]})
    actual_color, oracle_color, balanced_color = "#66758A", "#166F87", "#B56E20"
    labels = {"leaf_direct": "Direct", "leaf_owner": "Leaf owner", "dual_owner": "Dual owner"}
    for ax, scope, title in zip(axes[:2], ("given_document", "corpus_32"), ("Given document", "32-document stress test")):
        rows = [r for r in data["methods"] if r["scope"] == scope]
        if len(rows) != 3: raise ValueError("missing method")
        for index, row in enumerate(rows):
            actual = row["metrics"]["actual_source_qualified_evidence_f1"]["question_weighted"]
            oracle = row["metrics"]["oracle_source_qualified_evidence_f1"]["question_weighted"]
            ax.plot([actual, oracle], [index, index], color="#C9D4DC", linewidth=3, zorder=1)
            ax.scatter(actual, index, color=actual_color, s=60, zorder=2, label="Actual" if index == 0 else None)
            ax.scatter(oracle, index, color=oracle_color, marker="D", s=55, zorder=2, label="Gold oracle" if index == 0 else None)
            ax.annotate(f"{actual:.3f}", (actual, index), xytext=(0, 12), textcoords="offset points", ha="center", color=actual_color)
            ax.annotate(f"{oracle:.3f}", (oracle, index), xytext=(0, -21), textcoords="offset points", ha="center", color=oracle_color)
        ax.set_yticks(range(3), [labels[r["method"]] for r in rows])
        ax.set_ylim(3.0, -.6); ax.set_xlim(0, 1)
        ax.set_xlabel("Source-qualified Evidence F1\n(question-weighted)")
        ax.set_title(title, fontweight="bold", pad=20)
        ax.grid(axis="x", color="#E3E8EC"); ax.set_axisbelow(True)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax = axes[2]
    pairs = data["comparisons"]
    if len(pairs) != 4: raise ValueError("missing comparison")
    pair_labels = []
    for index, row in enumerate(pairs):
        scope = "Given" if row["scope"] == "given_document" else "Corpus"
        pair_labels.append(f"{scope}: {labels[row['plus']]}\n− {labels[row['minus']]}")
        for offset, weight, color, label in [(-.13, "question_weighted", oracle_color, "Question-weighted"),
                                              (.13, "family_balanced", balanced_color, "Family-balanced")]:
            values = row[weight]; mean = values["delta"]; low, high = values["bootstrap_percentile_95"]
            ax.plot([low, high], [index+offset]*2, color=color, linewidth=2)
            ax.scatter(mean, index+offset, color=color, s=25, label=label if index == 0 else None)
    ax.axvline(0, color="#54616E", linestyle="--", linewidth=1)
    ax.set_yticks(range(4), pair_labels); ax.set_ylim(4.0, -.7); ax.set_xlim(-.06, .075)
    ax.set_xlabel("Oracle F1 difference\n(family bootstrap 95% CI)")
    ax.set_title("All four fixed oracle comparisons", fontweight="bold", pad=20)
    ax.grid(axis="x", color="#E3E8EC"); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#B8C3CC")
        ax.tick_params(axis="y", length=0)
    figure.suptitle("Candidate coverage leaves a large selection gap", x=.055, y=.97, ha="left", fontsize=18, fontweight="bold")
    figure.text(.055, .905, "77 exposed development questions · 24 families · exact enumeration of 311,028 subsets · zero API calls", color="#536372")
    figure.text(.055, .075, "Oracle uses gold references and can choose empty evidence; it is not a deployable selector or Answer F1.\n"
                "Caps are fixed at 3 whole units / 1,024 BGE tokens; actual lengths differ. Intervals are unadjusted across comparisons.",
                color="#536372", fontsize=9, linespacing=1.5)
    figure.subplots_adjust(left=.085, right=.985, top=.8, bottom=.24, wspace=.65)
    for extension in ("png", "svg"):
        figure.savefig(output / f"candidate_oracle.{extension}", dpi=180, metadata={"Date": None} if extension == "svg" else {})
    plt.close(figure)
    return {"png": str((output/"candidate_oracle.png").resolve()), "svg": str((output/"candidate_oracle.svg").resolve())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.output)))
