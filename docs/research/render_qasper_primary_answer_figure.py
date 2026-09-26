"""Render all six primary Answer means and all fifteen predeclared F1 pairs."""
import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ("dense_k3", "I_jev_k3", "I_general_k3", "ordinal_then_p_yes_k3", "p_yes_only_k3", "empty")
PAIRS = tuple((plus, minus) for minus, plus in combinations(("empty", *METHODS[:-1]), 2))
LABELS = {"dense_k3": "Dense", "I_jev_k3": "I-JEV", "I_general_k3": "I-general",
          "ordinal_then_p_yes_k3": "Score ordinal*", "p_yes_only_k3": "Score p(yes)*", "empty": "Empty evidence"}
WEIGHTS = ("question_weighted", "family_balanced")


def render(source, output):
    raw = Path(source).read_bytes()
    data = json.loads(raw)
    if (data.get("status") != "completed" or data.get("record_count") != 462
            or data.get("question_count") != 77 or data.get("family_count") != 24
            or data["publication"].get("independent_mathematical_verification") is not True
            or [m["method"] for m in data["metrics"]] != list(METHODS)
            or [(p["plus"], p["minus"]) for p in data["paired_comparisons"]] != list(PAIRS)
            or any(v != 77 for v in data["publication"]["identical_score_arm_counts"].values())):
        raise ValueError("requires the complete independently verified public primary Answer aggregate")
    if sum(len(p["metrics"]) * 2 for p in data["paired_comparisons"]) != 60:
        raise ValueError("all sixty metric/weight intervals must be retained in source")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "svg.hashsalt": "qasper-primary-answers-20260927",
        "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "#fbfbf8"})
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 13.5), gridspec_kw={"height_ratios": [1, 2.35]})
    figure.patch.set_facecolor("#fbfbf8")
    colors, markers, shifts = ("#245b7a", "#bd7048"), ("o", "s"), (.12, -.12)
    for axis in axes:
        axis.set_facecolor("#fbfbf8")
        axis.grid(axis="x", color="#dce1e2", linewidth=.7)
    for i, method in enumerate(data["metrics"]):
        for w, color, marker, shift in zip(WEIGHTS, colors, markers, shifts, strict=True):
            point = method["metrics"]["official_answer_f1"][w]
            if not math.isfinite(point) or not 0 <= point <= 1:
                raise ValueError("invalid method mean")
            axes[0].scatter(point, 5 - i + shift, color=color, marker=marker, s=35,
                            label=w.replace("_", " ").capitalize() if i == 0 else None, zorder=3)
        axes[0].text(.64, 5 - i,
            f"{method['metrics']['official_answer_f1']['question_weighted']:.3f} / "
            f"{method['metrics']['official_answer_f1']['family_balanced']:.3f}",
            ha="right", va="center", fontsize=9, color="#444d52")
    axes[0].set_yticks(range(5, -1, -1), [LABELS[m] for m in METHODS])
    axes[0].set_xlim(0, .65); axes[0].set_ylim(-.6, 5.6)
    axes[0].set_xticks([0, .1, .2, .3, .4, .5, .6])
    axes[0].set_title("All six fixed methods: Answer F1 means", loc="left", fontsize=12, pad=13)
    axes[0].set_xlabel("Official maximum-reference Answer F1 (0–1)")
    axes[0].text(.64, 5.64, "QW / FB", ha="right", va="bottom", fontsize=9, color="#444d52")
    axes[0].legend(loc="lower right", bbox_to_anchor=(1, 1.05), ncol=2, frameon=False, fontsize=9)
    for i, pair in enumerate(data["paired_comparisons"]):
        for w, color, marker, shift in zip(WEIGHTS, colors, markers, shifts, strict=True):
            estimate = pair["metrics"]["official_answer_f1"][w]
            point = estimate["delta"]; lo, hi = estimate["bootstrap_percentile_95"]
            if not all(math.isfinite(x) for x in (point, lo, hi)) or lo > point or point > hi:
                raise ValueError("invalid paired interval")
            axes[1].errorbar(point, 14 - i + shift, xerr=[[point - lo], [hi - point]],
                fmt=marker, color=color, markersize=4.6, capsize=3, linewidth=1.2, zorder=3)
    axes[1].axvline(0, color="#626d74", linewidth=1, linestyle="--")
    for y in (9.5, 5.5, .5): axes[1].axhline(y, color="#d8dddd", linewidth=.7)
    axes[1].set_yticks(range(14, -1, -1), [f"{LABELS[p]} − {LABELS[m]}" for p, m in PAIRS])
    axes[1].set_xlim(-.10, .60); axes[1].set_ylim(-.6, 14.6)
    axes[1].set_xticks([-.1, 0, .1, .2, .3, .4, .5, .6])
    axes[1].set_title("All fifteen fixed pairs: Answer F1 difference and descriptive 95% intervals", loc="left", fontsize=12, pad=14)
    axes[1].set_xlabel("Named first method minus named second method")
    figure.suptitle("Primary answers: promising development changes, mixed comparative evidence",
                   x=.035, y=.978, ha="left", fontsize=15, fontweight="bold")
    figure.text(.035, .944, "77 questions / 24 families  |  Given paper, fixed candidates  |  Same Qwen generator  |  462 complete predictions",
                fontsize=10, color="#444d52")
    figure.text(.035, .063,
        "* Both score methods use identical evidence and shared responses on all 77 questions; they are not independent repetitions.\n"
        "Score vs dense intervals are positive; score vs I-JEV/I-general intervals cross zero. I-JEV vs dense crosses zero with family weighting.\n"
        "10,000 whole-family bootstrap draws; unadjusted development intervals. Evidence lengths differ. No superiority claim over BGE reranking.",
        fontsize=9, color="#444d52", linespacing=1.7)
    figure.subplots_adjust(left=.29, right=.965, top=.88, bottom=.15, hspace=.42)
    paths = []
    for extension in ("png", "svg"):
        path = output / ("primary_answers." + extension)
        figure.savefig(path, dpi=180, bbox_inches="tight", metadata={"Creator": "SLAC research aggregate renderer"})
        if extension == "svg":
            path.write_text("\n".join(s.rstrip() for s in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        paths.append(path)
    plt.close(figure)
    manifest = {"status": "rendered", "public_source_sha256": hashlib.sha256(raw).hexdigest(),
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "all_six_means": True, "all_fifteen_answer_f1_pairs_both_weights": True,
        "raw_data_read": False, "api_calls": 0, "model_loaded": False}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.input, args.output), indent=2))
