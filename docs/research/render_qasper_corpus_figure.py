"""Render the public aggregate only; no raw questions, credentials, or API."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("figure output directory already exists")
    raw = source.read_bytes()
    data = json.loads(raw)
    if (data.get("schema") != "slac-qasper-corpus-bridge-v1" or data.get("status") != "completed"
            or data.get("question_count") != 77 or data.get("family_count") != 24
            or data.get("independent_confirmation") is not False or data.get("api_calls") != 0):
        raise ValueError("requires the complete public corpus diagnostic")
    rows = {row["method"]: row for row in data["metrics"]}
    if set(rows) != {"given_document", "corpus_32"} or len(data["metrics"]) != 2:
        raise ValueError("unexpected comparison methods")
    fields = ("source_qualified_evidence_f1", "source_qualified_evidence_recall",
              "candidate_source_qualified_evidence_recall")
    quality = {method: [row[field + "_question_macro"] for field in fields] for method, row in rows.items()}
    hits = [rows["corpus_32"][f"source_doc_hit_at_{k}_units_question_macro"] for k in (1, 5, 8)]
    if not all(np.isfinite(value) and 0 <= value <= 1 for values in quality.values() for value in values):
        raise ValueError("quality metric outside unit interval")
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in hits):
        raise ValueError("source document hit outside unit interval")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "slac-corpus-20260927", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.1), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=.065, right=.985, top=.75, bottom=.24, wspace=.28)
    fig.suptitle("Evidence retrieval across documents", x=.065, y=.97, ha="left", fontsize=19, weight="bold")
    fig.text(.065, .89, "77 exposed development questions / 24 paper families / fixed 32-paper corpus", color="#4b5563")
    colors = {"given_document": "#718096", "corpus_32": "#126b87"}
    labels = {"given_document": "Given paper", "corpus_32": "Query-only corpus"}
    x = np.arange(len(fields))
    for index, method in enumerate(("given_document", "corpus_32")):
        bars = axes[0].bar(x + (index - .5) * .34, quality[method], .34,
                           color=colors[method], label=labels[method])
        axes[0].bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    axes[0].set_xticks(x, ["Evidence F1", "Evidence recall", "Candidate recall"])
    axes[0].set_title("Complete native evidence, with source matching", loc="left", fontsize=11, pad=16)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    bars = axes[1].bar(np.arange(3), hits, .57, color=colors["corpus_32"])
    axes[1].bar_label(bars, labels=[f"{int(round(value * 77))}/77" for value in hits], padding=4, fontsize=11)
    axes[1].set_xticks(np.arange(3), ["Top 1 unit", "Top 5 units", "Top 8 units"])
    axes[1].set_title("Correct paper appears in retrieved units", loc="left", fontsize=11, pad=16)
    for axis in axes:
        axis.set_ylim(0, 1)
        axis.set_yticks(np.arange(0, 1.01, .2))
        axis.grid(axis="y", color="#e5e7eb", linewidth=.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", length=0)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color("#d1d5db")
    tokens = [rows[method]["actual_evidence_tokens_question_macro"] for method in ("given_document", "corpus_32")]
    fig.text(.065, .14, f"Same cap: 3 whole units / 1,024 BGE tokens. Actual means: {tokens[0]:.1f} vs {tokens[1]:.1f} tokens.", fontsize=9)
    fig.text(.065, .09, "Qasper questions assume a known paper. The corpus setting is a stress diagnostic, not the standard task.", fontsize=9, color="#4b5563")
    fig.text(.065, .045, "Point estimates only. No JEV, reranker, answer generation, or independent confirmation in this comparison.", fontsize=9, color="#4b5563")
    if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(raw).digest():
        raise ValueError("public aggregate changed during rendering")
    output.mkdir(parents=True, exist_ok=False)
    fig.savefig(output / "corpus_bridge.png", dpi=180, facecolor="white")
    fig.savefig(output / "corpus_bridge.svg", facecolor="white", metadata={"Date": None, "Creator": "SLAC research"})
    plt.close(fig)
    (output / "figure_manifest.json").write_text(json.dumps({"source_sha256": hashlib.sha256(raw).hexdigest(),
        "matplotlib": matplotlib.__version__, "api_calls": 0, "raw_inputs_read": False,
        "output_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ("corpus_bridge.png", "corpus_bridge.svg")}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(args.input, args.output)
