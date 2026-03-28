import pandas as pd
import matplotlib.pyplot as plt
from textwrap import fill

# -----------------------------
# Table 4 data
# -----------------------------
rows = [
    ["F1", "Functional", "Heterogeneous document ingestion",
     "Support PDF, DOCX, TXT, and structured JSON inputs through a common preprocessing entry path."],

    ["F2", "Functional", "Preservation of structural information",
     "Retain headings, numbered clauses, list items, and inherited scope through a validated units-tree representation."],

    ["F3", "Functional", "Adaptive segmentation",
     "Combine stable initial boundaries with local boundary refinement under practical constraints rather than using a fixed splitter."],

    ["F4", "Functional", "Multi-granularity retrieval",
     "Provide both chunk-level and leaf-level access and combine semantic and lexical retrieval signals."],

    ["F5", "Functional", "Budget-aware evidence preparation",
     "Select, deduplicate, and organise evidence under token and item budgets before LLM invocation while preserving source traceability."],

    ["F6", "Functional", "Modular interoperability",
     "Expose well-defined stage inputs and outputs so that components can be rerun, debugged, or replaced independently."],

    ["F7", "Functional", "Interactive usability",
     "Support document upload, session-based indexing, and user-facing querying through an OpenWebUI bridge."],

    ["NF1", "Non-functional", "Reproducibility",
     "Use JSONL artefacts, YAML configuration, frozen schemas, and explicit run summaries so outputs can be reproduced and inspected."],

    ["NF2", "Non-functional", "Robustness to imperfect input",
     "Remain stable under OCR noise, formatting noise, and inconsistent structural cues through conservative early segmentation."],

    ["NF3", "Non-functional", "Controllability",
     "Expose practical controls such as atom size, chunk budget, retrieval top-k, and evidence token budgets."],

    ["NF4", "Non-functional", "Traceability and provenance",
     "Retain evidence identifiers and source links so the system can report which evidence items supported an answer."],

    ["NF5", "Non-functional", "Architectural separation of concerns",
     "Keep OpenWebUI as a thin protocol adapter and separate retrieval, reranking, and integration into distinct stages."]
]

columns = [
    "Requirement ID",
    "Type",
    "Requirement",
    "Design consequence in SLAC"
]

df = pd.DataFrame(rows, columns=columns)

# Save CSV
df.to_csv("table4_requirements.csv", index=False, encoding="utf-8-sig")

# -----------------------------
# Render table as figure
# -----------------------------
def wrap_text(text, width):
    return fill(str(text), width=width)

wrapped_df = df.copy()
wrapped_df["Requirement"] = wrapped_df["Requirement"].apply(lambda x: wrap_text(x, 28))
wrapped_df["Design consequence in SLAC"] = wrapped_df["Design consequence in SLAC"].apply(lambda x: wrap_text(x, 62))

cell_text = wrapped_df.values.tolist()
col_labels = wrapped_df.columns.tolist()

fig, ax = plt.subplots(figsize=(16, 9))
ax.axis("off")

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="left",
    colLoc="center",
    loc="center",
    colWidths=[0.10, 0.13, 0.24, 0.53]
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.15)

# Header + body styling
for (r, c), cell in table.get_celld().items():
    cell.set_linewidth(0.8)
    if r == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#DCE6F1")
    else:
        if c == 1 and wrapped_df.iloc[r-1, 1] == "Functional":
            cell.set_facecolor("#F7FBFF")
        elif c == 1 and wrapped_df.iloc[r-1, 1] == "Non-functional":
            cell.set_facecolor("#FFF8F0")
        else:
            cell.set_facecolor("white")

# Slightly increase row heights for long rows
nrows = len(wrapped_df) + 1
for r in range(1, nrows):
    text_len = max(
        len(str(wrapped_df.iloc[r-1, 2])),
        len(str(wrapped_df.iloc[r-1, 3]))
    )
    height = 0.055 if text_len < 80 else 0.072
    for c in range(len(columns)):
        table[(r, c)].set_height(height)

plt.tight_layout()
plt.savefig("table4_requirements.png", dpi=300, bbox_inches="tight")
plt.savefig("table4_requirements.pdf", bbox_inches="tight")
plt.close()

print("Saved:")
print("  table4_requirements.csv")
print("  table4_requirements.png")
print("  table4_requirements.pdf")