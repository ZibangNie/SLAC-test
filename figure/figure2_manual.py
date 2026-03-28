import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np

EDGE = "#94A3B8"
BORDER = "#CBD5E1"
TEXT = "#111827"
TITLE_BG = "#F8FAFC"
PROCESS_BG = "#FFFFFF"
ARTEFACT_BG = "#DBEAFE"
QUERY_BG = "#EFF6FF"
FINAL_BG = "#DCFCE7"

LW_BOX = 1.2
LW_EDGE = 1.35
TITLE_H = 0.38

fig, ax = plt.subplots(figsize=(16, 10.5))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis("off")

panels = {}
boxes = {}

def draw_panel(key, x, y, w, h, title):
    outer = Rectangle((x, y), w, h, linewidth=LW_BOX, edgecolor=BORDER, facecolor="white")
    band = Rectangle((x, y + h - TITLE_H), w, TITLE_H, linewidth=LW_BOX, edgecolor=BORDER, facecolor=TITLE_BG)
    ax.add_patch(outer)
    ax.add_patch(band)
    ax.text(x + 0.18, y + h - TITLE_H / 2, title, ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=TEXT)
    panels[key] = {
        "x": x, "y": y, "w": w, "h": h,
        "cx": x + 0.22, "cy": y + 0.18,
        "cw": w - 0.44, "ch": h - TITLE_H - 0.32,
    }

def add_box(panel_key, box_key, rx, ry, rw, rh, label, fill, fontsize=9.4):
    p = panels[panel_key]
    x = p["cx"] + rx * p["cw"]
    y = p["cy"] + ry * p["ch"]
    w = rw * p["cw"]
    h = rh * p["ch"]
    rect = Rectangle((x, y), w, h, linewidth=1.0, edgecolor="#64748B", facecolor=fill)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fontsize,
            color=TEXT, linespacing=1.02)
    boxes[f"{panel_key}.{box_key}"] = {"x": x, "y": y, "w": w, "h": h}


def edge_point(name, side, frac=0.5):
    b = boxes[name]
    if side == "n":
        return (b["x"] + frac * b["w"], b["y"] + b["h"])
    if side == "s":
        return (b["x"] + frac * b["w"], b["y"])
    if side == "e":
        return (b["x"] + b["w"], b["y"] + frac * b["h"])
    if side == "w":
        return (b["x"], b["y"] + frac * b["h"])
    raise ValueError(side)


def draw_polyline(points, color=EDGE, lw=LW_EDGE, dashed=False, head=True):
    ls = "--" if dashed else "-"
    pts = list(points)
    if len(pts) < 2:
        return

    if head:
        a = np.array(pts[-2], float)
        b = np.array(pts[-1], float)
        v = b - a
        L = np.hypot(v[0], v[1])
        if L == 0:
            return
        u = v / L
        head_len = 0.13
        head_w = 0.10
        pre = b - u * head_len
        pts2 = pts[:-1] + [tuple(pre)]
        for p, q in zip(pts2[:-1], pts2[1:]):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, linestyle=ls, solid_capstyle="butt")
        perp = np.array([-u[1], u[0]])
        base1 = pre + perp * (head_w / 2)
        base2 = pre - perp * (head_w / 2)
        poly = Polygon([tuple(b), tuple(base1), tuple(base2)], closed=True, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
    else:
        for p, q in zip(pts[:-1], pts[1:]):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, linestyle=ls, solid_capstyle="butt")

# Panels
draw_panel("p1", 0.95, 11.20, 7.10, 2.10, "1. Document Ingestion and Structural Normalisation")
draw_panel("p2", 11.95, 11.20, 7.10, 2.10, "2. Seed Chunking and Atom Construction")
draw_panel("p3", 0.85, 7.85, 7.55, 2.55, "3. Boundary Refinement under Structural Constraints")
draw_panel("p4", 0.85, 4.90, 7.55, 2.10, "4. Retrieval Build and Multi-Granularity Indexing")
draw_panel("p5", 11.05, 5.45, 7.95, 4.95, "5. Query-Time Retrieval and Structure-Aware Expansion")
draw_panel("p6", 0.95, 0.90, 7.10, 2.55, "6. Neural Reranking and Budget-Aware Evidence Packing")
draw_panel("p7", 11.05, 0.90, 7.95, 2.55, "7. Integration, LLM Invocation, and OpenWebUI-Oriented Orchestration")

# Boxes
add_box("p1", "raw",     0.02, 0.22, 0.28, 0.44, "Raw documents\nPDF / DOCX / TXT / JSON", QUERY_BG, 9.3)
add_box("p1", "readers", 0.36, 0.22, 0.30, 0.44, "Format-specific readers\nand content extraction", PROCESS_BG, 9.0)
add_box("p1", "norm",    0.72, 0.22, 0.26, 0.44, "Normalised\nstructure-aware documents", ARTEFACT_BG, 9.2)

add_box("p2", "chunk0",  0.02, 0.22, 0.28, 0.44, "Rule-based initial\nsegmentation (chunk0)", PROCESS_BG, 9.0)
add_box("p2", "atomise", 0.36, 0.22, 0.30, 0.44, "Atom construction\n(sentences / lines / units)", PROCESS_BG, 9.0)
add_box("p2", "b0",      0.72, 0.22, 0.26, 0.44, "Atoms + seed\nboundaries b₀", ARTEFACT_BG, 9.2)

add_box("p3", "refiner", 0.02, 0.56, 0.28, 0.24, "Boundary\nrefiner", PROCESS_BG, 9.3)
add_box("p3", "edits",   0.35, 0.53, 0.31, 0.29, "Local boundary edits\nkeep / delete /\nshift / insert", PROCESS_BG, 8.8)
add_box("p3", "decode",  0.70, 0.56, 0.28, 0.24, "Constrained decoding\nand projection", PROCESS_BG, 9.0)
add_box("p3", "leaf",    0.02, 0.08, 0.28, 0.23, "Leaf records", ARTEFACT_BG, 9.3)
add_box("p3", "chunks",  0.35, 0.08, 0.28, 0.23, "Refined chunks", ARTEFACT_BG, 9.3)
add_box("p3", "catalog", 0.67, 0.08, 0.31, 0.23, "Document catalog\nand hierarchy metadata", ARTEFACT_BG, 8.5)

add_box("p4", "leaf_idx",   0.02, 0.50, 0.28, 0.26, "Leaf-level\ndense index", PROCESS_BG, 9.1)
add_box("p4", "chunk_idx",  0.35, 0.50, 0.28, 0.26, "Chunk-level\ndense index", PROCESS_BG, 9.1)
add_box("p4", "anchor_idx", 0.68, 0.50, 0.30, 0.26, "Anchor-oriented\nlexical index", PROCESS_BG, 9.0)
add_box("p4", "lookup",     0.22, 0.08, 0.56, 0.20, "Lookup resources\nparent / child / sibling / provenance", ARTEFACT_BG, 8.7)

add_box("p5", "query",      0.06, 0.77, 0.36, 0.17, "User question", QUERY_BG, 9.6)
add_box("p5", "prep",       0.58, 0.77, 0.36, 0.17, "Query preparation\nkeywords / bilingual variants / anchors", PROCESS_BG, 8.7)
add_box("p5", "leaf_ret",   0.06, 0.49, 0.24, 0.17, "Leaf dense\nretrieval", PROCESS_BG, 9.2)
add_box("p5", "chunk_ret",  0.38, 0.49, 0.24, 0.17, "Chunk dense\nretrieval", PROCESS_BG, 9.2)
add_box("p5", "anchor_ret", 0.70, 0.49, 0.24, 0.17, "Anchor lexical\nretrieval", PROCESS_BG, 9.2)
add_box("p5", "fusion",     0.06, 0.14, 0.24, 0.20, "Cross-view score fusion\nand candidate aggregation", PROCESS_BG, 8.7)
add_box("p5", "expand",     0.38, 0.14, 0.24, 0.20, "Tree-aware expansion\nparent / child / sibling /\nbranch support", PROCESS_BG, 8.4)
add_box("p5", "pool",       0.70, 0.14, 0.24, 0.20, "Expanded candidate pool\nwith structural context", ARTEFACT_BG, 8.7)

add_box("p6", "pairs",      0.03, 0.55, 0.27, 0.21, "Query-candidate\npair builder", PROCESS_BG, 9.0)
add_box("p6", "reranker",   0.36, 0.55, 0.27, 0.21, "Neural reranker", PROCESS_BG, 9.3)
add_box("p6", "shortlist",  0.69, 0.55, 0.27, 0.21, "Reranked shortlist", ARTEFACT_BG, 9.0)
add_box("p6", "packing",    0.17, 0.11, 0.30, 0.22, "Evidence packing\nbudget / deduplication /\nordering", PROCESS_BG, 8.6)
add_box("p6", "evidence",   0.56, 0.11, 0.29, 0.22, "Packed evidence bundle", ARTEFACT_BG, 8.8)

add_box("p7", "req",        0.05, 0.72, 0.40, 0.18, "Integration request\nquestion + session context + evidence", PROCESS_BG, 8.5)
add_box("p7", "prompt",     0.55, 0.72, 0.40, 0.18, "Grounded prompt\nconstruction and answer policy", PROCESS_BG, 8.6)
add_box("p7", "llm",        0.05, 0.42, 0.40, 0.18, "LLM invocation", PROCESS_BG, 9.3)
add_box("p7", "answer",     0.55, 0.42, 0.40, 0.18, "Grounded answer\nor abstention", FINAL_BG, 9.0)
add_box("p7", "bridge",     0.05, 0.12, 0.40, 0.18, "OpenWebUI bridge\nsession-aware orchestration", PROCESS_BG, 8.6)
add_box("p7", "ui",         0.55, 0.12, 0.40, 0.18, "User-visible response\nin OpenWebUI", FINAL_BG, 8.9)

# Internal arrows

def h(a,b):
    draw_polyline([edge_point(a,'e'), edge_point(b,'w')])

h('p1.raw','p1.readers')
h('p1.readers','p1.norm')
h('p2.chunk0','p2.atomise')
h('p2.atomise','p2.b0')
h('p3.refiner','p3.edits')
h('p3.edits','p3.decode')

# decode -> outputs with distinct lanes
s = edge_point('p3.decode','s',0.5)
lanes = [('p3.leaf',0.20,-0.16),('p3.chunks',0.50,-0.28),('p3.catalog',0.80,-0.16)]
for target, frac, drop in lanes:
    t = edge_point(target,'n',frac)
    ymid = s[1] + drop
    draw_polyline([s,(s[0],ymid),(t[0],ymid),t])

h('p5.query','p5.prep')

# prep -> retrievals, using separate vertical lanes above each target
for src_frac, target, lift in [(0.18,'p5.leaf_ret',0.18),(0.50,'p5.chunk_ret',0.30),(0.82,'p5.anchor_ret',0.18)]:
    s = edge_point('p5.prep','s',src_frac)
    t = edge_point(target,'n',0.5)
    y = t[1] + lift
    draw_polyline([s,(s[0],y),(t[0],y),t])

# retrieval -> fusion with three distinct entry points and no shared verticals
for src, src_frac, dst_frac, y in [
    ('p5.leaf_ret',0.20,0.20, edge_point('p5.fusion','n')[1]+0.18),
    ('p5.chunk_ret',0.50,0.50, edge_point('p5.fusion','n')[1]+0.32),
    ('p5.anchor_ret',0.80,0.80, edge_point('p5.fusion','n')[1]+0.46),
]:
    s = edge_point(src,'s',src_frac)
    t = edge_point('p5.fusion','n',dst_frac)
    draw_polyline([s,(s[0],y),(t[0],y),t])

h('p5.fusion','p5.expand')
h('p5.expand','p5.pool')
h('p6.pairs','p6.reranker')
h('p6.reranker','p6.shortlist')

s = edge_point('p6.shortlist','s',0.55)
t = edge_point('p6.packing','n',0.55)
draw_polyline([s,(s[0],t[1]+0.18),(t[0],t[1]+0.18),t])
h('p6.packing','p6.evidence')

h('p7.req','p7.prompt')
s = edge_point('p7.prompt','s',0.35)
t = edge_point('p7.llm','n',0.50)
draw_polyline([s,(s[0],t[1]+0.16),(t[0],t[1]+0.16),t])
h('p7.llm','p7.answer')
s = edge_point('p7.answer','s',0.65)
t = edge_point('p7.bridge','n',0.50)
draw_polyline([s,(s[0],t[1]+0.16),(t[0],t[1]+0.16),t])
h('p7.bridge','p7.ui')

# Cross-panel arrows
# 1 -> 2
s = edge_point('p1.norm','e',0.52); t = edge_point('p2.chunk0','w',0.52)
draw_polyline([s,(10.20,s[1]),(10.20,t[1]),t])

# 2 -> 3 routed around panels, not across titles
s = edge_point('p2.b0','s',0.50); t = edge_point('p3.refiner','n',0.20)
draw_polyline([s,(s[0],10.95),(1.15,10.95),(1.15,t[1]),t])

# 3 -> 4 straight vertical
for src, dst in [('p3.leaf','p4.leaf_idx'),('p3.chunks','p4.chunk_idx'),('p3.catalog','p4.anchor_idx')]:
    draw_polyline([edge_point(src,'s',0.50), edge_point(dst,'n',0.50)])

# 4 -> 5 on three separate horizontal channels in the whitespace gap
channels = [9.55, 9.90, 10.25]
for (src, dst), x in zip([('p4.leaf_idx','p5.leaf_ret'),('p4.chunk_idx','p5.chunk_ret'),('p4.anchor_idx','p5.anchor_ret')], channels):
    s = edge_point(src,'e',0.50)
    t = edge_point(dst,'w',0.50)
    draw_polyline([s,(x,s[1]),(x,t[1]),t])

# lookup -> expand on its own dashed lane below the three main retrieval channels
s = edge_point('p4.lookup','e',0.72); t = edge_point('p5.expand','w',0.50)
draw_polyline([s,(9.15,s[1]),(9.15,t[1]),t], dashed=True)

# 5 -> 6: route outside panels, then into p6 from above
s = edge_point('p5.pool','s',0.55); t = edge_point('p6.pairs','n',0.50)
draw_polyline([s,(s[0],4.08),(3.95,4.08),(3.95,t[1]),t])

# 6 -> 7 horizontal between same-row panels
s = edge_point('p6.evidence','e',0.50); t = edge_point('p7.req','w',0.50)
draw_polyline([s,(10.10,s[1]),(10.10,t[1]),t])

plt.tight_layout(pad=0.25)
plt.savefig('figure2_manual_v4.svg', bbox_inches='tight')
plt.savefig('figure2_manual_v4.png', bbox_inches='tight', dpi=180)
plt.savefig('figure2_manual_v4.pdf', bbox_inches='tight')
plt.close()
