# -*- coding: utf-8 -*-
# SLAC-NumberChunker: 编号级弱监督 + 模型学习 + DP自适应分块 + PDF可视化
#
# 子命令：
#   prepare      : PDF 批量抽取文本为 .txt
#   train_num    : 仅用“编号行(2.1.1等)”做弱标签训练边界打分器（编号级分块）
#   eval_num     : 用“编号弱标”做代理评估（与弱标一致度）
#   segment_dp   : 不用任何显式规则，基于模型概率 + DP 进行自适应分块（无固定粒度）
#                  同时可输出 *.boundaries.tsv 以及 PDF 可视化
#   selftrain    : （可选）半监督自训练：用模型高置信修正弱标再训
#
# 依赖：numpy（必须）
#      可选：scikit-learn（更强分类器）、pdfminer.six/PyPDF2（prepare）、reportlab（PDF可视化）
#
# 训练/推理解读：训练阶段用“编号行”自动打弱标签；测试/推理阶段不再使用任何编号/规则，
# 仅依赖模型学到的“边界概率”以及 DP 代价最小化来自适应决定切分位置与粒度。

import os, re, glob, math, argparse, json
from typing import List, Tuple
import numpy as np

# =============== 可选依赖 ===============
USE_SKLEARN = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_recall_fscore_support
except Exception:
    USE_SKLEARN = False
    StandardScaler = None
    precision_recall_fscore_support = None

# PDF 文本抽取（prepare 用）
def extract_text_from_pdf(path: str) -> str:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        txt = pdfminer_extract_text(path)
        if txt and len(txt.strip()) > 0:
            return txt
    except Exception:
        pass
    try:
        import PyPDF2
        out = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                out.append(t)
        txt = "\n".join(out)
        if txt and len(txt.strip()) > 0:
            return txt
    except Exception:
        pass
    raise RuntimeError(
        "无法从 PDF 中提取文本：未检测到 pdfminer.six 或 PyPDF2。"
        " 请先 `pip install pdfminer.six`（推荐）或 `pip install PyPDF2`。"
    )

# =============== 文本预处理 ===============
def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t

def lines_of(text: str) -> List[str]:
    return normalize_text(text.strip("\n")).split("\n")

def indent_level(s: str) -> int:
    return len(s) - len(s.lstrip(" "))

def ratio_digit(s: str) -> float:
    digits = sum(ch.isdigit() for ch in s)
    return digits / max(1, len(s))

def ratio_punct(s: str) -> float:
    puncts = sum(ch in ".,;:!?，。；：？！[]" for ch in s)
    return puncts / max(1, len(s))

def starts_with_digit(s: str) -> float:
    s2 = s.lstrip()
    return 1.0 if (len(s2)>0 and s2[0].isdigit()) else 0.0

def dot_count_prefix(s: str, k:int=8) -> float:
    """前缀(默认8字符)里的 '.' 个数，帮助区分 2.1.1/3.2.4 这类编号，但不直接用规则分块。"""
    s2 = s.lstrip()[:k]
    return float(s2.count("."))

# =============== 特征工程（无“显式编号/标题规则”泄漏） ===============
def feature_for_line(s: str) -> np.ndarray:
    # 注意：测试时我们不再使用“显式规则”，这里只提取一般统计特征
    s = s.rstrip()
    feats = [
        len(s),                   # 行长
        indent_level(s),          # 左缩进
        ratio_digit(s),           # 数字占比（编号行通常较高）
        ratio_punct(s),           # 标点占比（编号行前缀常有 .）
        starts_with_digit(s),     # 是否以数字开头（作为可学习特征，而非规则）
        dot_count_prefix(s, 8),   # 前缀中 '.' 个数
    ]
    return np.array(feats, dtype=float)

def boundary_features(lines: List[str], i: int, k_ctx:int=2) -> np.ndarray:
    """为“行间隙 i”构造特征：前后各 k 行 + 当前两行 + 差分"""
    def get_line(idx):
        if 0 <= idx < len(lines): return lines[idx]
        return ""
    prev_ctx = [get_line(i - j) for j in range(k_ctx, 0, -1)]
    next_ctx = [get_line(i + j) for j in range(1, k_ctx + 1)]
    fprev = [feature_for_line(s) for s in prev_ctx]
    fnext = [feature_for_line(s) for s in next_ctx]
    fprev = np.concatenate(fprev) if fprev else np.zeros(6*k_ctx)
    fnext = np.concatenate(fnext) if fnext else np.zeros(6*k_ctx)
    cur = [feature_for_line(get_line(i)), feature_for_line(get_line(i+1))]
    fcur = np.concatenate(cur)
    # 差分信号（下行-上行），帮助检测“样式切换”
    delta = cur[1] - cur[0]
    return np.concatenate([fprev, fcur, fnext, delta])

# =============== 训练用编号弱标签（仅用于训练，不在测试使用） ===============
# 只把“下一行是 2.1.1 这类编号行”的位置作为切点（唯一标准）
DOT_NUM_PAT = re.compile(r"^\s*\d+(?:\.\d+){1,}\b")  # 2.1 / 2.1.1 / 3.2.4.5 等

def pseudo_label_number_only(lines: List[str], i: int) -> int:
    """若第 i+1 行为“点分式编号”则在 i 处切；否则不切。"""
    if i+1 >= len(lines):
        return 0
    nxt = lines[i+1]
    return 1 if DOT_NUM_PAT.match(nxt) else 0

# =============== 简易模型（LogReg / 感知机） ===============
class Perceptron1:
    def __init__(self, dim, lr=0.05, epochs=10):
        self.w = np.zeros(dim)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
    def fit(self, X, y):
        rng = np.random.default_rng(42)
        for _ in range(self.epochs):
            idx = rng.permutation(len(X))
            for i in idx:
                z = X[i].dot(self.w) + self.b
                p = 1.0/(1.0+math.exp(-z))
                g = p - y[i]
                self.w -= self.lr * g * X[i]
                self.b -= self.lr * g
    def predict_proba(self, X):
        z = X.dot(self.w) + self.b
        p = 1.0/(1.0+np.exp(-z))
        return np.vstack([1-p, p]).T

def fit_model(X, y):
    if USE_SKLEARN:
        scaler = StandardScaler() if StandardScaler is not None else None
        Xs = scaler.fit_transform(X) if scaler is not None else X
        clf = LogisticRegression(max_iter=500)
        clf.fit(Xs, y)
        return {"type":"sklogreg",
                "scaler_mean": (scaler.mean_.tolist() if scaler else None),
                "scaler_scale": (scaler.scale_.tolist() if scaler else None),
                "coef": clf.coef_.tolist(),
                "intercept": clf.intercept_.tolist()}
    else:
        clf = Perceptron1(X.shape[1], lr=0.03, epochs=8)
        clf.fit(X, y)
        return {"type":"perceptron", "w": clf.w.tolist(), "b": clf.b}

def predict_proba(model, X):
    if model["type"]=="sklogreg":
        coef = np.array(model["coef"])
        intercept = np.array(model["intercept"])
        if model["scaler_mean"] is not None:
            mean = np.array(model["scaler_mean"])
            scale = np.array(model["scaler_scale"])
            Xs = (X - mean)/scale
        else:
            Xs = X
        z = Xs.dot(coef.T) + intercept
        p = 1.0/(1.0+np.exp(-z))
        return np.hstack([1-p, p])
    else:
        w = np.array(model["w"]); b = model["b"]
        z = X.dot(w) + b
        p = 1.0/(1.0+np.exp(-z))
        return np.vstack([1-p, p]).T

# =============== 数据构建（编号弱标） ===============
def build_dataset_number_only(txt_paths: List[str], k_ctx:int=2):
    X, y, metas = [], [], []
    for path in txt_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        lines = lines_of(txt)
        for i in range(len(lines)-1):
            X.append(boundary_features(lines, i, k_ctx=k_ctx))
            y.append(pseudo_label_number_only(lines, i))
            metas.append((path, i))
    return np.vstack(X), np.array(y, dtype=int), metas

# =============== DP 自适应分块（无固定粒度） ===============
def _len_penalty(L, len_target=12.0, len_weight=0.2):
    # 柔性长度正则：接近 len_target 成本低，偏离越多越贵，但不“禁止”
    r = L / max(1.0, len_target)
    return float(len_weight) * (r - 1.0)**2 * L

def _dp_optimal_spans(pb, len_target=12.0, len_weight=0.2, cut_cost=0.3):
    """
    pb: [N-1]，第 i 个是“在 i 与 i+1 之间切”的概率（来自模型）
    目标：最小化 sum(边界代价 + 长度代价)
      边界代价 = -log(pb) + cut_cost
      长度代价 = _len_penalty(L)
    返回：[(start, end), ...] 半开区间
    """
    import math
    N = len(pb) + 1
    bc = [-math.log(max(1e-6, float(p))) + float(cut_cost) for p in pb] + [0.0]  # 末尾闭合成本0
    INF = 1e18
    dp  = [INF] * (N + 1)
    prv = [-1]  * (N + 1)
    dp[0] = 0.0
    for j in range(1, N + 1):
        best, arg = INF, -1
        for i in range(0, j):
            L = j - i
            cost = dp[i] + bc[j - 1] + _len_penalty(L, len_target, len_weight)
            if cost < best:
                best, arg = cost, i
        dp[j]  = best
        prv[j] = arg
    spans = []
    j = N
    while j > 0:
        i = prv[j]
        spans.append((i, j))
        j = i
    spans.reverse()
    return spans

# =============== PDF 可视化 ===============
def render_pdf_with_cuts(pdf_path: str, lines: List[str], spans: List[Tuple[int,int]],
                         pb: np.ndarray, title: str,
                         font_path: str = "", font_name: str = "Helvetica", font_size: int = 11):
    """
    生成“审阅用”PDF（不复原原版面），标注 CUT 与概率。
    - font_path: 指向 .ttf/.otf/.ttc（中文字体），如 NotoSansCJKsc-Regular.otf / simsun.ttc / msyh.ttc
    - font_name: 注册后的字体名（默认=文件名去扩展名）
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception as e:
        raise RuntimeError("需要 reportlab 以输出 PDF：pip install reportlab") from e

    # 注册中文字体（若提供）
    if font_path:
        try:
            base_name = font_name or os.path.splitext(os.path.basename(font_path))[0]
            pdfmetrics.registerFont(TTFont(base_name, font_path))
            font_name = base_name
        except Exception as e:
            # 注册失败则退回 Helvetica
            print(f"[VIZ] 字体注册失败，退回 Helvetica：{e}")
            font_name = "Helvetica"

    page_w, page_h = A4
    c = canvas.Canvas(pdf_path, pagesize=A4)
    left = 20*mm; right = page_w - 20*mm
    top = page_h - 20*mm; bottom = 20*mm
    line_h = font_size + 1

    def wrap_by_width(txt: str, max_width: float) -> List[str]:
        # 逐字符测量宽度（CJK无空格场景也OK）
        segs, buf = [], ""
        for ch in txt.replace("\t", "    "):
            w = c.stringWidth(buf + ch, font_name, font_size)
            if w <= max_width:
                buf += ch
            else:
                if buf:
                    segs.append(buf)
                buf = ch
        if buf:
            segs.append(buf)
        return segs or [""]

    # 收集切点索引：在 (b-1, b) 的缝处
    cut_set = set(b-1 for (a,b) in spans if b-1 >= 0)

    c.setTitle(title)
    c.setFont(font_name, font_size)
    y = top
    c.drawString(left, y, title); y -= (line_h*1.5)

    for j, raw in enumerate(lines):
        # 若在 (j-1, j) 处切，则打标签
        if j-1 in cut_set and j-1 < len(pb):
            p = float(pb[j-1])
            tag = f"CUT p={p:.2f}"
            c.setFillColor(colors.red if p>=0.5 else colors.orange)
            c.drawString(left, y, tag); y -= line_h*0.9
            c.setFillColor(colors.black)

        # 按像素宽度换行
        for seg in wrap_by_width(raw, right - left):
            if y < bottom:
                c.showPage(); y = top; c.setFont(font_name, font_size)
            c.drawString(left, y, seg); y -= line_h

        if y < bottom:
            c.showPage(); y = top; c.setFont(font_name, font_size)

    c.save()

# =============== 评估指标（代理：与弱标一致度） ===============
def boundary_metrics(y_true, y_pred):
    if precision_recall_fscore_support is None:
        tp = int(((y_true==1)&(y_pred==1)).sum())
        fp = int(((y_true==0)&(y_pred==1)).sum())
        fn = int(((y_true==1)&(y_pred==0)).sum())
        prec = tp / max(1, tp+fp); rec  = tp / max(1, tp+fn)
        f1   = 0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        return prec, rec, f1
    else:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        return float(p), float(r), float(f1)

# =============== 子命令实现 ===============
def cmd_prepare(args):
    pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    os.makedirs(args.out_txt_dir, exist_ok=True)
    for p in pdfs:
        try:
            txt = extract_text_from_pdf(p)
            txt = normalize_text(txt)
            stem = os.path.splitext(os.path.basename(p))[0]
            out = os.path.join(args.out_txt_dir, stem + ".txt")
            with open(out, "w", encoding="utf-8") as f:
                f.write(txt)
            print(f"[OK] {p} -> {out}")
        except Exception as e:
            print(f"[FAIL] {p}: {e}")

def cmd_train_num(args):
    paths = sorted(glob.glob(args.txt_glob))
    assert paths, f"No txt files matched: {args.txt_glob}"
    X, y, _ = build_dataset_number_only(paths, k_ctx=args.k_ctx)
    model = fit_model(X, y)
    # 训练阶段只做代理一致度查看（与弱标），不代表真实上限
    proba = predict_proba(model, X)[:,1]
    y_pred = (proba >= args.thresh_eval).astype(int)
    p, r, f1 = boundary_metrics(y, y_pred)
    print(f"Train(proxy vs number-only weak labels): P={p:.3f} R={r:.3f} F1={f1:.3f}")
    np.savez(args.model_out, model=json.dumps(model))
    print(f"Model saved to: {args.model_out}")

def cmd_eval_num(args):
    paths = sorted(glob.glob(args.txt_glob))
    assert paths, f"No txt files matched: {args.txt_glob}"
    data = np.load(args.model, allow_pickle=True)
    model = json.loads(str(data["model"]))
    X, y, _ = build_dataset_number_only(paths, k_ctx=args.k_ctx)
    proba = predict_proba(model, X)[:,1]
    y_pred = (proba >= args.thresh_eval).astype(int)
    p, r, f1 = boundary_metrics(y, y_pred)
    print(f"Eval(proxy vs number-only weak labels): P={p:.3f} R={r:.3f} F1={f1:.3f}")

def cmd_segment_dp(args):
    paths = sorted(glob.glob(args.txt_glob))
    assert paths, f"No txt files matched: {args.txt_glob}"
    data = np.load(args.model, allow_pickle=True)
    model = json.loads(str(data["model"]))

    os.makedirs(args.out_dir, exist_ok=True)
    if args.viz_pdf_dir:
        os.makedirs(args.viz_pdf_dir, exist_ok=True)

    for pth in paths:
        with open(pth, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        ls = lines_of(txt)
        if len(ls) <= 1:
            spans = [(0, len(ls))]; pb = np.array([])
        else:
            Xb = np.vstack([boundary_features(ls, i, k_ctx=args.k_ctx) for i in range(len(ls)-1)])
            pb = predict_proba(model, Xb)[:,1]
            spans = _dp_optimal_spans(pb, len_target=args.len_target,
                                      len_weight=args.len_weight, cut_cost=args.cut_cost)

        stem = os.path.splitext(os.path.basename(pth))[0]

        # 写 chunks
        out_path = os.path.join(args.out_dir, f"{stem}.chunks.txt")
        with open(out_path, "w", encoding="utf-8") as wf:
            for (a,b) in spans:
                wf.write("\n".join(ls[a:b]).strip("\n") + "\n" + "-"*60 + "\n")
        print(f"[DP-SEG] {pth} -> {out_path} ({len(spans)} chunks)")

        # 写边界概率/是否选中
        if args.save_probs:
            tsv_path = os.path.join(args.out_dir, f"{stem}.boundaries.tsv")
            with open(tsv_path, "w", encoding="utf-8") as tf:
                tf.write("gap_index\tprob\tcut\n")
                cut_idx = set(b-1 for (a,b) in spans if b-1 >= 0)
                for i in range(len(ls)-1):
                    prob = float(pb[i]) if len(ls)>1 else 0.0
                    cut  = 1 if i in cut_idx else 0
                    tf.write(f"{i}\t{prob:.6f}\t{cut}\n")

        # PDF 可视化
        if args.viz_pdf_dir:
            pdf_path = os.path.join(args.viz_pdf_dir, f"{stem}.viz.pdf")
            try:
                render_pdf_with_cuts(
                    pdf_path, ls, spans, pb,
                    title=f"{stem} (DP segmentation)",
                    font_path=args.viz_font_path, font_name=args.viz_font_name, font_size=args.viz_font_size
                )
                print(f"[VIZ] {pdf_path}")
            except Exception as e:
                print(f"[VIZ-FAIL] {stem}: {e}")

def cmd_selftrain(args):
    # 用“编号弱标”作为起点做自训练（可选）
    paths = sorted(glob.glob(args.txt_glob))
    assert paths, f"No txt files matched: {args.txt_glob}"
    X, y_pseudo, _ = build_dataset_number_only(paths, k_ctx=args.k_ctx)
    model = fit_model(X, y_pseudo)
    for it in range(args.iters):
        proba = predict_proba(model, X)[:,1]
        # 仅对高置信进行翻转/固化
        y_new = y_pseudo.copy()
        y_new[proba>=args.hi] = 1
        y_new[proba<=args.lo] = 0
        changed = int((y_new!=y_pseudo).sum())
        print(f"[Iter {it}] relabeled={changed}")
        if changed == 0:
            break
        model = fit_model(X, y_new)
        y_pseudo = y_new
    np.savez(args.model_out, model=json.dumps(model))
    print(f"Refined model saved to: {args.model_out}")

# =============== CLI 构建 ===============
def build_argparser():
    ap = argparse.ArgumentParser(description="SLAC-NumberChunker pipeline")
    sub = ap.add_subparsers()

    ap_prep = sub.add_parser("prepare", help="Extract text from PDFs into .txt files")
    ap_prep.add_argument("--pdf_dir", required=True)
    ap_prep.add_argument("--out_txt_dir", required=True)
    ap_prep.set_defaults(func=cmd_prepare)

    ap_train = sub.add_parser("train_num", help="Train boundary classifier using ONLY dotted-number weak labels")
    ap_train.add_argument("--txt_glob", required=True)
    ap_train.add_argument("--model_out", required=True)
    ap_train.add_argument("--k_ctx", type=int, default=2)
    ap_train.add_argument("--thresh_eval", type=float, default=0.5)
    ap_train.set_defaults(func=cmd_train_num)

    ap_eval = sub.add_parser("eval_num", help="Eval proxy vs dotted-number weak labels")
    ap_eval.add_argument("--txt_glob", required=True)
    ap_eval.add_argument("--model", required=True)
    ap_eval.add_argument("--k_ctx", type=int, default=2)
    ap_eval.add_argument("--thresh_eval", type=float, default=0.5)
    ap_eval.set_defaults(func=cmd_eval_num)

    ap_seg = sub.add_parser("segment_dp", help="DP-based adaptive segmentation (no explicit rules at test)")
    ap_seg.add_argument("--txt_glob", required=True)
    ap_seg.add_argument("--model", required=True)
    ap_seg.add_argument("--out_dir", required=True)
    ap_seg.add_argument("--k_ctx", type=int, default=2)
    ap_seg.add_argument("--len_target", type=float, default=12.0, help="软目标块长（不是硬约束）")
    ap_seg.add_argument("--len_weight", type=float, default=0.2, help="块长偏离的惩罚强度")
    ap_seg.add_argument("--cut_cost", type=float, default=0.3, help="每次切割的固定代价（大→更少切分）")
    ap_seg.add_argument("--save_probs", action="store_true", help="输出 *.boundaries.tsv（概率+是否切）")
    ap_seg.add_argument("--viz_pdf_dir", default="", help="若提供目录，则生成可视化 PDF")
    ap_seg.set_defaults(func=cmd_segment_dp)

    ap_st = sub.add_parser("selftrain", help="(optional) self-training starting from dotted-number weak labels")
    ap_st.add_argument("--txt_glob", required=True)
    ap_st.add_argument("--model_out", required=True)
    ap_st.add_argument("--k_ctx", type=int, default=2)
    ap_st.add_argument("--iters", type=int, default=2)
    ap_st.add_argument("--hi", type=float, default=0.9)
    ap_st.add_argument("--lo", type=float, default=0.1)
    ap_st.set_defaults(func=cmd_selftrain)

    ap_seg.add_argument("--viz_font_path", default="", help="PDF可视化使用的中文字体文件路径(.ttf/.otf/.ttc)")
    ap_seg.add_argument("--viz_font_name", default="", help="PDF字体名（不填自动用文件名）")
    ap_seg.add_argument("--viz_font_size", type=int, default=11, help="PDF字号")

    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
