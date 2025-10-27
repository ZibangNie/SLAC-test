# -*- coding: utf-8 -*-
# SLAC-Min: Self-Learning Adaptive Chunker (法规PDF分块原型)
# 功能：
#   prepare  从PDF批量抽取文本为TXT
#   train    基于结构先验伪标签训练边界分类器（线性模型/感知机）
#   eval     用伪标签做代理评估（P/R/F1）
#   segment  用训练好的模型把TXT切成 *.chunks.txt
#   selftrain半监督自训练：高置信分歧翻转伪标签，重复数轮
#
# 依赖：numpy（必须），scikit-learn / pdfminer.six / PyPDF2（可选但推荐）
# 用法示例见本文末尾“训练与使用步骤”。

import os, re, io, glob, math, argparse, json
from typing import List, Tuple
import numpy as np

# 可选依赖
USE_SKLEARN = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_recall_fscore_support
except Exception:
    USE_SKLEARN = False
    StandardScaler = None
    precision_recall_fscore_support = None


# -------------------- PDF 抽取 --------------------
def extract_text_from_pdf(path: str) -> str:
    """优先用 pdfminer.six；失败则用 PyPDF2；都没有就报错"""
    # pdfminer.six
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        txt = pdfminer_extract_text(path)
        if txt and len(txt.strip()) > 0:
            return txt
    except Exception:
        pass
    # PyPDF2
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


# -------------------- 文本与正则 --------------------
NUM_PAT = re.compile(r"""^\s*(
    (\d+(\.\d+)*[.)]?)       # 1. 或 1.1. 或 2)
  | ([IVXLCM]+[.)])          # 罗马数字 I. II)
  | (\([a-zA-Z0-9]+\))       # (a) (1)
)""", re.VERBOSE)

APPENDIX_PAT = re.compile(r"^\s*(Appendix|附录)\b", re.IGNORECASE)
CHAPTER_PAT  = re.compile(r"^\s*(Chapter|第一章|第二章|第三章|第[一二三四五六七八九十]+章)\b", re.IGNORECASE)
SECTION_PAT  = re.compile(r"^\s*(Section|条|节)\b", re.IGNORECASE)
BULLET_PAT   = re.compile(r"^\s*([-*•·]|[\(\[]?[a-zA-Z]\)|\d+[.)])\s+")

def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t

def lines_of(text: str) -> List[str]:
    return normalize_text(text.strip("\n")).split("\n")

def is_blank(s: str) -> bool:
    return len(s.strip()) == 0

def indent_level(s: str) -> int:
    return len(s) - len(s.lstrip(" "))

def ratio_upper(s: str) -> float:
    letters = [ch for ch in s if ch.isalpha()]
    if not letters: return 0.0
    return sum(1 for ch in letters if ch.isupper()) / len(letters)

def ratio_digit(s: str) -> float:
    digits = sum(ch.isdigit() for ch in s)
    return digits / max(1, len(s))

def ratio_punct(s: str) -> float:
    puncts = sum(ch in ",.;:!?，。；：？！[]" for ch in s)
    return puncts / max(1, len(s))

def is_heading_like(s: str) -> bool:
    s_clean = s.strip()
    return (
        bool(NUM_PAT.match(s_clean))
        or bool(APPENDIX_PAT.match(s_clean))
        or bool(CHAPTER_PAT.match(s_clean))
        or bool(SECTION_PAT.match(s_clean))
        or (s_clean.isupper() and 1 < len(s_clean) < 80)
    )

def is_bullet_like(s: str) -> bool:
    return bool(BULLET_PAT.match(s))


# -------------------- 特征工程 --------------------
def feature_for_line(s: str) -> np.ndarray:
    s = s.rstrip()
    feats = [
        len(s),
        indent_level(s),
        ratio_upper(s),
        ratio_digit(s),
        ratio_punct(s),
        1.0 if is_heading_like(s) else 0.0,
        1.0 if is_bullet_like(s) else 0.0,
        1.0 if s.endswith(":") else 0.0,
        1.0 if s.endswith("：") else 0.0,
        1.0 if is_blank(s) else 0.0,
    ]
    return np.array(feats, dtype=float)

def boundary_features(lines: List[str], i: int, k_ctx:int=2) -> np.ndarray:
    """构造“行间隙 i”的特征：前后各k行 + 当前两行 + 差分"""
    def get_line(idx):
        if 0 <= idx < len(lines): return lines[idx]
        return ""
    window_prev = [get_line(i - j) for j in range(k_ctx, 0, -1)]
    window_next = [get_line(i + j) for j in range(1, k_ctx + 1)]
    fprev = [feature_for_line(s) for s in window_prev]
    fnext = [feature_for_line(s) for s in window_next]
    fprev = np.concatenate(fprev) if len(fprev) else np.zeros(10*k_ctx)
    fnext = np.concatenate(fnext) if len(fnext) else np.zeros(10*k_ctx)
    cur = [feature_for_line(get_line(i)), feature_for_line(get_line(i+1))]
    fcur = np.concatenate(cur)
    delta_len = cur[1][0] - cur[0][0]
    delta_indent = cur[1][1] - cur[0][1]
    delta_upper = cur[1][2] - cur[0][2]
    pair = np.array([delta_len, delta_indent, delta_upper])
    return np.concatenate([fprev, fcur, fnext, pair])


# -------------------- 伪标签（结构先验） --------------------
def pseudo_label(lines: List[str], i: int) -> int:
    """
    规则：若满足其一则在 i 处切：
      - 当前行像标题/编号；
      - 下一行是空行；
      - 当前行以 :/： 结尾，且下一行像列表/编号；
      - 缩进显著回退（强烈结构提示）。
    """
    cur = lines[i] if i < len(lines) else ""
    nxt = lines[i+1] if i+1 < len(lines) else ""
    if is_heading_like(cur): return 1
    if is_blank(nxt): return 1
    if (cur.endswith(":") or cur.endswith("：")) and (is_bullet_like(nxt) or NUM_PAT.match(nxt.strip())):
        return 1
    if indent_level(nxt) < indent_level(cur) - 2 and len(nxt.strip())>0:
        return 1
    return 0


# -------------------- 模型（线性/感知机） --------------------
class Perceptron1:
    """无sklearn时的极简可训练分类器"""
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


# -------------------- 数据集构建 --------------------
def build_dataset_from_txts(txt_paths: List[str], k_ctx:int=2):
    X, y, metas = [], [], []
    for path in txt_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        lines = lines_of(txt)
        for i in range(len(lines)-1):
            X.append(boundary_features(lines, i, k_ctx=k_ctx))
            y.append(pseudo_label(lines, i))
            metas.append((path, i))
    return np.vstack(X), np.array(y, dtype=int), metas


# -------------------- 分块推理 --------------------
def segment_lines_with_probs(lines: List[str], pb: np.ndarray,
                             threshold=0.55, min_lines=2, max_lines=15):
    blocks = []
    start = 0
    i = 0
    while i < len(lines)-1:
        j = max(i, start + min_lines - 1)
        if (i - start + 1) >= max_lines:
            blocks.append((start, i+1))
            start = i+1
            i = start
            continue
        if j == i and pb[i] >= threshold:
            blocks.append((start, i+1))
            start = i+1
        i += 1
    if start < len(lines):
        blocks.append((start, len(lines)))
    return blocks

def segment_file(txt_path: str, model: dict, out_dir: str,
                 threshold=0.55, min_lines=2, max_lines=15):
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    lines = lines_of(txt)
    if len(lines) <= 1:
        spans = [(0, len(lines))]; pb = np.array([])
    else:
        Xb = np.vstack([boundary_features(lines, i) for i in range(len(lines)-1)])
        pb = predict_proba(model, Xb)[:,1]
        spans = segment_lines_with_probs(lines, pb, threshold=threshold,
                                         min_lines=min_lines, max_lines=max_lines)
    base = os.path.basename(txt_path); stem = os.path.splitext(base)[0]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}.chunks.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for (a,b) in spans:
            chunk = "\n".join(lines[a:b]).strip("\n")
            f.write(chunk + "\n" + "-"*60 + "\n")
    return out_path, spans


# -------------------- 评估与自训练 --------------------
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

def self_training_refine_labels(X, y_pseudo, proba, hi=0.9, lo=0.1):
    """高置信分歧翻转伪标签"""
    y_new = y_pseudo.copy()
    confident_pos = (proba>=hi); confident_neg = (proba<=lo)
    y_new[confident_pos] = 1; y_new[confident_neg] = 0
    changed = int((y_new!=y_pseudo).sum())
    return y_new, changed


# -------------------- 子命令 --------------------
def cmd_prepare(args):
    pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    os.makedirs(args.out_txt_dir, exist_ok=True)
    for p in pdfs:
        try:
            txt = extract_text_from_pdf(p)
            txt = normalize_text(txt)
            base = os.path.basename(p); stem = os.path.splitext(base)[0]
            out = os.path.join(args.out_txt_dir, stem + ".txt")
            with open(out, "w", encoding="utf-8") as f:
                f.write(txt)
            print(f"[OK] {p} -> {out}")
        except Exception as e:
            print(f"[FAIL] {p}: {e}")

def cmd_train(args):
    paths = sorted(glob.glob(args.txt_glob))
    assert paths, f"No txt files matched: {args.txt_glob}"
    X, y, _ = build_dataset_from_txts(paths, k_ctx=args.k_ctx)
    model = fit_model(X, y)
    proba = predict_proba(model, X)[:,1]
    y_pred = (proba >= args.thresh_eval).astype(int)
    p, r, f1 = boundary_metrics(y, y_pred)
    print(f"Train proxy (vs pseudo): P={p:.3f} R={r:.3f} F1={f1:.3f}")
    np.savez(args.model_out, model=json.dumps(model))
    print(f"Model saved to: {args.model_out}")

def cmd_eval(args):
    paths = sorted(glob.glob(args.txt_glob))
    assert paths, f"No txt files matched: {args.txt_glob}"
    data = np.load(args.model, allow_pickle=True)
    model = json.loads(str(data["model"]))
    X, y, _ = build_dataset_from_txts(paths, k_ctx=args.k_ctx)
    proba = predict_proba(model, X)[:,1]
    y_pred = (proba >= args.thresh_eval).astype(int)
    p, r, f1 = boundary_metrics(y, y_pred)
    print(f"Eval proxy (vs pseudo): P={p:.3f} R={r:.3f} F1={f1:.3f}")

def cmd_segment(args):
    paths = sorted(glob.glob(args.txt_glob))
    assert paths, f"No txt files matched: {args.txt_glob}"
    data = np.load(args.model, allow_pickle=True)
    model = json.loads(str(data["model"]))
    os.makedirs(args.out_dir, exist_ok=True)
    for pth in paths:
        out_path, spans = segment_file(
            pth, model, out_dir=args.out_dir,
            threshold=args.threshold, min_lines=args.min_lines, max_lines=args.max_lines
        )
        print(f"[SEG] {pth} -> {out_path} ({len(spans)} chunks)")

def cmd_selftrain(args):
    paths = sorted(glob.glob(args.txt_glob))
    assert paths, f"No txt files matched: {args.txt_glob}"
    X, y_pseudo, _ = build_dataset_from_txts(paths, k_ctx=args.k_ctx)
    model = fit_model(X, y_pseudo)
    for it in range(args.iters):
        proba = predict_proba(model, X)[:,1]
        y_new, changed = self_training_refine_labels(X, y_pseudo, proba, hi=args.hi, lo=args.lo)
        print(f"[Iter {it}] relabeled = {changed}")
        if changed == 0:
            break
        model = fit_model(X, y_new)
        y_pseudo = y_new
    np.savez(args.model_out, model=json.dumps(model))
    y_pred = (predict_proba(model, X)[:,1] >= args.thresh_eval).astype(int)
    p, r, f1 = boundary_metrics(y_pseudo, y_pred)
    print(f"Self-train proxy (vs refined): P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print(f"Refined model saved to: {args.model_out}")


def build_argparser():
    ap = argparse.ArgumentParser(description="SLAC-Min pipeline")
    sub = ap.add_subparsers()

    ap_prep = sub.add_parser("prepare", help="Extract text from PDFs into .txt files")
    ap_prep.add_argument("--pdf_dir", required=True, help="dir containing PDFs")
    ap_prep.add_argument("--out_txt_dir", required=True, help="output dir for .txt files")
    ap_prep.set_defaults(func=cmd_prepare)

    ap_train = sub.add_parser("train", help="Train boundary classifier on pseudo labels")
    ap_train.add_argument("--txt_glob", required=True, help='glob like "./txts/*.txt"')
    ap_train.add_argument("--model_out", required=True, help="path to save model npz")
    ap_train.add_argument("--k_ctx", type=int, default=2, help="context lines on each side")
    ap_train.add_argument("--thresh_eval", type=float, default=0.5, help="eval threshold for boundary classification")
    ap_train.set_defaults(func=cmd_train)

    ap_eval = sub.add_parser("eval", help="Evaluate (proxy) vs. pseudo labels")
    ap_eval.add_argument("--txt_glob", required=True)
    ap_eval.add_argument("--model", required=True, help="path to saved model npz")
    ap_eval.add_argument("--k_ctx", type=int, default=2)
    ap_eval.add_argument("--thresh_eval", type=float, default=0.5)
    ap_eval.set_defaults(func=cmd_eval)

    ap_seg = sub.add_parser("segment", help="Segment .txt files into chunks")
    ap_seg.add_argument("--txt_glob", required=True)
    ap_seg.add_argument("--model", required=True)
    ap_seg.add_argument("--out_dir", required=True)
    ap_seg.add_argument("--threshold", type=float, default=0.55)
    ap_seg.add_argument("--min_lines", type=int, default=2)
    ap_seg.add_argument("--max_lines", type=int, default=15)
    ap_seg.set_defaults(func=cmd_segment)

    ap_st = sub.add_parser("selftrain", help="Self-training refine labels")
    ap_st.add_argument("--txt_glob", required=True)
    ap_st.add_argument("--model_in", help="optional existing model (ignored)")
    ap_st.add_argument("--model_out", required=True)
    ap_st.add_argument("--k_ctx", type=int, default=2)
    ap_st.add_argument("--iters", type=int, default=3)
    ap_st.add_argument("--hi", type=float, default=0.9)
    ap_st.add_argument("--lo", type=float, default=0.1)
    ap_st.add_argument("--thresh_eval", type=float, default=0.5)
    ap_st.set_defaults(func=cmd_selftrain)

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
