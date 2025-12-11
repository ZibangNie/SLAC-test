import re
from pathlib import Path

import pdfplumber  # 需要: pip install pdfplumber


# ======== 一些可调参数 ========

MAX_PAGES = 8          # 每个 PDF 最多检查前几页，避免太慢
MIN_TEXT_LEN = 1000    # 至少要抽到这么多字符，才认为有分析意义
MIN_CHINESE_RATIO = 0.15  # 中文字符占比下限
MAX_WEIRD_RATIO = 0.5     # “异常字符”（非中英文、数字、常见标点）占比上限

# 用来定义“正常字符”的集合（除了中文、本身 isascii 的字母数字）
COMMON_PUNCT = set(" 。，、：；！？,.!?;:()（）[]【】<>《》…—-+/*=&%~“”‘’\"'")


def extract_text_from_pdf(pdf_path: Path, max_pages: int = MAX_PAGES) -> str:
    """抽取 PDF 前 max_pages 页文本，不会抛异常，失败时返回空字符串。"""
    try:
        text_parts = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"[ERROR] 打开或解析失败: {pdf_path.name} -> {e}")
        return ""


def char_stats(text: str):
    """统计中文、ASCII 字母数字、常见标点、异常字符数量。"""
    total = 0
    chinese = 0
    ascii_alnum = 0
    punct = 0
    weird = 0

    for ch in text:
        if ch.isspace():
            continue
        total += 1

        # 中文
        if "\u4e00" <= ch <= "\u9fff":
            chinese += 1
        # ASCII 字母数字
        elif ch.isascii() and (ch.isalpha() or ch.isdigit()):
            ascii_alnum += 1
        # 常见标点
        elif ch in COMMON_PUNCT:
            punct += 1
        else:
            weird += 1

    return {
        "total": total,
        "chinese": chinese,
        "ascii_alnum": ascii_alnum,
        "punct": punct,
        "weird": weird,
    }


def has_standard_headings(text: str) -> bool:
    """
    粗略检测是否有类似国标的结构编号：
    - '1 范围'
    - '2 规范性引用文件'
    - '3.1 总则'
    - '5.1.2 某某要求'
    """
    patterns = [
        r"\n\s*1\s+范围",
        r"\n\s*2\s+规范性引用文件",
        r"\n\s*\d+\s+[^\n]{2,20}",          # '3 术语和定义'
        r"\n\s*\d+\.\d+\s+[^\n]{2,30}",     # '5.1 总体要求'
        r"\n\s*\d+\.\d+\.\d+\s+[^\n]{2,30}" # '5.1.1 具体要求'
    ]

    hits = 0
    for pat in patterns:
        if re.search(pat, text):
            hits += 1
    return hits >= 2  # 至少命中两个模式就认为有明显结构


def is_structurable(text: str) -> (bool, str):
    """
    根据简单统计 + 结构特征，判断是否“可结构化”。
    返回 (是否可结构化, 诊断说明)。
    """
    if not text or len(text) < MIN_TEXT_LEN:
        return False, f"文本过短（len={len(text)}），可能是扫描版或解析失败"

    stats = char_stats(text)
    total = stats["total"] or 1  # 防止除零
    chinese_ratio = stats["chinese"] / total
    weird_ratio = stats["weird"] / total

    # 结构线索（是否检测到常见章节编号）
    structure_flag = has_standard_headings(text)

    reasons = []

    if chinese_ratio < MIN_CHINESE_RATIO:
        reasons.append(f"中文占比过低={chinese_ratio:.2f}")
    if weird_ratio > MAX_WEIRD_RATIO:
        reasons.append(f"异常字符占比偏高={weird_ratio:.2f}")
    if not structure_flag:
        reasons.append("未检测到明显国标章节编号模式")

    if reasons:
        diag = "；".join(reasons)
        return False, diag

    diag = (
        f"文本长度={len(text)}，中文占比={chinese_ratio:.2f}，"
        f"异常字符占比={weird_ratio:.2f}，检测到国标章节编号"
    )
    return True, diag


def main():
    pdf_files = sorted(Path(".").glob("*.pdf"))

    if not pdf_files:
        print("当前目录下没有找到 PDF 文件。")
        return

    ok_files = []
    bad_files = []

    print(f"在当前目录发现 {len(pdf_files)} 个 PDF，开始检测……\n")

    for pdf in pdf_files:
        print(f"=== 检查: {pdf.name} ===")
        text = extract_text_from_pdf(pdf, max_pages=MAX_PAGES)
        ok, diag = is_structurable(text)
        if ok:
            print(f"[OK]   认为可结构化：{diag}")
            ok_files.append(pdf.name)
        else:
            print(f"[WARN] 可能不适合直接结构化：{diag}")
            bad_files.append(pdf.name)
        print()

    print("======== 汇总结果 ========")
    print(f"可结构化（OK）：{len(ok_files)} 个")
    for name in ok_files:
        print(f"  - {name}")
    print(f"\n疑似有问题（需要人工检查或 OCR）：{len(bad_files)} 个")
    for name in bad_files:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
