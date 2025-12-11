"""
grobid_to_tei.py

批量调用本地 GROBID，将指定目录下的 PDF 论文转换为 TEI-XML。

假设：
- PDF 路径类似：D:\code\Github\SLAC-test\data\A_structure\papers\2512.01623.pdf
- TEI 输出目录：D:\code\Github\SLAC-test\data\A_structure\papers_tei\
- 本地 GROBID 服务已启动，地址为 http://localhost:8070/api/processFulltextDocument

依赖：
    pip install requests
"""

import os
import time
from pathlib import Path

import requests

# ======= 路径配置，根据你当前情况调整 =======

# 输入 PDF 目录
INPUT_DIR = Path(r"/data/A_structure/papers(deprecated)")

# TEI 输出目录
OUTPUT_DIR = Path(r"/data/A_structure/papers_tei(deprecated)")

# GROBID fulltext 接口
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

# 请求超时和间隔
REQUEST_TIMEOUT = 300      # 单个 PDF 处理最长等待秒数
SLEEP_BETWEEN = 2.0        # 两个 PDF 之间的间隔，给 GROBID 缓一缓


def process_pdf(pdf_path: Path) -> None:
    """
    调用 GROBID 处理单个 PDF，输出 TEI-XML 文件。
    """
    if not pdf_path.exists():
        print(f"[WARN] PDF not found: {pdf_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{pdf_path.stem}.tei.xml"

    # 已经处理过则跳过
    if out_path.exists():
        print(f"[SKIP] {pdf_path.name} -> {out_path} (already exists)")
        return

    print(f"[INFO] Processing {pdf_path} ...")

    with pdf_path.open("rb") as f:
        files = {
            "input": (pdf_path.name, f, "application/pdf"),
        }
        # GROBID 的一些常用参数，可以按需微调
        data = {
            "consolidateHeader": 1,
            "consolidateCitations": 0,
            "includeRawCitations": 0,
            "includeRawAffiliations": 0,
            # "segmentSentences": 1,  # 如果你想让 GROBID 顺便做句子分割，可以打开
        }

        try:
            resp = requests.post(
                GROBID_URL,
                files=files,
                data=data,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] GROBID request failed for {pdf_path.name}: {e}")
            return

    # 保存 TEI-XML
    try:
        out_path.write_text(resp.text, encoding="utf-8")
        print(f"[OK] {pdf_path.name} -> {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write TEI for {pdf_path.name}: {e}")


def main():
    if not INPUT_DIR.exists():
        print(f"[FATAL] INPUT_DIR does not exist: {INPUT_DIR}")
        return

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] No PDF files found under {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(pdf_files)} PDFs under {INPUT_DIR}")

    for pdf_path in pdf_files:
        process_pdf(pdf_path)
        time.sleep(SLEEP_BETWEEN)


if __name__ == "__main__":
    main()
