"""
fetch_paper.py

通过 arXiv API 获取不同领域的论文，并下载对应 PDF 到本地。

当前只获取还未覆盖的领域：
- cs.CL, cs.LG, cs.AI, cs.IR
- stat.ML, stat.AP, q-fin.*
- eess.SP, eess.SY

你之前已经下载过 math / physics，对应查询已经移除。
"""

import os
import re
import time
from typing import List, Tuple

import requests
import feedparser

# arXiv API 基础地址
BASE_API_URL = "https://export.arxiv.org/api/query"

# 按 arXiv 建议，使用包含联系方式的 User-Agent（请保持为你自己的邮箱）
USER_AGENT = "SLAC-data-builder/0.1 (mailto:2542881@dundee.ac.uk)"

# PDF 输出目录（相对当前脚本路径）
OUTPUT_DIR = "A_structure/papers(deprecated)"

# 访问频率与重试策略
REQUEST_INTERVAL = 5.0   # 每次 API 请求之间的间隔（秒）
DOWNLOAD_INTERVAL = 2.0  # 每次 PDF 下载之间的间隔（秒）
MAX_API_RETRIES = 5      # 遇到 429 / 超时 时的最大重试次数

# 只保留还未覆盖的领域（每个子类单独查询，避免复杂 OR 语法）
# (标签, search_query, max_results)
SEARCH_QUERIES: List[Tuple[str, str, int]] = [
    # CS / NLP / ML 相关
    ("cs_cl", "cat:cs.CL", 10),
    ("cs_lg", "cat:cs.LG", 10),
    ("cs_ai", "cat:cs.AI", 10),
    ("cs_ir", "cat:cs.IR", 10),

    # 统计 / 金融
    ("stat_ml", "cat:stat.ML", 5),
    ("stat_ap", "cat:stat.AP", 5),
    ("qfin", "cat:q-fin.*", 5),

    # 工程 / 信号处理
    ("eess_sp", "cat:eess.SP", 5),
    ("eess_sy", "cat:eess.SY", 5),
]


def query_arxiv(search_query: str, max_results: int):
    """
    调用 arXiv API，返回 feedparser 解析后的条目列表。
    带 429 / 超时 重试逻辑。
    """
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            resp = requests.get(BASE_API_URL, params=params, headers=headers, timeout=60)

            # 429: Too Many Requests
            if resp.status_code == 429:
                wait = 30 * attempt
                print(f"[WARN] 429 Too Many Requests, attempt {attempt}/{MAX_API_RETRIES}, sleep {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            return feed.entries

        except requests.exceptions.Timeout:
            wait = 30 * attempt
            print(f"[WARN] API timeout, attempt {attempt}/{MAX_API_RETRIES}, sleep {wait}s...")
            time.sleep(wait)

        except requests.exceptions.HTTPError as e:
            # 非 429 的 HTTP 错误直接抛出
            raise e

    raise RuntimeError("arXiv API repeatedly failed, please try again later or switch to manual ID mode.")


def extract_arxiv_id(entry) -> str:
    """
    从 API 返回的 entry 中提取不带版本号的 arXiv ID。
    例如: 'http://arxiv.org/abs/2301.01234v2' -> '2301.01234'
    """
    url = entry.get("id", "")
    if "/abs/" in url:
        id_part = url.split("/abs/")[-1]
    else:
        id_part = url.rsplit("/", 1)[-1]

    # 去掉末尾版本号 v1/v2/...
    id_part = re.sub(r"v\d+$", "", id_part)
    return id_part


def download_pdf(arxiv_id: str, out_dir: str) -> str:
    """
    根据 arXiv ID 下载 PDF 文件。
    如果文件已存在则跳过。
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    headers = {"User-Agent": USER_AGENT}

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{arxiv_id}.pdf")

    if os.path.exists(out_path):
        print(f"[SKIP] {arxiv_id} already downloaded")
        return out_path

    resp = requests.get(pdf_url, headers=headers, timeout=120)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(resp.content)

    print(f"[OK] {arxiv_id} -> {out_path}")
    return out_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label, search_query, max_results in SEARCH_QUERIES:
        print(f"\n=== Querying arXiv for topic '{label}' ({max_results} results) ===")

        try:
            entries = query_arxiv(search_query, max_results)
        except Exception as e:
            print(f"[ERROR] API query failed for '{label}': {e}")
            continue

        print(f"Received {len(entries)} entries for '{label}'")
        # 0 条就直接下一个 query
        if len(entries) == 0:
            continue

        time.sleep(REQUEST_INTERVAL)

        for entry in entries:
            arxiv_id = extract_arxiv_id(entry)
            try:
                download_pdf(arxiv_id, OUTPUT_DIR)
            except Exception as e:
                print(f"[FAIL] Download error for {arxiv_id}: {e}")
            time.sleep(DOWNLOAD_INTERVAL)


if __name__ == "__main__":
    main()
