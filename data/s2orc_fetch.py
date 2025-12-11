# API_KEY: rnSaOd2Z6O5vpOY6Unnp87B4aoN6mkrW1ZYBd2QN

import os
import json
import gzip
import time
from pathlib import Path

import requests

# ================= 配置 =================

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or "rnSaOd2Z6O5vpOY6Unnp87B4aoN6mkrW1ZYBd2QN"
DATASET_NAME = "s2orc"

TARGET_COUNT = 7000  # 目标论文数

OUT_DIR  = r"D:\code\Github\SLAC-test\data\A_structure\s2orc"  # 每篇论文一个 .json
LOG_DIR  = r"D:\code\Github\SLAC-test\log"
LOGFILE  = os.path.join(LOG_DIR, "s2orc_7000.log")

REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS = 1.0


# ================= 日志工具 =================

def log(msg: str):
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOGFILE, "a", encoding="utf-8") as lf:
        lf.write(line + "\n")


# ================= 基础函数 =================

def get_paper_key(obj: dict) -> str:
    """
    用来去重的 key：
    优先 paperId，其次 arxivId，再次 corpusid，最后 fall back 到 hash。
    """
    pid = obj.get("paperId")
    if pid:
        return f"paperId:{pid}"
    ext = obj.get("externalids") or {}
    if ext.get("arxiv"):
        return f"arxiv:{ext['arxiv']}"
    cid = obj.get("corpusid")
    if cid is not None:
        return f"corpusid:{cid}"
    # 极端 fallback（理论上不会大量出现）
    return f"anon:{hash(json.dumps(obj, sort_keys=True))}"


def key_to_path(out_dir: Path, key: str) -> Path:
    """
    把去重 key 转成文件路径：
    - paperId:12345      -> out_dir / "paperId_12345.json"
    - corpusid:17109151  -> out_dir / "corpusid_17109151.json"
    - arxiv:hep-ph/9209259
        -> out_dir / "arxiv_hep-ph" / "9209259.json"
    - arxiv:1504.04461
        -> out_dir / "arxiv_1504.04461.json"
    """
    prefix, _, rest = key.partition(":")
    # 正常有 prefix 和 rest
    if prefix == "arxiv":
        arxiv_id = rest
        if "/" in arxiv_id:
            cat, pid = arxiv_id.split("/", 1)  # hep-ph, 9209259
            subdir = f"arxiv_{cat}"
            filename = pid + ".json"
            return out_dir / subdir / filename
        else:
            filename = f"arxiv_{arxiv_id}.json"
            return out_dir / filename
    else:
        safe = key.replace(":", "_").replace("/", "_")
        filename = safe + ".json"
        return out_dir / filename


def load_existing(out_dir: Path):
    """
    读取 OUT_DIR 下已有的 *.json（包括子目录），返回 (seen_keys, count)：
      - seen_keys: 用于去重
      - count: 已保存的论文条数
    """
    seen = set()
    if not out_dir.exists():
        return seen, 0

    log(f"检测到已有输出目录: {out_dir}，开始加载已有记录用于断点续跑…")
    count = 0
    # 用 rglob 遍历子目录
    for fp in out_dir.rglob("*.json"):
        try:
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            log(f"WARN: 文件 {fp} 读取或解析 JSON 失败，已跳过。")
            continue
        key = get_paper_key(obj)
        seen.add(key)
        count += 1
    log(f"已有 {count} 篇论文，将在此基础上继续抓取。")
    return seen, count


def get_latest_release_id():
    log("请求最新 release_id …")
    resp = requests.get(
        "https://api.semanticscholar.org/datasets/v1/release/latest",
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    rid = data["release_id"]
    log(f"最新 release_id = {rid}")
    return rid


def get_dataset_files(release_id: str, dataset_name: str):
    log(f"获取 release {release_id} 下数据集 {dataset_name} 的分片列表…")
    headers = {"x-api-key": API_KEY}
    url = f"https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/{dataset_name}/"
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    files = data.get("files") or []
    log(f"共发现 {len(files)} 个分片 URL。")
    return files


# ================= 主逻辑 =================

def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError("请先设置 SEMANTIC_SCHOLAR_API_KEY 环境变量，或者把 API_KEY 改为你的真实 key。")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取已有结果（支持断点续跑 + 去重）
    seen_keys, saved_count = load_existing(out_dir)
    if saved_count >= TARGET_COUNT:
        log(f"已有 {saved_count} 篇论文，已达到或超过目标 {TARGET_COUNT}，无需再抓取。")
        return

    # 2. 拿 release_id 和分片列表
    release_id = get_latest_release_id()
    time.sleep(SLEEP_BETWEEN_REQUESTS)

    file_urls = get_dataset_files(release_id, DATASET_NAME)
    if not file_urls:
        log("ERROR: 没有找到任何 S2ORC 分片 URL，退出。")
        return

    # 3. 遍历分片，逐行读取，直到凑够 TARGET_COUNT
    for file_idx, url in enumerate(file_urls):
        if saved_count >= TARGET_COUNT:
            break

        log(f"开始处理分片 {file_idx+1}/{len(file_urls)}: {url}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)  # 控制请求频率

        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                r.raise_for_status()
                r.raw.decode_content = True
                with gzip.GzipFile(fileobj=r.raw) as gz:
                    for line_idx, line_bytes in enumerate(gz, start=1):
                        if saved_count >= TARGET_COUNT:
                            break
                        try:
                            line = line_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            log(f"WARN: 分片 {file_idx+1} 第 {line_idx} 行解码失败，跳过。")
                            continue

                        line = line.strip()
                        if not line:
                            continue

                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            log(f"WARN: 分片 {file_idx+1} 第 {line_idx} 行 JSON 解析失败，跳过。")
                            continue

                        key = get_paper_key(obj)
                        if key in seen_keys:
                            continue

                        # 生成文件路径（可能带子目录）
                        fp = key_to_path(out_dir, key)
                        fp.parent.mkdir(parents=True, exist_ok=True)

                        # 理论上不会冲突，如果已经存在就直接当已保存
                        if fp.exists():
                            log(f"WARN: 文件 {fp} 已存在，但 key 不在 seen_keys 内，视为已保存。")
                            seen_keys.add(key)
                            saved_count += 1
                            continue

                        with fp.open("w", encoding="utf-8") as f:
                            json.dump(obj, f, ensure_ascii=False, indent=2)

                        saved_count += 1
                        seen_keys.add(key)

                        if saved_count % 100 == 0:
                            log(f"已累计保存 {saved_count} 篇论文…")

                        if saved_count >= TARGET_COUNT:
                            break

        except requests.RequestException as e:
            log(f"ERROR: 请求分片 {file_idx+1} 失败: {e}")
            # 出错就跳过这个分片，继续下一个
            continue
        except Exception as e:
            log(f"ERROR: 处理分片 {file_idx+1} 时发生未知错误: {e}")
            continue

    log(f"完成：共保存 {saved_count} 篇论文（目标 {TARGET_COUNT}）。输出目录：{out_dir}")


if __name__ == "__main__":
    main()
