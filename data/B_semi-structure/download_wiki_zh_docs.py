"""
从 pleisto/wikipedia-cn-20230720-filtered 下载中文维基条目，
将每一条（完整词条）保存为一个本地 .txt 文件。

策略：
- 只保留长度在 [MIN_CHARS, MAX_CHARS] 区间的条目（过短/过长都跳过）
- 按“字符数≈token 数”累积，直到 TOTAL_TOKENS >= TARGET_TOKENS 为止
- 只有真正保存的 txt 文件才计入 token 统计

配置：
- 输出目录：D:\code\Github\SLAC-test\data\B_semi-structure\zh-tech
- 日志文件：D:\code\Github\SLAC-test\log\wiki-zh.log
"""

import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ================= 配置 =================

# 输出 txt 文件目录
OUTPUT_DIR = Path(r"D:\code\Github\SLAC-test\data\B_semi-structure\zh-tech")

# 日志文件路径
LOG_FILE = Path(r"D:\code\Github\SLAC-test\log\wiki-zh.log")

# 数据集信息
DATASET_NAME = "pleisto/wikipedia-cn-20230720-filtered"
CONFIG_NAME = None          # 该数据集无子配置，可设为 None
SPLIT = "train"             # 一般只有一个 split，叫 train

# 文本长度区间（字符数）
# 只保留 MIN_CHARS <= length <= MAX_CHARS 的条目
MIN_CHARS = 1500
MAX_CHARS = 3000

# 目标 token 数（近似按字符数计）
TARGET_TOKENS = 5_000_000

# 最多扫描多少条样本（不是保存数量，只是一个安全上限）
# None 表示遍历整个 split
MAX_SAMPLES_TO_SCAN = None  # 比如可以设为 45000

# 输出文件名前缀
FILE_PREFIX = "wiki_zh"


# ================= 日志初始化 =================

def setup_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(log_file),
        filemode="w",  # 每次运行重写日志，如需追加改为 "a"
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger("").addHandler(console)


# ================= 工具函数 =================

def detect_text_field(dataset) -> str:
    """
    自动检测数据集中哪个字段是“正文”字段。

    策略：
    1. 看第一条样本 dataset[0]
    2. 找出所有值是非空字符串的字段
    3. 选择长度最长的那个字段作为正文字段
    """
    if len(dataset) == 0:
        raise ValueError("数据集为空，无法自动检测文本字段。")

    sample = dataset[0]
    candidates = []

    for key, value in sample.items():
        if isinstance(value, str) and value.strip():
            candidates.append((key, len(value.strip())))

    if not candidates:
        raise ValueError(f"未在样本中找到字符串字段，样本 keys: {list(sample.keys())}")

    candidates.sort(key=lambda x: x[1], reverse=True)
    text_field, length = candidates[0]

    logging.info(
        "自动检测到文本字段为 '%s'（样本0长度=%d 字符）。所有字符串字段候选: %s",
        text_field,
        length,
        candidates,
    )
    return text_field


# ================= 主逻辑 =================

def main():
    setup_logger(LOG_FILE)

    logging.info(
        "开始加载数据集 %s (config=%s, split=%s)...",
        DATASET_NAME,
        CONFIG_NAME,
        SPLIT,
    )

    try:
        if CONFIG_NAME is None:
            ds = load_dataset(DATASET_NAME, split=SPLIT)
        else:
            ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT)
    except Exception as e:
        logging.exception("加载数据集失败: %r", e)
        return

    logging.info("数据集加载成功，样本数量: %d", len(ds))

    try:
        text_field = detect_text_field(ds)
    except Exception as e:
        logging.exception("自动检测文本字段失败: %r", e)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_scanned = 0       # 扫描过的样本数
    saved_docs = 0          # 实际保存的文档数
    skipped_too_short = 0   # < MIN_CHARS
    skipped_too_long = 0    # > MAX_CHARS
    skipped_invalid = 0     # 非字符串或缺字段
    errors = 0
    total_tokens = 0        # 仅对已保存的文档累积长度（近似 token 数）

    # 限制扫描数量
    if MAX_SAMPLES_TO_SCAN is not None:
        n = min(MAX_SAMPLES_TO_SCAN, len(ds))
        iterator = ds.select(range(n))
        logging.info("将扫描前 %d 条样本（MAX_SAMPLES_TO_SCAN=%d）。", n, MAX_SAMPLES_TO_SCAN)
    else:
        iterator = ds
        logging.info("将扫描所有样本，总数=%d。", len(ds))

    for idx, ex in tqdm(
        enumerate(iterator),
        total=len(iterator),
        desc=f"Saving zh wiki articles ({SPLIT})",
    ):
        total_scanned += 1

        # 如果已经达到目标 token 数，就直接停止
        if total_tokens >= TARGET_TOKENS:
            logging.info(
                "已达到目标 token 数：%d（TARGET_TOKENS=%d），停止扫描。",
                total_tokens, TARGET_TOKENS
            )
            break

        try:
            if text_field not in ex:
                logging.warning(
                    "样本 %d 不包含字段 '%s'，实际 keys=%s，跳过。",
                    idx,
                    text_field,
                    list(ex.keys()),
                )
                skipped_invalid += 1
                continue

            value = ex[text_field]

            if not isinstance(value, str):
                logging.warning(
                    "样本 %d 的字段 '%s' 不是字符串，类型=%s，跳过。",
                    idx,
                    text_field,
                    type(value),
                )
                skipped_invalid += 1
                continue

            text_stripped = value.strip()
            length = len(text_stripped)

            if length < MIN_CHARS:
                logging.info(
                    "样本 %d 文本过短，长度=%d，阈值 MIN_CHARS=%d，跳过。",
                    idx,
                    length,
                    MIN_CHARS,
                )
                skipped_too_short += 1
                continue

            if length > MAX_CHARS:
                logging.info(
                    "样本 %d 文本过长，长度=%d，阈值 MAX_CHARS=%d，跳过。",
                    idx,
                    length,
                    MAX_CHARS,
                )
                skipped_too_long += 1
                continue

            # 保存文件
            filename = OUTPUT_DIR / f"{FILE_PREFIX}_{idx:06d}.txt"
            filename.write_text(text_stripped + "\n", encoding="utf-8")

            saved_docs += 1
            total_tokens += length  # 近似 token 数

            logging.info(
                "已保存样本 %d -> %s（长度=%d 字符），当前累计 tokens=%d",
                idx,
                filename.name,
                length,
                total_tokens,
            )

        except Exception as e:
            logging.exception("处理样本 %d 时出错: %r", idx, e)
            errors += 1

    logging.info(
        "处理完成。扫描样本数: %d, 保存: %d, 过短跳过: %d, 过长跳过: %d, 无效: %d, 异常: %d",
        total_scanned,
        saved_docs,
        skipped_too_short,
        skipped_too_long,
        skipped_invalid,
        errors,
    )
    logging.info(
        "最终累计 token 数(≈字符数): %d（目标 TARGET_TOKENS=%d）",
        total_tokens,
        TARGET_TOKENS,
    )
    logging.info("输出目录: %s", OUTPUT_DIR.resolve())
    logging.info("日志文件: %s", LOG_FILE.resolve())


if __name__ == "__main__":
    main()
