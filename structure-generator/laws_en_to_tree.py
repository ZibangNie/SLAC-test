# -*- coding: utf-8 -*-
"""
将英文 EU 法律 JSON 转换为 SLAC 结构树 JSON（A 类数据，英文法律）

输入目录:
    D:\code\Github\SLAC-test\data\A_structure\laws\en

输出目录:
    D:\code\Github\SLAC-test\data\A_structure\laws_structure\en
    - 输出文件与输入文件保持相同的相对路径
    - 扩展名从 .json 变为 .tree.json

额外功能:
    - 随机抽取 10 篇结构树，调用 DeepSeek API 做结构检查
    - 将检查结果写入:
      D:\code\Github\SLAC-test\log\laws_en_tree.log

依赖:
    pip install openai  (用于调用 DeepSeek，若不安装则自动跳过检查步骤)
    Python 3.8+

环境变量:
    DEEPSEEK_API_KEY  : DeepSeek API 的密钥
    DEEPSEEK_API_BASE : 可选，DeepSeek OpenAI 兼容接口 base_url，
                        默认 "https://api.deepseek.com"
    DEEPSEEK_MODEL    : 可选，模型名，默认 "deepseek-chat"
"""

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
    # OpenAI 兼容客户端，用于 DeepSeek
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # 没有安装 openai 时，自动跳过 DeepSeek 检查


# ===================== 基本配置 =====================

INPUT_DIR = Path(r"D:\code\Github\SLAC-test\data\A_structure\laws\en")
OUTPUT_DIR = Path(r"D:\code\Github\SLAC-test\data\A_structure\laws_structure\en")

LOG_FILE = Path(r"/log/laws_en_structure.log")

# 随机抽查样本数量
SAMPLE_SIZE = 10

# A 类英文法律的统一元信息
SOURCE_TYPE = "A"
DEFAULT_SPLIT = "unknown"
DEFAULT_WEIGHT = 1.0


# ===================== 日志配置 =====================

def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清空已有 handler，避免重复
    logger.handlers.clear()

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    sh = logging.StreamHandler()

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)


# ===================== 结构树构建工具 =====================

class StructureBuilder:
    """
    负责构建单篇文档的结构树。
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.units: List[dict] = []
        self.next_id: int = 0

        # 栈: (unit_id, level, kind)
        self.stack: List[Tuple[int, int, str]] = []

        # 根节点（整篇文档主标题）的 unit_id
        self.root_id: Optional[int] = None

        # 最近的 CHAPTER / TITLE 顶层节点，用于挂 Article
        self.last_chapter_or_title_id: Optional[int] = None

    def add_unit(self, text: str, type_: str, level: int, parent_id: Optional[int]) -> int:
        text = text.strip()
        if not text:
            # 不创建空文本节点
            return -1

        uid = self.next_id
        self.next_id += 1

        unit = {
            "unit_id": uid,
            "text": text,
            "type": type_,
            "level": int(level),
            "parent_id": parent_id,
        }
        self.units.append(unit)
        return uid

    def set_root_title(self, text: str) -> int:
        """
        创建整篇文档主标题 (type=title, level=0, parent_id=null)
        """
        uid = self.add_unit(text, "title", 0, None)
        self.root_id = uid
        self.stack = [(uid, 0, "title")]
        return uid

    def reset_stack_to_root(self) -> None:
        if self.root_id is not None:
            self.stack = [(self.root_id, 0, "title")]
        else:
            self.stack = []

    def reset_stack_to_chapter(self) -> None:
        """
        将栈重置到 [root, last_chapter_or_title]，用于在 CHAPTER 下挂 Article。
        """
        self.reset_stack_to_root()
        if self.last_chapter_or_title_id is not None:
            self.stack.append((self.last_chapter_or_title_id, 1, "chapter"))

    def start_heading(self, text: str, level: int, kind: str = "heading", mark_chapter: bool = False) -> int:
        """
        添加一个 heading 节点，并按 level 更新栈。
        """
        while self.stack and self.stack[-1][1] >= level:
            self.stack.pop()

        parent_id = self.stack[-1][0] if self.stack else None
        uid = self.add_unit(text, "heading", level, parent_id)
        self.stack.append((uid, level, kind))

        if mark_chapter:
            self.last_chapter_or_title_id = uid

        return uid

    def add_paragraph_like(self, text: str, type_: str = "paragraph") -> int:
        """
        添加 paragraph / list-item / other 等节点，挂在当前栈顶下。
        """
        if self.stack:
            parent_id = self.stack[-1][0]
            level = self.stack[-1][1] + 1
        else:
            parent_id = None
            level = 0
        return self.add_unit(text, type_, level, parent_id)


# ===================== 模式匹配正则与辅助函数 =====================

CHAPTER_RE = re.compile(r"^CHAPTER\s+[IVXLCDM]+\b", re.IGNORECASE)
TITLE_RE = re.compile(r"^TITLE\s+[IVXLCDM]+\b", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^Article\s+\d+[A-Za-z]?\b", re.IGNORECASE)
INSTITUTION_RE = re.compile(
    r"^(THE )?(COMMISSION|COUNCIL|EUROPEAN PARLIAMENT|EUROPEAN CENTRAL BANK)",
    re.IGNORECASE,
)
HAVING_REGARD_RE = re.compile(r"^Having regard to", re.IGNORECASE)
WHEREAS_RE = re.compile(r"^Whereas:?$", re.IGNORECASE)
HAS_ADOPTED_RE = re.compile(
    r"HAS ADOPTED THIS (REGULATION|DECISION|DIRECTIVE|OPINION)",
    re.IGNORECASE,
)
LETTER_SECTION_RE = re.compile(r"^[A-Z]\.\s+[A-Z]", re.IGNORECASE)
ROMAN_SINGLE_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
NUMERIC_POINT_RE = re.compile(r"^\d+\.\s+")
PAREN_NUMERIC_RE = re.compile(r"^\(\d+\)")
LETTER_ITEM_RE = re.compile(r"^(\([a-z]\)|[a-z]\))\s+", re.IGNORECASE)
DASH_ITEM_RE = re.compile(r"^-\s+")


def is_decorative_line(line: str) -> bool:
    """
    判断是否是装饰行（例如 *****），可以直接过滤。
    """
    return bool(re.fullmatch(r"[*]+", line))


def is_mostly_upper(line: str) -> bool:
    """
    粗略判断一行是否“主要为大写”（用于识别章节标题、副标题）。
    """
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return False
    upper_count = sum(1 for ch in letters if ch.isupper())
    return upper_count / len(letters) >= 0.6


def is_post_title_start(line: str) -> bool:
    """
    判断是否是文首标题块之后的第一行（如 THE COMMISSION... / Having regard... / Article 1 等）。
    用于截断 title block。
    """
    if INSTITUTION_RE.search(line):
        return True
    if HAVING_REGARD_RE.match(line):
        return True
    if WHEREAS_RE.match(line):
        return True
    if ARTICLE_RE.match(line):
        return True
    if CHAPTER_RE.match(line):
        return True
    if TITLE_RE.match(line):
        return True
    if HAS_ADOPTED_RE.search(line):
        return True
    return False


def is_chapter_heading(line: str) -> bool:
    return bool(CHAPTER_RE.match(line))


def is_title_heading(line: str) -> bool:
    return bool(TITLE_RE.match(line))


def is_article_heading(line: str) -> bool:
    return bool(ARTICLE_RE.match(line))


def is_whereas_heading(line: str) -> bool:
    return bool(WHEREAS_RE.match(line))


def is_has_adopted_line(line: str) -> bool:
    return bool(HAS_ADOPTED_RE.search(line))


def is_institution_line(line: str) -> bool:
    return bool(INSTITUTION_RE.match(line))


def is_having_regard_line(line: str) -> bool:
    return bool(HAVING_REGARD_RE.match(line))


def is_letter_section_heading(line: str) -> bool:
    """
    A. PROCEDURE / B. PRODUCT UNDER CONSIDERATION...
    """
    return bool(LETTER_SECTION_RE.match(line) and is_mostly_upper(line))


def is_roman_heading(line: str) -> bool:
    """
    单独一行的罗马数字 I / II / III...
    """
    return bool(ROMAN_SINGLE_RE.fullmatch(line))


def is_numeric_point(line: str) -> bool:
    """
    形如 1. ... 的开头，用于区分 heading 或 list-item（看当前上下文）。
    """
    return bool(NUMERIC_POINT_RE.match(line))


def is_paren_numeric_item(line: str) -> bool:
    """
    (1) ... 作为列表条款。
    """
    return bool(PAREN_NUMERIC_RE.match(line))


def is_letter_list_item(line: str) -> bool:
    """
    (a) ... 或 a) ... 作为子列表条款。
    """
    return bool(LETTER_ITEM_RE.match(line))


def is_dash_list_item(line: str) -> bool:
    """
    - ... 列表条目。
    """
    return bool(DASH_ITEM_RE.match(line))


def is_potential_subtitle(line: str) -> bool:
    """
    用于检测 Article 后面的一行是否可以合并为副标题：
    - 不太长
    - 不像另一个 heading / 列表
    - 不以 Having regard / Whereas / Article / Chapter / Title 开头
    """
    if len(line) > 80:
        return False

    # 不能像其他结构标记
    if (
        is_article_heading(line)
        or is_chapter_heading(line)
        or is_title_heading(line)
        or is_letter_section_heading(line)
        or is_numeric_point(line)
        or is_paren_numeric_item(line)
        or is_letter_list_item(line)
        or is_dash_list_item(line)
        or is_institution_line(line)
        or is_having_regard_line(line)
        or is_whereas_heading(line)
        or is_has_adopted_line(line)
    ):
        return False

    return True


# ===================== 单篇文档结构树生成 =====================

def build_structure_tree_for_law(text: str, doc_id: str) -> dict:
    """
    给定 MultiEURLEX 英文法律文本，构建结构树 JSON（不含 split / source_type 等元信息）。
    """
    # 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = text.split("\n")

    # 预处理：去掉前后空白、过滤空行和纯 ***** 行
    lines: List[str] = []
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        if is_decorative_line(line):
            # 装饰行目前直接忽略（如果你想保留，可以改成 other 类型）
            continue
        lines.append(line)

    builder = StructureBuilder(doc_id=doc_id)

    if not lines:
        # 空文档兜底
        builder.set_root_title(doc_id)
        return {
            "doc_id": doc_id,
            "units": builder.units,
        }

    # ---------- 文首标题块 ----------
    title_lines: List[str] = []
    title_end_index = 0

    for idx, line in enumerate(lines):
        if is_post_title_start(line):
            title_end_index = idx
            break
        title_lines.append(line)
    else:
        # 没有明显的标题结束标记，则整个文档当成一行标题或至少第一行
        if not title_lines and lines:
            title_lines = [lines[0]]
            title_end_index = 1
        else:
            title_end_index = len(lines)

    if not title_lines:
        # 极端情况，至少用第一行
        title_lines = [lines[0]]
        title_end_index = max(title_end_index, 1)

    title_text = " ".join(title_lines)
    builder.set_root_title(title_text)

    # ---------- 主体内容扫描 ----------
    i = title_end_index

    while i < len(lines):
        line = lines[i]
        top_kind = builder.stack[-1][2] if builder.stack else None

        # 1) CHAPTER / TITLE 顶层章节
        if is_chapter_heading(line):
            # 可能存在下一行全大写副标题
            merged_text = line
            if (
                i + 1 < len(lines)
                and is_mostly_upper(lines[i + 1])
                and not is_chapter_heading(lines[i + 1])
                and not is_title_heading(lines[i + 1])
                and not is_article_heading(lines[i + 1])
            ):
                merged_text = f"{line} – {lines[i + 1]}"
                i += 1

            builder.start_heading(
                merged_text,
                level=1,
                kind="chapter",
                mark_chapter=True,
            )
            i += 1
            continue

        if is_title_heading(line):
            merged_text = line
            if (
                i + 1 < len(lines)
                and is_mostly_upper(lines[i + 1])
                and not is_chapter_heading(lines[i + 1])
                and not is_title_heading(lines[i + 1])
                and not is_article_heading(lines[i + 1])
            ):
                merged_text = f"{line} – {lines[i + 1]}"
                i += 1

            builder.start_heading(
                merged_text,
                level=1,
                kind="title",
                mark_chapter=True,
            )
            i += 1
            continue

        # 2) 字母章节 A. PROCEDURE / B. PRODUCT...
        if is_letter_section_heading(line):
            builder.start_heading(line, level=1, kind="section", mark_chapter=False)
            i += 1
            continue

        # 3) 罗马数字章节 I / II / III...
        if is_roman_heading(line):
            builder.start_heading(line, level=1, kind="roman", mark_chapter=False)
            i += 1
            continue

        # 4) 'Whereas:' 作为序言 heading
        if is_whereas_heading(line):
            builder.start_heading(line, level=1, kind="preamble", mark_chapter=False)
            i += 1
            continue

        # 5) HAS ADOPTED THIS REGULATION/DECISION...
        if is_has_adopted_line(line):
            # 简单作为段落处理，不改变结构栈
            builder.add_paragraph_like(line, type_="paragraph")
            i += 1
            continue

        # 6) Article X / Article 1a...
        if is_article_heading(line):
            merged_text = line
            if i + 1 < len(lines) and is_potential_subtitle(lines[i + 1]):
                merged_text = f"{line} – {lines[i + 1]}"
                i += 1

            # Article 默认挂在最新的 CHAPTER/TITLE 下面，如果没有则挂 root
            if builder.last_chapter_or_title_id is not None:
                builder.reset_stack_to_chapter()
                level = 2
            else:
                builder.reset_stack_to_root()
                level = 1

            builder.start_heading(merged_text, level=level, kind="article", mark_chapter=False)
            i += 1
            continue

        # 7) 数字点开头：1. Measures in force / 1. The financial contribution...
        if is_numeric_point(line):
            if top_kind in {"section", "chapter", "roman", "title"}:
                # 在 A./CHAPTER 等章节下面，视为子 heading
                parent_level = builder.stack[-1][1] if builder.stack else 0
                level = parent_level + 1
                builder.start_heading(line, level=level, kind="subsection", mark_chapter=False)
            else:
                # 其他情况下，视为条款 list-item
                builder.add_paragraph_like(line, type_="list-item")
            i += 1
            continue

        # 8) 列表条款：(1) / (a) / a) / - ...
        if is_paren_numeric_item(line) or is_letter_list_item(line) or is_dash_list_item(line):
            builder.add_paragraph_like(line, type_="list-item")
            i += 1
            continue

        # 9) 机构行 + Having regard...
        if is_institution_line(line) or is_having_regard_line(line):
            builder.add_paragraph_like(line, type_="paragraph")
            i += 1
            continue

        # 10) 其他全部当作 paragraph
        builder.add_paragraph_like(line, type_="paragraph")
        i += 1

    return {
        "doc_id": doc_id,
        "units": builder.units,
    }


# ===================== DeepSeek 结构检查（随机抽样） =====================

def run_deepseek_check_on_samples(tree_files: List[Path]) -> None:
    """
    随机抽取部分结构树文件，调用 DeepSeek API 做简单结构检查。
    结果写到日志中。若未配置 openai/DeepSeek，则跳过。
    """
    if not tree_files:
        logging.info("没有生成任何结构树文件，跳过 DeepSeek 检查。")
        return

    if OpenAI is None:
        logging.info("未安装 openai 库，无法调用 DeepSeek，跳过检查步骤。")
        return

    api_key = "sk-403803e6e58941bab12e23eef49d6c3c"
    if not api_key:
        logging.info("未设置 DEEPSEEK_API_KEY 环境变量，跳过 DeepSeek 检查。")
        return

    base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    client = OpenAI(api_key=api_key, base_url=base_url)

    sample_files = random.sample(tree_files, k=min(SAMPLE_SIZE, len(tree_files)))
    logging.info("开始使用 DeepSeek 检查 %d 篇结构树样本。", len(sample_files))

    for path in sample_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                tree = json.load(f)
        except Exception as e:
            logging.exception("读取结构树文件失败: %s (%s)", path, e)
            continue

        doc_id = tree.get("doc_id", path.stem)
        units = tree.get("units", [])

        # 截取前若干个 unit 片段，防止 prompt 太长
        preview_units = units[:40]
        preview_text_lines = []
        for u in preview_units:
            preview_text_lines.append(
                f"[unit_id={u.get('unit_id')}, type={u.get('type')}, level={u.get('level')}, parent_id={u.get('parent_id')}] {u.get('text')}"
            )
        preview_text = "\n".join(preview_text_lines)

        prompt = (
            "You are checking the structural JSON tree extracted from an EU legal document.\n"
            "Please briefly check whether the following sequence of units looks like a reasonable "
            "hierarchical structure (titles, headings, articles, paragraphs, list-items) for such a document.\n"
            "Just answer in English with a short comment about potential issues (if any). "
            "If it looks fine overall, just say it is generally reasonable.\n\n"
            f"doc_id: {doc_id}\n"
            f"Preview of first units:\n{preview_text}"
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a careful inspector of document structure trees."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            reply = resp.choices[0].message.content if resp.choices else ""
            logging.info("DeepSeek 检查 doc_id=%s, file=%s", doc_id, path)
            logging.info("DeepSeek 回复: %s", reply)
        except Exception as e:
            logging.exception("调用 DeepSeek 检查失败: doc_id=%s, file=%s, error=%s", doc_id, path, e)


# ===================== 主流程：批量转换 =====================

def process_all_laws() -> None:
    setup_logging()

    logging.info("开始处理英文法律 JSON -> 结构树。")
    logging.info("输入目录: %s", INPUT_DIR)
    logging.info("输出目录: %s", OUTPUT_DIR)

    if not INPUT_DIR.exists():
        logging.error("输入目录不存在: %s", INPUT_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(INPUT_DIR.rglob("*.json"))
    logging.info("共发现 %d 个 JSON 文件。", len(json_files))

    tree_files: List[Path] = []

    for idx, json_path in enumerate(json_files, start=1):
        rel_path = json_path.relative_to(INPUT_DIR)
        out_path = (OUTPUT_DIR / rel_path).with_suffix(".tree.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logging.exception("读取 JSON 失败: %s (%s)", json_path, e)
            continue

        celex_id = data.get("celex_id") or json_path.stem
        text = data.get("text", "")
        if not text:
            logging.warning("文件缺少 text 字段或为空: %s", json_path)

        tree_core = build_structure_tree_for_law(text, celex_id)

        # 包装为统一的 SLAC 文档结构
        doc = {
            "doc_id": celex_id,
            "source_type": SOURCE_TYPE,
            "split": DEFAULT_SPLIT,
            "weight": DEFAULT_WEIGHT,
            "units": tree_core["units"],
        }

        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.exception("写入结构树 JSON 失败: %s (%s)", out_path, e)
            continue

        tree_files.append(out_path)

        if idx % 50 == 0 or idx == len(json_files):
            logging.info(
                "进度: %d / %d (当前: %s -> %s)",
                idx, len(json_files), json_path.name, out_path.name
            )

    logging.info("结构树生成完成，成功写出 %d 个文件。", len(tree_files))

    # 随机抽查样本送 DeepSeek API 检查
    run_deepseek_check_on_samples(tree_files)

    logging.info("全部处理结束。")


if __name__ == "__main__":
    process_all_laws()
