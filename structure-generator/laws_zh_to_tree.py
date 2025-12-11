# -*- coding: utf-8 -*-
"""
批量将中文法律 docx 转换为 SLAC 项目所需的结构树 JSON。

依赖：
    pip install python-docx

目录约定（可按需修改）：
    输入目录：D:\code\Github\SLAC-test\data\A_structure\laws\zh
    输出目录：D:\code\Github\SLAC-test\data\A_structure\laws_structure\zh
    日志文件：D:\code\Github\SLAC-test\log\laws_zh_structure.log
"""

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from docx import Document  # pip install python-docx

# ===================== 路径配置 =====================

INPUT_DIR = Path(r"D:\code\Github\SLAC-test\data\A_structure\laws\zh")
OUTPUT_DIR = Path(r"D:\code\Github\SLAC-test\data\A_structure\laws_structure\zh")
LOG_FILE = Path(r"D:\code\Github\SLAC-test\log\laws_zh_structure.log")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ===================== 日志配置 =====================

logger = logging.getLogger("laws_zh_structure")
logger.setLevel(logging.INFO)

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# ===================== 正则与工具函数 =====================

CH_NUM = "一二三四五六七八九十百千〇零0-9０-９"

RE_COMPILE = re.compile(rf"^第[{CH_NUM}]+编")
RE_CHAPTER = re.compile(rf"^第[{CH_NUM}]+章")
RE_SECTION = re.compile(rf"^第[{CH_NUM}]+节")
RE_ARTICLE = re.compile(rf"^第[{CH_NUM}]+条")

RE_TOC = re.compile(r"^目\s*录$")  # 目录
RE_LIST_ITEM = re.compile(r"^[（(][一二三四五六七八九十百千〇零0-9]+[)）]")  # （一） (一)

# 修正案中的“一、二、三、”顶层条目
RE_AMEND_ITEM = re.compile(r"^[一二三四五六七八九十百千〇零0-9０-９]+[、．\.]")

# 简单判断页眉页脚/页码（例如 “- 6 -” “6 / 10” 等）
RE_PAGENO = re.compile(r"^[-–—\d\s./]+$")

# 合并后的单个 unit 最大字符数，超过则强制新起一个 unit
MAX_UNIT_CHAR_LEN = 2000


def normalize_text(text: str) -> str:
    """统一处理空白（去掉首尾空白 & 全角空格），保留中间空白用于显示。"""
    if text is None:
        return ""
    t = text.strip().replace("\u3000", " ")
    return t


def compact_text(text: str) -> str:
    """去掉所有空白，用于结构模式匹配（编/章/节/条/修正案项/目录）。"""
    t = normalize_text(text)
    return re.sub(r"\s+", "", t)


def classify_heading(text: str):
    """
    识别是否为 编 / 章 / 节 / 条 标题。
    返回: "compile" | "chapter" | "section" | "article" | None
    """
    norm = compact_text(text)
    if RE_COMPILE.match(norm):
        return "compile"
    if RE_CHAPTER.match(norm):
        return "chapter"
    if RE_SECTION.match(norm):
        return "section"
    if RE_ARTICLE.match(norm):
        return "article"
    return None


def is_amend_item_title(text: str) -> bool:
    """修正案中形如“一、…… ”的顶层条目标题"""
    norm = compact_text(text)
    return bool(RE_AMEND_ITEM.match(norm))


def is_toc_line(text: str) -> bool:
    """识别目录起始行"""
    t = normalize_text(text)
    t_comp = compact_text(t)
    # 兼容“目    录”和“目录”
    return bool(RE_TOC.match(t)) or t_comp == "目录"


def is_list_item(text: str) -> bool:
    t = normalize_text(text)
    return bool(RE_LIST_ITEM.match(t))


def is_page_number_like(text: str) -> bool:
    t = normalize_text(text)
    if len(t) > 15:
        return False
    return bool(RE_PAGENO.match(t))


# ===================== 核心解析函数 =====================


def parse_law_docx(docx_path: Path) -> dict:
    """
    解析单个法律 docx 文件为结构树 JSON 对象。
    """
    logger.info(f"Start parsing: {docx_path}")

    document = Document(docx_path)
    units = []
    next_unit_id = 0

    def add_unit(text: str, type_: str, level: int, parent_id):
        nonlocal next_unit_id
        unit = {
            "unit_id": next_unit_id,
            "text": text,
            "type": type_,
            "level": level,
            "parent_id": parent_id,
        }
        units.append(unit)
        uid = next_unit_id
        next_unit_id += 1
        return uid

    title_id = None

    toc_unit_id = None
    toc_unit_index = None
    in_toc = False  # 是否在“目录”区域内（目录内容将全部合并到一个 unit）

    current_compile_id = None  # 当前“编”
    current_chapter_id = None  # 当前“章”
    current_section_id = None  # 当前“节”

    last_article_id = None      # 最近的“条/修正项目”对应的 unit_id
    last_article_index = None   # units 中的索引

    filename_stem = docx_path.stem
    doc_id = filename_stem

    paragraphs = document.paragraphs

    # 是否为“修正案”类文档：标题或文件名包含“修正案”
    is_amendment_doc = False

    for para in paragraphs:
        raw = para.text
        text = normalize_text(raw)

        if not text:
            # 空行：通常可以安全忽略
            continue

        # 跳过明显的页码/页眉页脚
        if is_page_number_like(text):
            logger.debug(f"Skip page-like line in {docx_path.name}: {text!r}")
            continue

        # 设置文档标题：第一行非空内容
        if title_id is None:
            title_id = add_unit(text=text, type_="title", level=0, parent_id=None)
            is_amendment_doc = ("修正案" in text) or ("修正案" in docx_path.stem)
            logger.info(f"{docx_path.name}: detected title -> {text}")
            logger.info(f"{docx_path.name}: is_amendment_doc={is_amendment_doc}")
            continue

        # 目录起始
        if toc_unit_id is None and is_toc_line(text):
            toc_unit_id = add_unit(
                text=text,            # 起始写“目 录”或“目录”
                type_="other",
                level=1,
                parent_id=title_id,
            )
            toc_unit_index = len(units) - 1
            in_toc = True
            logger.info(f"{docx_path.name}: detected TOC start")
            continue

        # 目录区域：将所有内容合并到一个 unit，直到遇到首个正文结构标题
        if in_toc:
            heading_kind = classify_heading(text)
            # 编 / 章 / 节 / 条 视为正文开始
            if heading_kind in {"compile", "chapter", "section", "article"}:
                in_toc = False
                logger.info(f"{docx_path.name}: leave TOC region on line: {text}")
                # fall through，继续用正文逻辑处理这行
            else:
                # 仍在目录区域，把当前行拼到 toc_unit 的 text 中
                if toc_unit_index is not None:
                    prev = units[toc_unit_index]["text"]
                    units[toc_unit_index]["text"] = prev.rstrip() + "\n" + text
                else:
                    logger.warning(
                        f"{docx_path.name}: in_toc=True but toc_unit_index is None."
                    )
                continue

        # 特殊：附件标题（例如“附件：……”）
        if text.startswith("附件"):
            attach_unit_id = add_unit(
                text=text,
                type_="heading",
                level=1,          # 作为一级标题挂在法名下
                parent_id=title_id,
            )
            # 进入附件后，可以清空当前上下文（防止条目继续挂在原章节下）
            current_compile_id = None
            current_chapter_id = None
            current_section_id = None
            last_article_id = None
            last_article_index = None
            logger.info(f"{docx_path.name}: detected attachment heading -> {text}")
            continue

        # 结构标题或条文识别
        heading_kind = classify_heading(text)

        # 修正案中的“一、二、三……”视作条级起始
        if heading_kind is None and is_amendment_doc and is_amend_item_title(text):
            heading_kind = "article"
            logger.debug(f"{docx_path.name}: detected amendment item as article -> {text}")

        if heading_kind == "compile":
            parent_id = title_id
            level = 1
            current_compile_id = add_unit(
                text=text,
                type_="heading",
                level=level,
                parent_id=parent_id,
            )
            current_chapter_id = None
            current_section_id = None
            last_article_id = None
            last_article_index = None
            logger.debug(f"{docx_path.name}: compile heading -> {text}")
            continue

        if heading_kind == "chapter":
            parent_id = current_compile_id if current_compile_id is not None else title_id
            parent_level = units[parent_id]["level"] if parent_id is not None else 0
            level = parent_level + 1
            current_chapter_id = add_unit(
                text=text,
                type_="heading",
                level=level,
                parent_id=parent_id,
            )
            current_section_id = None
            last_article_id = None
            last_article_index = None
            logger.debug(f"{docx_path.name}: chapter heading -> {text}")
            continue

        if heading_kind == "section":
            if current_chapter_id is not None:
                parent_id = current_chapter_id
            elif current_compile_id is not None:
                parent_id = current_compile_id
            else:
                parent_id = title_id
            parent_level = units[parent_id]["level"] if parent_id is not None else 0
            level = parent_level + 1
            current_section_id = add_unit(
                text=text,
                type_="heading",
                level=level,
                parent_id=parent_id,
            )
            last_article_id = None
            last_article_index = None
            logger.debug(f"{docx_path.name}: section heading -> {text}")
            continue

        if heading_kind == "article":
            # 条文/修正项目：作为 paragraph；父节点优先级：节 > 章 > 编 > 法名
            if current_section_id is not None:
                parent_id = current_section_id
            elif current_chapter_id is not None:
                parent_id = current_chapter_id
            elif current_compile_id is not None:
                parent_id = current_compile_id
            else:
                parent_id = title_id

            parent_level = units[parent_id]["level"] if parent_id is not None else 0
            level = parent_level + 1

            last_article_id = add_unit(
                text=text,
                type_="paragraph",
                level=level,
                parent_id=parent_id,
            )
            last_article_index = len(units) - 1

            logger.debug(f"{docx_path.name}: article -> {text}")
            continue

        # 列举款（（一）（二）……）
        if is_list_item(text):
            if last_article_id is not None:
                parent_id = last_article_id
                parent_level = units[parent_id]["level"]
            else:
                # 理论上不太会出现“没有条就有款”，保险起见挂在最近章节/法名下
                parent_id = (
                    current_section_id
                    or current_chapter_id
                    or current_compile_id
                    or title_id
                )
                parent_level = units[parent_id]["level"] if parent_id is not None else 0

            level = parent_level + 1
            add_unit(
                text=text,
                type_="list-item",
                level=level,
                parent_id=parent_id,
            )
            logger.debug(f"{docx_path.name}: list item -> {text}")
            continue

        # 普通段落：
        # 若已在某一条文/修正项目内，则视为该条文的续行，优先尝试合并到上一条的 text 中
        if last_article_id is not None and last_article_index is not None:
            prev_text = units[last_article_index]["text"]
            merged_text = prev_text.rstrip() + "\n" + text

            if len(merged_text) <= MAX_UNIT_CHAR_LEN:
                # 长度可接受：继续合并到同一个 unit
                units[last_article_index]["text"] = merged_text
                logger.debug(
                    f"{docx_path.name}: append to article {last_article_id} -> {text}"
                )
            else:
                # 太长：新起一个同级段落 unit，仍然挂在同一父节点下
                parent_id = units[last_article_index]["parent_id"]
                level = units[last_article_index]["level"]
                new_uid = add_unit(
                    text=text,
                    type_="paragraph",
                    level=level,
                    parent_id=parent_id,
                )
                last_article_id = new_uid
                last_article_index = len(units) - 1
                logger.debug(
                    f"{docx_path.name}: start new paragraph under same article due to length -> {text}"
                )
        else:
            # 例如法条前/后的说明性段落，挂在法名下
            add_unit(
                text=text,
                type_="paragraph",
                level=1,
                parent_id=title_id,
            )
            logger.debug(f"{docx_path.name}: root-level paragraph -> {text}")

    # 统计信息
    type_counts = Counter(u["type"] for u in units)
    logger.info(
        f"Finished {docx_path.name}: units={len(units)}, "
        f"types={dict(type_counts)}"
    )

    # 构造最终文档对象
    doc_obj = {
        "doc_id": doc_id,
        "source_type": "A",
        "lang": "zh",
        "domain": "law",
        "split": "train",
        "weight": 1.0,
        "units": units,
    }
    return doc_obj


# ===================== 批处理主函数 =====================


def process_all_laws():
    if not INPUT_DIR.exists():
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return

    docx_files = sorted(INPUT_DIR.glob("*.docx"))
    if not docx_files:
        logger.warning(f"No .docx files found in {INPUT_DIR}")
        return

    logger.info(f"Found {len(docx_files)} .docx files in {INPUT_DIR}")

    for docx_path in docx_files:
        try:
            doc_obj = parse_law_docx(docx_path)
            out_path = OUTPUT_DIR / f"{docx_path.stem}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(doc_obj, f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote JSON: {out_path}")
        except Exception as e:
            logger.exception(f"Failed to process {docx_path}: {e}")


if __name__ == "__main__":
    process_all_laws()
