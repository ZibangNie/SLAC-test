# -*- coding: utf-8 -*-
"""
en_standards_pdf_to_tree_v2_deepseek.py

英文国家/行业标准（ISO/IEC/ITU-T/DICOM/RFC 等）PDF -> SLAC 结构树 JSON（A 类数据）

整体流程：
  1）规则脚本从 PDF 抽取行文本，构建一个初始结构树（rule-based）；
  2）将初始结构树交给 DeepSeek-Reasoner，让其根据英文标准体例重写/优化结构树；
  3）将优化后的结构树再发给 DeepSeek 做“结构评估”，要求返回 JSON（is_ok / issues / suggestions）；
  4）如果 is_ok=true，则接受该结构树；
     如果 is_ok=false，则把“评估结果 + 修改意见”再发给 DeepSeek，让其在此基础上二次修正结构树；
  5）对 DeepSeek 输出做一次规范化（unit_id / parent_id / level 等），然后写出最终 .tree.json。

并发调度：
  - 输入可以是单个 PDF 文件，也可以是包含多个 PDF 的目录；
  - 使用 ThreadPoolExecutor 并发处理，最大并发数 MAX_WORKERS = 64；
  - 为了防止 API 被打爆，保留“每秒最多提交一个新任务”的限速逻辑。
"""

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

import pdfplumber
from openai import OpenAI, APIStatusError, RateLimitError


# ======================== 硬编码路径 & 基本配置 ========================

# 输入：英文标准 PDF 根目录（可为单个文件或目录）
PDF_INPUT_PATH = Path(r"D:\code\Github\SLAC-test\data\A_structure\national_standards\en")

# 输出：结构树 JSON 根目录
OUT_ROOT = Path(r"D:\code\Github\SLAC-test\data\A_structure\national_standards_structure\en")

# 日志文件
LOG_FILE = Path(r"D:\code\Github\SLAC-test\log\en_standards_structure_with_deepseek.log")

# DeepSeek 配置
# 建议改为从环境变量读取：os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip() or "sk-403803e6e58941bab12e23eef49d6c3c"
DEEPSEEK_MODEL = "deepseek-reasoner"
DEEPSEEK_MAX_TOKENS = 32000
ENABLE_DEEPSEEK_REFINE = True  # 是否启用 DeepSeek 结构优化与评估

# 并发调度配置
MAX_WORKERS = 64          # 最大并发数
SUBMIT_INTERVAL = 1.0     # 提交任务的时间间隔（秒），用于限速

# DeepSeek 多轮“优化-评估-再优化”的最大轮次
# round 1: 规则树 -> 第一次优化
# round 2..N: 基于上一轮结构树 + 评估反馈进行再次优化
MAX_DEEPSEEK_ROUNDS = 5


# ======================== 日志 ========================

def setup_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("en_standards_deepseek")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


logger = setup_logger()


# ======================== DeepSeek 客户端 ========================

def get_deepseek_client() -> OpenAI:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    return client


# ======================== 英文规则解析：数据结构 ========================

@dataclass
class LineRecord:
    page: int            # 页号，从 0 开始
    index_in_page: int   # 页内行号，从 0 开始
    text: str            # 原始文本（已 strip）


@dataclass
class Block:
    kind: str            # 'heading' | 'paragraph' | 'list-item' | 'other'
    text: str            # 合并后的文本
    page_start: int
    page_end: int
    line_start: int      # 清洗后行的全局行号
    line_end: int


# ======================== 英文规则解析：正则 & 工具函数 ========================

# 数字编号标题：
#   1 Scope
#   1.1 General
#   2.3.4 Detailed requirements
HEADING_NUM_RE = re.compile(r'^\s*(\d+(?:\.\d+)*)\.?\s+\S')

# Section / Clause 形式标题：
#   Section 3.2 Title
#   Clause 4 General
SECTION_HEADING_RE = re.compile(r'^(Section|Clause)\s+(\d+(?:\.\d+)*)\s+\S', re.IGNORECASE)

# Annex / Appendix 标题：
#   Annex A (normative) ...
#   Appendix B Other ...
ANNEX_RE = re.compile(r'^(Annex|Appendix)\s+[A-Z]\b', re.IGNORECASE)

# Annex 内部编号：
#   A.1 Title
#   A.1.1 Subsection
ANNEX_SUBSECTION_RE = re.compile(r'^([A-Z])\.(\d+(?:\.\d+)*)\s+\S')

# 列表项：
#   1) ...
#   (a) ...
#   a. ...
#   - ...
LIST_ITEM_RE = re.compile(
    r'^\s*(?:[-*•+]\s+|\(?[a-zA-Z0-9]+\)\s+|[a-zA-Z0-9]+[.)]\s+)'
)

# 表格 / 图标题：
#   Table 1 – ...
#   Figure 2. ...
TABLE_OR_FIGURE_RE = re.compile(
    r'^(Table|Figure)\s+\d+(\s*[-–.:].*)?$', re.IGNORECASE
)

# 目录页典型行：
#   1 Scope ............................... 1
DOT_LEADER_RE = re.compile(r'\.{2,}\s*\d+\s*$')


def normalize_header_footer(text: str) -> str:
    """对页眉页脚做轻度归一化，去掉页码等，以便统计跨页重复。"""
    t = text.strip()
    t = re.sub(r'\bPage\s+\d+\b', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\(\d{2}/\d{4}\)\s*\d*$', '', t)  # 如 (06/2024) 1
    t = re.sub(r'\d+\s*$', '', t)                 # 末尾纯数字
    return t.strip()


def looks_like_heading_without_number(text: str) -> bool:
    """
    识别无编号但像标题的行，例如：
      "Scope and Field of Application"
      "Normative references"
    """
    t = text.strip()
    if not t:
        return False

    if t.endswith(('.', '!', '?', '：', ':')):
        return False

    if len(t) > 80:
        return False

    words = t.split()
    if len(words) == 0 or len(words) > 12:
        return False

    cap_like = 0
    for w in words:
        if w[0].isupper():
            cap_like += 1
        elif w.isupper() and any(ch.isalpha() for ch in w):
            cap_like += 1

    if cap_like >= max(2, int(len(words) * 0.6)):
        return True
    return False


def merge_text(prev: str, nxt: str) -> str:
    """
    合并同一 block 内的两行：
      - 若上一行以单个 "-" 结尾，则视为断词连接；
      - 否则在中间加空格。
    """
    prev = prev.rstrip()
    nxt = nxt.lstrip()
    if prev.endswith('-') and not prev.endswith('--'):
        return prev[:-1] + nxt
    return prev + ' ' + nxt


# ======================== 英文规则解析：PDF 抽取与清洗 ========================

def extract_lines_from_pdf(pdf_path: Path) -> List[LineRecord]:
    """使用 pdfplumber 抽取每一页的行文本。"""
    lines: List[LineRecord] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
            for idx, raw_line in enumerate(text.split('\n')):
                lines.append(
                    LineRecord(
                        page=pno,
                        index_in_page=idx,
                        text=raw_line.rstrip('\n')
                    )
                )
    return lines


def detect_toc_pages(lines: List[LineRecord]) -> Set[int]:
    """
    检测目录页：如果某页中大量行为 "...... 1" 这种模式，则视为 TOC。
    """
    page_to_texts: Dict[int, List[str]] = {}
    for rec in lines:
        page_to_texts.setdefault(rec.page, []).append(rec.text)

    toc_pages: Set[int] = set()
    for page, texts in page_to_texts.items():
        total = len(texts)
        if total == 0:
            continue
        dot_like = sum(1 for t in texts if DOT_LEADER_RE.search(t))
        if dot_like >= 5 and dot_like / total > 0.3:
            toc_pages.add(page)

    return toc_pages


def detect_headers_footers(
    lines: List[LineRecord],
    toc_pages: Set[int]
) -> Tuple[Set[str], Set[str]]:
    """
    从非 TOC 页中统计跨页重复的顶部/底部行，作为页眉/页脚候选。
    """
    page_to_lines: Dict[int, List[LineRecord]] = {}
    for rec in lines:
        if rec.page in toc_pages:
            continue
        page_to_lines.setdefault(rec.page, []).append(rec)

    num_pages = len(page_to_lines)
    if num_pages <= 1:
        return set(), set()

    header_counter: Dict[str, int] = {}
    footer_counter: Dict[str, int] = {}

    for page, recs in page_to_lines.items():
        if len(recs) < 5:
            continue

        top = [r for r in recs[:5] if r.text.strip()]
        bottom = [r for r in recs[-5:] if r.text.strip()]

        for r in top[:3]:
            key = normalize_header_footer(r.text)
            if not key:
                continue
            header_counter[key] = header_counter.get(key, 0) + 1
        for r in bottom[:3]:
            key = normalize_header_footer(r.text)
            if not key:
                continue
            footer_counter[key] = footer_counter.get(key, 0) + 1

    header_texts = {
        t for t, c in header_counter.items()
        if c >= max(3, num_pages // 2)
    }
    footer_texts = {
        t for t, c in footer_counter.items()
        if c >= max(3, num_pages // 2)
    }
    return header_texts, footer_texts


def _is_hardcoded_header_footer(t: str) -> bool:
    """
    针对目前样本中暴露出的几类特殊页眉/页脚，做硬编码过滤：
      - DICOM PS3.1 页眉
      - RFC Standards Track 页眉
      - ISO/IEC DIR 2 页脚
      - 装饰性 "- Standard -" 行
    """
    plain = t.replace('\u200b', '').strip()  # 去掉零宽空格等

    # DICOM PS3.1 页眉，例如：
    #   "DICOM PS3.1 2025e - Introduction and Overview Page 25"
    #   "Page 28 DICOM PS3.1 2025e - Introduction and Overview"
    if "DICOM PS3.1" in plain and "Page" in plain:
        return True
    if re.search(r'^Page\s+\d+\s+DICOM PS3\.1', plain):
        return True

    # RFC 页眉，例如："Hinden Standards Track [Page 3]"
    if "Standards Track" in plain and "[Page " in plain:
        return True

    # ISO/IEC DIR 2 页脚，例如："– 18 – ISO/IEC DIR 2 RLV © ISO/IEC 2021"
    if "ISO/IEC DIR 2" in plain and "© ISO/IEC" in plain:
        return True

    # 装饰行 "- Standard -"
    if re.match(r'^-+\s*Standard\s*-+\s*$', plain, flags=re.IGNORECASE):
        return True

    return False


def filter_lines(
    lines: List[LineRecord],
    toc_pages: Set[int],
    header_texts: Set[str],
    footer_texts: Set[str]
) -> List[LineRecord]:
    """
    删除目录页、页眉页脚、空行和部分硬编码装饰行，得到“干净行序列”。
    """
    result: List[LineRecord] = []
    for rec in lines:
        if rec.page in toc_pages:
            continue
        t = rec.text.strip()
        if not t:
            continue

        # 硬编码过滤（DICOM / RFC / ISO 页眉页脚等）
        if _is_hardcoded_header_footer(t):
            continue

        norm = normalize_header_footer(t)
        if norm and (norm in header_texts or norm in footer_texts):
            continue

        result.append(
            LineRecord(
                page=rec.page,
                index_in_page=rec.index_in_page,
                text=t
            )
        )
    return result


# ======================== 英文规则解析：行 -> Block ========================

def classify_line(text: str) -> str:
    """
    单行粗分类：
      'heading' | 'list-item' | 'paragraph' | 'other' | 'blank'
    """
    t = text.rstrip()
    if not t.strip():
        return "blank"

    if LIST_ITEM_RE.match(t):
        return "list-item"

    if HEADING_NUM_RE.match(t) or SECTION_HEADING_RE.match(t) \
            or ANNEX_RE.match(t) or ANNEX_SUBSECTION_RE.match(t):
        return "heading"

    if TABLE_OR_FIGURE_RE.match(t):
        return "other"

    if looks_like_heading_without_number(t):
        return "heading"

    return "paragraph"


def merge_lines_to_blocks(lines: List[LineRecord]) -> List[Block]:
    """
    将清洗后的行合并为 block：
      - heading：一般单行成块；
      - paragraph：连续正文合并；
      - list-item：同一项可能跨行，合并；
      - other：单行成块。
    """
    blocks: List[Block] = []
    current: Optional[Block] = None

    for global_idx, rec in enumerate(lines):
        raw = rec.text
        kind = classify_line(raw)

        if kind == "blank":
            if current is not None and current.kind in ("paragraph", "list-item"):
                blocks.append(current)
                current = None
            continue

        if kind == "heading":
            if current is not None:
                blocks.append(current)
                current = None
            blocks.append(
                Block(
                    kind="heading",
                    text=raw.strip(),
                    page_start=rec.page,
                    page_end=rec.page,
                    line_start=global_idx,
                    line_end=global_idx,
                )
            )
            continue

        if kind == "list-item":
            if current is not None and current.kind == "list-item":
                current.text = merge_text(current.text, raw)
                current.page_end = rec.page
                current.line_end = global_idx
            else:
                if current is not None:
                    blocks.append(current)
                current = Block(
                    kind="list-item",
                    text=raw.strip(),
                    page_start=rec.page,
                    page_end=rec.page,
                    line_start=global_idx,
                    line_end=global_idx,
                )
            continue

        if kind == "other":
            if current is not None:
                blocks.append(current)
                current = None
            blocks.append(
                Block(
                    kind="other",
                    text=raw.strip(),
                    page_start=rec.page,
                    page_end=rec.page,
                    line_start=global_idx,
                    line_end=global_idx,
                )
            )
            continue

        if kind == "paragraph":
            if current is not None and current.kind in ("paragraph", "list-item"):
                current.text = merge_text(current.text, raw)
                current.page_end = rec.page
                current.line_end = global_idx
            else:
                if current is not None:
                    blocks.append(current)
                current = Block(
                    kind="paragraph",
                    text=raw.strip(),
                    page_start=rec.page,
                    page_end=rec.page,
                    line_start=global_idx,
                    line_end=global_idx,
                )
            continue

    if current is not None:
        blocks.append(current)

    return blocks


def merge_heading_with_next_short_block(blocks: List[Block]) -> List[Block]:
    """
    在 block 级别做一次“heading + 短尾行”合并，例如：
      [heading] "PS3.10 PS3.4 Transfer"
      [paragraph] "Syntaxes"
    合并为：
      [heading] "PS3.10 PS3.4 Transfer Syntaxes"
    """
    if not blocks:
        return blocks

    merged: List[Block] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.kind == "heading" and i + 1 < len(blocks):
            nb = blocks[i + 1]
            if nb.kind in ("paragraph", "other"):
                word_cnt = len(nb.text.strip().split())
                if 0 < word_cnt <= 3:
                    # 合并
                    b.text = merge_text(b.text, nb.text)
                    b.page_end = nb.page_end
                    b.line_end = nb.line_end
                    merged.append(b)
                    i += 2
                    continue
        merged.append(b)
        i += 1

    return merged


# ======================== 英文规则解析：Block -> 结构树 units ========================

def infer_heading_level(text: str) -> int:
    """
    根据标题文本推断层级 level (>=1)。
    """
    t = text.strip()

    m = HEADING_NUM_RE.match(t)
    if m:
        m2 = re.match(r'^(\d+(?:\.\d+)*)', t)
        if m2:
            num_str = m2.group(1)
            depth = num_str.count('.') + 1
            return max(1, depth)

    m = SECTION_HEADING_RE.match(t)
    if m:
        num_str = m.group(2)
        depth = num_str.count('.') + 1
        return max(1, depth)

    if ANNEX_RE.match(t):
        return 1

    m = ANNEX_SUBSECTION_RE.match(t)
    if m:
        num_str = m.group(2)
        depth = num_str.count('.') + 1
        return max(2, 1 + depth)

    return 1


def build_units_from_blocks(blocks: List[Block]) -> List[Dict]:
    """
    将 block 序列转为 unit 列表：
      - 第一个 block 强行视为 title；
      - heading 通过 infer_heading_level 给 level 并挂父节点；
      - paragraph / list-item / other 挂在当前栈顶节点下面。
    """
    units: List[Dict] = []
    if not blocks:
        return units

    def clean_block_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    stack: List[Tuple[int, int]] = []  # (unit_id, level)

    for idx, block in enumerate(blocks):
        unit_id = len(units)
        text = clean_block_text(block.text)

        if idx == 0:
            unit_type = "title"
            level = 0
            parent_id = None
            units.append(
                {
                    "unit_id": unit_id,
                    "text": text,
                    "type": unit_type,
                    "level": level,
                    "parent_id": parent_id,
                }
            )
            stack = [(unit_id, level)]
            continue

        if block.kind == "heading":
            level = infer_heading_level(text)
            if level < 1:
                level = 1

            while stack and stack[-1][1] >= level:
                stack.pop()
            if not stack:
                stack = [(0, 0)]
            parent_id = stack[-1][0]
            unit_type = "heading"

            units.append(
                {
                    "unit_id": unit_id,
                    "text": text,
                    "type": unit_type,
                    "level": level,
                    "parent_id": parent_id,
                }
            )
            stack.append((unit_id, level))
            continue

        if not stack:
            stack = [(0, 0)]
        parent_id = stack[-1][0]
        parent_level = stack[-1][1]
        level = parent_level + 1

        if block.kind == "list-item":
            unit_type = "list-item"
        elif block.kind == "paragraph":
            unit_type = "paragraph"
        else:
            unit_type = "other"

        units.append(
            {
                "unit_id": unit_id,
                "text": text,
                "type": unit_type,
                "level": level,
                "parent_id": parent_id,
            }
        )

    return units


def build_rule_doc_from_pdf(pdf_path: Path) -> Dict:
    """
    英文规则版：从 PDF 抽取 -> 清洗 -> 行合并 -> 标题合并 -> 构建 doc 对象。
    顶层结构：
      {
        "doc_id": "...",
        "source_type": "A",
        "split": "train",
        "weight": 1.0,
        "units": [...]
      }
    """
    logger.info("[rule] start parse PDF: %s", pdf_path)
    print(f"[rule] {pdf_path.name}")

    try:
        lines = extract_lines_from_pdf(pdf_path)
    except Exception as e:
        logger.error("extract_lines_from_pdf 失败: %s, error=%s", pdf_path, e)
        lines = []

    if not lines:
        logger.warning("未能从 PDF 抽取到任何文本: %s", pdf_path)
        units = [
            {
                "unit_id": 0,
                "text": pdf_path.stem,
                "type": "title",
                "level": 0,
                "parent_id": None,
            }
        ]
    else:
        toc_pages = detect_toc_pages(lines)
        header_texts, footer_texts = detect_headers_footers(lines, toc_pages)
        cleaned_lines = filter_lines(lines, toc_pages, header_texts, footer_texts)
        if not cleaned_lines:
            logger.warning("清洗后无内容: %s", pdf_path)
            units = [
                {
                    "unit_id": 0,
                    "text": pdf_path.stem,
                    "type": "title",
                    "level": 0,
                    "parent_id": None,
                }
            ]
        else:
            blocks = merge_lines_to_blocks(cleaned_lines)
            if not blocks:
                logger.warning("未生成任何 block: %s", pdf_path)
                units = [
                    {
                        "unit_id": 0,
                        "text": pdf_path.stem,
                        "type": "title",
                        "level": 0,
                        "parent_id": None,
                    }
                ]
            else:
                blocks = merge_heading_with_next_short_block(blocks)
                units = build_units_from_blocks(blocks)

    logger.info("[rule] %s: %d units", pdf_path.name, len(units))
    doc = {
        "doc_id": pdf_path.stem.replace("+", "_").replace("-", "_"),
        "source_type": "A",
        "split": "train",
        "weight": 1.0,
        "units": units,
    }
    return doc


# ======================== 结构树规范化 & 校验（通用） ========================

ALLOWED_TOP_KEYS = {"doc_id", "source_type", "split", "weight", "units"}


def canonicalize_tree(data: Dict, fallback_doc_id: str) -> Dict:
    """
    对模型返回的结构树做一轮“规范化”，确保：
      - 仅包含允许的顶层字段；
      - 存在 doc_id/source_type/split/weight/units；
      - units 内的 unit_id 从 0 到 n-1 连续；
      - 只有一个根 title 节点（unit 0），其 parent_id=None, level=0；
      - 其他节点 parent_id 合法（至少指向根），level 合理，type 在允许范围内。
    """
    if not isinstance(data, dict):
        raise ValueError("结构树顶层结果不是 JSON 对象")

    # 删掉多余顶层字段
    for k in list(data.keys()):
        if k not in ALLOWED_TOP_KEYS:
            data.pop(k, None)

    # 顶层字段补全
    if not data.get("doc_id"):
        data["doc_id"] = fallback_doc_id
    data.setdefault("source_type", "A")
    data.setdefault("split", "train")
    data.setdefault("weight", 1.0)

    units = data.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("结果中 units 为空或不存在")

    # 先确保有一个 title 节点在首位
    root_idx = 0
    for i, u in enumerate(units):
        if u.get("type") == "title":
            root_idx = i
            break
    if root_idx != 0:
        root_unit = units[root_idx]
        others = units[:root_idx] + units[root_idx + 1:]
        units = [root_unit] + others

    n = len(units)

    # 旧 id -> 新 id
    old_ids = []
    for idx, u in enumerate(units):
        old_ids.append(u.get("unit_id", idx))
    old_to_new = {old: new for new, old in enumerate(old_ids)}

    for new_id, u in enumerate(units):
        u["unit_id"] = new_id

    root_id = 0
    units[root_id].setdefault("type", "title")
    units[root_id]["parent_id"] = None
    units[root_id]["level"] = 0

    for i, u in enumerate(units):
        if i == root_id:
            continue

        p = u.get("parent_id")
        if isinstance(p, int) and p in old_to_new:
            parent_id = old_to_new[p]
        elif isinstance(p, int) and 0 <= p < n:
            # 已是新 id，但为安全依然保留
            parent_id = p
        else:
            parent_id = root_id

        if parent_id == i:
            parent_id = root_id
        u["parent_id"] = parent_id

        parent_level = units[parent_id].get("level", 0)
        lvl = u.get("level")
        if not isinstance(lvl, int) or lvl <= parent_level:
            u["level"] = parent_level + 1

        if u.get("type") not in ("title", "heading", "paragraph", "list-item", "other"):
            u["type"] = "paragraph"

        if not isinstance(u.get("text"), str):
            u["text"] = ""

    data["units"] = units
    return data


def validate_structure_tree(data: Dict) -> Tuple[bool, List[str]]:
    """
    简单结构校验：
      - units 存在且非空；
      - unit_id 连续；
      - parent_id 合法且无明显 parent 环；
      - 至少有一个 type='title' 的根节点。
    """
    issues: List[str] = []

    if not isinstance(data, dict):
        issues.append("顶层不是 JSON 对象")
        return False, issues

    units = data.get("units")
    if not isinstance(units, list) or not units:
        issues.append("units 为空或不是数组")
        return False, issues

    n = len(units)
    ids: List[int] = []
    title_count = 0
    for i, u in enumerate(units):
        uid = u.get("unit_id")
        if not isinstance(uid, int):
            issues.append(f"unit[{i}] 的 unit_id 非整数或缺失")
        else:
            ids.append(uid)
        if u.get("type") == "title" and u.get("parent_id") is None:
            title_count += 1

    if sorted(ids) != list(range(n)):
        issues.append("unit_id 未从 0 到 n-1 连续编号")

    if title_count == 0:
        issues.append("未检测到任何 parent_id=null 的 title 根节点")
    elif title_count > 1:
        issues.append(f"存在多个 title 根节点: {title_count}")

    id2unit = {u.get("unit_id"): u for u in units if isinstance(u.get("unit_id"), int)}

    for u in units:
        uid = u.get("unit_id")
        p = u.get("parent_id")
        if uid is None:
            continue
        if p is None:
            continue
        if not isinstance(p, int) or p < 0 or p >= n:
            issues.append(f"unit_id={uid} 的 parent_id 非法: {p}")
            continue

        visited = set()
        cur = uid
        steps = 0
        while True:
            parent_id = id2unit.get(cur, {}).get("parent_id")
            if parent_id is None:
                break
            if parent_id in visited:
                issues.append(f"检测到 parent 环，起点 unit_id={uid}")
                break
            visited.add(parent_id)
            cur = parent_id
            steps += 1
            if steps > n:
                issues.append(f"parent 链长度超过 n，可能存在环，起点 unit_id={uid}")
                break

    is_valid = len(issues) == 0
    return is_valid, issues


# ======================== DeepSeek Prompt（英文标准结构重建） ========================

DEEPSEEK_SYSTEM_PROMPT = r"""
You are the structure annotator in the SLAC project for **English technical standards and specifications**:
ISO, IEC, IEEE, ITU-T Recommendations, IETF RFCs, DICOM PS/Parts/Supplements, etc.

You receive an **initial structure tree JSON** that was produced by a rule-based parser from a PDF.
Each unit already contains text snippets of the original document, but:
- titles may be wrong,
- some headings are mis-typed as paragraphs or vice versa,
- heading levels and parent_id may be incorrect,
- page headers/footers or table rows may appear as headings,
- important sections such as "Abstract", "Scope", "Normative references" may not be modeled cleanly.

Your job is to **rewrite this structure tree into a clean, well-formed hierarchy** that follows typical
structure of English standards while preserving all useful content.

--------------------
[Top-level JSON]

You MUST output a single JSON object with EXACTLY the following fields:

{
  "doc_id": "string",
  "source_type": "A",
  "split": "train",
  "weight": 1.0,
  "units": [ ... ]
}

- doc_id: any identifier for the document (file name or standard number is OK). The caller may override it.
- source_type: always "A".
- split: always "train".
- weight: always 1.0.
- units: an array of unit objects described below.

Do NOT output any other top-level keys.

--------------------
[Unit objects]

Each element in "units" MUST be an object with ALL of the following fields:

- "unit_id": integer, unique inside the document, MUST be 0..N-1 in order.
- "text": string, the text content of this structural unit.
- "type": one of:
    - "title"     : the single main title of the whole standard;
    - "heading"   : any other section heading (e.g., "1 Scope", "3 Terms and definitions",
                    "Annex A (normative)", "Appendix B", "2.1 Addressing model", etc.);
    - "paragraph" : normal body paragraphs (possibly merged from several lines);
    - "list-item" : items of a list (bullets, numbered clauses, reference entries like "[1] ...");
    - "other"     : any structural content that must be retained but is neither of the above
                    (e.g., cover page copyright block, whole TOC block, table captions, etc.).
- "level": integer depth level (0, 1, 2, 3, ...). Level 0 is reserved for the single root title.
- "parent_id": integer or null. Root title MUST have parent_id = null.
               All other units MUST point to some parent unit_id in the same array.

If a unit has an unknown or exotic type in the input, map it to one of the 5 allowed types above.

--------------------
[Structural conventions for English standards]

Follow these conventions when rebuilding the tree:

1. **Root title**
   - There MUST be exactly one root unit with:
       type="title", level=0, parent_id=null.
   - Its text should be the main title of the document, such as
       "IP Version 6 Addressing Architecture"
       "Digital Imaging and Communications in Medicine (DICOM)"
       "Recommendation ITU-T F.781.1"

2. **Main top-level sections (level=1, parent=root)**
   - Typical section headings include:
       "Abstract"
       "1 Scope"
       "2 Normative references"
       "3 Terms and definitions"
       "4 Abbreviations and acronyms"
       "Introduction"
       "Annex A (normative)" / "Annex B (informative)"
       "Appendix A"
       "References" / "Bibliography"
   - Numbered clauses like "1", "2", "3", "4", "5", "6", ... should be level=1 under root.
   - "Abstract" may appear before numbered clauses but should also be a level=1 heading.

3. **Numbered subclauses**
   - Use the numbering depth to define the level:
       "2.1"        -> deeper than "2"
       "2.1.1"      -> deeper than "2.1"
       "A.1"        -> deeper than "Annex A"
       "A.1.1"      -> deeper than "A.1"
   - Child clauses must have parent_id pointing to the immediately enclosing clause.

4. **Annexes / Appendices**
   - "Annex A", "Annex B", "Appendix A", etc. should be level=1 headings under the root.
   - Their internal numbered sections (e.g., "A.1", "A.1.1") should form a proper hierarchy
     inside that annex.

5. **Paragraphs and lists**
   - Normal sentences or merged blocks of sentences should be type="paragraph".
   - Bullet/numbered items and reference list entries (like "[1] ...") should be type="list-item"
     under the corresponding section, with level one deeper than the section heading.

6. **Page headers/footers and pure noise**
   - Pure page headers/footers such as:
       "Hinden             Standards Track             [Page 3]"
       "DICOM PS3.1  2025e  - Introduction and Overview  Page 25"
       "– 18 – ISO/IEC DIR 2 RLV © ISO/IEC 2021"
     should usually be **removed**.
   - Only keep them as type="other" if they contain essential information (for example,
     an important "This part supersedes ISO/IEC xxxx" statement). Otherwise they can be dropped.

7. **Tables / Figures**
   - Table or figure captions (e.g., "Table 1 – Summary of ...", "Figure 2 – ...") can be type="other"
     with a reasonable parent section.
   - Internal table rows do not need a precise structural hierarchy; it is acceptable to keep them
     as paragraphs or list-items under the surrounding section.

8. **Front matter**
   - Cover page metadata (publisher, copyright, document type) can be grouped into a few "other" units
     under the root, but should not be mistaken for numbered clauses or main sections.

--------------------
[Task]

You are given an **initial tree JSON** produced by a rule-based parser.
Your job is to produce a **clean final tree JSON** that:

- obeys all the schema constraints above,
- has exactly one root title (level=0, parent_id=null),
- has reasonable heading levels and parent_id,
- removes or downplays noise such as page headers/footers or duplicated titles,
- keeps all important textual content of the standard.

Return ONLY the final JSON object. Do NOT include explanations or comments.
"""


# ======================== DeepSeek 结构评估 Prompt（JSON 结果） ========================

DEEPSEEK_CHECK_SYSTEM_PROMPT = r"""
You are a structural QA checker for **English technical standards** (ISO, IEC, ITU-T, IETF RFC, DICOM, etc.).

You will receive a **structure tree JSON** for one document. Your task is to judge whether the tree is
structurally good enough to be used as training data in an NLP system.

You MUST respond with a single JSON object of the form:

{
  "is_ok": true or false,
  "summary": "one or two sentences summarizing your verdict",
  "issues": ["issue 1", "issue 2", "..."],
  "suggestions": ["concrete fix 1", "concrete fix 2", "..."]
}

Definition of "is_ok":
- true  -> the tree is globally reasonable; remaining issues are minor and do NOT require further
           automatic fixing before using this sample for training.
- false -> there are important structural problems that SHOULD be fixed (for example:
           wrong root title, broken parent_id links, missing main sections, many mis-typed headings/
           levels, heavy noise such as page headers/footers treated as headings, etc.).

Requirements:
- "summary" must be a short free-text explanation (English).
- "issues" is a list of concrete structural problems. It MAY be empty only if is_ok=true.
- "suggestions" is a list of concrete action items that a model could follow to fix the issues.
- Do not include any fields other than: is_ok, summary, issues, suggestions.
"""


# ======================== DeepSeek 调用：结构优化（支持反馈） ========================

def refine_tree_with_deepseek(initial_doc: Dict, pdf_path: Path,
                              feedback: Optional[Dict] = None) -> Dict:
    """
    将当前结构树交给 deepseek-reasoner，根据 System Prompt 做整体重构 / 修复。
    如果 feedback 不为 None，则其中包含上一步结构评估的结果（issues / suggestions），
    只使用“当前结构树 + 评估反馈”，不再引入原始规则树或原文。

    如 deepseek 调用或解析失败，则回退为 initial_doc（最近一次结构版本）。
    """
    if not ENABLE_DEEPSEEK_REFINE:
        return initial_doc

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("REPLACE_WITH"):
        logger.warning("未正确配置 DEEPSEEK_API_KEY，跳过 DeepSeek 优化: %s", pdf_path)
        return initial_doc

    client = get_deepseek_client()
    tree_str = json.dumps(initial_doc, ensure_ascii=False)

    if feedback is None:
        user_prompt = (
            "You are given an initial structure tree JSON for one English technical standard.\n"
            "Please rewrite and improve the tree according to the system instructions:\n"
            "- keep exactly one root title (type='title', level=0, parent_id=null);\n"
            "- fix heading levels and parent_id based on numbering and typical section structure;\n"
            "- treat front matter and noise correctly (page headers/footers should normally be removed);\n"
            "- keep all important content as paragraphs or list-items under appropriate sections.\n\n"
            "Here is the current initial tree JSON:\n"
            f"{tree_str}\n"
        )
    else:
        feedback_str = json.dumps(feedback, ensure_ascii=False)
        user_prompt = (
            "You are given an already optimized structure tree JSON for an English technical standard,\n"
            "together with a structural QA review that lists remaining issues and suggestions.\n"
            "Your job is to produce a **revised** structure tree that resolves those issues while still\n"
            "respecting the schema and conventions in the system message.\n\n"
            "1) Read the current tree carefully.\n"
            "2) Read the QA review JSON (issues + suggestions).\n"
            "3) Apply the suggestions when they are reasonable, but you may also fix additional problems\n"
            "   you detect.\n"
            "4) Output a clean final tree JSON, WITHOUT including the QA review itself.\n\n"
            "Current tree JSON:\n"
            f"{tree_str}\n\n"
            "QA review JSON:\n"
            f"{feedback_str}\n"
        )

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=DEEPSEEK_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        refined_raw = json.loads(content)
    except APIStatusError as e:
        msg = str(e)
        if "400" in msg:
            logger.error(
                "DeepSeek 调用返回 400 错误，保留当前结构树作为最近一次修改结果。文件: %s, error=%s",
                pdf_path,
                e,
            )
        else:
            logger.error(
                "DeepSeek 调用失败（APIStatusError），使用当前结构树作为回退。文件: %s, error=%s",
                pdf_path,
                e,
            )
        return initial_doc
    except RateLimitError as e:
        logger.error(
            "DeepSeek 调用失败（RateLimit），使用当前结构树作为回退。文件: %s, error=%s",
            pdf_path,
            e,
        )
        return initial_doc
    except json.JSONDecodeError as e:
        logger.error("DeepSeek 返回内容无法解析为 JSON，使用当前结构树作为回退。文件: %s, error=%s", pdf_path, e)
        return initial_doc
    except Exception as e:
        logger.error("DeepSeek 调用过程中异常，使用当前结构树作为回退。文件: %s, error=%s", pdf_path, e)
        return initial_doc

    # 规范化 + 校验
    try:
        refined = canonicalize_tree(refined_raw, fallback_doc_id=initial_doc.get("doc_id", pdf_path.stem))
        ok, issues = validate_structure_tree(refined)
        if not ok:
            logger.warning(
                "DeepSeek 优化后的结构树本地校验存在问题，将保留该结果但记录 issues。文件: %s, issues: %s",
                pdf_path,
                "; ".join(issues),
            )
        else:
            logger.info("DeepSeek 优化后的结构树通过本地校验。文件: %s", pdf_path)
        return refined
    except Exception as e:
        logger.error(
            "DeepSeek 结果在规范化/校验阶段失败，回退到当前结构树。文件: %s, error=%s",
            pdf_path,
            e,
        )
        return initial_doc


# ======================== DeepSeek 结构评估（逐文档） ========================

def deepseek_check_tree(tree_doc: Dict, pdf_path: Path) -> Optional[Dict]:
    """
    调用 DeepSeek 对当前结构树进行“结构合理性评估”，返回 JSON：
      {
        "is_ok": true/false,
        "summary": "...",
        "issues": [...],
        "suggestions": [...]
      }
    如调用失败则返回 None。
    """
    if not ENABLE_DEEPSEEK_REFINE:
        return None

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("REPLACE_WITH"):
        logger.warning("未正确配置 DEEPSEEK_API_KEY，跳过 DeepSeek 结构评估: %s", pdf_path)
        return None

    client = get_deepseek_client()
    tree_str = json.dumps(tree_doc, ensure_ascii=False)
    # 防止 prompt 过长，必要时截断
    if len(tree_str) > 60000:
        tree_str = tree_str[:60000] + "\n...(truncated)"

    user_prompt = (
        f"Document file name: {pdf_path.name}\n\n"
        f"Here is the structure tree JSON (possibly truncated):\n\n"
        f"{tree_str}\n\n"
        "Please evaluate the structural soundness ONLY, and respond strictly as a JSON object following "
        "the schema in the system message."
    )

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": DEEPSEEK_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        review = json.loads(content)
        # 粗略校验字段
        if not isinstance(review, dict):
            raise ValueError("check response is not a JSON object")
        if "is_ok" not in review:
            raise ValueError("check response missing 'is_ok'")
        logger.info(
            "[DeepSeek-Check] %s: is_ok=%s, summary=%s",
            pdf_path.name,
            review.get("is_ok"),
            review.get("summary", ""),
        )
        return review
    except APIStatusError as e:
        msg = str(e)
        if "400" in msg:
            logger.error(
                "[DeepSeek-Check] API 400 错误，停止本次评估，保留当前结构树。文件: %s, error=%s",
                pdf_path,
                e,
            )
        else:
            logger.error("[DeepSeek-Check] API 错误 (%s): %s", pdf_path, e)
        return None
    except RateLimitError as e:
        logger.error("[DeepSeek-Check] RateLimit 错误 (%s): %s", pdf_path, e)
        return None
    except json.JSONDecodeError as e:
        logger.error("[DeepSeek-Check] JSON 解析失败 (%s): %s", pdf_path, e)
        return None
    except Exception as e:
        logger.error("[DeepSeek-Check] 调用失败 (%s): %s", pdf_path, e)
        return None


# ======================== 多轮 DeepSeek：规则 -> 多轮生成 + 评估 + 修正 ========================

def refine_with_two_stage_deepseek(rule_doc: Dict, pdf_path: Path) -> Dict:
    """
    对单个文档执行完整的 DeepSeek 多轮流程：
      - round 1: 基于规则结构树做第一次 DeepSeek 重构；
      - round 2..N: 使用上一轮结构树 + 上一轮评估反馈进行再次优化；
      - 每一轮优化后立即进行结构评估；
      - 若某轮评估 is_ok=true，则停止并返回该轮结构树；
      - 若评估失败（包括 HTTP 400）、或达最大轮次，则返回最近一次优化结果。

    注意：第二轮及之后的优化只依赖“当前结构树 JSON + 评估反馈”，不会再引入原始规则树或原文。
    """
    current_doc = rule_doc
    last_review: Optional[Dict] = None

    if not ENABLE_DEEPSEEK_REFINE:
        # 理论上不会走到这里（外层已判断），留个兜底
        try:
            current_doc = canonicalize_tree(
                current_doc,
                fallback_doc_id=rule_doc.get("doc_id", pdf_path.stem)
            )
        except Exception as e:
            logger.error("规则版结构树规范化失败（未使用 DeepSeek）。文件: %s, error=%s", pdf_path, e)
        return current_doc

    for round_idx in range(MAX_DEEPSEEK_ROUNDS):
        feedback = None if round_idx == 0 else last_review

        logger.info(
            "DeepSeek 优化轮次 %d/%d，feedback=%s，文件: %s",
            round_idx + 1,
            MAX_DEEPSEEK_ROUNDS,
            "None" if feedback is None else "present",
            pdf_path,
        )

        # 优化：当前结构树 + 可选反馈
        optimized_doc = refine_tree_with_deepseek(current_doc, pdf_path, feedback=feedback)
        current_doc = optimized_doc

        # 评估
        review = deepseek_check_tree(current_doc, pdf_path)
        if not review:
            logger.info(
                "DeepSeek 结构评估失败或返回空，停止进一步修正，使用当前结构树。文件: %s",
                pdf_path,
            )
            break

        last_review = review
        is_ok = bool(review.get("is_ok", False))
        if is_ok:
            logger.info(
                "DeepSeek 评估通过 (is_ok=true)，在第 %d 轮结束优化。文件: %s",
                round_idx + 1,
                pdf_path,
            )
            break
        else:
            logger.info(
                "DeepSeek 评估 is_ok=false，将在下一轮继续基于本轮评估结果修正结构树（若未达最大轮次）。文件: %s",
                pdf_path,
            )

    # 循环结束后，将最后一轮评估结果完整写入日志（如果存在）
    if last_review is not None:
        try:
            logger.info(
                "[DeepSeek-Final-Review] %s: is_ok=%s, summary=%s, issues=%s, suggestions=%s",
                pdf_path.name,
                last_review.get("is_ok"),
                last_review.get("summary", ""),
                "; ".join(last_review.get("issues") or []),
                "; ".join(last_review.get("suggestions") or []),
            )
        except Exception:
            logger.info(
                "[DeepSeek-Final-Review-RAW] %s: %s",
                pdf_path.name,
                json.dumps(last_review, ensure_ascii=False),
            )

    return current_doc


# ======================== 文件收集 & 保存 ========================

def collect_pdfs(input_path: Path) -> Tuple[Path, List[Path]]:
    """
    输入可以是目录或单个 PDF，返回 (root_dir, pdf_list)。
    """
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"输入文件不是 PDF: {input_path}")
        root_dir = input_path.parent
        pdfs = [input_path]
    else:
        root_dir = input_path
        pdfs = sorted(root_dir.rglob("*.pdf"))
    return root_dir, pdfs


def save_tree_json(doc: Dict, pdf_path: Path, root_dir: Path, out_root: Path) -> Path:
    """
    根据 pdf_path 相对 root_dir 的路径，确定输出 JSON 路径。
    """
    try:
        rel = pdf_path.relative_to(root_dir)
    except ValueError:
        rel = Path(pdf_path.name)

    out_path = out_root / rel.with_suffix(".tree.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    logger.info("已保存结构树 JSON: %s", out_path)
    return out_path


# ======================== 并发任务封装 ========================

def process_one_pdf(pdf_path: Path, root_dir: Path, out_root: Path) -> Optional[Path]:
    """
    单个 PDF 的完整处理流程：
      1）规则构建结构树；
      2）DeepSeek 多轮“重构 + 评估 + 再修正”（如启用）；
      3）规范化 + 校验已在 DeepSeek 流程中完成；
      4）保存最终 JSON。
    由线程池并发调用。
    """
    logger.info("开始处理 PDF: %s", pdf_path)
    print(f"[task] start {pdf_path.name}")

    # 1) 规则版结构树
    rule_doc = build_rule_doc_from_pdf(pdf_path)

    # 2) DeepSeek 优化 + 评估 + 多轮修正
    if ENABLE_DEEPSEEK_REFINE:
        final_doc = refine_with_two_stage_deepseek(rule_doc, pdf_path)
    else:
        final_doc = canonicalize_tree(rule_doc, fallback_doc_id=rule_doc.get("doc_id", pdf_path.stem))
        ok, issues = validate_structure_tree(final_doc)
        if not ok:
            logger.warning(
                "规则版结构树本地校验存在问题（未使用 DeepSeek）。文件: %s, issues: %s",
                pdf_path,
                "; ".join(issues),
            )

    # 3) 保存
    out_path = save_tree_json(final_doc, pdf_path, root_dir=root_dir, out_root=out_root)
    return out_path


# ======================== 主流程（并发调度，限速） ========================

def main() -> None:
    root_dir, pdfs = collect_pdfs(PDF_INPUT_PATH)
    out_root = OUT_ROOT

    if not pdfs:
        logger.warning("未在 %s 下找到任何 PDF 文件。", PDF_INPUT_PATH)
        print(f"[convert] no pdf files found under {PDF_INPUT_PATH}")
        return

    total = len(pdfs)
    logger.info(
        "共找到 %d 个 PDF。输入根：%s，输出根：%s，MAX_WORKERS=%d",
        total,
        root_dir,
        out_root,
        MAX_WORKERS,
    )
    print(f"[convert] found {total} pdf file(s) under {PDF_INPUT_PATH}")

    out_paths: List[Path] = []

    submitted = 0
    completed = 0
    next_index = 0
    last_submit_time = 0.0

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            in_flight = {}  # future -> (idx, pdf_path)

            while completed < total:
                now = time.time()

                # 1) 在有空位时，按“每秒最多 1 个”的速率提交新任务
                if (
                    next_index < total
                    and len(in_flight) < MAX_WORKERS
                    and (now - last_submit_time) >= SUBMIT_INTERVAL
                ):
                    pdf_path = pdfs[next_index]
                    idx = next_index + 1  # 仅用于日志
                    logger.info(
                        "提交任务: 第 %d/%d 个文件: %s (当前在途任务: %d)",
                        idx,
                        total,
                        pdf_path,
                        len(in_flight),
                    )
                    fut = executor.submit(process_one_pdf, pdf_path, root_dir, out_root)
                    in_flight[fut] = (idx, pdf_path)
                    submitted += 1
                    next_index += 1
                    last_submit_time = now

                # 2) 检查已完成的任务
                done_futs = [f for f in list(in_flight.keys()) if f.done()]
                for fut in done_futs:
                    idx, pdf_path = in_flight.pop(fut)
                    try:
                        result_path = fut.result()
                        if result_path is not None:
                            out_paths.append(result_path)
                        else:
                            logger.error("处理 PDF 失败（返回 None）: %s", pdf_path)
                    except Exception as e:
                        logger.exception(
                            "并发任务执行时出现未捕获异常: 文件 %s, 错误: %s",
                            pdf_path,
                            e,
                        )
                    finally:
                        completed += 1
                        logger.info(
                            "进度汇总: 已完成 %d/%d 个文件（当前在途: %d）",
                            completed,
                            total,
                            len(in_flight),
                        )

                # 3) 所有任务都已提交且执行完毕，退出循环
                if next_index >= total and not in_flight:
                    break

                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.warning(
            "检测到手动中断 (KeyboardInterrupt)。当前状态: 已提交 %d 个任务，已完成 %d 个，next_index=%d。",
            submitted,
            completed,
            next_index,
        )
    except Exception as e:
        logger.exception(
            "主调度循环发生未捕获异常，程序中断。当前状态: 已提交 %d 个任务，已完成 %d 个，next_index=%d。错误: %s",
            submitted,
            completed,
            next_index,
            e,
        )

    logger.info(
        "PDF -> 结构树转换完成，共成功 %d 个（总文件: %d）。",
        len(out_paths),
        total,
    )
    print(f"[convert] done, {len(out_paths)} file(s) converted (total {total})")


if __name__ == "__main__":
    main()
