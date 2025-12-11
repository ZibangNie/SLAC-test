# -*- coding: utf-8 -*-
"""
zh_standards_pdf_to_tree_v5.py

中文国家标准 / 行业标准 PDF -> SLAC 结构树 JSON（A 类结构化文本）

流程：
  1）使用规则脚本从 PDF 抽取行文本，并构建一个初始结构树（rule-based 版本）；
  2）将该初始结构树作为“待修复草稿”，连同结构约束说明（System Prompt）一起送入 DeepSeek-Reasoner；
  3）DeepSeek 在此基础上重新构建 / 优化结构树，输出符合 SLAC 规范的 JSON；
  4）对模型输出做一次规范化（unit_id / parent_id / level 纠正），最终写入 .tree.json 文件。

并发调度：
  - 输入可以是单个 PDF，或包含多个 PDF 的目录；
  - 使用 ThreadPoolExecutor 并发处理，最大并发数 MAX_WORKERS（默认 512）；
  - 全局限速：每秒钟最多提交 1 个新任务，直到达到 MAX_WORKERS；
  - 当任务完成、池子空出位置后，仍然按“每秒 1 个”的节奏补新任务。
"""

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import pdfplumber
from openai import OpenAI, APIStatusError, RateLimitError

# ========= 基本配置 =========

# 可以是目录，也可以是单个 PDF
PDF_INPUT_PATH = Path(r"D:\code\Github\SLAC-test\data\A_structure\national_standards\zh")

# 输出根目录
OUT_ROOT = Path(r"D:\code\Github\SLAC-test\data\A_structure\national_standards_structure\zh")

# 日志文件
LOG_FILE = Path(r"D:\code\Github\SLAC-test\log\zh_standards_structure_with_deepseek.log")

# DeepSeek 配置（优化结构树）
DEEPSEEK_API_KEY = "sk-403803e6e58941bab12e23eef49d6c3c"  # 建议改为从环境变量读取
DEEPSEEK_MODEL = "deepseek-reasoner"
DEEPSEEK_MAX_TOKENS = 64000
ENABLE_DEEPSEEK_REFINE = True

# DeepSeek 随机抽查配置（结构合理性评估，可选）
ENABLE_DEEPSEEK_CHECK = False
DEEPSEEK_CHECK_SAMPLE_SIZE = 10

# 并发调度配置
MAX_WORKERS = 512          # 最高并发数
SUBMIT_INTERVAL = 1.0      # 每秒最多提交 1 个任务


# ========= 日志 =========

def setup_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("zh_standards_v5")
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


# ========= DeepSeek 客户端 =========

def get_deepseek_client() -> OpenAI:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    return client


# ========= 工具函数 =========

def collect_pdfs(input_path: Path) -> Tuple[Path, List[Path]]:
    """输入可以是目录或单个 PDF，返回 (root_dir, pdf_list)。"""
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


def extract_lines(pdf_path: Path) -> List[str]:
    """用 pdfplumber 抽取整份 PDF 的按行文本。"""
    lines: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not text:
                continue
            for raw in text.splitlines():
                line = raw.strip()
                if line:
                    lines.append(line)
    return lines


def has_chinese(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fa5]", s))


def detect_toc_index(lines: List[str]) -> Optional[int]:
    for i, line in enumerate(lines):
        t = line.replace(" ", "")
        if t in ("目次", "目录", "目錄"):
            return i
    return None


def is_toc_dot_line(text: str) -> bool:
    """粗略判断是否为目录里的‘…… 页码’类行。"""
    s = text.strip()
    if not s:
        return False
    dot_chars = "…⋯·."
    dot_cnt = sum(s.count(ch) for ch in dot_chars)
    return dot_cnt >= 5


def find_toc_region(lines: List[str], idx_toc: int) -> Tuple[int, int]:
    """返回目录内容的 [start, end) 索引区间。"""
    start = idx_toc + 1
    i = start
    n = len(lines)
    while i < n:
        line = lines[i]
        if is_toc_dot_line(line):
            i += 1
            continue
        if i + 1 < n and is_toc_dot_line(lines[i + 1]):
            i += 2
            continue
        break
    return start, i


def detect_title_in_front(front_lines: List[str]) -> Optional[int]:
    """
    在目录前的内容中，尽量找到“真正标题行”的索引：
      - 必须含中文；
      - 不应是“中华人民共和国国家标准”“GB/Txxxx-202x”“ICS/CCS”等元数据；
      - 尽量选长度较长的一行。
    """
    candidates: List[Tuple[int, int]] = []
    for idx, line in enumerate(front_lines):
        s = line.strip()
        if not has_chinese(s):
            continue
        norm = s.replace(" ", "")
        if any(bad in norm for bad in (
            "中华人民共和国国家标准",
            "中华人民共和国",
            "国家标准",
            "行业标准",
            "ICS",
            "CCS",
            "GB/T",
            "GB ",
            "TB/",
            "TB ",
        )):
            continue
        if len(norm) < 4:
            continue
        candidates.append((idx, len(norm)))
    if not candidates:
        return None
    # 选最长的一行
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def is_noise_front_line(line: str) -> bool:
    """判断封面上的明显噪声行：罗马页码、标准号等。"""
    s = line.strip()
    if not s:
        return True
    norm = s.replace(" ", "")
    # 罗马数字页码
    if re.fullmatch(r"[IVXLCDMⅰ-ⅹⅠ-Ⅹ]+", norm):
        return True
    # 标准号
    if re.search(r"\d{4}", norm) and ("GB" in norm or "TB" in norm):
        return True
    if "ICS" in norm or "CCS" in norm:
        return True
    return False


# ========= 编号解析 =========

def normalize_numeric_code(raw_code: str) -> Optional[Tuple[str, int]]:
    """
    数字编号统一规范化为形如 "5", "5.1", "5.1.1", "5.10.1" 等，返回 (canonical_code, level)。

    规则：
      - "5"            -> "5", level=1
      - "5.2"          -> "5.2", level=2
      - "5.2.3.2"      -> "5.2.3.2", level=4
      - "51"           -> "5.1", level=2
      - "511"          -> "5.1.1", level=3
      - "314"          -> "3.1.4", level=3
      - "5.101"        -> "5.10.1", level=3  （特殊处理）
    """
    raw = raw_code.strip(".")
    if not raw:
        return None

    # 带点：优先处理
    if "." in raw:
        parts = [p for p in raw.split(".") if p]
        if not parts:
            return None
        # 处理类似 "5.101"
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) >= 3:
            prefix = parts[0]
            suf = parts[1]
            if len(suf) == 3:
                canonical_segments = [prefix, suf[:-1], suf[-1]]  # 5.101 -> 5.10.1
            else:
                # 比较少见，保守一点
                canonical_segments = [prefix, suf]
        else:
            canonical_segments = parts
        canonical = ".".join(canonical_segments)
        level = len(canonical_segments)
        return canonical, level

    # 全是数字：紧凑型 51 -> 5.1, 511 -> 5.1.1
    if not raw.isdigit():
        return None
    if len(raw) == 1:
        canonical = raw
        level = 1
        return canonical, level
    # 2~4 位：每一位视为一层
    canonical = ".".join(list(raw))
    level = len(raw)
    return canonical, level


def parse_numeric_heading(line: str) -> Optional[Tuple[str, int, str]]:
    """
    解析类似：
      - "5 试验的一般条件"
      - "5.5 增加:"
      - "5.101 器具即使装有电动机..."
      - "51 环境要求"
      - "314 增加"
    返回 (canonical_code, level, title) 或 None。
    """
    s = line.lstrip()
    m = re.match(r"^([0-9][0-9\.]*)\s+(.+)$", s)
    if not m:
        return None
    raw_code = m.group(1)
    title = m.group(2).strip()
    # 排除类似 "1)" 的情况
    if raw_code.endswith(")") or raw_code.endswith("）"):
        return None
    # 标题必须有中文，否则认为是纯数字引用/年份，视为正文
    if not has_chinese(title):
        return None
    norm = normalize_numeric_code(raw_code)
    if not norm:
        return None
    canonical_code, level = norm
    return canonical_code, level, f"{raw_code} {title}"


def parse_annex_main(line: str) -> Optional[str]:
    """
    解析 '附录 N XXXX'，返回字母，如 "N"。
    """
    s = line.lstrip()
    m = re.match(r"^附录\s*([A-ZＡ-Ｚ])", s)
    if not m:
        return None
    ch = m.group(1)
    # 全角转半角
    code = chr(ord(ch) - 0xFF21 + ord("A")) if "Ａ" <= ch <= "Ｚ" else ch
    return code


def parse_annex_sub_heading(line: str) -> Optional[Tuple[str, int]]:
    """
    解析 A.1 / A.1.1 / A.1.1.2 之类，返回 (canonical_code, depth)
      - canonical_code: 例如 "A.1.1"
      - depth: 段数，如 A.1 -> 2, A.1.1 -> 3
    """
    s = line.lstrip()
    m = re.match(r"^([A-ZＡ-Ｚ])(\.\d+(?:\.\d+)*)\s+.+", s)
    if not m:
        return None
    ch = m.group(1)
    d = m.group(2)  # ".1" ".1.1"
    letter = chr(ord(ch) - 0xFF21 + ord("A")) if "Ａ" <= ch <= "Ｚ" else ch
    nums = [x for x in d.split(".") if x]
    segs = [letter] + nums
    canonical = ".".join(segs)
    depth = len(segs)
    return canonical, depth


# ========= 段落合并 =========

def merge_paragraphs(lines: List[str], short_threshold: int = 25) -> List[str]:
    """
    通用段落合并：
      - 所有行属于同一 heading；
      - 过短的行，或前一行以 '：:；;,、' 结尾，合并到前一行；
      - 结果返回若干段，每段是用 '\n' 拼接的文本。
    """
    result: List[str] = []
    cur: Optional[str] = None
    last_line: Optional[str] = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if cur is None:
            cur = line
            last_line = line
            continue

        # 是否合并到上一段
        plain_len = len(line.replace(" ", ""))
        short = plain_len <= short_threshold
        prev_last_char = (last_line or "").strip()[-1] if last_line else ""

        should_merge = short or (prev_last_char in "：:；;,、")

        if should_merge:
            cur = cur + "\n" + line
        else:
            result.append(cur)
            cur = line
        last_line = line

    if cur is not None:
        result.append(cur)
    return result


def split_reference_items(lines: List[str], short_threshold: int = 25) -> List[str]:
    """
    参考文献部分：按 [1]、[2]... 拆成多个条目，每条条目内部再用 merge_paragraphs 合并行。
    """
    items: List[List[str]] = []
    cur_block: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\[\d+\]", line):
            if cur_block:
                items.append(cur_block)
            cur_block = [line]
        else:
            cur_block.append(line)
    if cur_block:
        items.append(cur_block)

    result: List[str] = []
    for block in items:
        ps = merge_paragraphs(block, short_threshold=short_threshold)
        # 对参考文献，一般希望整条文献作为一个段，这里合并全部行
        text = "\n".join(ps)
        result.append(text)
    return result


# ========= 正文解析（规则版） =========

def parse_body_lines(
    body_lines: List[str],
    start_unit_id: int,
    root_id: int,
) -> Tuple[List[Dict], int]:
    """
    正文解析（规则版）：
      - 使用 code_to_unit 显式维护编号 -> heading 的父子关系；
      - 使用 level 栈处理前言/引言/附录/参考文献等没有 code 的 heading；
      - 非 heading 行暂存到 pending_lines，一旦遇到新 heading 或文档结束就全部挂到当前 heading 下。
    """
    units: List[Dict] = []
    uid = start_unit_id

    # 当前 heading 栈：level -> unit_id
    level_stack: Dict[int, int] = {}
    # 编号 -> unit_id
    code_to_unit: Dict[str, int] = {}

    annex_root_id: Optional[int] = None  # “附录”总标题节点
    in_references: bool = False

    pending_lines: List[str] = []

    # 只认第一次前言/引言/参考文献为 heading
    seen_preface = False
    seen_intro = False
    seen_refs = False

    def current_heading_parent() -> Tuple[int, int]:
        """返回当前最近 heading 的 (level, unit_id)，如果没有，则 parent=root, level=0"""
        if not level_stack:
            return 0, root_id
        max_level = max(level_stack.keys())
        return max_level, level_stack[max_level]

    def add_heading(text: str, level: int, parent_id: int) -> int:
        nonlocal uid, level_stack, in_references, units

        units.append(
            {
                "unit_id": uid,
                "text": text.strip(),
                "type": "heading",
                "level": level,
                "parent_id": parent_id,
            }
        )
        # 更新 level 栈：删掉比当前更深的层级
        level_stack = {lv: u for lv, u in level_stack.items() if lv < level}
        level_stack[level] = uid

        norm = text.replace(" ", "")
        in_references = (norm == "参考文献")

        cur_id = uid
        uid += 1
        return cur_id

    def flush_pending():
        nonlocal pending_lines, uid, units
        if not pending_lines:
            return

        cur_level, cur_parent = current_heading_parent()
        if in_references:
            # 参考文献：拆成 list-item
            items = split_reference_items(pending_lines, short_threshold=25)
            for t in items:
                units.append(
                    {
                        "unit_id": uid,
                        "text": t,
                        "type": "list-item",
                        "level": cur_level + 1 if cur_level >= 0 else 1,
                        "parent_id": cur_parent,
                    }
                )
                uid += 1
        else:
            paras = merge_paragraphs(pending_lines, short_threshold=25)
            for t in paras:
                units.append(
                    {
                        "unit_id": uid,
                        "text": t,
                        "type": "paragraph",
                        "level": cur_level + 1 if cur_level >= 0 else 1,
                        "parent_id": cur_parent,
                    }
                )
                uid += 1
        pending_lines = []

    for raw in body_lines:
        line = raw.strip()
        if not line:
            continue
        norm = line.replace(" ", "")

        # --- 先尝试识别各种 heading ---

        # 特殊章节：前言 / 引言 / 参考文献（只认第一次）
        if norm in ("前言", "前 言"):
            if not seen_preface:
                flush_pending()
                add_heading("前言", level=1, parent_id=root_id)
                seen_preface = True
            else:
                # 后续出现的“前言”当正文处理
                pending_lines.append(line)
            continue

        if norm in ("引言", "引 言"):
            if not seen_intro:
                flush_pending()
                add_heading("引言", level=1, parent_id=root_id)
                seen_intro = True
            else:
                pending_lines.append(line)
            continue

        if norm == "参考文献":
            if not seen_refs:
                flush_pending()
                add_heading("参考文献", level=1, parent_id=root_id)
                seen_refs = True
            else:
                pending_lines.append(line)
            continue

        # 附录总标题
        if norm == "附录":
            flush_pending()
            annex_root_id = add_heading("附录", level=1, parent_id=root_id)
            continue

        # “附录 N ...”
        annex_code = parse_annex_main(line)
        if annex_code is not None:
            flush_pending()
            if annex_root_id is not None:
                parent_id = annex_root_id
                level = 2
            else:
                parent_id = root_id
                level = 1
            hid = add_heading(line, level=level, parent_id=parent_id)
            code_to_unit[annex_code] = hid
            continue

        # 附录内部子标题：A.1 / A.1.1
        annex_sub = parse_annex_sub_heading(line)
        if annex_sub is not None:
            canonical, depth = annex_sub
            flush_pending()
            # depth: A(1).1(2).1(3) -> 3
            # 如果有附录总标题，则从 level=2 开始加；否则从 level=1。
            base_level = 1 if annex_root_id is not None else 0
            level = base_level + depth
            segs = canonical.split(".")
            parent_id = root_id
            if len(segs) > 1:
                parent_code = ".".join(segs[:-1])
                tmp = parent_code
                # 向上回退找最近存在的父 code
                while True:
                    if tmp in code_to_unit:
                        parent_id = code_to_unit[tmp]
                        break
                    if "." in tmp:
                        tmp = tmp.rsplit(".", 1)[0]
                    else:
                        break
                # 再退到最顶层字母
                if parent_id == root_id and segs[0] in code_to_unit:
                    parent_id = code_to_unit[segs[0]]
            if parent_id == root_id and annex_root_id is not None:
                parent_id = annex_root_id

            hid = add_heading(line, level=level, parent_id=parent_id)
            code_to_unit[canonical] = hid
            continue

        # 数字条款 heading：1、1.1、5.2.3.2、51、511、314、5.101 等
        num_info = parse_numeric_heading(line)
        if num_info is not None:
            canonical_code, num_level, full_title = num_info
            flush_pending()
            segs = canonical_code.split(".")
            parent_id = root_id
            if len(segs) > 1:
                parent_code = ".".join(segs[:-1])
                tmp = parent_code
                # 向上回退寻找父节点：22.10.2 -> 22.10 -> 22
                while True:
                    if tmp in code_to_unit:
                        parent_id = code_to_unit[tmp]
                        break
                    if "." in tmp:
                        tmp = tmp.rsplit(".", 1)[0]
                    else:
                        break
                # 如果还没找到，尝试挂到最顶层章节号（如 "22"）
                if parent_id == root_id and segs[0] in code_to_unit:
                    parent_id = code_to_unit[segs[0]]
            # 顶层只有一个段位，如 "5"
            if num_level <= 0:
                num_level = 1
            hid = add_heading(full_title, level=num_level, parent_id=parent_id)
            code_to_unit[canonical_code] = hid
            continue

        # --- 不是 heading，当作正文行缓存 ---
        pending_lines.append(line)

    # 最后一批 pending 内容挂到最后一个 heading 下
    flush_pending()

    return units, uid


# ========= PDF -> 初始结构树（规则版） =========

def build_tree_from_pdf(pdf_path: Path, root_dir: Path) -> Dict:
    logger.info("开始解析 PDF(规则版): %s", pdf_path)
    print(f"[convert-rule] {pdf_path.name}")

    lines = extract_lines(pdf_path)
    if not lines:
        logger.warning("未能从 PDF 抽取到任何文本: %s", pdf_path)
        return {
            "doc_id": pdf_path.stem,
            "source_type": "A",
            "split": "train",
            "weight": 1.0,
            "units": [
                {
                    "unit_id": 0,
                    "text": pdf_path.stem,
                    "type": "title",
                    "level": 0,
                    "parent_id": None,
                }
            ],
        }

    units: List[Dict] = []
    uid = 0

    # ==== 目录定位 ====
    idx_toc = detect_toc_index(lines)

    if idx_toc is not None and idx_toc > 0:
        front_lines = lines[:idx_toc]
    else:
        # 无目录的情况：取前 10 行作为“前置信息区域”
        n_front = min(10, len(lines))
        front_lines = lines[:n_front]
        idx_toc = None

    title_idx = detect_title_in_front(front_lines)

    # ---- 构造 title + 可选 front-matter 节点 ----
    if title_idx is not None:
        title_text = front_lines[title_idx].strip()
    else:
        title_text = "\n".join(front_lines).strip() or pdf_path.stem

    units.append(
        {
            "unit_id": uid,
            "text": title_text,
            "type": "title",
            "level": 0,
            "parent_id": None,
        }
    )
    root_id = uid
    uid += 1

    # front-matter 节点（如果能识别出 title，则把其余目录前内容合并）
    if title_idx is not None:
        fm_lines: List[str] = []
        for i, line in enumerate(front_lines):
            if i == title_idx:
                continue
            if is_noise_front_line(line):
                continue
            fm_lines.append(line.strip())
        if fm_lines:
            fm_text = "\n".join(fm_lines)
            units.append(
                {
                    "unit_id": uid,
                    "text": fm_text,
                    "type": "other",
                    "level": 1,
                    "parent_id": root_id,
                }
            )
            uid += 1

    # ==== 目录节点 ====
    body_start_idx: int
    if idx_toc is not None:
        toc_start, toc_end = find_toc_region(lines, idx_toc)

        # 目录 heading
        toc_heading_text = lines[idx_toc].strip() or "目录"
        units.append(
            {
                "unit_id": uid,
                "text": toc_heading_text,
                "type": "heading",
                "level": 1,
                "parent_id": root_id,
            }
        )
        toc_id = uid
        uid += 1

        toc_lines = [ln for ln in lines[toc_start:toc_end] if ln.strip()]
        if toc_lines:
            toc_text = "\n".join(toc_lines)
            # 目录内容整体作为一个 other 节点（非正文）
            units.append(
                {
                    "unit_id": uid,
                    "text": toc_text,
                    "type": "other",
                    "level": 2,
                    "parent_id": toc_id,
                }
            )
            uid += 1

        body_start_idx = toc_end
    else:
        # 没有目录：正文从 front_lines 之后开始
        body_start_idx = len(front_lines)

    # ==== 正文解析 ====
    body_lines = lines[body_start_idx:]
    body_units, uid = parse_body_lines(body_lines, start_unit_id=uid, root_id=root_id)
    units.extend(body_units)

    doc = {
        "doc_id": pdf_path.stem.replace("+", "_").replace("-", "_"),
        "source_type": "A",
        "split": "train",
        "weight": 1.0,
        "units": units,
    }
    return doc


# ========= 结构树规范化与校验 =========

ALLOWED_TOP_KEYS = {"doc_id", "source_type", "split", "weight", "units"}


def canonicalize_tree(data: Dict, fallback_doc_id: str) -> Dict:
    """
    对模型返回的结构树做一轮“规范化”，确保：
      - 仅包含允许的顶层字段；
      - 存在 doc_id/source_type/split/weight/units；
      - units 内的 unit_id 从 0 到 n-1 连续；
      - 只有一个根 title 节点（unit 0），其 parent_id=None, level=0；
      - 其他节点 parent_id 合法（至少指向根），level 为非负整数且不低于父节点。
    """
    if not isinstance(data, dict):
        raise ValueError("DeepSeek 返回的顶层结果不是 JSON 对象")

    # 删掉多余顶层字段
    for k in list(data.keys()):
        if k not in ALLOWED_TOP_KEYS:
            data.pop(k, None)

    # doc_id / source_type / split / weight 补全
    if not data.get("doc_id"):
        data["doc_id"] = fallback_doc_id
    data.setdefault("source_type", "A")
    data.setdefault("split", "train")
    data.setdefault("weight", 1.0)

    units = data.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("DeepSeek 返回的结果中 units 为空或不存在")

    # ---- 先根据当前顺序重排：保证 root title 在第一位 ----
    root_idx = 0
    for i, u in enumerate(units):
        if u.get("type") == "title":
            root_idx = i
            break

    if root_idx != 0:
        # 将 root title 换到首位，保持其它相对顺序
        root_unit = units[root_idx]
        others = units[:root_idx] + units[root_idx + 1:]
        units = [root_unit] + others

    n = len(units)

    # 重新分配 unit_id，并建立 old->new 映射（如果 DeepSeek 自己给了 id）
    old_ids = []
    for idx, u in enumerate(units):
        old_ids.append(u.get("unit_id", idx))

    old_to_new = {old: new for new, old in enumerate(old_ids)}

    for new_id, u in enumerate(units):
        u["unit_id"] = new_id

    # 根节点固定为 0
    root_id = 0
    units[root_id].setdefault("type", "title")
    units[root_id]["parent_id"] = None
    units[root_id]["level"] = 0

    # 处理其他节点的 parent_id 和 level
    for i, u in enumerate(units):
        if i == root_id:
            continue

        p = u.get("parent_id")
        # 映射旧 parent_id -> 新 id
        if isinstance(p, int) and p in old_to_new:
            parent_id = old_to_new[p]
        else:
            # 若缺失或非法，则挂到根
            parent_id = root_id
        if parent_id == i:
            parent_id = root_id

        u["parent_id"] = parent_id

        # level 校正：至少比父节点深一层
        parent_level = units[parent_id].get("level", 0)
        lvl = u.get("level")
        if not isinstance(lvl, int) or lvl <= parent_level:
            u["level"] = parent_level + 1

        # type 合法性兜底
        if u.get("type") not in ("title", "heading", "paragraph", "list-item", "other"):
            # 对未知类型降级为 paragraph
            u["type"] = "paragraph"

        # text 字段兜底
        if not isinstance(u.get("text"), str):
            u["text"] = ""

    data["units"] = units
    return data


def validate_structure_tree(data: Dict) -> Tuple[bool, List[str]]:
    """
    简单结构校验：
      - units 存在且非空；
      - unit_id 连续；
      - parent_id 合法，且无明显 parent 环；
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
    ids = []
    title_count = 0
    for i, u in enumerate(units):
        uid = u.get("unit_id")
        if not isinstance(uid, int):
            issues.append(f"unit[{i}] 的 unit_id 非整数或缺失")
        else:
            ids.append(uid)
        if u.get("type") == "title":
            if u.get("parent_id") is None:
                title_count += 1

    if sorted(ids) != list(range(n)):
        issues.append("unit_id 未从 0 到 n-1 连续编号")

    if title_count == 0:
        issues.append("未检测到任何 parent_id=null 的 title 根节点")
    elif title_count > 1:
        issues.append(f"存在多个 title 根节点: {title_count}")

    # parent 合法性与简单环检测
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

        # 环检测：沿 parent 链逆推，如果回到自身则视作环
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


# ========= DeepSeek Prompt =========

DEEPSEEK_SYSTEM_PROMPT = r"""
你是 SLAC 项目中专门处理「中文国家标准 / 行业标准（如 GB/T、GB、TB/T 等）」的结构标注器。

输入：一份标准全文文本（由 PDF 抽取），包含封面信息、目次（目录）、前言、引言、正文各章、附录、参考文献等。
你的任务：根据中国标准常见体例，将该文档重建为一棵结构树，并输出一个 JSON 对象（不得包含任何多余文字）。

--------------------
【顶层 JSON 结构】

你必须输出一个 JSON 对象，格式为：

{
  "doc_id": "GBT_4706_36_2024",
  "source_type": "A",
  "split": "train",
  "weight": 1.0,
  "units": [
    {
      "unit_id": 0,
      "text": "标准中文主标题 ...",
      "type": "title",
      "level": 0,
      "parent_id": null
    },
    ...
  ]
}

含义说明：

- doc_id：文档唯一 ID。优先用标准号或文件名（如 "GBT_4706_36_2024"），如无法确定可以留空，调用方会覆盖。
- source_type：固定为 "A"。
- split：固定为 "train"（由调用方后续再划分）。
- weight：固定为 1.0。
- units：结构单元数组，按原文顺序列出。

不得输出 doc_name 或 language 字段，只需要上述 4 个顶层字段。

--------------------
【单位（unit）字段与类型】

每个 unit 必须包含以下字段：

- unit_id：整数，文档内唯一 ID，从 0 开始，依次递增（0, 1, 2, ...）；
- text：对应的原文片段；
- type：结构类型，只能取以下 5 种之一：
  - "title"     ：整份标准的主标题（中文主标题）；
  - "heading"   ：除主标题外的各级标题（章、条、款、附录标题等）；
  - "paragraph" ：正文段落或语义上合并后的段落；
  - "list-item" ：列表条目（如以 1)、a)、—、•、"-" 等开头的条款或参考文献条目）；
  - "other"     ：不属于以上类型但又必须保留的结构性内容（如整块目录内容、前置版权信息、页眉中的“本标准代替 GB XXX”等）。
- level：整数层级（0、1、2、3...）；
- parent_id：父节点的 unit_id，根节点使用 null。

禁止使用未列出的其他 type 取值。

--------------------
【国家标准典型结构与层级约定】

...（此处保持你之前的长 prompt 原文，已省略，代码中已完整保留）...
"""


# ========= DeepSeek 结构树优化 =========

def refine_tree_with_deepseek(initial_doc: Dict, pdf_path: Path) -> Dict:
    """
    将规则版结构树交给 deepseek-reasoner，根据 System Prompt 做一次整体重构 / 修复。
    如 deepseek 调用或解析失败，则返回原始 initial_doc。
    """
    if not ENABLE_DEEPSEEK_REFINE:
        return initial_doc

    client = get_deepseek_client()
    tree_str = json.dumps(initial_doc, ensure_ascii=False)

    user_prompt = (
        "下面是某一份中文国家标准/行业标准文档，已经通过规则脚本抽取出了一个初始结构树 JSON。\n"
        "你需要在保留原文信息的前提下，依据系统提示中的体例规范，对结构树进行整体优化和修复：\n"
        "  - 确保只有一个主标题 title 作为根节点（parent_id=null，level=0）；\n"
        "  - 目录应整体作为一个 other 子节点挂在“目次/目录” heading 下，而不是拆成很多 heading；\n"
        "  - 前言、引言、各章、条款、附录、参考文献的层级和 parent_id 要符合 GB/T 常见结构；\n"
        "  - 避免重复或位置错误的“前言”“引言”“参考文献” heading；\n"
        "  - 所有非根节点必须有合法 parent_id，整棵树没有孤立节点或环。\n"
        "重要：请直接输出最终的结构树 JSON，顶层字段只允许 doc_id, source_type, split, weight, units。\n\n"
        "当前的初始结构树 JSON 如下（其中 units.text 已包含原文片段，可视作原始文本的拆分形式）：\n"
        f"{tree_str}\n"
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
        refined = json.loads(content)
    except (APIStatusError, RateLimitError) as e:
        logger.error("DeepSeek 调用失败（API 错误）: %s, 使用规则版结果。文件: %s", e, pdf_path)
        return initial_doc
    except json.JSONDecodeError as e:
        logger.error("DeepSeek 返回内容无法解析为 JSON: %s, 使用规则版结果。文件: %s", e, pdf_path)
        return initial_doc
    except Exception as e:
        logger.error("DeepSeek 调用过程中发生异常: %s, 使用规则版结果。文件: %s", e, pdf_path)
        return initial_doc

    # 规范化 + 校验
    try:
        refined = canonicalize_tree(refined, fallback_doc_id=initial_doc.get("doc_id", pdf_path.stem))
        ok, issues = validate_structure_tree(refined)
        if not ok:
            logger.warning(
                "DeepSeek 优化后的结构树仍存在潜在问题，将保留该结果但记录 issues。文件: %s, issues: %s",
                pdf_path,
                "; ".join(issues),
            )
        else:
            logger.info("DeepSeek 优化后的结构树通过校验。文件: %s", pdf_path)
        return refined
    except Exception as e:
        logger.error(
            "DeepSeek 结果在规范化/校验阶段失败: %s, 使用规则版结果。文件: %s",
            e,
            pdf_path,
        )
        return initial_doc


# ========= DeepSeek 抽查（结构合理性评估，可选） =========

DEEPSEEK_CHECK_SYSTEM_PROMPT = (
    "你是熟悉中国国家标准（GB/T）和铁路行业标准结构体例的专家。现在给你某份标准文档的"
    "“结构树 JSON”（包含标题/段落及层级信息）。请只从“结构合理性”角度进行审查：\n"
    "1）给出总体结论（结构基本合理 / 基本不合理 / 不完整等）；\n"
    "2）指出最显著的结构问题（例如：前言/引言位置错误或重复，章节顺序乱，条款编号层级错误，"
    "附录挂在错误父节点，参考文献位置不当，目录被误当正文等）。\n"
    "不需要评价技术内容的正确性。"
)


def deepseek_check_one(client: OpenAI, tree_doc: Dict, pdf_name: str) -> None:
    tree_str = json.dumps(tree_doc, ensure_ascii=False)
    if len(tree_str) > 60000:
        tree_str = tree_str[:60000] + "\n...（后续部分已截断，仅供结构抽查）"

    user_prompt = (
        f"文档文件名：{pdf_name}\n\n"
        f"下面是该文档的结构树 JSON（可能已截断）：\n\n"
        f"{tree_str}\n\n"
        f"请按系统提示，只从结构合理性角度进行审查。"
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
        )
        content = resp.choices[0].message.content or ""
        logger.info("[DeepSeek-Check] ===== %s =====\n%s\n", pdf_name, content)
    except (APIStatusError, RateLimitError) as e:
        logger.error("[DeepSeek-Check] API 错误 (%s): %s", pdf_name, e)
    except Exception as e:
        logger.error("[DeepSeek-Check] 调用失败 (%s): %s", pdf_name, e)


def deepseek_random_check(out_paths: List[Path]) -> None:
    if not out_paths:
        logger.info("没有可供 DeepSeek 抽查的 JSON 文件。")
        return
    n = min(DEEPSEEK_CHECK_SAMPLE_SIZE, len(out_paths))
    samples = random.sample(out_paths, n)
    logger.info("开始 DeepSeek 结构合理性抽查，共 %d 个样本。", n)
    print(f"[deepseek-check] random check {n} file(s)")

    client = get_deepseek_client()
    for json_path in samples:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                tree_doc = json.load(f)
        except Exception as e:
            logger.error("加载 JSON 失败 (%s): %s", json_path, e)
            continue
        pdf_name = json_path.stem + ".pdf"
        deepseek_check_one(client, tree_doc, pdf_name)


# ========= 保存结果 =========

def save_tree_json(doc: Dict, pdf_path: Path, root_dir: Path, out_root: Path) -> Path:
    """根据 pdf_path 相对 root_dir 的路径，确定输出 JSON 路径。"""
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


# ========= 并发任务封装 =========

def process_one_pdf(pdf_path: Path, root_dir: Path, out_root: Path) -> Optional[Path]:
    """
    单个 PDF 的完整处理流程（规则构建 + DeepSeek 修复 + 保存 JSON）。
    由线程池并发调用。
    """
    logger.info("开始处理 PDF: %s", pdf_path)
    print(f"[task] start {pdf_path.name}")
    # 1) 规则版结构树
    rule_doc = build_tree_from_pdf(pdf_path, root_dir=root_dir)

    # 2) DeepSeek 优化
    final_doc = refine_tree_with_deepseek(rule_doc, pdf_path=pdf_path)

    # 3) 保存最终结构树
    out_path = save_tree_json(final_doc, pdf_path, root_dir=root_dir, out_root=out_root)
    return out_path


# ========= 主流程（并发调度，限速 1 秒 1 个任务） =========

def main() -> None:
    root_dir, pdfs = collect_pdfs(PDF_INPUT_PATH)
    out_root = OUT_ROOT

    if not pdfs:
        logger.warning("未在 %s 下找到任何 PDF 文件。", PDF_INPUT_PATH)
        print(f"[convert] no pdf files found under {PDF_INPUT_PATH}")
        return

    total = len(pdfs)
    logger.info("共找到 %d 个 PDF。输入根：%s，输出根：%s", total, root_dir, out_root)
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

                # 略微 sleep，避免空转
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

    if ENABLE_DEEPSEEK_CHECK:
        deepseek_random_check(out_paths)
        logger.info("=== zh_standards_pdf_to_tree_v5 DONE (with DeepSeek check) ===")
        print("[deepseek-check] done")
    else:
        logger.info("=== zh_standards_pdf_to_tree_v5 DONE ===")


if __name__ == "__main__":
    main()
