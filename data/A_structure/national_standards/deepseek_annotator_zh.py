# -*- coding: utf-8 -*-
"""
zh_standards_model_to_tree.py

使用 DeepSeek-Reasoner 将「中文国家标准 / 行业标准 PDF」直接标注为
SLAC A 类结构树 JSON。

- 输入：PDF_INPUT_PATH（可以是单个 PDF 文件，也可以是目录）
- 输出：D:\code\Github\SLAC-test\data\A_structure\national_standards_structure\zh
- 模型：deepseek-reasoner（max_tokens=64000）
- 并发：最多 512 个线程并行处理不同 PDF
- 机制：
    * 每个 PDF 调用 reasoner 生成结构树 JSON；
    * 本地做结构校验 validate_structure_tree；
    * 若不通过，重新调用 reasoner 再跑一次；
    * 若仍失败，记录错误日志并跳过该文件；
    * 所有 JSON 生成完毕后，随机抽样部分文件，
      用 DeepSeek 再做一次“结构合理性审查”，结果写到主日志。
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pdfplumber
from openai import OpenAI, APIStatusError, RateLimitError

# ================== 基本配置 ==================

API_KEY = "sk-403803e6e58941bab12e23eef49d6c3c"

# 输入可以是单个 PDF 文件，也可以是目录
PDF_INPUT_PATH = Path(
    r"D:\code\Github\SLAC-test\data\A_structure\national_standards\zh\GBT+4706.36-2024.pdf"
)

# 输出目录（A 类结构树）
OUT_ROOT = Path(
    r"D:\code\Github\SLAC-test\data\A_structure\national_standards_structure\zh"
)

# 日志目录 / 文件
LOG_DIR = Path(r"D:\code\Github\SLAC-test\log")
LOG_FILE = LOG_DIR / "zh_standards_model_structure.log"
ERROR_LOG_FILE = LOG_DIR / "zh_standards_model_structure_error.log"

# DeepSeek 配置
DEEPSEEK_MODEL = "deepseek-reasoner"
DEEPSEEK_MAX_TOKENS = 64000

# 并发线程数（最多同时处理 512 个文件）
MAX_WORKERS = 512

# DeepSeek 抽查数量
DEEPSEEK_SAMPLE_SIZE = 10

# ================== 标注 System Prompt（专门针对国家标准） ==================

ANNOTATE_SYSTEM_PROMPT = r"""
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

请按中国国家标准 / 行业标准的典型编排习惯来建树。常见顺序为：

1. 封面信息：标准号、发布机关、发布日期、实施日期等；
2. 标准主标题（中文）以及可能的英文标题；
3. 目次 / 目录（“目次”“目录”等）；
4. 前言（“前言”）；
5. 引言（如有，“引言”）；
6. 正文章节：例如
   - 1 范围
   - 2 规范性引用文件
   - 3 术语和定义
   - 4 总则 / 一般要求
   - 5 试验的一般条件
   - ...
7. 附录：如“附录 A（规范性）XXXX”、“附录 N”、“附录 P”等；
8. 参考文献。

结构约定：

- 整个文档应有且只有一个主标题节点：
  - type = "title"，level = 0，parent_id = null；
  - text 使用标准的中文主标题（如能识别）。
- “目次 / 目录”：
  - 作为一个 heading 节点，通常 level = 1，parent_id 指向 title；
  - 目录具体项目（章节标题 + 页码）不要逐条当做 heading；
  - 建议将整个目录内容作为一个整体 children 单元（type="other" 或 "paragraph" 均可，但推荐 "other"）。
- “前言” / “引言”：
  - 各自作为一个 heading 节点，通常 level = 1，parent_id 指向 title；
  - 前言正文、引言正文的段落全部作为 paragraph 子节点挂在相应 heading 下。
  - 注意：正文中不应出现多个前言 / 引言 heading，避免重复。
- 正文一级章节（章）：
  - 形如 “1 范围”“2 规范性引用文件”“3 术语和定义”等独立成行内容：
    - type = "heading"；
    - level 通常 = 1，parent_id 指向 title；
  - 同一层级的章节应该使用相同的 level。
- 下级条款（条 / 款）：
  - 形如 “3.1 XXXX”“3.1.2 XXXX”“5.5 增加:”“7.1 增加”等：
    - 作为 heading 节点；
    - level = 父章节的 level + 1；
    - parent_id 指向上一级对应的 heading，例如：
      - “3.1” 的父节点是 “3 术语和定义”；
      - “3.1.2” 的父节点是 “3.1”；
      - “5.5 增加” 的父节点是 “5 试验的一般条件”；
      - “5.101 器具即使装有电动机...” 的父节点是 “5.10 增加”。
  - 当出现紧凑编号（无点号）如 “51 纸吸管”“55 增加”“61 代替”“314 增加”等时：
      - 结合上下文判断它们属于哪个章节，例如 “5 要求” 下的 “51 纸吸管”“55 增加” 应作为 “5” 的子 heading；
      - 3 位数字如 314 往往表示 3.1.4，应作为 “3.1” 或 “3 总则”下的子条款；
      - 重点是父子关系正确：这些 heading 的 parent_id 应指向对应章 / 条，而不是直接挂在根节点。
- 条款正文：
  - 某条款下的所有说明性文本、试验步骤、注释（“注 101: ...”）等，直到下一个条款标题出现为止，都挂在该条款 heading 下面，作为 paragraph 子节点。
  - 对于非常细碎的行（只有几个字或一个短语），如有必要可以与前一行合并为同一个 paragraph，以避免过度碎片化。
- 列表条目与参考文献：
  - 以 1)、2)、a)、b)、—、• 等开头的行，标记为 type="list-item"，其 parent_id 指向最近的相关 heading 或 paragraph；
  - “参考文献”作为一个 heading 节点，level 通常 = 1 或 2；
  - 每一条参考文献（例如以“[1]”开头）作为一个 list-item 子节点挂在“参考文献”下。

--------------------
【层级（level）和 parent_id 要求】

- 所有非根节点必须有合法的 parent_id，指向已经出现过的某个 unit_id；
- 层级规则：
  - 根节点（title）：level = 0；
  - 直接从属于文档的顶层内容（如“目次”“前言”“1 范围”等）：level 通常 = 1；
  - 子标题：level = 父节点 level + 1；
  - 段落与列表条目：level 通常等于父节点 level 或父节点 level + 1，但不得小于父节点；
  - 不要出现一次跳 2 级以上（例如从 1 直接跳到 4）的层级。
- 整体结构必须是一棵有根树，不允许：
  - 孤立节点（parent_id 指向不存在的 unit_id）；
  - 环（某个节点通过 parent 链回到自身）。

--------------------
【text 字段的要求】

- text 必须是原文的子串，可以拼接多个相邻行，但不得发明新的内容。
- 不允许：
  - 改写、总结、续写或扩展原文；
  - 翻译语言（中文必须保持中文；如原文含英文则原样保留）。
- 允许的轻微处理：
  - 去除首尾多余空白；
  - 在合并多行内容时，用换行符 "\n" 连接。

--------------------
【输出格式与约束】

- 你必须只输出一个 JSON 对象；
- 不要输出任何解释性文字、注释或 Markdown 代码块；
- JSON 必须是合法语法：
  - 所有键和值使用双引号；
  - 字段之间用逗号分隔，无多余尾逗号；
  - 内容为合法 UTF-8 文本。
"""

# ================== DeepSeek 抽查 System Prompt（结构合理性检查） ==================

CHECK_SYSTEM_PROMPT = (
    "你是熟悉中国国家标准（GB/T）和行业标准结构体例的专家。"
    "现在给你某份标准文档的“结构树 JSON”（包含标题/段落及层级信息）。"
    "请只从“结构合理性”角度进行审查：\n"
    "1）先给出总体结论（结构基本合理 / 基本不合理 / 不完整等）；\n"
    "2）指出最显著的结构问题（例如：前言/引言位置错误或重复，章节顺序混乱，"
    "条款编号层级错误，附录挂在错误父节点，参考文献位置不当，目录被误当正文等）；\n"
    "3）如有必要，可简要建议如何修改；\n"
    "不需要评价技术内容的正确性。"
)

# ================== 日志配置 ==================


def setup_loggers() -> Tuple[logging.Logger, logging.Logger]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 主 logger
    logger = logging.getLogger("zh_standards_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh_main = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8-sig")
    fh_main.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh_main.setFormatter(fmt)
    logger.addHandler(fh_main)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 错误 logger
    err_logger = logging.getLogger("zh_standards_model_error")
    err_logger.setLevel(logging.INFO)
    err_logger.handlers.clear()
    fh_err = logging.FileHandler(ERROR_LOG_FILE, mode="a", encoding="utf-8-sig")
    fh_err.setLevel(logging.INFO)
    fh_err.setFormatter(fmt)
    err_logger.addHandler(fh_err)

    return logger, err_logger


logger, err_logger = setup_loggers()

# ================== DeepSeek 客户端 ==================

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

# ================== 工具函数 ==================


def extract_text_from_pdf(pdf_path: Path) -> str:
    """使用 pdfplumber 从 PDF 中抽取整篇文本，按页面顺序拼接。"""
    texts: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    texts.append(t)
    except Exception as e:
        logger.error("pdfplumber 读取 PDF 失败: %s, 错误: %r", pdf_path, e)
        err_logger.error("pdfplumber 读取 PDF 失败: %s, 错误: %r", pdf_path, e)
        return ""
    return "\n\n".join(texts)


def estimate_tokens(text: str) -> int:
    """粗略估计 token 数，用于日志记录（不参与模型选择）。"""
    chinese_count = 0
    other_count = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            chinese_count += 1
        else:
            other_count += 1
    est_tokens = chinese_count + other_count / 4.0
    return int(est_tokens)


# ================== 结构树校验 ==================


def validate_structure_tree(data: Dict) -> Tuple[bool, List[str]]:
    """
    对模型返回的 A 类结构树做基本一致性检查：
    - 顶层必须为 dict，包含 units；
    - units 为非空数组；
    - unit_id 从 0 到 n-1 连续；
    - type 只能是 {title, heading, paragraph, list-item, other}；
    - parent_id 合法且无环；
    - level 合理（>=0，子节点不比父节点更浅，不一次跳两级以上）；
    - 至少有一个 title，且通常不应超过 2 个。
    """
    issues: List[str] = []

    if not isinstance(data, dict):
        issues.append("顶层 JSON 不是对象(dict)")
        return False, issues

    units = data.get("units")
    if not isinstance(units, list) or not units:
        issues.append("units 缺失或为空")
        return False, issues

    n = len(units)
    ids: List[int] = []
    id2unit: Dict[int, Dict] = {}

    allowed_types = {"title", "heading", "paragraph", "list-item", "other"}

    root_count = 0
    title_count = 0

    # 收集 unit_id、检查类型
    for i, u in enumerate(units):
        uid = u.get("unit_id")
        if not isinstance(uid, int):
            issues.append(f"unit[{i}] 的 unit_id 不是整数或缺失")
            continue
        ids.append(uid)
        id2unit[uid] = u

        t = u.get("type")
        if t not in allowed_types:
            issues.append(f"unit_id={uid} 的 type 非法: {t}")

    # unit_id 应该从 0 到 n-1 连续
    if sorted(ids) != list(range(n)):
        issues.append("unit_id 未从 0 到 n-1 连续编号")

    # 检查 parent_id 和 level
    for u in units:
        uid = u.get("unit_id")
        if uid is None:
            continue

        lvl = u.get("level")
        p = u.get("parent_id")
        t = u.get("type")

        if not isinstance(lvl, int) or lvl < 0:
            issues.append(f"unit_id={uid} 的 level 非法: {lvl}")

        if t == "title":
            title_count += 1

        if p is None:
            root_count += 1
            # 根节点建议 level 为 0 或 1
            if isinstance(lvl, int) and lvl not in (0, 1):
                issues.append(f"根节点 unit_id={uid} 的 level 不在 [0,1]")
        else:
            if not isinstance(p, int) or p < 0 or p >= n:
                issues.append(f"unit_id={uid} 的 parent_id 非法: {p}")
                continue

    if root_count == 0:
        issues.append("未检测到任何 parent_id=null 的根节点")
    if root_count > 5:
        issues.append(f"根节点数量过多: {root_count}")

    if title_count == 0:
        issues.append("未找到任何 type='title' 的单元")
    if title_count > 2:
        issues.append(f"type='title' 的单元数量过多: {title_count}")

    # 检查父子 level 关系 + 检查是否有环
    for u in units:
        uid = u.get("unit_id")
        if uid is None:
            continue
        lvl = u.get("level")
        p = u.get("parent_id")

        # 父子 level 关系
        if p is not None and isinstance(p, int) and p in id2unit:
            parent = id2unit[p]
            pl = parent.get("level", 0)

            if isinstance(lvl, int) and isinstance(pl, int):
                if lvl < pl:
                    issues.append(
                        f"unit_id={uid} 的 level={lvl} 小于父节点 {p} 的 level={pl}"
                    )
                if lvl > pl + 2:
                    issues.append(
                        f"unit_id={uid} 的 level={lvl} 相对父节点 {p} 的 level={pl} 跳级过大"
                    )

        # 检查环
        visited = set()
        cur = uid
        steps = 0
        while True:
            parent_id = id2unit.get(cur, {}).get("parent_id")
            if parent_id is None:
                break
            if parent_id in visited:
                issues.append(f"检测到 parent 环，涉及 unit_id={uid}")
                break
            visited.add(parent_id)
            cur = parent_id
            steps += 1
            if steps > n:
                issues.append(f"parent 链长度超过 n，可能存在环，起点 unit_id={uid}")
                break

    is_valid = len(issues) == 0
    return is_valid, issues


# ================== 调用 DeepSeek 标注 ==================


def call_deepseek_annotate(text: str, pdf_name: str) -> Dict:
    """
    调用 deepseek-reasoner 生成结构树 JSON。
    只用一个模型，不做模型切换，必要时由上层重试第二次。
    """
    messages = [
        {"role": "system", "content": ANNOTATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"文件名：{pdf_name}\n\n以下是该标准的全文文本，请为其生成 A 类结构树 JSON：\n\n{text}",
        },
    ]

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=DEEPSEEK_MAX_TOKENS,
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    return data


# ================== DeepSeek 结构抽查 ==================


def deepseek_check_one(tree_doc: Dict, pdf_name: str) -> None:
    tree_str = json.dumps(tree_doc, ensure_ascii=False)
    # 为了防止 prompt 过长，必要时截断一部分（只是抽查）
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
                {"role": "system", "content": CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=DEEPSEEK_MAX_TOKENS,
        )
        content = resp.choices[0].message.content or ""
        logger.info("[DeepSeek-Check] ===== %s =====\n%s\n", pdf_name, content)
    except Exception as e:
        logger.error("[DeepSeek-Check] 调用失败 (%s): %r", pdf_name, e)
        err_logger.error("[DeepSeek-Check] 调用失败 (%s): %r", pdf_name, e)


def deepseek_random_check(out_paths: List[Path]) -> None:
    if not out_paths:
        logger.info("没有可供 DeepSeek 抽查的 JSON 文件。")
        return
    n = min(DEEPSEEK_SAMPLE_SIZE, len(out_paths))
    samples = random.sample(out_paths, n)
    logger.info("开始 DeepSeek 抽查，共 %d 个样本。", n)

    for json_path in samples:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                tree_doc = json.load(f)
        except Exception as e:
            logger.error("加载 JSON 失败 (%s): %r", json_path, e)
            err_logger.error("加载 JSON 失败 (%s): %r", json_path, e)
            continue
        pdf_name = json_path.stem + ".pdf"
        deepseek_check_one(tree_doc, pdf_name)


# ================== 单文件处理 ==================


def annotate_pdf_file(
    pdf_path: Path,
    root_dir: Path,
) -> Optional[Path]:
    """
    对单个 PDF 进行结构标注：
    - 抽取文本；
    - 调用 deepseek-reasoner 生成结构树；
    - 做结构校验，不通过则重跑一次；
    - 最终通过则写 JSON，并返回输出路径；否则返回 None。
    """
    logger.info("开始处理 PDF: %s", pdf_path)

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        logger.error("PDF 文本为空，跳过: %s", pdf_path)
        err_logger.error("PDF 文本为空，跳过: %s", pdf_path)
        return None

    est_tokens = estimate_tokens(text)
    logger.info("PDF %s 粗略估计 token 数: %d", pdf_path, est_tokens)

    data: Optional[Dict] = None
    first_error: Optional[Exception] = None

    # 第一次调用
    try:
        data = call_deepseek_annotate(text, pdf_name=pdf_path.name)
    except (RateLimitError, APIStatusError) as e:
        first_error = e
        logger.error("第一次调用 DeepSeek 失败 (%s): %r", pdf_path, e)
        err_logger.error("第一次调用 DeepSeek 失败 (%s): %r", pdf_path, e)
    except json.JSONDecodeError as e:
        first_error = e
        logger.error("DeepSeek 返回内容无法解析为 JSON (%s): %r", pdf_path, e)
        err_logger.error("DeepSeek 返回内容无法解析为 JSON (%s): %r", pdf_path, e)
    except Exception as e:
        first_error = e
        logger.error("调用 DeepSeek 标注时出现未知错误 (%s): %r", pdf_path, e)
        err_logger.error("调用 DeepSeek 标注时出现未知错误 (%s): %r", pdf_path, e)

    # 结构校验
    def run_validation(doc: Dict) -> Tuple[bool, List[str]]:
        ok, issues = validate_structure_tree(doc)
        if ok:
            logger.info("结构校验通过: %s", pdf_path)
        else:
            logger.warning(
                "结构校验未通过: %s; 问题: %s",
                pdf_path,
                "; ".join(issues),
            )
        return ok, issues

    # 若第一次有结果，先校验
    if data is not None:
        ok, issues = run_validation(data)
    else:
        ok, issues = False, ["DeepSeek 第一次调用失败或未返回数据"]

    # 若不通过，再重跑一次
    if not ok:
        logger.info("准备对 PDF 进行第二次 DeepSeek 重跑: %s", pdf_path)
        try:
            data2 = call_deepseek_annotate(text, pdf_name=pdf_path.name)
        except Exception as e:
            logger.error(
                "第二次调用 DeepSeek 标注失败 (%s): %r; 首次错误: %r; 首次问题: %s",
                pdf_path,
                e,
                first_error,
                "; ".join(issues),
            )
            err_logger.error(
                "第二次调用 DeepSeek 标注失败 (%s): %r; 首次错误: %r; 首次问题: %s",
                pdf_path,
                e,
                first_error,
                "; ".join(issues),
            )
            return None

        ok2, issues2 = run_validation(data2)
        if not ok2:
            logger.error(
                "PDF 结构两次校验均失败，放弃该文件: %s; 首次问题: %s; 第二次问题: %s",
                pdf_path,
                "; ".join(issues),
                "; ".join(issues2),
            )
            err_logger.error(
                "PDF 结构两次校验均失败，放弃该文件: %s; 首次问题: %s; 第二次问题: %s",
                pdf_path,
                "; ".join(issues),
                "; ".join(issues2),
            )
            return None
        else:
            data = data2

    # 到这里 data 一定是通过校验的结构树
    assert data is not None

    # 补全 doc_id / source_type / split / weight
    try:
        if "doc_id" not in data or not data.get("doc_id"):
            data["doc_id"] = pdf_path.stem
        data.setdefault("source_type", "A")
        data.setdefault("split", "train")
        data.setdefault("weight", 1.0)
    except Exception as e:
        logger.warning("补全顶层字段时出错: %r; 文件: %s", e, pdf_path)

    # 确定输出路径，保持和输入目录的相对结构
    try:
        rel = pdf_path.relative_to(root_dir)
    except ValueError:
        rel = Path(pdf_path.name)

    out_path = OUT_ROOT / rel.with_suffix(".tree.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("已保存结构树 JSON: %s", out_path)
        return out_path
    except Exception as e:
        logger.error("写入 JSON 文件失败: %s, 错误: %r", out_path, e)
        err_logger.error("写入 JSON 文件失败: %s, 错误: %r", out_path, e)
        return None


# ================== 文件收集 ==================


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


# ================== 主流程（并发调度） ==================


def main() -> None:
    root_dir, pdfs = collect_pdfs(PDF_INPUT_PATH)

    if not pdfs:
        logger.warning("未在 %s 下找到任何 PDF 文件。", PDF_INPUT_PATH)
        return

    total = len(pdfs)
    logger.info(
        "共找到待处理 PDF 文件 %d 个。root_dir=%s, 输出根目录=%s, 最大并发=%d",
        total,
        root_dir,
        OUT_ROOT,
        MAX_WORKERS,
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    out_paths: List[Path] = []

    start_time = time.time()

    # 并发提交任务
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pdf = {
            executor.submit(annotate_pdf_file, pdf_path, root_dir): pdf_path
            for pdf_path in pdfs
        }

        completed = 0
        for fut in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[fut]
            try:
                result = fut.result()
                if result is not None:
                    out_paths.append(result)
            except Exception as e:
                logger.error(
                    "并发任务执行时出现未捕获异常: 文件 %s, 错误: %r",
                    pdf_path,
                    e,
                )
                err_logger.error(
                    "并发任务执行时出现未捕获异常: 文件 %s, 错误: %r",
                    pdf_path,
                    e,
                )
            finally:
                completed += 1
                logger.info("进度: 已完成 %d/%d 个文件", completed, total)

    elapsed = time.time() - start_time
    logger.info(
        "PDF -> 结构树转换完成，共成功 %d 个，耗时约 %.1f 秒。",
        len(out_paths),
        elapsed,
    )

    # 随机抽查部分结构树
    if out_paths:
        deepseek_random_check(out_paths)
        logger.info("=== zh_standards_model_to_tree DONE ===")
    else:
        logger.warning("没有成功生成的 JSON 文件，跳过 DeepSeek 抽查。")


if __name__ == "__main__":
    main()
