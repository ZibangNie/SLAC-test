import json
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI, APIStatusError, RateLimitError

# ============== 基本配置 ==================

# 建议改为从环境变量读取，而不是硬编码
API_KEY = "sk-403803e6e58941bab12e23eef49d6c3c"

# 目录配置
EN_TXT_DIR = Path(r"D:\code\Github\SLAC-test\data\B_semi-structure\en-tech\txt")
EN_JSON_DIR = Path(r"D:\code\Github\SLAC-test\data\B_semi-structure\en-tech\json")

ZH_TXT_DIR = Path(r"D:\code\Github\SLAC-test\data\B_semi-structure\zh-tech\txt")
ZH_JSON_DIR = Path(r"D:\code\Github\SLAC-test\data\B_semi-structure\zh-tech\json")

# 日志文件
LOG_FILE = Path(r"D:\code\Github\SLAC-test\log\structure_annotation.log")
ERROR_LOG_FILE = Path(r"D:\code\Github\SLAC-test\log\structure_annotation_error.log")

# 模型名称与 token 上限（来自官方文档）
CHAT_MODEL = "deepseek-chat"          # 上限约 8k tokens
REASONER_MODEL = "deepseek-reasoner"  # 上限约 32k tokens

CHAT_TOKEN_LIMIT = 8000
# 阈值设为 6000：估计 token 小于 6000 -> 优先 chat，大于则优先 reasoner
TOKEN_THRESHOLD_FOR_CHAT = 6000

# 重试配置（针对常见 4xx/5xx）
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0  # 秒

# 并发线程数 / 最大同时调用数 —— 按你的要求：512
MAX_WORKERS = 512

# ============== System Prompt（中文结构标注说明） ==================

SYSTEM_PROMPT = r"""
你是 SLAC 项目的文档结构标注器。

你的任务：给定一篇原始文章（弱结构或非结构化文本），综合利用结构信号（空行、缩进、编号、标题格式等）和语义线索（话题切换、层次关系等），重建其层次结构，并输出一个描述结构树的 JSON 对象。

你必须只输出一个合法的 JSON 对象，整体格式示例如下（字段名保持英文，注意输出时不要包含本示例的注释）：

{
  "doc_id": "doc_001",
  "doc_name": "Document title or name if available",
  "language": "en",
  "units": [
    {
      "unit_id": 0,
      "text": "Document title or first main heading",
      "type": "title",
      "level": 0,
      "parent_id": null
    },
    {
      "unit_id": 1,
      "text": "First paragraph ...",
      "type": "paragraph",
      "level": 1,
      "parent_id": 0
    }
  ]
}

不要输出任何额外说明文字，不要使用 Markdown 代码块，只输出 JSON。

--------------------
顶层字段说明

- doc_id：字符串，文档唯一 ID；如无法从上下文获取，可使用占位符（如 "doc_001"）。
- doc_name：文档名称或主标题：
  - 优先使用原文中明显的标题行；
  - 若原文无标题，可使用简短占位名称（例如 "unknown_title"），但不要发明不存在的内容。
- language：文档主要语言，取 "zh"、"en" 或 "mixed" 等简单标记。
- units：结构单元数组，按原文出现顺序依次列出。

--------------------
单元字段与类型（unit）

每个单元对象必须包含字段：

- unit_id：整数，文档内唯一 ID，从 0 开始，逐一递增；
- text：该单元对应的原文片段；
- type：结构类型，只能取以下值之一：
  - "title"     ：整篇文档主标题；
  - "heading"   ：除主标题外的各级标题（章节、小节等）；
  - "paragraph" ：正文段落或语义上合并后的段落；
  - "list-item" ：列表条目（如以 1.、2.、-、*、• 等开头）；
  - "other"     ：不属于以上类型但又必须保留的结构性内容；
- level：整数层级；
- parent_id：父节点的 unit_id，根节点用 null。

禁止使用未列出的其他 type 取值。

--------------------
层级（level）与父子关系（parent_id）

- 根节点（通常对应整个文档的主标题）：
  - type = "title"，level = 0，parent_id = null。
- 直接从属于文档的顶层内容（章节标题或正文块）：
  - 通常 level = 1，parent_id 指向根节点的 unit_id；
  - 如你决定不显式创建根标题节点，可令这些顶层单元 parent_id = null。
- 更深层级：
  - 子标题、子部分等使用 level = 2, 3, ...。

一般建议：
- 子标题的 level = 父标题的 level + 1；
- 正文段落的 level 通常等于父节点的 level 或 level + 1；
- 所有单元都必须满足 level >= 父节点的 level，且不要一次跳 2 级以上（例如从 1 直接跳到 4 是不合理的）。

父子关系规则：

- 每个非根单元都必须有一个明确的父节点，parent_id 为已出现单元的 unit_id；
- 标题下面的段落、子标题，其 parent_id 指向该标题；
- 列表条目的 parent_id 应指向最近的相关标题或引导段落；
- units 数组中的顺序必须与原文顺序一致，父节点必须先于子节点出现；
- 整体结构必须构成一棵有根树，不允许环或孤立节点。

--------------------
文本字段（text）的要求

- text 必须是原文的子串：
  - 可以将相邻的多行/多段按原始顺序拼接为一个单元；
  - 可以在一个长段落内部按语义边界拆分为多个单元，但每个单元对应连续片段。
- 不允许：
  - 改写、压缩、扩展或总结内容；
  - 翻译或改变语言；
  - 添加原文中不存在的新句子或新标题文本。
- 仅允许的轻微修改：
  - 去除首尾多余空白字符；
  - 在合并片段时插入单个换行或空行作为分隔。
- 原文中的拼写错误、标点问题、特殊符号（如 "@-@"）一律照原样保留。

--------------------
结构与语义综合划分原则

切分结构单元时，必须同时考虑形式结构与语义结构，而不是只按物理段落机械切分。

- 段落与块级内容：
  - 空行（连续换行）通常是自然段边界；
  - 如果原文是一整块长文本，可以按句子和话题变化，将多句语义紧密的句子合并为一个 "paragraph" 单元；
  - 非常短且显然是上一句延续的行，可以与前文合并；
  - 对特别长的段落（例如超过数百字或包含多个明显话题的小节），应优先考虑在恰当的语义边界处拆分为 2–3 个 "paragraph" 单元，而不是全部合并在一个单元中。
- 标题识别：
  - 文首独立成行、类似文档名的内容通常可视为 "title"（level = 0）；
  - 形如 "Background"、"Introduction"、"3. Methods" 等独立行，且后面跟随解释说明的，可视为 "heading"；
  - 只能使用原文中存在的文本作为标题内容，不得凭空造标题。
- 列表识别：
  - 以 1.、2.、(a)、-、*、• 等开头的行标为 "list-item"；
  - 列表整体的语义归属由上方最近的相关标题或引导段落决定，将其设为这些单元的父节点。
- 语义驱动的合并 / 拆分（可以跨原始段落边界）：
  - 合并：若多个连续文本片段很短且属于同一主题，可合并为一个 "paragraph"；
  - 拆分：若某段很长且内部存在明显话题切换，可在语义边界处拆成多个 "paragraph"；
  - 在需要跨原始段落边界时，可以将属于同一主题的连续句子视为一个单元，但必须保持全文中句子的先后顺序不变，不得重复或遗漏任何句子。
  - 无论合并或拆分，都必须保证 text 是原文片段的严格拼接或子串。

--------------------
unit_id 编号规则

- unit_id 必须从 0 开始，依次递增（0, 1, 2, ...）；
- 每个单元的 unit_id 必须唯一；
- units 数组中单元的顺序必须与原文出现顺序一致，并与 unit_id 从小到大一致。

--------------------
语言与输出约束

- 文本可能是中文、英文或中英混合；
- text 字段必须保持原文语言，不得翻译或混用；
- 你必须只输出一个 JSON 对象：
  - 不要输出任何解释、注释、示例文字；
  - JSON 语法必须完全正确：键和值使用双引号，内部双引号正确转义，逗号和括号位置正确，无多余尾逗号；
  - 整体内容必须是合法 UTF-8 文本。

牢记：你的职责是依据原文中的实际内容与语义，将文章划分为有层次的结构树，并通过 units 精确记录每个结构单元。不得发明任何不存在的内容，只输出合法 JSON。
"""

# ============== 日志配置 ==================


def setup_loggers():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ERROR_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 主 logger
    logger = logging.getLogger("structure")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 用 utf-8-sig，方便 Windows 记事本正确识别中文
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
    err_logger = logging.getLogger("structure_error")
    err_logger.setLevel(logging.INFO)
    if not err_logger.handlers:
        fh_err = logging.FileHandler(ERROR_LOG_FILE, mode="a", encoding="utf-8-sig")
        fh_err.setLevel(logging.INFO)
        fh_err.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        err_logger.addHandler(fh_err)

    return logger, err_logger


# ============== DeepSeek 客户端 ==================

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

# ============== Token 数估计 ==================


def estimate_tokens(text: str) -> int:
    """
    粗略估计 token 数：
    - 中文：大致 1 字 ≈ 1 token
    - 其它字符（字母、空格、标点等）：大致 4 个字符 ≈ 1 token

    只是为了在 chat / reasoner 之间做分类，不要求特别精确。
    """
    chinese_count = 0
    other_count = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            chinese_count += 1
        else:
            other_count += 1

    est_tokens = chinese_count + other_count / 4.0
    return int(est_tokens)


# ============== 结构树校验（闭环 + level + 超长段落） ==================


def validate_structure_tree(data) -> (bool, list):
    """
    对模型返回的 JSON 结构树做基本一致性检查：
    - units 存在且非空；
    - unit_id 从 0 到 n-1 连续；
    - parent_id 合法且无环；
    - level 合理（>=0，子节点不比父节点更浅，不一次跳两级以上）；
    - 段落超长时给出提示（用于判定是否需要 reasoner 深度重跑）。

    返回:
      (is_valid: bool, issues: List[str])
    """
    issues = []

    if not isinstance(data, dict):
        issues.append("顶层不是 JSON 对象")
        return False, issues

    units = data.get("units")
    if not isinstance(units, list) or not units:
        issues.append("units 为空或不是数组")
        return False, issues

    n = len(units)

    # 检查 unit_id 连续性
    ids = []
    for i, u in enumerate(units):
        uid = u.get("unit_id")
        if not isinstance(uid, int):
            issues.append(f"unit[{i}] 的 unit_id 非整数或缺失")
        else:
            ids.append(uid)
    if sorted(ids) != list(range(n)):
        issues.append("unit_id 未从 0 到 n-1 连续编号")

    # 构建 id -> unit 映射
    id2unit = {}
    for u in units:
        uid = u.get("unit_id")
        if isinstance(uid, int):
            id2unit[uid] = u

    # 检查 parent_id、level 和根节点数量
    root_count = 0
    for u in units:
        uid = u.get("unit_id")
        lvl = u.get("level")
        p = u.get("parent_id")
        utype = u.get("type")

        if not isinstance(lvl, int) or lvl < 0:
            issues.append(f"unit_id={uid} 的 level 非法: {lvl}")

        if p is None:
            root_count += 1
            # 根节点通常 level 为 0 或 1
            if lvl not in (0, 1):
                issues.append(f"根节点 unit_id={uid} 的 level 不在 [0,1]")
        else:
            if not isinstance(p, int) or p < 0 or p >= n:
                issues.append(f"unit_id={uid} 的 parent_id 非法: {p}")
                continue

    if root_count == 0:
        issues.append("未检测到任何 parent_id=null 的根节点")
    elif root_count > 3:
        issues.append(f"根节点数量过多: {root_count}")

    # 检查无环和 level 相对关系
    for u in units:
        uid = u.get("unit_id")
        lvl = u.get("level")
        p = u.get("parent_id")

        # 父子 level 关系
        if p is not None and isinstance(p, int) and p in id2unit:
            parent = id2unit[p]
            pl = parent.get("level", 0)
            ptype = parent.get("type")

            if isinstance(lvl, int) and isinstance(pl, int):
                if lvl < pl:
                    issues.append(
                        f"unit_id={uid} 的 level={lvl} 小于父节点 {p} 的 level={pl}"
                    )
                if lvl > pl + 2:
                    issues.append(
                        f"unit_id={uid} 的 level={lvl} 相对父节点 {p} 的 level={pl} 跳级过大"
                    )
                # 额外：段落挂在 heading 下时，限制更严格一些
                if u.get("type") == "paragraph" and ptype == "heading":
                    if lvl < pl or lvl > pl + 1:
                        issues.append(
                            f"paragraph unit_id={uid} 相对 heading 父节点 {p} 的 level 不合理"
                        )

        # 简单检测是否存在 parent 环（祖先链回到自身）
        visited = set()
        cur = uid
        steps = 0
        while True:
            parent_id = id2unit.get(cur, {}).get("parent_id")
            if parent_id is None:
                break
            if parent_id in visited:
                issues.append(f"检测到环状 parent 链，涉及 unit_id={uid}")
                break
            visited.add(parent_id)
            cur = parent_id
            steps += 1
            if steps > n:
                issues.append(f"parent 链长度超过 n，可能存在环，起点 unit_id={uid}")
                break

    # 检查是否存在“明显超长”的段落，用于触发 reasoner 深度重跑
    LONG_PARAGRAPH_THRESHOLD = 4000  # 字符级阈值，保守一点
    has_long_paragraph = False
    for u in units:
        if u.get("type") == "paragraph":
            t = u.get("text", "")
            if isinstance(t, str) and len(t) > LONG_PARAGRAPH_THRESHOLD:
                has_long_paragraph = True
                issues.append(
                    f"检测到超长段落 unit_id={u.get('unit_id')}，长度约 {len(t)} 字符"
                )
                break

    # 只要有 issues 就认为需要进一步处理（用 reasoner 重跑）
    is_valid = len(issues) == 0
    return is_valid, issues


# ============== 调用模型（含常见 HTTP 码重试） ==================


def call_model(model_name: str, text: str, logger: logging.Logger):
    """
    调用指定模型，返回解析后的 JSON。
    - 对常见的 429 / 5xx 做重试（指数退避）；
    - 其他异常直接抛出，由上层 fallback / 记录。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("调用模型 %s，第 %d 次尝试", model_name, attempt)

            kwargs = dict(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            # deepseek-chat 显式设置 max_tokens=8000
            if model_name == CHAT_MODEL:
                kwargs["max_tokens"] = CHAT_TOKEN_LIMIT

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            data = json.loads(content)
            return data

        except RateLimitError as e:
            # 对应 429 — 频率限制
            logger.warning(
                "模型 %s 调用触发 RateLimitError（可能是 429），第 %d/%d 次重试。错误: %r",
                model_name,
                attempt,
                MAX_RETRIES,
                e,
            )
            if attempt == MAX_RETRIES:
                raise
            # 指数退避
            time.sleep(BASE_RETRY_DELAY * (2 ** (attempt - 1)))

        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            # 常见需要重试的 HTTP 码：429 + 一些 5xx
            if status in (408, 429, 500, 502, 503, 504):
                logger.warning(
                    "模型 %s 返回状态码 %s，第 %d/%d 次重试。错误: %r",
                    model_name,
                    status,
                    attempt,
                    MAX_RETRIES,
                    e,
                )
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(BASE_RETRY_DELAY * (2 ** (attempt - 1)))
            else:
                logger.error(
                    "模型 %s 调用失败，状态码 %s，不重试。错误: %r",
                    model_name,
                    status,
                    e,
                )
                raise

        except json.JSONDecodeError as e:
            snippet = ""
            try:
                snippet = content[:500].replace("\n", " ")
            except Exception:
                pass
            logger.error(
                "模型 %s 返回内容无法解析为 JSON（不重试）。错误: %r, 片段: %s",
                model_name,
                e,
                snippet,
            )
            raise

        except Exception as e:
            logger.error(
                "模型 %s 调用出现未知错误（不重试）。错误: %r",
                model_name,
                e,
            )
            raise


# ============== 单文件处理（含结构校验 + reasoner 重跑） ==================


def annotate_file(
    input_path: Path,
    output_dir: Path,
    logger: logging.Logger,
    err_logger: logging.Logger,
) -> None:
    """对单个 txt 文件进行结构标注，自动选择模型 + 结构校验 + reasoner 修复。"""
    try:
        text = input_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("读取文件失败: %s, 错误: %r", input_path, e)
        err_logger.error("读取文件失败: %s, 错误: %r", input_path, e)
        return

    if not text.strip():
        logger.warning("文件内容为空，跳过: %s", input_path)
        err_logger.error("文件内容为空，跳过: %s", input_path)
        return

    est_tokens = estimate_tokens(text)
    logger.info("文件 %s 估计 token 数: %d", input_path, est_tokens)

    # 根据估计 token 数决定 primary / fallback
    if est_tokens <= TOKEN_THRESHOLD_FOR_CHAT:
        primary_model = CHAT_MODEL
        fallback_model = REASONER_MODEL
    else:
        primary_model = REASONER_MODEL
        fallback_model = CHAT_MODEL

    logger.info(
        "文件 %s 选择 primary=%s, fallback=%s",
        input_path,
        primary_model,
        fallback_model,
    )

    primary_error = None
    fallback_error = None
    data = None

    # 1) 先用 primary 模型
    try:
        data = call_model(primary_model, text, logger)
    except Exception as e:
        primary_error = e
        logger.error(
            "使用模型 %s 处理失败，将尝试备用模型 %s。文件: %s, 错误: %r",
            primary_model,
            fallback_model,
            input_path,
            e,
        )

    # 2) 若失败，再用 fallback 模型
    if data is None:
        try:
            data = call_model(fallback_model, text, logger)
        except Exception as e2:
            fallback_error = e2
            logger.error(
                "备用模型 %s 处理仍然失败，放弃该文件: %s, 错误: %r",
                fallback_model,
                input_path,
                e2,
            )
            err_logger.error(
                "文件处理失败（两个模型均失败）: %s; primary=%s, fallback=%s; primary_error=%r; fallback_error=%r",
                input_path,
                primary_model,
                fallback_model,
                primary_error,
                fallback_error,
            )
            return

    # 3) 对结构结果做自动校验，如不合格则交给 reasoner 深度重跑
    is_valid, issues = validate_structure_tree(data)
    if not is_valid:
        logger.warning(
            "文件 %s 初次结构校验未通过，将使用 %s 重新标注。问题: %s",
            input_path,
            REASONER_MODEL,
            "; ".join(issues),
        )
        try:
            data2 = call_model(REASONER_MODEL, text, logger)
            is_valid2, issues2 = validate_structure_tree(data2)
            if not is_valid2:
                logger.error(
                    "文件 %s 使用 %s 重跑后结构仍然异常，将放弃该文件。问题: %s",
                    input_path,
                    REASONER_MODEL,
                    "; ".join(issues2),
                )
                err_logger.error(
                    "结构树校验失败（两次）: %s; 首次问题: %s; reasoner 问题: %s",
                    input_path,
                    "; ".join(issues),
                    "; ".join(issues2),
                )
                return
            else:
                data = data2
                logger.info(
                    "文件 %s 经过 %s 深度重跑后结构校验通过，使用修复后的结果。",
                    input_path,
                    REASONER_MODEL,
                )
        except Exception as e:
            logger.error(
                "文件 %s 调用 %s 修复结构时异常，将放弃该文件。错误: %r; 原始问题: %s",
                input_path,
                REASONER_MODEL,
                e,
                "; ".join(issues),
            )
            err_logger.error(
                "调用 reasoner 修复结构时异常: %s; 错误: %r; 原始问题: %s",
                input_path,
                e,
                "; ".join(issues),
            )
            return

    # 4) 有一个结构校验通过的结果，写 JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + ".json")

    # 尝试用文件名补全 doc_id
    try:
        if isinstance(data, dict):
            if "doc_id" not in data or not data.get("doc_id"):
                data["doc_id"] = input_path.stem
    except Exception as e:
        logger.warning("补全 doc_id 时出错: %r; 文件: %s", e, input_path)

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("已保存结构树 JSON: %s", output_path)
    except Exception as e:
        logger.error("写入 JSON 文件失败: %s, 错误: %r", output_path, e)
        err_logger.error("写入 JSON 文件失败: %s, 错误: %r", output_path, e)


# ============== 文件收集 ==================


def collect_input_files():
    """收集 en 和 zh 两个目录下所有 txt 文件，返回 (path, 对应输出目录) 列表。"""
    files_with_outdir = []

    if EN_TXT_DIR.is_dir():
        for p in sorted(EN_TXT_DIR.glob("*.txt")):
            files_with_outdir.append((p, EN_JSON_DIR))

    if ZH_TXT_DIR.is_dir():
        for p in sorted(ZH_TXT_DIR.glob("*.txt")):
            files_with_outdir.append((p, ZH_JSON_DIR))

    return files_with_outdir


# ============== 并行调度（限速 + 最大512并发） ==================


def main():
    logger, err_logger = setup_loggers()

    files = collect_input_files()
    total = len(files)

    if total == 0:
        logger.warning(
            "未在以下目录找到任何 txt 文件: %s, %s",
            EN_TXT_DIR,
            ZH_TXT_DIR,
        )
        return

    logger.info(
        "共找到待处理文件 %d 个。计划最大并发 worker 数: %d（每秒最多启动 1 个任务）",
        total,
        MAX_WORKERS,
    )

    submitted = 0
    completed = 0
    next_index = 0  # 下一个待提交文件的索引
    last_submit_time = 0.0

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            in_flight = {}  # future -> (idx, path)

            while completed < total:
                now = time.time()

                # 1) 如有空位，且距离上次提交已 >= 1 秒，则提交一个新任务
                if (
                    next_index < total
                    and len(in_flight) < MAX_WORKERS
                    and now - last_submit_time >= 1.0
                ):
                    path, out_dir = files[next_index]
                    idx = next_index + 1  # 1-based 计数，仅用于日志
                    submitted += 1
                    logger.info("提交任务: 第 %d/%d 个文件: %s", idx, total, path)

                    fut = executor.submit(
                        annotate_file, path, out_dir, logger, err_logger
                    )
                    in_flight[fut] = (idx, path)

                    next_index += 1
                    last_submit_time = now

                # 2) 检查已有任务是否完成
                done_futs = [f for f in list(in_flight.keys()) if f.done()]
                for fut in done_futs:
                    idx, path = in_flight.pop(fut)
                    try:
                        fut.result()
                    except Exception as e:
                        # annotate_file 内部已经记录了大部分错误，这里再兜一层
                        logger.error(
                            "并发任务执行时出现未捕获异常: 文件 %s, 错误: %r",
                            path,
                            e,
                        )
                        err_logger.error(
                            "并发任务执行时出现未捕获异常: 文件 %s, 错误: %r",
                            path,
                            e,
                        )
                    finally:
                        completed += 1
                        logger.info("进度汇总: 已完成 %d/%d 个文件", completed, total)

                # 3) 所有文件都已提交且都处理完成，退出循环
                if next_index >= total and not in_flight:
                    break

                # 略微 sleep，避免空转占满 CPU
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.warning(
            "检测到手动中断 (KeyboardInterrupt)。当前状态: 已提交 %d 个任务，已完成 %d 个，next_index=%d。",
            submitted,
            completed,
            next_index,
        )
    except Exception as e:
        logger.error(
            "主线程中发生未捕获异常，程序中断。当前状态: 已提交 %d 个任务，已完成 %d 个，next_index=%d。错误: %r",
            submitted,
            completed,
            next_index,
            e,
        )


if __name__ == "__main__":
    main()
