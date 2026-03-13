#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

API_BASE = "https://api.deepseek.com"
CHAT_URL = f"{API_BASE}/chat/completions"
MODEL_NAME = "deepseek-reasoner"


SYSTEM_PROMPT = r"""
你是一个面向铁路标准、铁路规范、工程验收标准、技术规程、法规条文和技术文档的“全文结构标注器”。

你的任务不是摘要，不是翻译，不是解释，也不是改写。
你的任务是：把输入的“原文全文”重组成一个严格、可校验、可程序消费的树形 JSON 结构。

====================
一、输出目标
====================

你必须输出一个完整 JSON object，表示整篇文档的树形结构分段结果。

顶层 JSON 必须具有以下字段：
- doc_id: string
- doc_name: string
- language: string
- units: array

其中 units 是按原文顺序排列的节点数组。每个节点必须包含：
- unit_id: int
- text: string
- type: string
- level: int
- parent_id: int|null

禁止输出：
- markdown
- 解释文字
- 注释
- 代码块围栏
- 任何 JSON 之外的内容

你只能输出“完整 JSON 文件本体”。

====================
二、树结构硬约束
====================

你必须严格满足以下约束：

1. 必须存在且仅存在一个顶层 root 节点：
   - unit_id = 0
   - type = "root"
   - level = 0
   - parent_id = null
   - text 不能为空；如果无法确定标题，可写 "unknown_title"

2. unit_id 必须从 0 开始，连续递增，不能跳号，不能重复。

3. 所有非 root 节点都必须有父节点：
   - parent_id 不能为 null
   - parent_id 必须引用一个更早出现的节点

4. level 必须严格满足：
   - child.level = parent.level + 1
   不能相等，不能跳级。

5. 整个 units 必须能构成一棵单根树：
   - 不能有孤儿节点
   - 不能有多个根
   - 不能断裂
   - 不能形成环

6. 所有节点必须按文档原文顺序输出。

====================
三、文本抽取要求
====================

1. 节点 text 必须尽量保留原文中的“连续文本片段”。
2. 不要总结，不要润色，不要翻译，不要改写。
3. 不要省略关键内容。
4. 不要随意拼接两个相距很远的片段。
5. 如果一个标题、条款号、短标签本来就很短，也允许它单独成为节点。
6. 标准/规范文本中的短结构锚点是正常现象，不要因为短就强行并入别的段落。
7. 尽量不要把一个巨大章节整体塞进一个 paragraph。
8. 也不要把普通正文切得极其碎。

====================
四、类型标注建议
====================

优先使用这些 type：
- root
- title
- chapter
- appendix
- paragraph
- list_item
- table_title
- figure_title
- other

使用建议：
- 文档主标题、标准名：title
- 章/篇/部分：chapter
- 附录：annex / appendix
- 正文说明段：paragraph
- 列表项、枚举项：list_item
- 表题：table_title
- 图题：figure_title
- 注：note
- 无法确定：other

如果不确定类型，宁可用 other，也不要破坏树结构。

====================
五、结构识别偏好（非常重要）
====================

对于铁路标准、工程规范、法规文本，优先识别这些结构信号：

1. 编号标题：
   - 1
   - 1.1
   - 1.1.1
   - 4.4.1
   - 第1章 / 第3节 / 第5条
   - (1) / （一）
   - A.1 / B.2

2. 结构标题：
   - 范围 / 术语 / 定义 / 要求 / 检验 / 施工 / 验收 / 附录
   - Scope / Definitions / Requirements / Annex / Appendix / Note / Table / Figure

3. 枚举或列表项：
   - 以 - / • / · / 数字编号 / 字母编号 开头的行

4. 表题、图题、注释行：
   - 表1、图2、注1、Table 3、Figure 5、Note 等

短编号、短标题、短标签在规范文本中是合法结构，不应被消解掉。

====================
六、全文覆盖要求
====================

1. 输出应覆盖整篇文档的重要结构和正文顺序。
2. 不允许只标少量标题而忽略大量正文。
3. 要求逐字符 100% 覆盖，必须体现“全文结构”。
4. 标题、章节、条款、正文、附录、表题/图题/注释等重要内容应纳入树中。
5. 如果某些 OCR 或原文噪声较重，仍需尽量构造合理树结构，但不要编造新内容。

====================
七、输出模板
====================

输出格式示意如下（仅示意，不要照抄内容）：

{
  "doc_id": "...",
  "doc_name": "...",
  "language": "zh|en|other",
  "units": [
    {
      "unit_id": 0,
      "text": "unknown_title",
      "type": "root",
      "level": 0,
      "parent_id": null
    },
    {
      "unit_id": 1,
      "text": "1 Scope",
      "type": "section",
      "level": 1,
      "parent_id": 0
    },
    {
      "unit_id": 2,
      "text": "This standard specifies ...",
      "type": "paragraph",
      "level": 2,
      "parent_id": 1
    }
  ]
}

再次强调：
你只能输出一个完整 JSON object，不要输出任何解释性文字。
""".strip()


FIRST_USER_PROMPT_TEMPLATE = r"""
请对下面这篇文档进行“全文树形结构标注”，并返回严格 JSON。

文档元信息：
- doc_id: {doc_id}
- doc_name: {doc_name}
- language: {language}

要求：
1. 必须输出完整 JSON object。
2. 必须包含唯一 root 节点。
3. 所有非 root 节点必须有父节点。
4. level 必须严格满足 child.level = parent.level + 1。
5. 节点 text 尽量保留原文连续文本片段。
6. 不要总结、改写、翻译。
7. 短标题、短编号、短标签在规范文本中是合法结构，允许单独成节点。
8. 要尽量覆盖全文的重要结构和正文，而不是只标少数标题。

下面是原始全文：

===== BEGIN DOCUMENT =====
{full_text}
===== END DOCUMENT =====

请直接输出完整 JSON 文件。
""".strip()


REPAIR_PROMPT_TEMPLATE = r"""
你上一次输出的 JSON 没有通过程序校验。请你基于“同一篇原始文档”，修复并重新输出一个能通过校验的完整 JSON 文件。

本次必须修复的问题：
{errors}

你必须特别注意：
1. 输出必须是完整 JSON object。
2. 必须有且仅有一个 root 节点，且 root:
   - unit_id = 0
   - type = "root"
   - level = 0
   - parent_id = null
3. unit_id 必须从 0 连续递增。
4. 所有非 root 节点必须有合法 parent_id。
5. 必须满足 child.level = parent.level + 1。
6. 所有节点必须按原文顺序输出。
7. text 尽量保留原文连续文本，不要改写。
8. 短标题、短编号、短标签允许单独成节点。
9. 要尽量覆盖全文重要结构和正文。

你上一次输出如下：
===== BEGIN PREVIOUS OUTPUT =====
{previous_output}
===== END PREVIOUS OUTPUT =====

原始全文如下：
===== BEGIN DOCUMENT =====
{full_text}
===== END DOCUMENT =====

请只输出修复后的完整 JSON 文件。
""".strip()


@dataclass
class SourceDoc:
    path: Path
    doc_id: str
    doc_name: str
    language: str
    full_text: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate full-text source documents into tree JSON via DeepSeek.")
    p.add_argument("--input_dir", type=str, required=True, help="Directory containing raw text files.")
    p.add_argument("--output_dir", type=str, required=True, help="Directory for outputs.")
    p.add_argument("--glob", type=str, default="*.txt", help="Glob for input files. Example: *.txt")
    p.add_argument("--launch_interval", type=float, default=1.0, help="Launch one new request every N seconds.")
    p.add_argument("--request_timeout", type=float, default=3600.0, help="Per-request timeout in seconds.")
    p.add_argument("--skip_existing", action="store_true", help="Skip docs that already have validated outputs.")

    # 只重试一次：第一次 + 一次 repair
    p.add_argument("--max_attempts", type=int, default=2)

    # 第一次尽量大；如果失败，第二次更大
    p.add_argument("--first_input_chars", type=int, default=120000)
    p.add_argument("--retry_input_chars", type=int, default=220000)
    p.add_argument("--first_max_tokens", type=int, default=32000)
    p.add_argument("--retry_max_tokens", type=int, default=48000)
    return p.parse_args()


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def safe_slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    s = s.strip("._")
    return s[:160] if s else "doc"


def detect_language(text: str) -> str:
    if not text:
        return "other"
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cjk > max(200, latin * 0.5):
        return "zh"
    if latin > max(200, cjk * 0.5):
        return "en"
    return "other"


def guess_doc_name(text: str, fallback: str) -> str:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        return fallback
    first = lines[0]
    if len(first) > 120:
        return first[:120]
    return first


def load_source_doc(path: Path) -> SourceDoc:
    full_text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not full_text:
        raise ValueError(f"{path}: file is empty")

    doc_id = path.stem
    doc_name = guess_doc_name(full_text, fallback=path.stem)
    language = detect_language(full_text)

    return SourceDoc(
        path=path,
        doc_id=doc_id,
        doc_name=doc_name,
        language=language,
        full_text=full_text,
    )


def list_source_docs(input_dir: Path, pattern: str) -> List[SourceDoc]:
    docs: List[SourceDoc] = []
    for path in sorted(input_dir.rglob(pattern)):
        if path.is_file():
            docs.append(load_source_doc(path))
    return docs


def prepare_input_text(full_text: str, char_cap: int) -> str:
    if char_cap <= 0 or len(full_text) <= char_cap:
        return full_text
    return full_text[:char_cap]


def check_text_presence_ratio(source_text: str, units: List[Dict[str, Any]]) -> float:
    src = normalize_ws(source_text).lower()
    if not src:
        return 0.0
    checked = 0
    ok = 0
    for u in units:
        if int(u["unit_id"]) == 0:
            continue
        txt = normalize_ws(str(u.get("text") or "")).lower()
        if not txt:
            continue
        checked += 1
        if txt in src:
            ok += 1
    if checked == 0:
        return 0.0
    return ok / checked


def try_parse_json_object(text: str) -> Tuple[Optional[Any], Optional[str]]:
    text = (text or "").strip()
    if not text:
        return None, "empty_output"

    # 先直接 parse
    try:
        return json.loads(text), None
    except Exception:
        pass

    # 再尝试从第一个 { 到最后一个 } 截取
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        candidate = text[l:r + 1]
        try:
            return json.loads(candidate), None
        except Exception as e:
            return None, f"invalid_json_after_brace_trim: {repr(e)}"

    return None, "no_complete_json_object_found"


def validate_tree_json(data: Any, source_doc: SourceDoc) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    errors: List[str] = []

    if not isinstance(data, dict):
        return False, ["Top-level JSON must be an object."], None

    for key in ["doc_id", "doc_name", "language", "units"]:
        if key not in data:
            errors.append(f"Missing top-level key: {key}")

    if errors:
        return False, errors, None

    units = data.get("units")
    if not isinstance(units, list) or not units:
        errors.append("units must be a non-empty list")
        return False, errors, None

    required_unit_keys = {"unit_id", "text", "type", "level", "parent_id"}
    norm_units: List[Dict[str, Any]] = []
    seen_ids = set()

    for idx, u in enumerate(units):
        if not isinstance(u, dict):
            errors.append(f"units[{idx}] must be an object")
            continue

        miss = required_unit_keys - set(u.keys())
        if miss:
            errors.append(f"units[{idx}] missing keys: {sorted(miss)}")
            continue

        try:
            unit_id = int(u["unit_id"])
        except Exception:
            errors.append(f"units[{idx}].unit_id must be int")
            continue

        try:
            level = int(u["level"])
        except Exception:
            errors.append(f"units[{idx}].level must be int")
            continue

        parent_id = u["parent_id"]
        if parent_id is not None:
            try:
                parent_id = int(parent_id)
            except Exception:
                errors.append(f"units[{idx}].parent_id must be int or null")
                continue

        text = str(u.get("text") or "").strip()
        typ = str(u.get("type") or "").strip()

        if not text:
            errors.append(f"units[{idx}].text is empty")
        if not typ:
            errors.append(f"units[{idx}].type is empty")
        if unit_id in seen_ids:
            errors.append(f"Duplicate unit_id: {unit_id}")
        seen_ids.add(unit_id)

        norm_units.append(
            {
                "unit_id": unit_id,
                "text": text,
                "type": typ,
                "level": level,
                "parent_id": parent_id,
            }
        )

    if errors:
        return False, errors, None

    norm_units.sort(key=lambda x: x["unit_id"])

    expected_ids = list(range(len(norm_units)))
    actual_ids = [u["unit_id"] for u in norm_units]
    if actual_ids != expected_ids:
        errors.append(f"unit_id must be contiguous from 0. got={actual_ids[:20]}...")

    roots = [u for u in norm_units if u["parent_id"] is None]
    if len(roots) != 1:
        errors.append(f"There must be exactly one root node. got={len(roots)}")
    else:
        root = roots[0]
        if root["unit_id"] != 0:
            errors.append("Root node must have unit_id = 0")
        if root["level"] != 0:
            errors.append("Root node must have level = 0")
        if root["type"] != "root":
            errors.append("Root node must have type = 'root'")

    by_id = {u["unit_id"]: u for u in norm_units}
    children_of: Dict[int, List[int]] = defaultdict(list)

    for u in norm_units:
        uid = u["unit_id"]
        pid = u["parent_id"]
        if uid == 0:
            continue
        if pid is None:
            errors.append(f"Non-root node {uid} has null parent_id")
            continue
        if pid not in by_id:
            errors.append(f"Node {uid} parent_id={pid} does not exist")
            continue
        if pid >= uid:
            errors.append(f"Node {uid} parent_id must refer to an earlier node")
            continue

        parent = by_id[pid]
        if u["level"] != parent["level"] + 1:
            errors.append(
                f"Node {uid} level must equal parent.level + 1; got child={u['level']} parent={parent['level']}"
            )
        children_of[pid].append(uid)

    reachable = set()
    stack = [0]
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        stack.extend(children_of.get(cur, []))

    if len(reachable) != len(norm_units):
        missing = sorted(set(by_id.keys()) - reachable)
        errors.append(f"Tree is disconnected. unreachable nodes: {missing[:20]}")

    presence_ratio = check_text_presence_ratio(source_doc.full_text, norm_units)
    if presence_ratio < 0.70:
        errors.append(
            f"Too many unit texts are not verbatim substrings of source text. presence_ratio={presence_ratio:.3f}"
        )

    normalized = {
        "doc_id": source_doc.doc_id,
        "doc_name": source_doc.doc_name,
        "language": source_doc.language,
        "units": norm_units,
    }

    return len(errors) == 0, errors, normalized


async def call_deepseek(
    client: httpx.AsyncClient,
    api_key: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = await client.post(CHAT_URL, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


def extract_content(resp_json: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    choices = resp_json.get("choices") or []
    if not choices:
        raise ValueError("API response missing choices")
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return str(content), str(reasoning), msg


def write_failure_payload(
    fail_dir: Path,
    doc_slug: str,
    source_doc: SourceDoc,
    failure_reason: List[str],
    final_model_output: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    fail_payload = {
        "doc_id": source_doc.doc_id,
        "doc_name": source_doc.doc_name,
        "language": source_doc.language,
        "source_path": str(source_doc.path),
        "failure_reason": failure_reason,
        "final_model_output": final_model_output,
        "timestamp": int(time.time()),
    }
    if extra:
        fail_payload["extra"] = extra

    (fail_dir / f"{doc_slug}.failure.json").write_text(
        json.dumps(fail_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def annotate_one_doc(
    client: httpx.AsyncClient,
    api_key: str,
    source_doc: SourceDoc,
    output_dir: Path,
    args: argparse.Namespace,
    stats_lock: asyncio.Lock,
    stats: Dict[str, int],
) -> None:
    doc_slug = safe_slug(source_doc.doc_id)
    valid_dir = output_dir / "validated"
    raw_dir = output_dir / "raw_responses"
    fail_dir = output_dir / "failed"
    valid_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    previous_output: Optional[str] = None
    previous_errors: List[str] = []

    try:
        for attempt in range(1, 3):  # 固定只尝试两次
            if attempt == 1:
                input_text = prepare_input_text(source_doc.full_text, args.first_input_chars)
                user_prompt = FIRST_USER_PROMPT_TEMPLATE.format(
                    doc_id=source_doc.doc_id,
                    doc_name=source_doc.doc_name,
                    language=source_doc.language,
                    full_text=input_text,
                )
                max_tokens = args.first_max_tokens
            else:
                input_text = prepare_input_text(source_doc.full_text, args.retry_input_chars)
                user_prompt = REPAIR_PROMPT_TEMPLATE.format(
                    errors="\n".join(f"- {e}" for e in previous_errors),
                    previous_output=previous_output or "",
                    full_text=input_text,
                )
                max_tokens = args.retry_max_tokens

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            try:
                resp_json = await call_deepseek(
                    client=client,
                    api_key=api_key,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                content, reasoning, raw_msg = extract_content(resp_json)
            except Exception as e:
                previous_output = None
                previous_errors = [f"HTTP/API error on attempt {attempt}: {repr(e)}"]
                if attempt == 1:
                    continue
                break

            (raw_dir / f"{doc_slug}.attempt{attempt}.json").write_text(
                json.dumps(resp_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            parsed, parse_err = try_parse_json_object(content)
            if parsed is None:
                previous_output = content
                previous_errors = [f"Output is not a complete valid JSON document: {parse_err}"]
                if attempt == 1:
                    continue
                break

            ok, errors, normalized = validate_tree_json(parsed, source_doc)
            if ok and normalized is not None:
                out_path = valid_dir / f"{doc_slug}.json"
                out_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

                meta = {
                    "doc_id": source_doc.doc_id,
                    "doc_name": source_doc.doc_name,
                    "language": source_doc.language,
                    "source_path": str(source_doc.path),
                    "validated_path": str(out_path),
                    "attempt": attempt,
                    "reasoning_chars": len(reasoning),
                    "response_id": resp_json.get("id"),
                    "finish_reason": ((resp_json.get("choices") or [{}])[0]).get("finish_reason"),
                    "usage": resp_json.get("usage"),
                    "model": resp_json.get("model", MODEL_NAME),
                    "input_chars_used": len(input_text),
                    "max_tokens_used": max_tokens,
                    "timestamp": int(time.time()),
                }
                (valid_dir / f"{doc_slug}.meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                async with stats_lock:
                    stats["success"] += 1
                return

            previous_output = content
            previous_errors = errors

            if attempt == 1:
                continue
            break

        write_failure_payload(
            fail_dir=fail_dir,
            doc_slug=doc_slug,
            source_doc=source_doc,
            failure_reason=previous_errors,
            final_model_output=previous_output,
        )
        async with stats_lock:
            stats["failed"] += 1

    except Exception as e:
        tb = traceback.format_exc()
        write_failure_payload(
            fail_dir=fail_dir,
            doc_slug=doc_slug,
            source_doc=source_doc,
            failure_reason=[f"Unhandled exception: {repr(e)}"],
            final_model_output=previous_output,
            extra={"traceback": tb},
        )
        async with stats_lock:
            stats["failed"] += 1


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Please export DEEPSEEK_API_KEY before running this script.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = list_source_docs(input_dir, args.glob)

    if args.skip_existing:
        validated_dir = output_dir / "validated"
        filtered: List[SourceDoc] = []
        for d in docs:
            slug = safe_slug(d.doc_id)
            if not (validated_dir / f"{slug}.json").exists():
                filtered.append(d)
        docs = filtered

    stats = {"success": 0, "failed": 0}
    stats_lock = asyncio.Lock()
    tasks: List[asyncio.Task] = []

    timeout = httpx.Timeout(args.request_timeout)
    limits = httpx.Limits(max_keepalive_connections=1000, max_connections=None)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for idx, doc in enumerate(docs, start=1):
            tasks.append(
                asyncio.create_task(
                    annotate_one_doc(
                        client=client,
                        api_key=api_key,
                        source_doc=doc,
                        output_dir=output_dir,
                        args=args,
                        stats_lock=stats_lock,
                        stats=stats,
                    )
                )
            )
            print(f"[launch] {idx}/{len(docs)} doc_id={doc.doc_id}")
            await asyncio.sleep(args.launch_interval)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, r in enumerate(results, start=1):
                if isinstance(r, Exception):
                    print(f"[task-exception] task={idx} err={repr(r)}")

    summary = {
        "num_docs": len(docs),
        "success": stats["success"],
        "failed": stats["failed"],
        "validated_dir": str((output_dir / "validated").resolve()),
        "failed_dir": str((output_dir / "failed").resolve()),
        "raw_responses_dir": str((output_dir / "raw_responses").resolve()),
        "model": MODEL_NAME,
        "api_base": API_BASE,
        "launch_interval": args.launch_interval,
        "max_attempts": 2,
        "first_input_chars": args.first_input_chars,
        "retry_input_chars": args.retry_input_chars,
        "first_max_tokens": args.first_max_tokens,
        "retry_max_tokens": args.retry_max_tokens,
    }
    dump_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def dump_json(path: str | Path, obj: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    args.max_attempts = 2
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()