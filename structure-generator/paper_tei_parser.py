"""
tei_to_tree.py

功能：
  将 TEI-XML 论文转换为统一的结构树格式：
  每篇文档 -> { "doc_id": ..., "units": [ {unit_id, text, type, level, parent_id}, ... ] }

配置：
  - 输入 TEI 路径：D:\code\Github\SLAC-test\data\A_structure\papers_tei\*.tei.xml
  - 输出 JSON 路径：D:\code\Github\SLAC-test\data\A_structure\papers_structure\*.tree.json

字段约定：
  - unit_id: 文档内顺序编号（0,1,2,...）
  - text: 单元文本
  - type: "heading" 或 "paragraph"
  - level: 整数层级，0 为文档根标题，1 为一级标题/内容，2 为二级...
  - parent_id: 父节点的 unit_id，根为 null

改进点：
  1）利用 <head n="1.2.3"> 的编号计算层级和父子关系；
  2）Definition / Assumption / Proposition 拆分为干净的 heading + paragraph；
  3）过滤明显无意义的超短段（公式编号、end for 等）；
  4）跳过图表内的 head / p（figure/table caption）；
  5）显式抽取 Abstract：支持 header 里的 <abstract> 和 text/front 里的 abstract/summary。
"""

import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

# ===== 路径配置 =====

TEI_DIR = Path(r"/data/A_structure/papers_tei(deprecated)")
OUT_DIR = Path(r"D:\code\Github\SLAC-test\data\A_structure\papers_structure")

# 是否只测试一个文件
TEST_MODE = True

# 如果你想指定某个文件测试，可以填文件名（比如 "2512.01623.tei.xml"）
# 留空字符串表示不指定，自动取目录里的第一个
TEST_FILENAME = "2512.01623.tei.xml"  # 不想指定就改成 ""


NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def normalize_text(elem) -> str:
    """把 XML 节点下所有文本拼在一起，并压缩空白。"""
    if elem is None:
        return ""
    text = "".join(elem.itertext())
    return " ".join(text.split())


def level_from_n(n_str: str) -> int:
    """
    根据 <head n="2.2.3"> 的 n 属性计算层级：
      - "1" -> 1
      - "2.2" -> 2
      - "2.2.3" -> 3
    """
    if not n_str:
        return 1
    # 去掉非数字和点，比如奇怪的符号
    cleaned = re.sub(r"[^0-9\.]", "", n_str)
    parts = [p for p in cleaned.split(".") if p]
    return len(parts) if parts else 1


def split_special_heading(text: str):
    """
    专门处理 Definition / Assumption / Proposition 这一类：
      "Definition 1. (XXX). A function ..." ->
          heading_text = "Definition 1. (XXX)."
          tail_text    = "A function ..."
    如果不匹配，返回 (text, None)
    """
    patterns = [
        r"^(Definition\s+\d+\.?(?:\s*\([^)]*\))?\.?)\s*(.*)$",
        r"^(Assumption\s+\d+\.?(?:\s*\([^)]*\))?\.?)\s*(.*)$",
        r"^(Proposition\s+\d+\.?(?:\s*\([^)]*\))?\.?)\s*(.*)$",
    ]
    for pat in patterns:
        m = re.match(pat, text)
        if m:
            head_text = m.group(1).strip()
            tail = m.group(2).strip()
            return head_text, (tail if tail else None)
    return text, None


def is_trivial_paragraph(text: str) -> bool:
    """
    判断一个段落是否“明显没啥结构价值”的短碎片：
      - 特别短且全是符号/数字，比如 "(15)", "2.", ") ."
      - 少数字符的控制词，比如 "end for", "Here:", "analysis."
    返回 True 则跳过。
    """
    s = text.strip()
    if not s:
        return True

    # 特别短的直接扔掉
    if len(s) <= 3:
        return True

    # 纯数字或括号包裹的数字
    if len(s) <= 8 and re.fullmatch(r"\(?\d+\)?\.?", s):
        return True

    # 去掉两边标点后再看
    core = s.strip(" .:;()[]{}").lower()
    trivial_tokens = {
        "end", "end for", "here", "analysis", "where", "where:",
        "figure", "table"
    }
    if len(core) <= 12 and core in trivial_tokens:
        return True

    return False


def parse_tei_file(tei_path: Path) -> dict:
    """
    单篇 TEI -> 结构树 dict:
    {
      "doc_id": ...,
      "units": [ {...}, ... ]
    }
    """
    tree = ET.parse(tei_path)
    root = tree.getroot()

    # unit_id 自增计数器
    counter = 0

    def new_uid():
        nonlocal counter
        uid = counter
        counter += 1
        return uid

    units = []

    # 1. 文档级标题（根节点，level=0）
    title_el = root.find(
        ".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title", NS
    )
    doc_title = normalize_text(title_el) if title_el is not None else tei_path.stem

    root_id = new_uid()
    units.append({
        "unit_id": root_id,
        "text": doc_title,
        "type": "heading",
        "level": 0,
        "parent_id": None,
    })

    # 2. Abstract（如果有）——不参与后续 heading_stack，单独挂在 root 下
    abstract_texts = []

    # 2.1 优先从 header 里的 profileDesc/abstract 取
    header_abstract = root.find(
        ".//tei:teiHeader/tei:profileDesc/tei:abstract", NS
    )
    if header_abstract is not None:
        ps = header_abstract.findall(".//tei:p", NS)
        if ps:
            for p in ps:
                t = normalize_text(p)
                if t:
                    abstract_texts.append(t)
        else:
            t = normalize_text(header_abstract)
            if t:
                abstract_texts.append(t)

    # 2.2 如果 header 里没有，再尝试 text/front 里的 abstract/summary
    if not abstract_texts:
        front = root.find(".//tei:text/tei:front", NS)
        if front is not None:
            for div in front.findall(".//tei:div", NS):
                div_type = (div.get("type") or "").lower()
                if div_type in ("abstract", "summary"):
                    for p in div.findall(".//tei:p", NS):
                        t = normalize_text(p)
                        if t:
                            abstract_texts.append(t)

    # 2.3 把 Abstract 写入 units
    if abstract_texts:
        abs_head_id = new_uid()
        units.append({
            "unit_id": abs_head_id,
            "text": "Abstract",
            "type": "heading",
            "level": 1,
            "parent_id": root_id,
        })
        for txt in abstract_texts:
            if is_trivial_paragraph(txt):
                continue
            pid = new_uid()
            units.append({
                "unit_id": pid,
                "text": txt,
                "type": "paragraph",
                "level": 1,
                "parent_id": abs_head_id,
            })

    # 3. 正文 body
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return {"doc_id": tei_path.stem, "units": units}

    # 3.1 标记所有 figure 内的 head / p，后面迭代时跳过
    figure_heads = set()
    figure_paras = set()
    for fig in body.findall(".//tei:figure", NS):
        for h in fig.findall(".//tei:head", NS):
            figure_heads.add(h)
        for p in fig.findall(".//tei:p", NS):
            figure_paras.add(p)

    # 3.2 维护一个 “当前 heading 栈”：level -> unit_id（只用于 body）
    heading_stack = {0: root_id}
    last_heading_level = 0  # 最近一个 heading 的层级

    # 3.3 按文档顺序遍历 body 下所有子元素
    for elem in body.iter():
        tag = elem.tag
        if not isinstance(tag, str):
            continue
        # 去掉命名空间
        if tag.startswith("{"):
            tag = tag.split("}", 1)[1]

        # === 处理 heading ===
        if tag == "head":
            if elem in figure_heads:
                # 图表题注，跳过
                continue

            raw_text = normalize_text(elem)
            if not raw_text:
                continue

            n_attr = elem.get("n")
            if n_attr:
                lvl = level_from_n(n_attr)
                # 找 parent：先找 lvl-1，再往上回退
                parent_id = None
                for pl in range(lvl - 1, -1, -1):
                    if pl in heading_stack:
                        parent_id = heading_stack[pl]
                        break
                if parent_id is None:
                    parent_id = root_id
            else:
                # 无编号的 heading（Definition, Assumption 等），挂在最近的 heading 下面
                if last_heading_level > 0:
                    lvl = last_heading_level + 1
                    parent_id = heading_stack.get(last_heading_level, root_id)
                else:
                    lvl = 1
                    parent_id = root_id

            # Definition / Assumption / Proposition 拆分
            heading_text, tail_text = split_special_heading(raw_text)

            # 先写 heading 单元
            heading_id = new_uid()
            units.append({
                "unit_id": heading_id,
                "text": heading_text,
                "type": "heading",
                "level": lvl,
                "parent_id": parent_id,
            })

            # 如果有拆出的尾部内容，作为紧随其后的 paragraph
            if tail_text:
                if not is_trivial_paragraph(tail_text):
                    pid = new_uid()
                    units.append({
                        "unit_id": pid,
                        "text": tail_text,
                        "type": "paragraph",
                        "level": lvl,       # 视为该层级的内容
                        "parent_id": heading_id,
                    })

            # 更新 heading 栈
            # 去掉比当前层级更深的
            for k in list(heading_stack.keys()):
                if k > lvl:
                    del heading_stack[k]
            heading_stack[lvl] = heading_id
            last_heading_level = lvl

        # === 处理正文段落 ===
        elif tag == "p":
            if elem in figure_paras:
                # 图表中的 caption 段落，跳过
                continue

            text = normalize_text(elem)
            if not text:
                continue
            if is_trivial_paragraph(text):
                continue

            # 段落挂在“当前最深的 heading”下面
            current_level = max(heading_stack.keys())  # 至少有 0
            if current_level == 0:
                lvl = 1
                parent_id = root_id
            else:
                lvl = current_level
                parent_id = heading_stack[current_level]

            pid = new_uid()
            units.append({
                "unit_id": pid,
                "text": text,
                "type": "paragraph",
                "level": lvl,
                "parent_id": parent_id,
            })

        # 其他 tag 目前忽略（formula/note 等以后需要再补）

    doc = {
        "doc_id": tei_path.stem,  # 比如 "2512.01623"
        "units": units,
    }
    return doc


def main():
    if not TEI_DIR.exists():
        print(f"[FATAL] TEI_DIR not found: {TEI_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tei_files = sorted(TEI_DIR.glob("*.tei.xml"))
    if not tei_files:
        print(f"[WARN] No TEI files under {TEI_DIR}")
        return

    # TEST_MODE 下，只取一个文件
    if TEST_MODE:
        if TEST_FILENAME:
            target = TEI_DIR / TEST_FILENAME
            if not target.exists():
                print(f"[FATAL] TEST file not found: {target}")
                return
            tei_files = [target]
        else:
            tei_files = [tei_files[0]]
        print(f"[INFO] TEST_MODE = True, only processing: {tei_files[0].name}")
    else:
        print(f"[INFO] Found {len(tei_files)} TEI files under {TEI_DIR}")

    for idx, tei_path in enumerate(tei_files, start=1):
        try:
            doc = parse_tei_file(tei_path)
        except Exception as e:
            print(f"[ERROR] Failed to parse {tei_path.name}: {e}")
            continue

        out_path = OUT_DIR / f"{tei_path.stem}.tree.json"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            print(f"[OK] {tei_path.name} -> {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write {out_path.name}: {e}")

        if not TEST_MODE and idx % 10 == 0:
            print(f"[INFO] Processed {idx} files...")


if __name__ == "__main__":
    main()
