import json
import re
from pathlib import Path
from typing import Optional

# ===== 路径配置 =====
JSON_ROOT = r"D:\code\Github\SLAC-test\data\A_structure\s2orc"
OUT_ROOT = r"D:\code\Github\SLAC-test\data\A_structure\papers_structure"
LOG_FILE = r"D:\code\Github\SLAC-test\log\structure_json_tree.log"


# ===== 日志工具 =====

def log_line(msg: str):
    print(msg)
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def log_bad(path: Path, reason: str):
    line = f"{reason}\t{path}"
    log_line(line)


# ===== 公共工具函数 =====

def load_spans(ann: dict, key: str):
    """把 annotations[key] 解析成 list[dict]。"""
    raw = ann.get(key)
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        return raw
    return []


def level_from_n(n: str) -> int:
    """attributes.n -> 编号层级，仅用于定位父节点。"""
    if not n:
        return 1
    cleaned = re.sub(r"[^0-9\.]", "", n)
    parts = [p for p in cleaned.split(".") if p]
    return len(parts) if parts else 1


def normalize_span_bounds(span: dict, text_len: int) -> Optional[tuple[int, int]]:
    """
    把 span["start"], span["end"] 规范成合法的 (st, ed)：
      - 转成 int
      - 裁剪到 [0, text_len]
      - 保证 st < ed
    如果无法获得合法边界，返回 None。
    """
    try:
        st = int(span["start"])
        ed = int(span["end"])
    except Exception:
        return None

    if text_len <= 0:
        return None

    if st < 0:
        st = 0
    if ed < 0:
        return None
    if st >= text_len:
        return None
    if ed > text_len:
        ed = text_len
    if ed <= st:
        return None
    return st, ed


def inside_ranges(start: int, end: int, ranges):
    for a, b in ranges:
        if start >= a and end <= b:
            return True
    return False


# ===== 单篇 JSON -> 结构树 =====

def build_tree_from_obj(obj: dict) -> Optional[dict]:
    """
    给定一条 S2ORC JSON 记录，构建结构树：
      { "doc_id": ..., "units": [...] }
    如果 content.text 缺失或为空，返回 None。
    """
    content = obj.get("content") or {}
    text = content.get("text")
    if not isinstance(text, str):
        return None
    text = text
    if not text.strip():
        return None

    text_len = len(text)
    ann = content.get("annotations") or {}

    # ---- 取出各类 span ----
    title_spans    = load_spans(ann, "title")
    section_spans  = load_spans(ann, "sectionheader")
    para_spans     = load_spans(ann, "paragraph")
    figcap_spans   = load_spans(ann, "figurecaption")
    bibentry_spans = load_spans(ann, "bibentry")
    formula_spans  = load_spans(ann, "formula")
    figure_spans   = load_spans(ann, "figure")
    table_spans    = load_spans(ann, "table")

    # ---- figure / table 范围，用于过滤假 sectionheader ----
    skip_ranges = []
    for s in figure_spans + table_spans:
        bounds = normalize_span_bounds(s, text_len)
        if not bounds:
            continue
        skip_ranges.append(bounds)

    # 过滤掉落在 figure/table 里的 sectionheader
    clean_sections = []
    for s in section_spans:
        bounds = normalize_span_bounds(s, text_len)
        if not bounds:
            continue
        st, ed = bounds
        seg = text[st:ed].strip()
        if not seg:
            continue
        if inside_ranges(st, ed, skip_ranges):
            continue
        # 把合法的边界写回 span 里，后面直接用
        s["__start"] = st
        s["__end"] = ed
        clean_sections.append(s)
    section_spans = clean_sections

    # ===== 结构树构建 =====
    units = []
    uid = 0
    id2level: dict[int, int] = {}

    def new_unit(t, utype, level, parent_id, subtype=None):
        nonlocal uid, units, id2level
        unit = {
            "unit_id": uid,
            "text": t,
            "type": utype,
            "level": int(level),
            "parent_id": parent_id,
        }
        if subtype:
            unit["subtype"] = subtype
        units.append(unit)
        id2level[uid] = int(level)
        uid += 1
        return unit["unit_id"]

    # ---- 文档根标题 ----
    if title_spans:
        t0 = title_spans[0]
        bounds = normalize_span_bounds(t0, text_len)
        if bounds:
            st, ed = bounds
            root_text = text[st:ed].strip()
        else:
            root_text = f"corpus_{obj.get('corpusid', '')}"
    else:
        root_text = f"corpus_{obj.get('corpusid', '')}"

    root_id = new_unit(root_text, "heading", 0, None, subtype="doc_title")

    # ---- 汇总所有需要建树的 span，统一构 items ----
    items = []

    def add_item(kind: str, span: dict):
        bounds = normalize_span_bounds(span, text_len)
        if not bounds:
            return
        st, ed = bounds
        items.append({"kind": kind, "span": span, "start": st, "end": ed})

    for s in section_spans:
        # 已经在上面写了 __start/__end，但这里统一再走 normalize_span_bounds，防御性一点
        add_item("sectionheader", s)

    for p in para_spans:
        add_item("paragraph", p)

    for fc in figcap_spans:
        add_item("figurecaption", fc)

    for be in bibentry_spans:
        add_item("bibentry", be)

    for fm in formula_spans:
        add_item("formula", fm)

    items.sort(key=lambda x: x["start"])

    # ---- 按文档顺序遍历，维护 heading 栈 ----
    heading_stack = {0: root_id}   # 编号层级 -> unit_id
    last_heading_hier_level = 0
    refs_heading_id = None

    for item in items:
        start = int(item["start"])
        end = int(item["end"])
        seg = text[start:end].strip()
        if not seg:
            continue

        kind = item["kind"]

        if kind == "sectionheader":
            s = item["span"]
            attrs = s.get("attributes") or {}
            n = attrs.get("n")

            if n:
                hier_level = level_from_n(n)
            else:
                hier_level = (last_heading_hier_level + 1) if last_heading_hier_level > 0 else 1

            parent_id = root_id
            for lv in range(hier_level - 1, -1, -1):
                if lv in heading_stack:
                    parent_id = heading_stack[lv]
                    break

            parent_level = id2level.get(parent_id, 0)
            out_level = parent_level + 1

            hid = new_unit(seg, "heading", out_level, parent_id, subtype="sectionheader")

            for lv in list(heading_stack.keys()):
                if lv > hier_level:
                    del heading_stack[lv]
            heading_stack[hier_level] = hid
            last_heading_hier_level = hier_level

            low = seg.lower()
            if refs_heading_id is None and (low.startswith("references") or low.startswith("bibliography")):
                refs_heading_id = hid

        elif kind == "paragraph":
            current_hier_level = max(heading_stack.keys())
            parent_id = heading_stack.get(current_hier_level, root_id)
            parent_level = id2level.get(parent_id, 0)
            out_level = parent_level + 1
            new_unit(seg, "paragraph", out_level, parent_id, subtype="body")

        elif kind == "figurecaption":
            current_hier_level = max(heading_stack.keys())
            parent_id = heading_stack.get(current_hier_level, root_id)
            parent_level = id2level.get(parent_id, 0)
            out_level = parent_level + 1
            new_unit(seg, "figure", out_level, parent_id, subtype="figurecaption")

        elif kind == "formula":
            current_hier_level = max(heading_stack.keys())
            parent_id = heading_stack.get(current_hier_level, root_id)
            parent_level = id2level.get(parent_id, 0)
            out_level = parent_level + 1
            new_unit(seg, "equation", out_level, parent_id, subtype="formula")

        elif kind == "bibentry":
            if refs_heading_id is None:
                parent_id = root_id
                parent_level = id2level.get(parent_id, 0)
                refs_level = parent_level + 1
                refs_heading_id = new_unit("References", "heading", refs_level,
                                           parent_id, subtype="references")

            parent_id = refs_heading_id
            parent_level = id2level.get(parent_id, 0)
            out_level = parent_level + 1
            new_unit(seg, "reference", out_level, parent_id, subtype="bibentry")

    doc = {
        "doc_id": f"corpus_{obj.get('corpusid')}",
        "units": units,
    }
    return doc


# ===== 批处理主逻辑 =====

def main():
    json_root = Path(JSON_ROOT)
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    if not json_root.exists():
        log_line(f"[FATAL] JSON_ROOT 不存在: {json_root}")
        return

    total = 0
    built = 0
    skipped_no_text = 0
    json_error = 0
    other_error = 0

    log_line(f"[INFO] 开始扫描 JSON 根目录: {json_root}")

    for json_path in json_root.rglob("*.json"):
        total += 1

        rel_path = json_path.relative_to(json_root)
        out_path = (out_root / rel_path).with_suffix(".tree.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue

        try:
            with json_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except json.JSONDecodeError:
            json_error += 1
            log_bad(json_path, "json_decode_error")
            continue
        except Exception as e:
            other_error += 1
            log_bad(json_path, f"json_open_error:{e}")
            continue

        doc = build_tree_from_obj(obj)
        if doc is None:
            skipped_no_text += 1
            log_bad(json_path, "no_text")
            continue

        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            built += 1
        except Exception as e:
            other_error += 1
            log_bad(json_path, f"write_error:{e}")
            continue

        if built % 100 == 0:
            log_line(f"[INFO] 已生成结构树 {built} 篇 / 已扫描 {total} 文件")

    log_line(f"[DONE] 总 JSON 文件数: {total}")
    log_line(f"       成功生成结构树: {built}")
    log_line(f"       无 text 内容(跳过): {skipped_no_text}")
    log_line(f"       JSON 解析错误: {json_error}")
    log_line(f"       其它错误: {other_error}")
    log_line(f"       输出根目录: {out_root}")
    log_line(f"       日志文件: {LOG_FILE}")


if __name__ == "__main__":
    main()
