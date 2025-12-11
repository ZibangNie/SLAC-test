import docx
import re
import json
from pathlib import Path

# ✅ 读取 Word 文本（.doc 文件需先另存为 .docx）
def load_docx_text(docx_path):
    doc = docx.Document(docx_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

# ✅ 匹配章节编号（1.，1.1，1.2.3 等）及标题
section_pattern = re.compile(r"^(\d+(?:\.\d+)*)([\u4e00-\u9fa5A-Za-z0-9\s\-\（\）\(\)《》“”、．\.\:：]*)$")

# ✅ 构造平铺结构树（扁平化结构，带 parent_id）
def build_flat_structure(paragraphs):
    units = []
    id_counter = 1
    stack = []  # 存放当前路径的 unit_id（按层级）

    for para in paragraphs:
        match = section_pattern.match(para)
        if match:
            sec_num, title = match.groups()
            level = sec_num.count('.') + 1  # 层级从 1 开始
            parent_id = stack[level - 2] if level > 1 and len(stack) >= level - 1 else None

            unit = {
                "unit_id": id_counter,
                "text": f"{sec_num} {title.strip()}",
                "type": "heading",
                "level": level,
                "parent_id": parent_id
            }
            units.append(unit)

            if len(stack) >= level:
                stack = stack[:level - 1]
            stack.append(id_counter)
            id_counter += 1
        else:
            parent_id = stack[-1] if stack else None
            unit = {
                "unit_id": id_counter,
                "text": para,
                "type": "paragraph",
                "level": (len(stack) + 1),
                "parent_id": parent_id
            }
            units.append(unit)
            id_counter += 1

    return units

# ✅ 主函数：加载、构造、保存
if __name__ == "__main__":
    input_path = Path(r"D:\code\Github\SLAC-test\data\docx\TB10623-2014 城际铁路设计规范（条文说明）.docx")  # ⚠️ 替换为你的 Word 文件路径
    output_path = Path("tb10623_flat_units.json")

    print("🔍 正在读取文档…")
    paras = load_docx_text(input_path)

    print("📄 正在解析文档结构为扁平单元…")
    flat_units = build_flat_structure(paras)

    print("💾 正在写入 JSON…")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(flat_units, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成：已生成结构单元文件 {output_path}")
