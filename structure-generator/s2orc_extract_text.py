import os
import json
from pathlib import Path
from typing import Optional

# ========== 配置区域 ==========

# JSON 根目录（每篇论文一个 .json，包含子目录）
JSON_ROOT = r"D:\code\Github\SLAC-test\data\A_structure\s2orc"

# 输出 TXT 根目录（会复制 JSON 的目录结构）
TXT_ROOT = r"D:\code\Github\SLAC-test\data\A_structure\papers"

# 日志文件路径：记录没有 text 或提取失败的 JSON 文件
LOG_FILE = r"D:\code\Github\SLAC-test\log\extract_text.log"

# 过滤掉特别短的文本（字符数），为 0 则不过滤
MIN_CHARS = 0


# ========== 日志函数 ==========

def log_bad(json_path: Path, reason: str):
    """
    记录提取失败或无文本的文件路径到日志。
    reason 用于标注原因，如 'no_content', 'short_text', 'json_error' 等。
    """
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{reason}\t{json_path}\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


# ========== 抽取函数 ==========

def extract_text(obj: dict) -> Optional[str]:
    """
    从一条 S2ORC JSON 记录中抽取全文文本。
    当前版本仅使用 content.text。
    """
    content = obj.get("content")
    if not content or not isinstance(content, dict):
        return None

    txt = content.get("text")
    if not isinstance(txt, str):
        return None

    txt = txt.strip()
    if len(txt) < MIN_CHARS:
        return None

    return txt


# ========== 主逻辑 ==========

def main():
    json_root = Path(JSON_ROOT)
    txt_root = Path(TXT_ROOT)

    if not json_root.exists():
        print(f"[FATAL] JSON_ROOT 不存在: {json_root}")
        return

    # 确保 TXT 根目录存在
    txt_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    saved = 0
    skipped_no_text = 0
    skipped_short = 0
    skipped_decode_err = 0

    print(f"[INFO] 扫描 JSON 根目录: {json_root}")

    # 遍历所有子目录下的 .json 文件
    for json_path in json_root.rglob("*.json"):
        total_files += 1

        # 计算相对路径，并映射到 TXT 根目录
        rel_path = json_path.relative_to(json_root)          # 例如 arxiv_hep-ph/9209259.json
        out_txt_path = (txt_root / rel_path).with_suffix(".txt")

        # 确保输出目录存在
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)

        # 如果已经有 txt，可以选择跳过或覆盖
        # 当前策略：如果已存在就跳过，避免重复工作
        if out_txt_path.exists():
            continue

        # 读取 JSON
        try:
            with json_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] JSON 解析失败，跳过: {json_path}")
            skipped_decode_err += 1
            log_bad(json_path, "json_error")
            continue
        except Exception as e:
            print(f"[WARN] 打开文件失败 {json_path}: {e}")
            skipped_decode_err += 1
            log_bad(json_path, "open_error")
            continue

        # 抽取文本
        txt = extract_text(obj)
        if txt is None:
            content = obj.get("content")
            # 没有 content.text
            if (not content
                or not isinstance(content, dict)
                or not isinstance((content or {}).get("text"), str)):
                skipped_no_text += 1
                log_bad(json_path, "no_content_text")
            else:
                # 有 text，但长度不足 MIN_CHARS
                skipped_short += 1
                log_bad(json_path, "short_text")
            continue

        # 写出 txt
        with out_txt_path.open("w", encoding="utf-8") as fo:
            fo.write(txt)

        saved += 1
        if saved % 100 == 0:
            print(f"[INFO] 已保存 {saved} 篇文本…")

    print(f"[DONE] 共扫描 JSON 文件：{total_files}")
    print(f"       成功提取并保存：{saved}")
    print(f"       无 content.text 的：{skipped_no_text}")
    print(f"       JSON 损坏或读取失败：{skipped_decode_err}")
    if MIN_CHARS > 0:
        print(f"       文本过短(<{MIN_CHARS} chars) 的：{skipped_short}")
    print(f"       TXT 输出根目录：{txt_root}")
    print(f"       日志文件：{LOG_FILE}")


if __name__ == "__main__":
    main()
