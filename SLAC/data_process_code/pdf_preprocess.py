import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List

import fitz  # PyMuPDF

# -------------------- 配置 --------------------
EN_DIR = Path(r"D:\code\Github\SLAC-test\SLAC\data\en")
ZH_DIR = Path(r"D:\code\Github\SLAC-test\SLAC\data\zh")

EN_ORIGIN = Path(r"D:\code\Github\SLAC-test\SLAC\data\en_origin")
ZH_ORIGIN = Path(r"D:\code\Github\SLAC-test\SLAC\data\zh_origin")

# OCR 输出语言
LANG_EN = "eng"
LANG_ZH = "chi_sim+eng"

# OCR 并行
JOBS = max(1, min(4, os.cpu_count() or 2))

# “可提取文本”判定阈值（可按你数据调）
MIN_TEXT_CHARS_PER_PAGE = 20     # 单页文本 >= 20 字符算“有文本页”
MIN_TOTAL_TEXT_CHARS = 200       # 全文文本 >= 200 字符算“整体有文本”

# “OCR 候选”判定：默认只要 text_pages==0 就认为需要 OCR（更稳）
OCR_IF_NO_TEXT_PAGES = True

# 输出日志
LOG_DIR = Path(r"D:\code\Github\SLAC-test\SLAC\data\_pdf_clean_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 若 True：遇到备份目标已存在，会生成唯一文件名避免覆盖
ALLOW_BACKUP_RENAME = True

# 若 True：对无法解析/加密的 PDF，也复制回工作目录（仅备份和记录，OCR不做）
COPY_UNPARSEABLE_BACK_TO_WORKDIR = False

# -------------------- 工具函数 --------------------

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def unique_path(p: Path) -> Path:
    """如果 p 已存在，生成不冲突的新路径：name__dup1.ext"""
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    for k in range(1, 10_000):
        cand = p.with_name(f"{stem}__dup{k}{suf}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Cannot find unique name for {p}")

def analyze_pdf(pdf_path: Path) -> Tuple[str, str, Dict[str, Any]]:
    """
    解析 PDF，返回 (status, reason, stats)
    status:
      - "text_ok"         : 可提取文本
      - "no_text"         : 提不出文本（疑似扫描/图片/或提取失败）
      - "unparseable"     : 打不开/加密/损坏/0页等
    """
    stats: Dict[str, Any] = {
        "pages": None,
        "text_pages": 0,
        "image_pages": 0,
        "total_text_chars": 0,
        "total_images": 0,
        "needs_password": False,
        "open_error": "",
    }

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        stats["open_error"] = f"{type(e).__name__}: {e}"
        return "unparseable", "open_failed", stats

    try:
        if getattr(doc, "needs_pass", False):
            stats["needs_password"] = True
            return "unparseable", "needs_password", stats

        n = doc.page_count
        stats["pages"] = n
        if n == 0:
            return "unparseable", "zero_pages", stats

        for i in range(n):
            page = doc.load_page(i)

            txt = (page.get_text("text") or "").strip()
            c = len(txt)
            stats["total_text_chars"] += c
            if c >= MIN_TEXT_CHARS_PER_PAGE:
                stats["text_pages"] += 1

            imgs = page.get_images(full=True)
            ni = len(imgs)
            stats["total_images"] += ni
            if ni > 0:
                stats["image_pages"] += 1

        # 判定 text_ok
        if stats["text_pages"] > 0 and stats["total_text_chars"] >= MIN_TOTAL_TEXT_CHARS:
            return "text_ok", "has_text", stats

        # 否则认为 no_text（可能是扫描件/图片/提取失败/矢量轮廓等）
        return "no_text", "no_text_extracted", stats

    finally:
        doc.close()

def run_ocrmypdf(in_pdf: Path, out_pdf: Path, lang: str) -> Tuple[bool, str]:
    """
    调用 ocrmypdf 生成可复制文本 PDF。
    使用 sys.executable -m ocrmypdf，避免 PATH 问题。
    解决 Windows 解码问题：encoding='utf-8', errors='replace'
    """
    safe_mkdir(out_pdf.parent)

    cmd = [
        sys.executable, "-m", "ocrmypdf",
        "-l", lang,
        "--deskew",
        "--rotate-pages",
        "--force-ocr",
        "--jobs", str(JOBS),
        str(in_pdf),
        str(out_pdf),
    ]

    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if p.returncode == 0:
            return True, "ok"
        else:
            # 收敛错误信息（末行更有用）
            err_lines = (p.stderr or "").strip().splitlines()
            last = err_lines[-1] if err_lines else ""
            return False, f"ret={p.returncode} err={last}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def write_lines(path: Path, lines: List[str]):
    safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def append_jsonl(path: Path, obj: Dict[str, Any]):
    safe_mkdir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def list_pdfs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.pdf"))

# -------------------- 主流程 --------------------

def backup_and_process_one_root(work_dir: Path, origin_dir: Path, lang: str, tag: str):
    """
    1) 扫描 work_dir 下所有 PDF，记录 image/no_text/unparseable
    2) 移动全部 PDF 到 origin_dir（保留相对路径）
    3) 从 origin_dir 构建新的 work_dir：
        - text_ok: 复制回 work_dir
        - no_text: OCR 输出到 work_dir（同名）
        - unparseable: 仅记录；可选复制回 work_dir
    4) 最后对 work_dir 再扫描，输出仍不可提取文本清单
    """
    log_jsonl = LOG_DIR / f"{tag}_process.jsonl"
    img_list_txt = LOG_DIR / f"{tag}_no_text_candidates.txt"
    unparseable_txt = LOG_DIR / f"{tag}_unparseable.txt"
    ocr_fail_txt = LOG_DIR / f"{tag}_ocr_failed.txt"
    final_bad_txt = LOG_DIR / f"{tag}_final_still_bad.txt"

    pdfs = list_pdfs(work_dir)
    print(f"[{tag}] found PDFs in work_dir: {len(pdfs)}")

    no_text_candidates = []
    unparseable = []
    ocr_failed = []

    # 先扫描并分类（在移动前记录原路径）
    analysis_results = []
    for pdf in pdfs:
        status, reason, stats = analyze_pdf(pdf)
        analysis_results.append((pdf, status, reason, stats))
        rec = {
            "stage": "pre_scan",
            "tag": tag,
            "path": str(pdf),
            "status": status,
            "reason": reason,
            "stats": stats,
        }
        append_jsonl(log_jsonl, rec)

        if status == "no_text":
            no_text_candidates.append(f"{pdf}\t{reason}\tpages={stats.get('pages')}\timages={stats.get('total_images')}\ttotal_text={stats.get('total_text_chars')}")
        elif status == "unparseable":
            unparseable.append(f"{pdf}\t{reason}\tneeds_password={stats.get('needs_password')}\topen_error={stats.get('open_error')}")

    write_lines(img_list_txt, no_text_candidates)
    write_lines(unparseable_txt, unparseable)

    # 1) 移动所有 PDF 到 origin_dir（备份）
    moved_map = []  # (orig_pdf_path, backup_pdf_path, status, reason, stats)
    for pdf, status, reason, stats in analysis_results:
        rel = pdf.relative_to(work_dir)
        dst = origin_dir / rel
        safe_mkdir(dst.parent)

        final_dst = dst
        if final_dst.exists():
            if ALLOW_BACKUP_RENAME:
                final_dst = unique_path(final_dst)
            else:
                # 不覆盖：直接记录失败并跳过
                append_jsonl(log_jsonl, {
                    "stage": "backup_move",
                    "tag": tag,
                    "path": str(pdf),
                    "ok": False,
                    "msg": f"backup_target_exists: {dst}",
                })
                continue

        try:
            shutil.move(str(pdf), str(final_dst))
            append_jsonl(log_jsonl, {
                "stage": "backup_move",
                "tag": tag,
                "src": str(pdf),
                "dst": str(final_dst),
                "ok": True,
            })
            moved_map.append((pdf, final_dst, status, reason, stats))
        except Exception as e:
            append_jsonl(log_jsonl, {
                "stage": "backup_move",
                "tag": tag,
                "src": str(pdf),
                "dst": str(final_dst),
                "ok": False,
                "msg": f"{type(e).__name__}: {e}",
            })

    # 2) 从备份构建工作目录（复制/ OCR 输出）
    for orig_pdf_path, backup_pdf_path, status, reason, stats in moved_map:
        rel = orig_pdf_path.relative_to(work_dir)
        work_target = work_dir / rel
        safe_mkdir(work_target.parent)

        if status == "text_ok":
            # 复制回工作目录
            try:
                shutil.copy2(str(backup_pdf_path), str(work_target))
                append_jsonl(log_jsonl, {
                    "stage": "restore_text_ok",
                    "tag": tag,
                    "backup": str(backup_pdf_path),
                    "work": str(work_target),
                    "ok": True,
                })
            except Exception as e:
                append_jsonl(log_jsonl, {
                    "stage": "restore_text_ok",
                    "tag": tag,
                    "backup": str(backup_pdf_path),
                    "work": str(work_target),
                    "ok": False,
                    "msg": f"{type(e).__name__}: {e}",
                })
            continue

        if status == "unparseable":
            # 只记录；可选复制回工作目录
            if COPY_UNPARSEABLE_BACK_TO_WORKDIR:
                try:
                    shutil.copy2(str(backup_pdf_path), str(work_target))
                    append_jsonl(log_jsonl, {
                        "stage": "restore_unparseable",
                        "tag": tag,
                        "backup": str(backup_pdf_path),
                        "work": str(work_target),
                        "ok": True,
                    })
                except Exception as e:
                    append_jsonl(log_jsonl, {
                        "stage": "restore_unparseable",
                        "tag": tag,
                        "backup": str(backup_pdf_path),
                        "work": str(work_target),
                        "ok": False,
                        "msg": f"{type(e).__name__}: {e}",
                    })
            continue

        # status == "no_text"
        # 是否 OCR
        if OCR_IF_NO_TEXT_PAGES:
            ok, msg = run_ocrmypdf(backup_pdf_path, work_target, lang)
            append_jsonl(log_jsonl, {
                "stage": "ocr",
                "tag": tag,
                "backup": str(backup_pdf_path),
                "work": str(work_target),
                "ok": ok,
                "msg": msg,
            })
            if not ok:
                ocr_failed.append(f"{backup_pdf_path}\t{msg}")
                # 回退：OCR 失败则把原 PDF 复制回工作目录（保证工作集完整）
                try:
                    shutil.copy2(str(backup_pdf_path), str(work_target))
                except Exception:
                    pass
        else:
            # 不 OCR：直接复制回工作目录
            try:
                shutil.copy2(str(backup_pdf_path), str(work_target))
            except Exception:
                pass

    write_lines(ocr_fail_txt, ocr_failed)

    # 3) 再扫描工作目录，确认是否已经可提取文本
    final_bad = []
    final_pdfs = list_pdfs(work_dir)
    for pdf in final_pdfs:
        status, reason, stats = analyze_pdf(pdf)
        append_jsonl(log_jsonl, {
            "stage": "final_scan",
            "tag": tag,
            "path": str(pdf),
            "status": status,
            "reason": reason,
            "stats": stats,
        })
        if status != "text_ok":
            final_bad.append(f"{pdf}\t{status}\t{reason}\tpages={stats.get('pages')}\ttext_pages={stats.get('text_pages')}\ttotal_text={stats.get('total_text_chars')}\tneeds_password={stats.get('needs_password')}\topen_error={stats.get('open_error')}")

    write_lines(final_bad_txt, final_bad)

    print(f"[{tag}] done.")
    print(f"  pre_scan no_text candidates: {len(no_text_candidates)} -> {img_list_txt}")
    print(f"  pre_scan unparseable       : {len(unparseable)} -> {unparseable_txt}")
    print(f"  ocr failed                 : {len(ocr_failed)} -> {ocr_fail_txt}")
    print(f"  final still bad            : {len(final_bad)} -> {final_bad_txt}")
    print(f"  jsonl log                  : {log_jsonl}")

def main():
    safe_mkdir(EN_ORIGIN)
    safe_mkdir(ZH_ORIGIN)

    backup_and_process_one_root(EN_DIR, EN_ORIGIN, LANG_EN, tag="en")
    backup_and_process_one_root(ZH_DIR, ZH_ORIGIN, LANG_ZH, tag="zh")

    print("\nALL DONE.")
    print(f"Logs at: {LOG_DIR}")

if __name__ == "__main__":
    main()