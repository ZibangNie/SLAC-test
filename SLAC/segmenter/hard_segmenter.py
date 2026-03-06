# -*- coding: utf-8 -*-
"""hard_segmenter.py

Main entry for SLAC hard-coded coarse segmentation pipeline (decoupled).

Pipeline
  1) pdf_to_text_preprocessor.py : PDF -> robust UTF-8 text (.txt) with diagnostics
  2) txt_to_json_segmenter.py    : TXT -> structure-tree JSON (units + strict tree)

This file orchestrates both steps so you can run a single command.

Recommended directory layout (your project)
  PDFs:
    D:\\code\\Github\\SLAC-test\\SLAC\\data\\en
    D:\\code\\Github\\SLAC-test\\SLAC\\data\\zh
  Intermediate text:
    D:\\code\\Github\\SLAC-test\\SLAC\\data\\extracted_text
  Output JSON:
    D:\\code\\Github\\SLAC-test\\SLAC\\data\\parsed_json

Notes
  - For best robustness on garbled PDFs, install:
      * Poppler (pdftotext)
      * Tesseract OCR + pytesseract + pillow
  - The preprocessor will attempt: span-filter extraction -> pdftotext -> OCR.

"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _best_effort_utf8_console() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_preprocess(
    input_dirs: List[Path],
    output_text_dir: Path,
    output_meta_dir: Path,
    log_dir: Path,
    *,
    max_pages: int = 0,
    enable_pdftotext: bool = True,
    ocr_on_images: bool = True,
    ocr_always_on_images: bool = True,
    ocr_lang: str = "chi_sim+eng",
    force: bool = False,
) -> Path:
    """Run PDF -> TXT preprocessor over input dirs.

    Returns diagnostics CSV path.
    """
    from pdf_to_text_preprocessor import Config as PreCfg
    from pdf_to_text_preprocessor import iter_pdfs, process_pdf

    output_text_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    diag_csv = output_text_dir / "diagnostics_preprocess.csv"

    cfg = PreCfg(
        input_dirs=input_dirs,
        output_text_dir=output_text_dir,
        output_meta_dir=output_meta_dir,
        log_dir=log_dir,
        max_pages=max_pages,
        # span filtering
        min_font_size=5.0,
        max_weird_ratio_span=0.35,
        drop_tiny_spans=True,
        # garble thresholds
        garbled_page_weird_ratio=0.25,
        garbled_min_len=10,
        # OCR routing
        ocr_on_images=ocr_on_images,
        ocr_always_on_images=ocr_always_on_images,
        ocr_dpi=300,
        ocr_lang=ocr_lang,
        ocr_backend="tesseract",
        enable_pdftotext=enable_pdftotext,
        ocr_when_text_short=120,
        image_area_ratio_for_ocr=0.35,
        page_separator="\n\n\f\n\n",
    )

    pdfs = iter_pdfs(input_dirs)
    if not pdfs:
        print("[INFO] No PDFs found in input dirs.")
        return diag_csv

    # write diagnostics
    keys = [
        "pdf_path",
        "pages",
        "scanned_pages",
        "ocr_pages",
        "pdftotext_pages",
        "garbled_pages",
        "garbled_ratio",
        "skipped_image_only_pages",
        "dropped_noise_spans_total",
        "ocr_backend",
        "ocr_available",
        "skipped",
        "error",
    ]

    with open(diag_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for idx, pdf in enumerate(pdfs, start=1):
            try:
                out_txt = output_text_dir / f"{pdf.stem}.txt"
                if (not force) and out_txt.exists() and out_txt.stat().st_mtime >= pdf.stat().st_mtime:
                    w.writerow({"pdf_path": str(pdf), "skipped": True})
                    continue
                print(f"[PRE {idx}/{len(pdfs)}] {pdf}")
                d = process_pdf(pdf, cfg)
                d["skipped"] = False
                w.writerow(d)
            except Exception as e:
                w.writerow({"pdf_path": str(pdf), "error": f"{type(e).__name__}: {e}", "skipped": False})

    print(f"[DONE] Preprocess diagnostics: {diag_csv}")
    return diag_csv


def run_segment(
    input_txt_dir: Path,
    output_json_dir: Path,
    log_dir: Path,
    *,
    force: bool = False,
    sample_dir: Optional[Path] = None,
) -> Path:
    """Run TXT -> JSON segmentation."""
    from txt_to_json_segmenter import Config as SegCfg
    from txt_to_json_segmenter import run_batch

    output_json_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if sample_dir is not None:
        sample_dir.mkdir(parents=True, exist_ok=True)

    cfg = SegCfg()
    run_batch(
        input_txt_dir=input_txt_dir,
        output_json_dir=output_json_dir,
        log_dir=log_dir,
        cfg=cfg,
        sample_dir=sample_dir,
        force=force,
    )

    diag_csv = output_json_dir / "diagnostics_segment.csv"
    print(f"[DONE] Segment diagnostics: {diag_csv}")
    return diag_csv


def merge_diagnostics(pre_csv: Path, seg_csv: Path, out_csv: Path) -> None:
    """Optional: merge preprocess & segment diagnostics by stem (best-effort)."""
    if not pre_csv.exists() or not seg_csv.exists():
        return

    # preprocess: key by stem
    pre_map: Dict[str, Dict[str, Any]] = {}
    with open(pre_csv, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            p = row.get("pdf_path") or ""
            stem = Path(p).stem if p else ""
            if stem:
                pre_map[stem] = row

    merged_rows: List[Dict[str, Any]] = []
    with open(seg_csv, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            p = row.get("txt_path") or ""
            stem = Path(p).stem if p else ""
            out = dict(row)
            pre = pre_map.get(stem)
            if pre:
                for k, v in pre.items():
                    out[f"pre_{k}"] = v
            merged_rows.append(out)

    keys = set()
    for r in merged_rows:
        keys |= set(r.keys())
    keys = sorted(keys)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in merged_rows:
            w.writerow(row)


def main() -> None:
    _best_effort_utf8_console()

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input_dirs",
        nargs="+",
        default=[
            r"D:\\code\\Github\\SLAC-test\\SLAC\\data\\en",
            r"D:\\code\\Github\\SLAC-test\\SLAC\\data\\zh",
        ],
        help="PDF input dirs (recursive).",
    )

    ap.add_argument(
        "--output_text_dir",
        default=r"D:\\code\\Github\\SLAC-test\\SLAC\\data\\extracted_text",
        help="Intermediate .txt output dir.",
    )
    ap.add_argument(
        "--output_meta_dir",
        default=r"D:\\code\\Github\\SLAC-test\\SLAC\\data\\extracted_text_meta",
        help="Per-PDF meta json output dir.",
    )
    ap.add_argument(
        "--preprocess_log_dir",
        default=r"D:\\code\\Github\\SLAC-test\\SLAC\\data\\extracted_text_logs",
        help="Per-PDF preprocessing log dir.",
    )

    ap.add_argument(
        "--output_json_dir",
        default=r"D:\\code\\Github\\SLAC-test\\SLAC\\data\\parsed_json",
        help="Final JSON output dir.",
    )
    ap.add_argument(
        "--segment_log_dir",
        default=r"D:\\code\\Github\\SLAC-test\\SLAC\\data\\parsed_json\\_logs",
        help="Per-doc segmentation log dir.",
    )
    ap.add_argument(
        "--sample_dir",
        default=r"D:\\code\\Github\\SLAC-test\\SLAC\\data\\parsed_json\\_samples",
        help="Optional sample markdown output dir.",
    )

    ap.add_argument("--max_pages", type=int, default=0, help="Preprocess: scan at most N pages per PDF (0=all)")
    ap.add_argument("--ocr_lang", default="chi_sim+eng")

    ap.add_argument("--skip_preprocess", action="store_true")
    ap.add_argument("--skip_segment", action="store_true")
    ap.add_argument("--force", action="store_true", help="Force rerun (ignore timestamps)")

    args = ap.parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    output_text_dir = Path(args.output_text_dir)
    output_meta_dir = Path(args.output_meta_dir)
    preprocess_log_dir = Path(args.preprocess_log_dir)
    output_json_dir = Path(args.output_json_dir)
    segment_log_dir = Path(args.segment_log_dir)
    sample_dir = Path(args.sample_dir) if args.sample_dir else None

    pre_csv = output_text_dir / "diagnostics_preprocess.csv"
    seg_csv = output_json_dir / "diagnostics_segment.csv"

    if not args.skip_preprocess:
        pre_csv = run_preprocess(
            input_dirs=input_dirs,
            output_text_dir=output_text_dir,
            output_meta_dir=output_meta_dir,
            log_dir=preprocess_log_dir,
            max_pages=args.max_pages,
            ocr_lang=args.ocr_lang,
            force=args.force,
        )

    if not args.skip_segment:
        seg_csv = run_segment(
            input_txt_dir=output_text_dir,
            output_json_dir=output_json_dir,
            log_dir=segment_log_dir,
            sample_dir=sample_dir,
            force=args.force,
        )

    # merged diagnostics
    try:
        merged_csv = output_json_dir / "diagnostics_all.csv"
        merge_diagnostics(pre_csv, seg_csv, merged_csv)
        print(f"[DONE] Merged diagnostics: {merged_csv}")
    except Exception as e:
        print(f"[WARN] Failed to merge diagnostics: {e}")


if __name__ == "__main__":
    main()
