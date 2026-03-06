"""pdf_to_text_preprocessor.py

Robust PDF -> UTF-8 text preprocessor for SLAC.

Design goals
  - Decouple "PDF decoding" from downstream hard-coded segmentation.
  - Prefer fast, layout-aware text extraction.
  - Detect garbling / hidden-noise text layers.
  - If needed, fallback to external extractors (pdftotext) and then OCR.
  - If a page contains images (incl. diagrams), enable OCR fallback logic.

Outputs (per PDF)
  - <output_text_dir>/<stem>.txt         (combined text, page-separated)
  - <output_meta_dir>/<stem>.meta.json   (per-page stats + chosen method)
  - <log_dir>/<stem>.log                 (detailed log)
  - <output_text_dir>/diagnostics.csv    (doc-level summary)

Notes
  - OCR requires a backend:
      * pytesseract + Tesseract OCR installed (recommended)
        - Windows: install Tesseract, ensure tesseract.exe is on PATH
      * or easyocr / paddleocr (optional; not enabled by default)
  - pdftotext fallback requires Poppler (pdftotext in PATH).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF


# -----------------------------
# Text quality / garble detection
# -----------------------------

CID_RE = re.compile(r"cid:\d+", re.I)


def _is_allowed_char(ch: str) -> bool:
    o = ord(ch)
    if ch.isspace():
        return True
    # ASCII printable
    if 0x20 <= o <= 0x7E:
        return True
    # CJK
    if 0x4E00 <= o <= 0x9FFF:
        return True
    # CJK punctuation
    if 0x3000 <= o <= 0x303F:
        return True
    # Fullwidth forms
    if 0xFF00 <= o <= 0xFFEF:
        return True
    # common punctuation / symbols frequently in standards
    if o in {
        0x2010,
        0x2011,
        0x2012,
        0x2013,
        0x2014,
        0x2018,
        0x2019,
        0x201C,
        0x201D,
        0x2026,
        0x00B7,
        0x00D7,
        0x2264,
        0x2265,
        0x00B1,
        0x2212,
        0x03B1,
        0x03B2,
        0x03B3,
        0x03C0,
        0x00A0,
        0x221E,
        0x2192,
        0x2190,
        0x2191,
        0x2193,
        0x25CB,
        0x25CF,
        0x25A1,
        0x25A0,
        0x2030,
        0x2032,
        0x2033,
        0x00B0,
        0x22EF,  # ⋯
    }:
        return True
    return False


def _weird_ratio(text: str) -> float:
    if not text:
        return 1.0
    weird = sum(1 for ch in text if not _is_allowed_char(ch))
    return weird / max(1, len(text))


def score_text_quality(text: str) -> Dict[str, Any]:
    """Return a dict used for routing / diagnostics."""
    t = text or ""
    has_cid = bool(CID_RE.search(t))
    has_repl = "\ufffd" in t
    wr = _weird_ratio(t)
    garbled = has_cid or has_repl or wr >= 0.25
    return {
        "len": len(t),
        "has_cid": has_cid,
        "has_repl": has_repl,
        "weird_ratio": round(wr, 4),
        "garbled": bool(garbled),
    }


def normalize_text_basic(text: str) -> str:
    # remove NUL and other control chars except \n\t\f
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # normalize non-breaking space
    text = text.replace("\u00a0", " ")
    # collapse excessive spaces (but keep newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # trim trailing spaces per line
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    return text


# -----------------------------
# OCR backend
# -----------------------------


class OCRError(RuntimeError):
    pass


@dataclass
class OCRBackend:
    name: str
    available: bool
    detail: str

    def image_to_string(self, pil_image, lang: str) -> str:  # pragma: no cover
        raise NotImplementedError


def get_ocr_backend(prefer: str = "tesseract") -> OCRBackend:
    prefer = (prefer or "tesseract").lower().strip()

    if prefer == "tesseract":
        try:
            import pytesseract  # type: ignore
            from pytesseract import TesseractNotFoundError  # type: ignore

            # Ensure tesseract is reachable
            try:
                _ = pytesseract.get_tesseract_version()
            except TesseractNotFoundError as e:
                return OCRBackend(
                    name="tesseract",
                    available=False,
                    detail=f"TesseractNotFoundError: {e}",
                )

            class _Tess(OCRBackend):
                def image_to_string(self, pil_image, lang: str) -> str:
                    # psm 6 is a decent default for blocks of text
                    config = "--psm 6"
                    return pytesseract.image_to_string(pil_image, lang=lang, config=config)

            return _Tess(name="tesseract", available=True, detail="pytesseract+Tesseract OK")
        except Exception as e:
            return OCRBackend(name="tesseract", available=False, detail=f"import_error: {e}")

    # Placeholders for future expansion
    return OCRBackend(name=prefer, available=False, detail="unsupported backend")


def _pil_from_pixmap(pix: fitz.Pixmap):
    # PIL is optional; required for OCR backend.
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise OCRError(f"PIL not available: {e}")
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    return img


def _preprocess_for_ocr(pil_img):
    """Light preprocessing to improve OCR on scanned PDFs."""
    try:
        from PIL import ImageOps  # type: ignore
    except Exception:
        return pil_img
    img = pil_img
    img = ImageOps.grayscale(img)
    # Simple autocontrast
    img = ImageOps.autocontrast(img)
    return img


# -----------------------------
# Extractors
# -----------------------------


def extract_text_pymupdf_spans(
    page: fitz.Page,
    *,
    min_font_size: float,
    max_weird_ratio_span: float,
    drop_tiny_spans: bool,
    logger: logging.Logger,
) -> Tuple[str, Dict[str, Any]]:
    """Layout-aware extraction with span filtering (helps against hidden-noise layers)."""

    raw = page.get_text("rawdict")
    spans_out: List[Tuple[float, float, str, float]] = []  # (y, x, text, size)
    dropped = 0
    kept = 0
    kept_len = 0

    for block in raw.get("blocks", []) or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            for sp in line.get("spans", []) or []:
                t = sp.get("text") or ""
                if not t.strip():
                    continue
                size = float(sp.get("size") or 0)
                bbox = sp.get("bbox") or [0, 0, 0, 0]
                x0, y0 = float(bbox[0]), float(bbox[1])

                # span-level garble filter
                wr = _weird_ratio(t)
                if CID_RE.search(t) or "\ufffd" in t:
                    dropped += 1
                    continue
                if wr > max_weird_ratio_span:
                    dropped += 1
                    continue
                if drop_tiny_spans and size > 0 and size < min_font_size:
                    dropped += 1
                    continue

                spans_out.append((y0, x0, t, size))
                kept += 1
                kept_len += len(t)

    spans_out.sort(key=lambda z: (z[0], z[1]))

    # Join with heuristic: newline when y-gap large, else space.
    out_lines: List[str] = []
    cur_line: List[str] = []
    last_y: Optional[float] = None
    for y, x, t, size in spans_out:
        if last_y is None:
            cur_line = [t]
            last_y = y
            continue
        if abs(y - last_y) >= 4.0:  # line break threshold
            out_lines.append("".join(cur_line).rstrip())
            cur_line = [t]
            last_y = y
        else:
            # same line
            if cur_line and (cur_line[-1].endswith("-") and re.match(r"^[A-Za-z]", t)):
                # de-hyphenate
                cur_line[-1] = cur_line[-1][:-1] + t.lstrip()
            else:
                # add a space between ASCII chunks; no space for CJK adjacency
                if (cur_line and not re.search(r"[\u4e00-\u9fff]$", cur_line[-1])) and (
                    not re.match(r"^[\u4e00-\u9fff]", t)
                ):
                    cur_line.append(" " + t.lstrip())
                else:
                    cur_line.append(t)

    if cur_line:
        out_lines.append("".join(cur_line).rstrip())

    text = "\n".join(out_lines)
    text = normalize_text_basic(text)

    meta = {
        "kept_spans": kept,
        "dropped_spans": dropped,
        "kept_len": kept_len,
    }
    if dropped > 0 and kept == 0:
        logger.debug("All spans dropped by filters (page=%s)", page.number)
    return text, meta


def page_image_stats(page: fitz.Page) -> Dict[str, Any]:
    """Estimate whether a page is image-heavy."""
    try:
        d = page.get_text("dict")
    except Exception:
        d = {}

    w = float(page.rect.width)
    h = float(page.rect.height)
    page_area = max(1.0, w * h)
    img_area = 0.0
    img_blocks = 0
    for b in d.get("blocks", []) or []:
        if b.get("type") == 1:  # image
            img_blocks += 1
            bbox = b.get("bbox") or [0, 0, 0, 0]
            x0, y0, x1, y1 = map(float, bbox)
            img_area += max(0.0, (x1 - x0) * (y1 - y0))

    images = page.get_images(full=True) or []
    return {
        "images_count": len(images),
        "image_blocks": img_blocks,
        "image_area_ratio": round(img_area / page_area, 4),
    }


def pdftotext_available() -> bool:
    return shutil.which("pdftotext") is not None


def extract_text_pdftotext(pdf_path: Path, page_no: int, logger: logging.Logger) -> Optional[str]:
    """Use Poppler pdftotext for a single page via -f/-l. Returns None if unavailable/fails."""
    exe = shutil.which("pdftotext")
    if not exe:
        return None

    # Use a temp file next to output; keep in memory best-effort.
    # Windows: create in same drive to avoid permission issues.
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out_txt = Path(td) / "out.txt"
            cmd = [
                exe,
                "-enc",
                "UTF-8",
                "-layout",
                "-f",
                str(page_no + 1),
                "-l",
                str(page_no + 1),
                str(pdf_path),
                str(out_txt),
            ]
            cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if cp.returncode != 0:
                logger.debug("pdftotext failed (page=%d): %s", page_no, cp.stderr.strip()[:200])
                return None
            if not out_txt.exists():
                return None
            txt = out_txt.read_text("utf-8", errors="ignore")
            return normalize_text_basic(txt)
    except Exception as e:
        logger.debug("pdftotext exception: %s", e)
        return None


def ocr_page(
    page: fitz.Page,
    *,
    dpi: int,
    backend: OCRBackend,
    lang: str,
    logger: logging.Logger,
) -> Tuple[str, Dict[str, Any]]:
    if not backend.available:
        raise OCRError(f"OCR backend not available: {backend.name} ({backend.detail})")

    pix = page.get_pixmap(dpi=dpi)
    pil_img = _pil_from_pixmap(pix)
    pil_img = _preprocess_for_ocr(pil_img)

    txt = backend.image_to_string(pil_img, lang=lang) or ""
    txt = normalize_text_basic(txt)
    meta = {
        "ocr_backend": backend.name,
        "ocr_lang": lang,
        "dpi": dpi,
    }
    return txt, meta


def ocr_text_is_useful(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 20:
        return False
    q = score_text_quality(t)
    if q["garbled"]:
        return False
    # must contain some letters or CJK
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", t):
        return False
    return True


# -----------------------------
# End-to-end per PDF
# -----------------------------


@dataclass
class Config:
    input_dirs: List[Path]
    output_text_dir: Path
    output_meta_dir: Path
    log_dir: Path
    max_pages: int
    min_font_size: float
    max_weird_ratio_span: float
    drop_tiny_spans: bool
    garbled_page_weird_ratio: float
    garbled_min_len: int
    ocr_on_images: bool
    ocr_always_on_images: bool
    ocr_dpi: int
    ocr_lang: str
    ocr_backend: str
    enable_pdftotext: bool
    ocr_when_text_short: int
    image_area_ratio_for_ocr: float
    page_separator: str


def setup_logger(log_file: Path, level: str) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_file.stem + "_" + str(os.getpid()))
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Remove old handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def decide_page_method(
    *,
    baseline_text: str,
    span_text: str,
    img_stats: Dict[str, Any],
    cfg: Config,
) -> Dict[str, Any]:
    """Return decision dict: {need_ocr, try_pdftotext, use_span_text_as_base, reason}"""
    bq = score_text_quality(baseline_text)
    sq = score_text_quality(span_text)

    has_images = (img_stats.get("images_count", 0) > 0) or (img_stats.get("image_blocks", 0) > 0)
    img_ratio = float(img_stats.get("image_area_ratio", 0.0) or 0.0)

    # Determine page garble
    baseline_garbled = bq["garbled"] or (bq["weird_ratio"] >= cfg.garbled_page_weird_ratio)
    span_garbled = sq["garbled"] or (sq["weird_ratio"] >= cfg.garbled_page_weird_ratio)

    # Base text: prefer span_text if it is longer and cleaner
    use_span = (not span_garbled and (len(span_text) >= len(baseline_text) * 0.6)) or (
        baseline_garbled and not span_garbled
    )

    base_text = span_text if use_span else baseline_text
    base_q = sq if use_span else bq

    need_ocr = False
    try_pdft = False
    reason = []

    if has_images and cfg.ocr_on_images:
        # Enable OCR for image pages. Default behavior can be "always run OCR" and
        # then decide whether to use/append the OCR result.
        if cfg.ocr_always_on_images:
            need_ocr = True
            reason.append("images")
        else:
            # Only run OCR if likely useful.
            if len(base_text.strip()) < cfg.ocr_when_text_short:
                need_ocr = True
                reason.append(f"images+short_text<{cfg.ocr_when_text_short}")
            elif img_ratio >= cfg.image_area_ratio_for_ocr:
                need_ocr = True
                reason.append(f"images+img_ratio>={cfg.image_area_ratio_for_ocr}")

    if base_q["garbled"] or len(base_text.strip()) < cfg.garbled_min_len:
        reason.append("garbled_or_too_short")
        if cfg.enable_pdftotext:
            try_pdft = True
            reason.append("try_pdftotext")
        # If still bad after pdftotext, OCR as last resort.
        need_ocr = True

    return {
        "use_span": use_span,
        "base_q": base_q,
        "need_ocr": need_ocr,
        "try_pdftotext": try_pdft,
        "reason": "+".join(reason) if reason else "ok",
    }


def process_pdf(pdf_path: Path, cfg: Config, diag_writer: Optional[csv.DictWriter] = None) -> Dict[str, Any]:
    log_file = cfg.log_dir / (pdf_path.stem + ".log")
    logger = setup_logger(log_file, "INFO")

    logger.info("Start: %s", pdf_path)

    backend = get_ocr_backend(cfg.ocr_backend)
    if cfg.ocr_backend and not backend.available:
        logger.warning("OCR backend unavailable: %s (%s)", backend.name, backend.detail)

    doc = fitz.open(str(pdf_path))
    try:
        pages = doc.page_count
        scan_pages = pages if cfg.max_pages <= 0 else min(pages, cfg.max_pages)

        page_texts: List[str] = []
        page_meta: List[Dict[str, Any]] = []

        ocr_pages = 0
        pdft_pages = 0
        garbled_pages = 0
        skipped_image_only_pages = 0
        dropped_noise_spans_total = 0

        for i in range(scan_pages):
            page = doc.load_page(i)

            # Baseline extraction
            baseline = page.get_text("text") or ""
            baseline = normalize_text_basic(baseline)

            # Span-filter extraction
            span_text, span_meta = extract_text_pymupdf_spans(
                page,
                min_font_size=cfg.min_font_size,
                max_weird_ratio_span=cfg.max_weird_ratio_span,
                drop_tiny_spans=cfg.drop_tiny_spans,
                logger=logger,
            )
            dropped_noise_spans_total += int(span_meta.get("dropped_spans", 0) or 0)

            img_stats = page_image_stats(page)
            empty_text = len(baseline.strip()) == 0 and len(span_text.strip()) == 0
            image_only = empty_text and (img_stats.get("images_count", 0) > 0)

            if image_only:
                # Page is likely a cover/figure-only page. Keep empty, but record.
                skipped_image_only_pages += 1
                page_texts.append("")
                page_meta.append(
                    {
                        "page": i,
                        "method": "skip_image_only",
                        "img": img_stats,
                        "baseline_q": score_text_quality(baseline),
                        "span_q": score_text_quality(span_text),
                        **span_meta,
                    }
                )
                continue

            decision = decide_page_method(
                baseline_text=baseline,
                span_text=span_text,
                img_stats=img_stats,
                cfg=cfg,
            )
            use_span = bool(decision["use_span"])
            base_text = span_text if use_span else baseline
            base_q = decision["base_q"]

            method = "pymupdf_spans" if use_span else "pymupdf_text"
            chosen = base_text

            # pdftotext fallback
            pdft_txt = None
            if decision["try_pdftotext"]:
                pdft_txt = extract_text_pdftotext(pdf_path, i, logger)
                if pdft_txt is not None:
                    pq = score_text_quality(pdft_txt)
                    # choose if clearly better
                    if (not pq["garbled"] and (pq["len"] >= base_q["len"] * 0.8)) or (
                        base_q["garbled"] and not pq["garbled"]
                    ):
                        chosen = pdft_txt
                        method = "pdftotext"
                        base_q = pq
                        pdft_pages += 1

            # OCR fallback / hybrid
            ocr_txt = None
            ocr_meta = None
            if decision["need_ocr"]:
                if not backend.available:
                    # Backend unavailable: do not spam per-page warnings.
                    ocr_meta = {
                        "skipped": True,
                        "reason": "backend_unavailable",
                        "backend": backend.name,
                        "detail": backend.detail,
                    }
                else:
                    try:
                        ocr_txt, ocr_meta = ocr_page(
                            page,
                            dpi=cfg.ocr_dpi,
                            backend=backend,
                            lang=cfg.ocr_lang,
                            logger=logger,
                        )
                        ocr_pages += 1
                    except Exception as e:
                        logger.warning("OCR failed page=%d: %s", i, e)
                        ocr_txt = None

            # Combine: if chosen is garbled/short -> replace by OCR if useful
            if ocr_txt is not None and ocr_text_is_useful(ocr_txt):
                cq = score_text_quality(chosen)
                if cq["garbled"] or len(chosen.strip()) < cfg.ocr_when_text_short:
                    chosen = ocr_txt
                    method = "ocr"
                else:
                    # Hybrid: append OCR text only if it likely adds content.
                    # Avoid duplication if OCR is basically the same.
                    if ocr_txt.strip() and ocr_txt.strip() not in chosen:
                        chosen = chosen.rstrip() + "\n" + "[OCR]" + "\n" + ocr_txt.strip()
                        method = method + "+ocr"

            chosen = normalize_text_basic(chosen)

            if score_text_quality(chosen)["garbled"]:
                garbled_pages += 1

            page_texts.append(chosen)
            page_meta.append(
                {
                    "page": i,
                    "method": method,
                    "decision": {k: v for k, v in decision.items() if k != "base_q"},
                    "img": img_stats,
                    "baseline_q": score_text_quality(baseline),
                    "span_q": score_text_quality(span_text),
                    "chosen_q": score_text_quality(chosen),
                    "pdft_q": score_text_quality(pdft_txt) if pdft_txt is not None else None,
                    "ocr_meta": ocr_meta,
                    **span_meta,
                }
            )

        # Write outputs
        cfg.output_text_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_meta_dir.mkdir(parents=True, exist_ok=True)

        out_txt = cfg.output_text_dir / f"{pdf_path.stem}.txt"
        out_meta = cfg.output_meta_dir / f"{pdf_path.stem}.meta.json"

        combined = cfg.page_separator.join(page_texts)
        combined = normalize_text_basic(combined)

        out_txt.write_text(combined, "utf-8")
        out_meta.write_text(
            json.dumps(
                {
                    "pdf_path": str(pdf_path),
                    "pages": pages,
                    "scanned_pages": scan_pages,
                    "ocr_pages": ocr_pages,
                    "pdftotext_pages": pdft_pages,
                    "garbled_pages": garbled_pages,
                    "skipped_image_only_pages": skipped_image_only_pages,
                    "dropped_noise_spans_total": dropped_noise_spans_total,
                    "ocr_backend": {
                        "name": backend.name,
                        "available": backend.available,
                        "detail": backend.detail,
                    },
                    "page_meta": page_meta,
                },
                ensure_ascii=False,
                indent=2,
            ),
            "utf-8",
        )

        doc_diag = {
            "pdf_path": str(pdf_path),
            "pages": pages,
            "scanned_pages": scan_pages,
            "ocr_pages": ocr_pages,
            "pdftotext_pages": pdft_pages,
            "garbled_pages": garbled_pages,
            "garbled_ratio": round(garbled_pages / max(1, scan_pages), 6),
            "skipped_image_only_pages": skipped_image_only_pages,
            "dropped_noise_spans_total": dropped_noise_spans_total,
            "ocr_backend": backend.name,
            "ocr_available": backend.available,
        }

        logger.info(
            "Done: pages=%d scanned=%d ocr=%d pdft=%d garbled=%d skip_img_only=%d",
            pages,
            scan_pages,
            ocr_pages,
            pdft_pages,
            garbled_pages,
            skipped_image_only_pages,
        )

        if diag_writer is not None:
            diag_writer.writerow(doc_diag)
        return doc_diag
    finally:
        doc.close()


def iter_pdfs(input_dirs: List[Path]) -> List[Path]:
    pdfs: List[Path] = []
    for d in input_dirs:
        if not d.exists():
            continue
        pdfs.extend(sorted(d.rglob("*.pdf")))
    return pdfs


def main() -> None:
    # Best-effort: improve Windows console utf-8 output
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()

    # Boolean flags: support both --flag and --no-flag where possible.
    try:
        BoolAction = argparse.BooleanOptionalAction  # py>=3.9
    except Exception:  # pragma: no cover
        BoolAction = None
    ap.add_argument(
        "--input_dirs",
        nargs="+",
        default=[
            r"D:\code\Github\SLAC-test\SLAC\data\en",
            r"D:\code\Github\SLAC-test\SLAC\data\zh",
        ],
        help="One or more directories containing PDFs (recursive).",
    )
    ap.add_argument(
        "--output_text_dir",
        default=r"D:\code\Github\SLAC-test\SLAC\data\extracted_text",
        help="Where to write .txt outputs.",
    )
    ap.add_argument(
        "--output_meta_dir",
        default=r"D:\code\Github\SLAC-test\SLAC\data\extracted_text_meta",
        help="Where to write per-PDF meta JSON outputs.",
    )
    ap.add_argument(
        "--log_dir",
        default=r"D:\code\Github\SLAC-test\SLAC\data\extracted_text_logs",
        help="Where to write per-PDF logs.",
    )
    ap.add_argument(
        "--max_pages",
        type=int,
        default=0,
        help="Scan at most N pages per PDF (0=all).",
    )
    ap.add_argument("--log_level", default="INFO", help="INFO|DEBUG|WARNING|ERROR")

    # span filtering
    ap.add_argument("--min_font_size", type=float, default=5.0)
    ap.add_argument("--max_weird_ratio_span", type=float, default=0.35)
    ap.add_argument("--drop_tiny_spans", action="store_true", default=True)

    # garble thresholds
    ap.add_argument("--garbled_page_weird_ratio", type=float, default=0.25)
    ap.add_argument("--garbled_min_len", type=int, default=10)

    # OCR routing
    if BoolAction:
        ap.add_argument("--ocr_on_images", action=BoolAction, default=True)
        ap.add_argument(
            "--ocr_always_on_images",
            action=BoolAction,
            default=False,
            help="If true, run OCR on any page that contains images (even if text exists).",
        )
        ap.add_argument("--enable_pdftotext", action=BoolAction, default=True)
    else:
        ap.add_argument("--ocr_on_images", action="store_true", default=True)
        ap.add_argument("--no_ocr_on_images", action="store_false", dest="ocr_on_images")
        ap.add_argument("--ocr_always_on_images", action="store_true", default=False)
        ap.add_argument(
            "--no_ocr_always_on_images",
            action="store_false",
            dest="ocr_always_on_images",
        )
        ap.add_argument("--enable_pdftotext", action="store_true", default=True)
        ap.add_argument("--no_enable_pdftotext", action="store_false", dest="enable_pdftotext")
    ap.add_argument("--ocr_dpi", type=int, default=300)
    ap.add_argument("--ocr_lang", default="chi_sim+eng")
    ap.add_argument("--ocr_backend", default="tesseract", help="tesseract")
    ap.add_argument(
        "--ocr_when_text_short",
        type=int,
        default=120,
        help="If base extracted text shorter than this and page has images, run OCR.",
    )
    ap.add_argument(
        "--image_area_ratio_for_ocr",
        type=float,
        default=0.35,
        help="If image area ratio >= this and OCR-on-images enabled, run OCR.",
    )

    args = ap.parse_args()

    cfg = Config(
        input_dirs=[Path(x) for x in args.input_dirs],
        output_text_dir=Path(args.output_text_dir),
        output_meta_dir=Path(args.output_meta_dir),
        log_dir=Path(args.log_dir),
        max_pages=int(args.max_pages),
        min_font_size=float(args.min_font_size),
        max_weird_ratio_span=float(args.max_weird_ratio_span),
        drop_tiny_spans=bool(args.drop_tiny_spans),
        garbled_page_weird_ratio=float(args.garbled_page_weird_ratio),
        garbled_min_len=int(args.garbled_min_len),
        ocr_on_images=bool(args.ocr_on_images),
        ocr_always_on_images=bool(args.ocr_always_on_images),
        ocr_dpi=int(args.ocr_dpi),
        ocr_lang=str(args.ocr_lang),
        ocr_backend=str(args.ocr_backend),
        enable_pdftotext=bool(args.enable_pdftotext) and pdftotext_available(),
        ocr_when_text_short=int(args.ocr_when_text_short),
        image_area_ratio_for_ocr=float(args.image_area_ratio_for_ocr),
        page_separator="\n\n\f\n\n",  # form feed between pages
    )

    pdfs = iter_pdfs(cfg.input_dirs)
    if not pdfs:
        print("[INFO] No PDFs found.")
        return

    cfg.output_text_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = cfg.output_text_dir / "diagnostics.csv"
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
    ]
    with open(diag_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for idx, pdf in enumerate(pdfs, start=1):
            print(f"[{idx}/{len(pdfs)}] {pdf}")
            try:
                process_pdf(pdf, cfg, diag_writer=w)
            except Exception as e:
                # Do not crash the batch
                w.writerow(
                    {
                        "pdf_path": str(pdf),
                        "pages": "",
                        "scanned_pages": "",
                        "ocr_pages": "",
                        "pdftotext_pages": "",
                        "garbled_pages": "",
                        "garbled_ratio": "",
                        "skipped_image_only_pages": "",
                        "dropped_noise_spans_total": "",
                        "ocr_backend": cfg.ocr_backend,
                        "ocr_available": get_ocr_backend(cfg.ocr_backend).available,
                    }
                )
                # also print
                print(f"[ERROR] {pdf}: {type(e).__name__}: {e}")

    print("\n[DONE] Text outputs:", cfg.output_text_dir)
    print("[DONE] Meta outputs:", cfg.output_meta_dir)
    print("[DONE] Logs:", cfg.log_dir)
    print("[DONE] Diagnostics:", diag_csv)


if __name__ == "__main__":
    main()
