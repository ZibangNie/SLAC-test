"""
PDF reader for refiner pipeline.

Main responsibility:
- read PDF
- extract raw text and metadata
- provide unified document object for downstream cleaning/segmenting

Primary source to reuse:
SLAC/segmenter/pdf_to_text_preprocessor.py
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from ..utils.ids import stable_doc_id
from ..utils.io import ensure_dir, iter_input_files, load_json, load_text


PathLike = Union[str, Path]


@dataclass
class PDFReaderConfig:
    max_pages: int = 0

    # span filtering
    min_font_size: float = 5.0
    max_weird_ratio_span: float = 0.35
    drop_tiny_spans: bool = True

    # garble thresholds
    garbled_page_weird_ratio: float = 0.25
    garbled_min_len: int = 10

    # OCR routing
    ocr_on_images: bool = True
    ocr_always_on_images: bool = False
    ocr_dpi: int = 300
    ocr_lang: str = "chi_sim+eng"
    ocr_backend: str = "tesseract"
    enable_pdftotext: bool = True
    ocr_when_text_short: int = 120
    image_area_ratio_for_ocr: float = 0.35

    # output formatting
    page_separator: str = "\n\n\f\n\n"


def _import_pdf_preprocessor():
    """
    Late import so the rest of the pipeline can still be imported on machines
    where PyMuPDF / OCR stack is not installed.
    """
    from SLAC.segmenter.pdf_to_text_preprocessor import (  # type: ignore
        Config as PreCfg,
        pdftotext_available,
        process_pdf,
    )
    return PreCfg, process_pdf, pdftotext_available


def _build_pdf_cfg(
    pdf_path: Path,
    work_dir: Path,
    cfg: PDFReaderConfig,
):
    PreCfg, _, pdftotext_available = _import_pdf_preprocessor()

    text_dir = ensure_dir(work_dir / "text")
    meta_dir = ensure_dir(work_dir / "meta")
    log_dir = ensure_dir(work_dir / "logs")

    pre_cfg = PreCfg(
        input_dirs=[pdf_path.parent],
        output_text_dir=text_dir,
        output_meta_dir=meta_dir,
        log_dir=log_dir,
        max_pages=int(cfg.max_pages),
        min_font_size=float(cfg.min_font_size),
        max_weird_ratio_span=float(cfg.max_weird_ratio_span),
        drop_tiny_spans=bool(cfg.drop_tiny_spans),
        garbled_page_weird_ratio=float(cfg.garbled_page_weird_ratio),
        garbled_min_len=int(cfg.garbled_min_len),
        ocr_on_images=bool(cfg.ocr_on_images),
        ocr_always_on_images=bool(cfg.ocr_always_on_images),
        ocr_dpi=int(cfg.ocr_dpi),
        ocr_lang=str(cfg.ocr_lang),
        ocr_backend=str(cfg.ocr_backend),
        enable_pdftotext=bool(cfg.enable_pdftotext) and bool(pdftotext_available()),
        ocr_when_text_short=int(cfg.ocr_when_text_short),
        image_area_ratio_for_ocr=float(cfg.image_area_ratio_for_ocr),
        page_separator=str(cfg.page_separator),
    )
    return pre_cfg, text_dir, meta_dir, log_dir


def read_pdf_document(
    path: PathLike,
    *,
    doc_id: Optional[str] = None,
    work_dir: PathLike,
    cfg: Optional[PDFReaderConfig] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Read one PDF and return a unified document record:
    {
      "doc_id": "...",
      "source_path": "...",
      "source_type": "pdf",
      "raw_text": "...",
      "meta": {...}
    }

    This reader wraps the old robust PDF preprocessor instead of re-implementing
    PDF extraction logic from scratch.
    """
    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    cfg = cfg or PDFReaderConfig()

    final_doc_id = stable_doc_id(
        explicit_doc_id=doc_id,
        source_path=pdf_path,
        title_hint=None,
    )

    # isolate extraction cache per document to avoid same-stem collisions
    doc_work_dir = ensure_dir(Path(work_dir).expanduser().resolve() / final_doc_id)

    pre_cfg, text_dir, meta_dir, log_dir = _build_pdf_cfg(pdf_path, doc_work_dir, cfg)
    _, process_pdf, _ = _import_pdf_preprocessor()

    # process_pdf writes <stem>.txt and <stem>.meta.json
    diag = process_pdf(pdf_path, pre_cfg, diag_writer=None)

    txt_path = text_dir / f"{pdf_path.stem}.txt"
    meta_path = meta_dir / f"{pdf_path.stem}.meta.json"
    log_path = log_dir / f"{pdf_path.stem}.log"

    if not txt_path.exists():
        raise FileNotFoundError(
            f"PDF preprocessing finished but text output not found: {txt_path}"
        )

    raw_text = load_text(txt_path, encoding="utf-8", errors="ignore")
    page_meta = load_json(meta_path) if meta_path.exists() else None

    meta: Dict[str, Any] = {
        "reader": "pdf_reader",
        "num_chars": len(raw_text),
        "pdf_reader_cache_dir": doc_work_dir.as_posix(),
        "pdf_text_path": txt_path.as_posix(),
        "pdf_meta_path": meta_path.as_posix() if meta_path.exists() else None,
        "pdf_log_path": log_path.as_posix() if log_path.exists() else None,
        "pdf_extract_diag": diag,
        "pdf_reader_cfg": asdict(cfg),
    }
    if page_meta is not None:
        meta["pdf_page_meta"] = page_meta
    if extra_meta:
        meta.update(extra_meta)

    return {
        "doc_id": final_doc_id,
        "source_path": pdf_path.as_posix(),
        "source_type": "pdf",
        "raw_text": raw_text,
        "meta": meta,
    }


def iter_pdf_documents(
    input_paths: Sequence[PathLike],
    *,
    work_dir: PathLike,
    cfg: Optional[PDFReaderConfig] = None,
    recursive: bool = True,
) -> Iterator[Dict[str, Any]]:
    files = iter_input_files(
        input_paths,
        suffixes=[".pdf"],
        recursive=recursive,
        sort_result=True,
        allow_hidden=False,
    )
    for fp in files:
        yield read_pdf_document(
            fp,
            work_dir=work_dir,
            cfg=cfg,
        )