from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Union

from .docx_reader import read_docx_document
from .json_reader import read_json_document
from .pdf_reader import PDFReaderConfig, read_pdf_document
from .txt_reader import read_txt_document
from ..utils.io import iter_input_files


PathLike = Union[str, Path]


SUPPORTED_SUFFIXES = [".pdf", ".docx", ".txt", ".text", ".md", ".json"]


def read_document(
    path: PathLike,
    *,
    pdf_work_dir: Optional[PathLike] = None,
    pdf_cfg: Optional[PDFReaderConfig] = None,
) -> Dict:
    p = Path(path).expanduser().resolve()
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        if pdf_work_dir is None:
            raise ValueError("pdf_work_dir is required when reading PDF files")
        return read_pdf_document(
            p,
            work_dir=pdf_work_dir,
            cfg=pdf_cfg,
        )

    if suffix == ".docx":
        return read_docx_document(p)

    if suffix in {".txt", ".text", ".md"}:
        return read_txt_document(p)

    if suffix == ".json":
        return read_json_document(p)

    raise ValueError(f"Unsupported input suffix: {suffix} ({p})")


def iter_documents(
    input_paths: Sequence[PathLike],
    *,
    recursive: bool = True,
    pdf_work_dir: Optional[PathLike] = None,
    pdf_cfg: Optional[PDFReaderConfig] = None,
) -> Iterator[Dict]:
    files = iter_input_files(
        input_paths,
        suffixes=SUPPORTED_SUFFIXES,
        recursive=recursive,
        sort_result=True,
        allow_hidden=False,
    )
    for fp in files:
        yield read_document(
            fp,
            pdf_work_dir=pdf_work_dir,
            pdf_cfg=pdf_cfg,
        )