import os
from pathlib import Path
import traceback

import pythoncom
import win32com.client


INPUT_DIR = r"C:\Users\86173\Desktop\铁道数据\文本内容\zh-docx_doc"


def doc_to_pdf_batch(input_dir: str, overwrite: bool = False) -> None:
    """
    Convert all .doc/.docx under input_dir to PDF, saving PDFs alongside originals.
    Does NOT modify originals (opens ReadOnly, closes without saving).
    Requires Windows + Microsoft Word installed.
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in [".doc", ".docx"]])
    if not files:
        print(f"[INFO] No .doc/.docx files found in: {input_dir}")
        return

    pythoncom.CoInitialize()
    word = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0  # wdAlertsNone

        total = len(files)
        ok = 0
        skipped = 0
        failed = 0

        print(f"[INFO] Found {total} files. Starting conversion...")

        for idx, doc_path in enumerate(files, start=1):
            pdf_path = doc_path.with_suffix(".pdf")

            if pdf_path.exists() and not overwrite:
                print(f"[{idx}/{total}] SKIP (pdf exists): {doc_path.name}")
                skipped += 1
                continue

            try:
                # Open as ReadOnly to avoid modifying original
                doc = word.Documents.Open(
                    str(doc_path),
                    ReadOnly=True,
                    AddToRecentFiles=False,
                    ConfirmConversions=False
                )

                # Export to PDF
                # 17 = wdExportFormatPDF
                doc.ExportAsFixedFormat(
                    OutputFileName=str(pdf_path),
                    ExportFormat=17,
                    OpenAfterExport=False,
                    OptimizeFor=0,         # wdExportOptimizeForPrint
                    Range=0,               # wdExportAllDocument
                    Item=0,                # wdExportDocumentContent
                    IncludeDocProps=True,
                    KeepIRM=True,
                    CreateBookmarks=1,     # wdExportCreateHeadingBookmarks
                    DocStructureTags=True,
                    BitmapMissingFonts=True,
                    UseISO19005_1=False
                )

                # Close without saving
                doc.Close(SaveChanges=False)
                ok += 1
                print(f"[{idx}/{total}] OK: {doc_path.name} -> {pdf_path.name}")

            except Exception:
                failed += 1
                print(f"[{idx}/{total}] FAIL: {doc_path.name}")
                traceback.print_exc()

        print("\n[SUMMARY]")
        print(f"  OK      : {ok}")
        print(f"  SKIPPED : {skipped}")
        print(f"  FAILED  : {failed}")

    finally:
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    # overwrite=False: if PDF already exists, skip it
    # set overwrite=True if you want to regenerate PDFs
    doc_to_pdf_batch(INPUT_DIR, overwrite=False)