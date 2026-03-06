import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

import fitz  # PyMuPDF


# -------------------------
# Heuristics
# -------------------------
CID_RE = re.compile(r"cid:\d+", re.I)

def is_allowed_char(ch: str) -> bool:
    o = ord(ch)
    # whitespace
    if ch.isspace():
        return True
    # basic ascii printable
    if 0x20 <= o <= 0x7E:
        return True
    # CJK Unified Ideographs
    if 0x4E00 <= o <= 0x9FFF:
        return True
    # CJK punctuation / fullwidth forms (rough)
    if 0x3000 <= o <= 0x303F:  # CJK symbols & punctuation
        return True
    if 0xFF00 <= o <= 0xFFEF:  # Halfwidth and Fullwidth Forms
        return True
    # common punctuation (dash, quotes, etc.)
    if o in {0x2010,0x2011,0x2012,0x2013,0x2014,0x2018,0x2019,0x201C,0x201D,0x2026,0x00B7}:
        return True
    return False

def garbled_score(text: str) -> Dict[str, Any]:
    if text is None:
        text = ""
    t = text
    has_cid = bool(CID_RE.search(t))
    has_repl = "\ufffd" in t  # replacement char
    n = len(t)
    if n == 0:
        return {
            "len": 0,
            "has_cid": has_cid,
            "has_repl": has_repl,
            "weird_ratio": 1.0,
            "garbled": True,
        }
    weird = sum(1 for ch in t if not is_allowed_char(ch))
    weird_ratio = weird / max(1, n)
    garbled = has_cid or has_repl or weird_ratio >= 0.25
    return {
        "len": n,
        "has_cid": has_cid,
        "has_repl": has_repl,
        "weird_ratio": round(weird_ratio, 4),
        "garbled": garbled,
    }

def font_has_tounicode(doc: fitz.Document, xref: int) -> bool:
    # best-effort: check presence of ToUnicode key in font object
    try:
        key, val = doc.xref_get_key(xref, "ToUnicode")
        if not val:
            return False
        v = str(val).strip().lower()
        if v in {"null", "none", ""}:
            return False
        # typical value: "123 0 R"
        return True
    except Exception:
        return False

def font_is_embedded(font_tuple: Tuple) -> bool:
    # PyMuPDF page.get_fonts(full=True) may include "stream" xref as last item (0 if not embedded)
    # Different versions vary, so we try a robust approach.
    try:
        # often: (xref, ext, type, basefont, name, encoding, stream)
        if len(font_tuple) >= 7:
            stream = font_tuple[6]
            if isinstance(stream, int):
                return stream != 0
            # sometimes it's a string
            s = str(stream).strip().lower()
            return s not in {"0", "null", "none", ""}
    except Exception:
        pass
    # fallback unknown => False
    return False


# -------------------------
# Main
# -------------------------
def audit_pdf(pdf_path: Path, max_pages: int = 0) -> Dict[str, Any]:
    doc = fitz.open(str(pdf_path))
    try:
        total_pages = doc.page_count
        pages_to_scan = total_pages if max_pages <= 0 else min(total_pages, max_pages)

        garbled_pages = 0
        cid_pages = 0
        repl_pages = 0
        empty_text_pages = 0
        image_only_pages = 0

        # font stats
        seen_font_xrefs: Set[int] = set()
        embedded_font_xrefs: Set[int] = set()
        tounicode_font_xrefs: Set[int] = set()
        font_name_samples: Dict[int, str] = {}

        sample_garbled_snippet = ""
        sample_garbled_page = -1

        for i in range(pages_to_scan):
            page = doc.load_page(i)
            txt = page.get_text("text") or ""
            sc = garbled_score(txt)

            if sc["len"] == 0:
                empty_text_pages += 1
                # if page has images and no text, likely scanned/image-based
                if page.get_images(full=True):
                    image_only_pages += 1

            if sc["garbled"]:
                garbled_pages += 1
                if sc["has_cid"]:
                    cid_pages += 1
                if sc["has_repl"]:
                    repl_pages += 1
                if not sample_garbled_snippet and sc["len"] > 0:
                    sample_garbled_page = i
                    sample_garbled_snippet = txt[:200].replace("\n", " ")

            # fonts used on this page
            fonts = page.get_fonts(full=True)
            for ft in fonts:
                try:
                    xref = int(ft[0])
                except Exception:
                    continue
                seen_font_xrefs.add(xref)
                if xref not in font_name_samples:
                    # basefont/name fields vary by version; try best-effort
                    name = ""
                    if len(ft) >= 5:
                        name = str(ft[3] or "") + " | " + str(ft[4] or "")
                    else:
                        name = str(ft)
                    font_name_samples[xref] = name[:200]

                if font_is_embedded(ft):
                    embedded_font_xrefs.add(xref)
                if font_has_tounicode(doc, xref):
                    tounicode_font_xrefs.add(xref)

        # Summaries
        unique_fonts = len(seen_font_xrefs)
        embedded_ratio = (len(embedded_font_xrefs) / unique_fonts) if unique_fonts else 0.0
        tounicode_ratio = (len(tounicode_font_xrefs) / unique_fonts) if unique_fonts else 0.0

        return {
            "pdf_path": str(pdf_path),
            "pages": total_pages,
            "scanned_pages": pages_to_scan,
            "garbled_pages": garbled_pages,
            "garbled_ratio": round(garbled_pages / max(1, pages_to_scan), 4),
            "cid_pages": cid_pages,
            "repl_pages": repl_pages,
            "empty_text_pages": empty_text_pages,
            "image_only_pages": image_only_pages,
            "unique_fonts": unique_fonts,
            "embedded_fonts": len(embedded_font_xrefs),
            "embedded_ratio": round(embedded_ratio, 4),
            "fonts_with_tounicode": len(tounicode_font_xrefs),
            "tounicode_ratio": round(tounicode_ratio, 4),
            "sample_garbled_page": sample_garbled_page,
            "sample_garbled_snippet": sample_garbled_snippet,
        }
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dirs",
        nargs="+",
        default=[
            r"D:\code\Github\SLAC-test\SLAC\data\en",
            r"D:\code\Github\SLAC-test\SLAC\data\zh",
        ],
        help="PDF dirs (recursive).",
    )
    ap.add_argument(
        "--out_csv",
        default=r"D:\code\Github\SLAC-test\SLAC\data\pdf_audit_report.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--max_pages",
        type=int,
        default=5,
        help="Scan first N pages per PDF (0=all).",
    )
    args = ap.parse_args()

    pdfs: List[Path] = []
    for d in args.input_dirs:
        p = Path(d)
        if p.exists():
            pdfs.extend(sorted(p.rglob("*.pdf")))

    if not pdfs:
        print("[INFO] No PDFs found.")
        return

    rows: List[Dict[str, Any]] = []
    for idx, pdf in enumerate(pdfs, start=1):
        print(f"[{idx}/{len(pdfs)}] {pdf}")
        try:
            rows.append(audit_pdf(pdf, max_pages=args.max_pages))
        except Exception as e:
            rows.append({
                "pdf_path": str(pdf),
                "error": f"{type(e).__name__}: {e}",
            })

    # write CSV
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    keys = sorted(keys)

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n[DONE] Report:", outp)


if __name__ == "__main__":
    main()