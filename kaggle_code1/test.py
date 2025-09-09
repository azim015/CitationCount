from pathlib import Path
import re
import unicodedata
import pymupdf  # aka PyMuPDF

# Tunables
HEADER_FRACTION = 0.08   # top 8% of the page considered header
FOOTER_FRACTION = 0.08   # bottom 8% considered footer
ADD_PAGE_DIVIDERS = True

# --- helpers ---------------------------------------------------------------
#PDF_DIR = Path("files")
#PDF_DIR = Path(r"C:\Users\arezaei\OneDrive - Texas A&M University-Corpus Christi\Research & Academic\Kaggle\Citation Count\files")
PDF_DIR = Path(r"C:\Users\arezaei\OneDrive - Texas A&M University-Corpus Christi\Research & Academic\Kaggle\Citation Count\kaggle_code1\files")
output_dir = Path(r"C:\Users\arezaei\OneDrive - Texas A&M University-Corpus Christi\Research & Academic\Kaggle\Citation Count\kaggle_code1\output_texts")
_PAGE_NUM_RE = re.compile(
    r"^\s*(?:page\s*)?\d{1,4}(?:\s*/\s*\d{1,4}|\s+of\s+\d{1,4})?\s*$",
    flags=re.IGNORECASE
)

def _dehyphenate_and_normalize(s: str) -> str:
    # remove hyphen + linebreak splits: trans-\nmission -> transmission
    s = re.sub(r'-\s*\n\s*', '', s)
    # remove soft hyphens and normalize Unicode (fi/fl ligatures, etc.)
    s = s.replace('\u00ad', '')
    s = unicodedata.normalize('NFKC', s)
    # collapse runs of spaces/tabs (keep newlines as block separators)
    s = re.sub(r'[ \t]+', ' ', s)
    # trim right spaces per line
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s.strip()

def _strip_header_footer_blocks(page) -> str:
    """Extract text blocks and drop header/footer bands + page-number-only lines."""
    blocks = page.get_text("blocks")  # [(x0,y0,x1,y1,text,...), ...]
    page_h = page.rect.height
    header_y = page_h * HEADER_FRACTION
    footer_y = page_h * (1 - FOOTER_FRACTION)

    kept_lines = []
    for x0, y0, x1, y1, text, *rest in blocks:
        # Skip header/footer regions
        if y1 < header_y or y0 > footer_y:
            continue
        if not text or not text.strip():
            continue
        # Split into lines and drop isolated page-number lines
        for line in text.splitlines():
            if _PAGE_NUM_RE.match(line.strip()):
                continue
            kept_lines.append(line)

    return "\n".join(kept_lines)

def _extract_clean_page_text(page) -> str:
    """Best-effort clean text for a single page."""
    try:
        txt = _strip_header_footer_blocks(page)
        if not txt.strip():
            # fallback to plain extraction if blocks look empty
            txt = page.get_text() or ""
    except Exception:
        # fallback if block extraction crashes (e.g., bad annotations)
        txt = page.get_text() or ""
    return _dehyphenate_and_normalize(txt)

# --- main ------------------------------------------------------------------

def pdf_to_txt(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Case-insensitive find of PDFs
    pdf_files = [p for p in PDF_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    existing_txt_files = {f.stem for f in output_dir.glob("*.txt")}
    final_text = ""
    for pdf_path in sorted(pdf_files):
        txt_path = output_dir / f"{pdf_path.stem}.txt"
        if pdf_path.stem in existing_txt_files:
            continue  # already processed

        pages_out = []
        try:
            with pymupdf.open(pdf_path) as doc:
                for i, page in enumerate(doc, start=1):
                    page_text = _extract_clean_page_text(page)
                    if not page_text:
                        continue
                    if ADD_PAGE_DIVIDERS:
                        pages_out.append(f"=== [PAGE {i}] ===\n{page_text}")
                    else:
                        pages_out.append(page_text)
        except Exception as e:
            # Common when PDFs carry malformed annotation appearance streams, etc.
            print(f"[WARN] Failed to process {pdf_path.name}: {e}")
            continue

        final_text = "\n\n".join(pages_out).strip()

        final_text = re.sub(r'\s*\n\s*', '', final_text)
        final_text = re.sub(r'\s+', ' ', final_text).strip()
        
        print(final_text)
        if final_text:
            try:
                txt_path.write_text(final_text + "\n", encoding="utf-8")
            except Exception as e:
                print(f"[WARN] Could not write {txt_path.name}: {e}")
        else:
            print(f"[INFO] No extractable text in {pdf_path.name}; skipped.")
            
        #final_text = final_text.replace("\n", " ")
    

pdf_to_txt(output_dir)