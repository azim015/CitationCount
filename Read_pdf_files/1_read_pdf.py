import os, re, statistics
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
#import pandas as pd

# --- Heuristics you can tune ---
HEADER_FRAC = 0.10          # top 10% of page height
FOOTER_FRAC = 0.12          # bottom 12% of page height
REPEAT_THRESHOLD = 0.5      # appears in >= 50% of pages => header/footer
MIN_BLOCK_CHARS = 12        # ignore tiny noise blocks
CAPTION_PREFIX = re.compile(r'^\s*(figure|fig\.|table|tab\.)\s*[\d:.-]', re.I)

def block_text(block: Dict) -> str:
    """Concatenate text from a PyMuPDF text 'block' dict."""
    parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
    return "".join(parts).strip()

def block_avg_font(block: Dict) -> float:
    sizes = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            sz = span.get("size")
            if isinstance(sz, (int,float)):
                sizes.append(sz)
    return statistics.mean(sizes) if sizes else 0.0

def normalize_for_repetition(text: str) -> str:
    """Normalize text for repetition detection (remove numbers, trim whitespace)."""
    t = re.sub(r'\s+', ' ', text).strip()
    # Remove isolated page numbers / dates that vary but keep publishers/DOIs/etc.
    t = re.sub(r'\b\d{1,4}\b', '#', t)
    return t.lower()

def first_pass_collect(pdf_path: str):
    """
    First pass: collect candidate header/footer texts by page zones and by repetition.
    Returns:
      page_stats: per-page font stats,
      candidates: dict with normalized text counts for top and bottom zones
    """
    candidates_top = Counter()
    candidates_bot = Counter()
    page_font_sizes = []  # median font sizes per page for later caption heuristics

    with fitz.open(pdf_path) as doc:
        for page in doc:
            ph = page.rect.height
            top_cut = ph * HEADER_FRAC
            bot_cut = ph * (1 - FOOTER_FRAC)

            textpage = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
            page_sizes = []

            # scan blocks
            for block in textpage.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue  # only text blocks
                txt = block_text(block)
                if len(txt) < MIN_BLOCK_CHARS:
                    continue

                y0, y1 = block["bbox"][1], block["bbox"][3]
                avg_sz = block_avg_font(block)
                if avg_sz > 0:
                    page_sizes.append(avg_sz)

                norm = normalize_for_repetition(txt)
                if y0 <= top_cut:
                    candidates_top[norm] += 1
                elif y1 >= bot_cut:
                    candidates_bot[norm] += 1

            # store typical font size for this page
            page_font_sizes.append(statistics.median(page_sizes) if page_sizes else 0)

    return page_font_sizes, candidates_top, candidates_bot

def build_repeated_sets(candidates_top: Counter, candidates_bot: Counter, num_pages: int):
    """Select texts that appear on >= REPEAT_THRESHOLD of pages, mark as repeated header/footer."""
    top_repeated = set([t for t,c in candidates_top.items() if c >= num_pages * REPEAT_THRESHOLD])
    bot_repeated = set([t for t,c in candidates_bot.items() if c >= num_pages * REPEAT_THRESHOLD])
    return top_repeated, bot_repeated

def classify_block(block: Dict, page_num: int, page_height: float,
                   page_median_font: float,
                   top_repeated: set, bot_repeated: set) -> str:
    """
    Classify a block into header/footer/body/caption using:
      - position on page
      - repetition across pages
      - caption keywords and smaller font
    """
    txt = block_text(block)
    if len(txt) < MIN_BLOCK_CHARS:
        return None

    y0, y1 = block["bbox"][1], block["bbox"][3]
    top_cut = page_height * HEADER_FRAC
    bot_cut = page_height * (1 - FOOTER_FRAC)

    norm = normalize_for_repetition(txt)

    # Strong header/footer by repetition
    if norm in top_repeated:
        return "header"
    if norm in bot_repeated:
        return "footer"

    # Positional header/footer fallback
    if y0 <= top_cut:
        return "header"
    if y1 >= bot_cut:
        return "footer"

    # Caption heuristic: keyword prefix or significantly smaller font
    avg_sz = block_avg_font(block)
    looks_like_caption = CAPTION_PREFIX.match(txt) is not None
    much_smaller = (avg_sz > 0 and page_median_font > 0 and avg_sz <= 0.85 * page_median_font)
    if looks_like_caption or much_smaller:
        # Many captions sit just below figures/tables in body region.
        # We'll label as caption if it starts with keyword or font-size suggests captioning.
        return "caption"

    return "body"

def extract_sections_to_df(pdf_path: str, article_id: str = None) -> pd.DataFrame:
    """
    Extract sections from a PDF into a DataFrame with:
      article_id, page, section_type, content, bbox, avg_font_size
    """
    if article_id is None:
        article_id = os.path.splitext(os.path.basename(pdf_path))[0]

    # Pass 1: find repeated header/footer texts
    page_font_sizes, cand_top, cand_bot = first_pass_collect(pdf_path)
    with fitz.open(pdf_path) as doc:
        num_pages = len(doc)
    top_rep, bot_rep = build_repeated_sets(cand_top, cand_bot, num_pages)

    # Pass 2: classify blocks and collect rows
    rows = []
    with fitz.open(pdf_path) as doc:
        for pidx, page in enumerate(doc):
            ph = page.rect.height
            textpage = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
            page_median_font = page_font_sizes[pidx] if pidx < len(page_font_sizes) else 0

            for block in textpage.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                txt = block_text(block)
                if len(txt) < MIN_BLOCK_CHARS:
                    continue

                stype = classify_block(block, pidx+1, ph, page_median_font, top_rep, bot_rep)
                if not stype:
                    continue

                rows.append({
                    "article_id": article_id,
                    "page": pidx + 1,
                    "section_type": stype,
                    "content": txt,
                    "bbox": tuple(block["bbox"]),
                    "avg_font_size": round(block_avg_font(block), 2),
                })

    df = pd.DataFrame(rows).sort_values(["page", "section_type"]).reset_index(drop=True)

    # Optional: combine contiguous blocks of same section on the same page
    # (helps produce one header/footer/body/caption chunk per page)
    combined = []
    for (pg, st), grp in df.groupby(["page", "section_type"], sort=True):
        combined.append({
            "article_id": article_id,
            "page": pg,
            "section_type": st,
            "content": "\n".join(grp["content"].tolist()).strip(),
            "avg_font_size": round(statistics.mean([v for v in grp["avg_font_size"].tolist() if v]), 2) if len(grp) else 0,
            "bbox": None  # multiple blocks merged; omit bbox or compute union if needed
        })
    df2 = pd.DataFrame(combined).sort_values(["page", "section_type"]).reset_index(drop=True)
    return df2

# ---- Example usage ----
# if __name__ == "__main__":
#     pdf_file = "your_article.pdf"  # <- change to your file path
#     df = extract_sections_to_df(pdf_file)
#     print(df.head(10))
#     # Save if you like:
#     df.to_csv("sections_extracted.csv", index=False)



def _one(pdf_path: Path):
    return extract_sections_to_df(str(pdf_path))

if __name__ == "__main__":
    folder = Path(r"pdf_files")
    pdf_files = sorted(folder.rglob("*.pdf"))

    dfs = []
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(_one, p): p for p in pdf_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            try:
                dfs.append(fut.result())
            except Exception as e:
                print(f"[WARN] {futures[fut].name} failed: {e}")

    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv("sections_extracted_all.csv", index=False)
