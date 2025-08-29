#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import json
import time
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# --- Text extraction (pdfminer.six) ---
from pdfminer.high_level import extract_text

# Optional fuzzy helpers (useful if you later want to cluster near-duplicate refs)
from rapidfuzz import fuzz

# --- DOI patterns (robust) ---
DOI_CORE = r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'
RE_DOI_BARE = re.compile(rf'\b({DOI_CORE})\b')
RE_DOI_URL = re.compile(rf'https?://(?:dx\.)?doi\.org/({DOI_CORE})', re.IGNORECASE)
RE_DOI_PREFIX = re.compile(rf'\bdoi:\s*({DOI_CORE})', re.IGNORECASE)

# Headings that typically mark the start of the references section
REF_START_CUES = [
    r'\breferences\b',
    r'\bbibliography\b',
    r'\bworks\s+cited\b'
]
RE_REF_START = re.compile("|".join(REF_START_CUES), re.IGNORECASE)

# Headings that often appear *after* references and can mark the end
REF_END_CUES = [
    r'\bappendix\b',
    r'\bappendices\b',
    r'\backnowledg(e)?ments\b',
    r'\bsupplementary\b',
    r'\babout the authors\b',
    r'\bauthor biography\b',
    r'\bfootnotes\b'
]
RE_REF_END = re.compile("|".join(REF_END_CUES), re.IGNORECASE)

# Reference item markers (numbered, bracketed, or bullets)
RE_ITEM_MARKERS = [
    r'^\s*\[\d+\]\s+',          # [1] ...
    r'^\s*\d+\.\s+',            # 1. ...
    r'^\s*•\s+',                # bullets
    r'^\s*-\s+'                 # dashes
]
RE_ITEM_SPLIT = re.compile("|".join(RE_ITEM_MARKERS), re.MULTILINE)

def extract_pdf_text(pdf_path: Path) -> str:
    text = extract_text(str(pdf_path)) or ""
    return normalize_text(text)

def normalize_text(text: str) -> str:
    # Fix hyphenation at line breaks: "trans-\nformers" -> "transformers"
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    # Collapse excessive newlines but keep paragraph breaks
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalize unicode quotes/dashes (optional)
    return text

def slice_references_block(full_text: str) -> str:
    # Find start
    start_match = RE_REF_START.search(full_text)
    if not start_match:
        return ""  # No obvious References section detected
    start_idx = start_match.start()

    # Find end *after* start
    end_match = RE_REF_END.search(full_text, pos=start_idx + 1)
    end_idx = end_match.start() if end_match else len(full_text)

    ref_block = full_text[start_idx:end_idx].strip()
    return ref_block

def split_references(ref_block: str) -> List[str]:
    """
    Split the references block into individual references.
    Works for numbered/bracketed lists and fallback to paragraph-based split.
    """
    if not ref_block:
        return []

    # Remove the heading line itself (e.g., "References")
    #ref_block_wo_heading = re.sub(RE_REF_START, "", ref_block, count=1, flags=re.IGNORECASE).strip()

    # after
    ref_block_wo_heading = RE_REF_START.sub("", ref_block, count=1).strip()

    # If we find explicit markers, split by them and reattach the marker text for readability
    pieces = RE_ITEM_SPLIT.split(ref_block_wo_heading)
    if len(pieces) > 1:
        # RE_ITEM_SPLIT removes the marker; reconstruct items by joining text segments smartly
        # A simpler approach: split by lines that look like markers and then join until next marker.
        items = []
        current = []
        for line in ref_block_wo_heading.splitlines():
            if re.match(RE_ITEM_SPLIT, line):
                if current:
                    items.append(" ".join(current).strip())
                    current = []
                # Remove the marker from the start of this line
                line = re.sub(RE_ITEM_SPLIT, "", line).strip()
                current.append(line)
            else:
                current.append(line.strip())
        if current:
            items.append(" ".join(current).strip())
        # Clean empties
        items = [re.sub(r'\s+', ' ', it).strip(' .;') for it in items if it.strip()]
        return items

    # Fallback: split by blank lines / paragraph breaks
    paras = [p.strip() for p in re.split(r'\n\s*\n', ref_block_wo_heading) if p.strip()]
    # Merge multi-line refs: collapse internal newlines
    refs = [re.sub(r'\s+', ' ', p).strip(' .;') for p in paras]
    # Filter very short fragments
    refs = [r for r in refs if len(r) > 30]
    return refs

def extract_doi_from_text(s: str) -> Optional[str]:
    for pat in (RE_DOI_URL, RE_DOI_PREFIX, RE_DOI_BARE):
        m = pat.search(s)
        if m:
            doi = m.group(1)
            # Strip trailing punctuation/brackets
            doi = doi.rstrip('.,);]}>\"\'')
            doi = doi.lstrip('<[{(\"\'')
            return doi
    return None

def crossref_lookup_bibliographic(bibliographic: str, timeout=10) -> Optional[str]:
    """
    Try to resolve a DOI via Crossref using the full reference string.
    Note: Respect polite rate-limits; Crossref suggests ≤ 1 request/sec.
    """
    import requests
    url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic": bibliographic,
        "rows": 3,   # fetch a few, we'll pick the best
        "select": "DOI,title,author,issued,score"
    }
    headers = {
        "User-Agent": "doi-finder-script/1.0 (mailto:youremail@example.com)"
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        items = data.get("message", {}).get("items", [])
        # Heuristic: take the highest score with a valid DOI
        best = None
        best_score = -1
        for it in items:
            doi = it.get("DOI")
            score = it.get("score", 0)
            if doi and score > best_score:
                best = doi
                best_score = score
        return best
    except Exception:
        return None

def dedupe_close_strings(refs: List[str], threshold: int = 95) -> List[str]:
    """
    Optional: merge near-duplicate references (caused by PDF extraction noise).
    Keeps first occurrence of a near-duplicate cluster.
    """
    kept = []
    for r in refs:
        if not any(fuzz.token_sort_ratio(r, k) >= threshold for k in kept):
            kept.append(r)
    return kept

def extract_reference_dois_from_pdf(
    pdf_path: Path,
    lookup_missing: bool = False,
    sleep_between_lookups: float = 1.0
) -> List[Dict[str, Optional[str]]]:
    """
    Returns a list of dicts: {"reference": <string>, "doi": <doi or None>}
    """
    full_text = extract_pdf_text(pdf_path)
    ref_block = slice_references_block(full_text)
    refs = split_references(ref_block)

    # Optionally dedupe near-duplicates
    refs = dedupe_close_strings(refs, threshold=96)

    results = []
    for ref in refs:
        doi = extract_doi_from_text(ref)
        if not doi and lookup_missing:
            doi = crossref_lookup_bibliographic(ref)
            if sleep_between_lookups:
                time.sleep(sleep_between_lookups)
        results.append({"reference": ref, "doi": doi})
    return results

def save_outputs(basepath: Path, rows: List[Dict[str, Optional[str]]]) -> Tuple[Path, Path]:
    json_path = basepath.with_suffix(".citations.json")
    csv_path = basepath.with_suffix(".citations.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reference", "doi"])
        for r in rows:
            w.writerow([r["reference"], r["doi"] or ""])

    return json_path, csv_path

# def main():
#     import argparse
#     p = argparse.ArgumentParser(
#         description="Extract DOIs of citations from a research-paper PDF."
#     )
#     p.add_argument("pdf", type=str, help="Path to the PDF file")
#     p.add_argument("--lookup-missing", action="store_true",
#                    help="Use Crossref to find DOIs for references without explicit DOIs")
#     p.add_argument("--no-sleep", action="store_true",
#                    help="Do not sleep between Crossref lookups (not recommended)")
#     args = p.parse_args()

#     pdf_path = Path(args.pdf).expanduser().resolve()
#     if not pdf_path.exists():
#         print(f"ERROR: File not found: {pdf_path}", file=sys.stderr)
#         sys.exit(1)

#     rows = extract_reference_dois_from_pdf(
#         pdf_path,
#         lookup_missing=args.lookup_missing,
#         sleep_between_lookups=0.0 if args.no_sleep else 1.0,
#     )

#     # Summary
#     found = sum(1 for r in rows if r["doi"])
#     total = len(rows)
#     print(f"Parsed references: {total}")
#     print(f"DOIs found:        {found}")
#     print(f"Coverage:          {found/max(total,1):.0%}")

#     # Save
#     json_path, csv_path = save_outputs(pdf_path, rows)
#     print(f"\nSaved: {json_path}")
#     print(f"Saved: {csv_path}")

#     # Also print unique DOIs list:
#     unique_dois = sorted({r["doi"] for r in rows if r["doi"]})
#     if unique_dois:
#         print("\nUnique DOIs:")
#         for d in unique_dois:
#             print(d)

# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    folder = Path(".")   # current folder
    for pdf_file in folder.glob("*.pdf"):
        print(f"\n=== Processing {pdf_file.name} ===")
        rows = extract_reference_dois_from_pdf(pdf_file, lookup_missing=True)
        for r in rows:
            print(r)