from pathlib import Path

if __name__ == "__main__":
    folder = Path(".")   # current folder
    for pdf_file in folder.glob("*.pdf"):
        print(f"\n=== Processing {pdf_file.name} ===")
        rows = extract_reference_dois_from_pdf(pdf_file, lookup_missing=True)
        for r in rows:
            print(r)
