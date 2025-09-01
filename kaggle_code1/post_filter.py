import polars as pl
from helpers import *

"""
Fourth essence: Post-filter to cut FP DOIs that look like literature.
- Read /kaggle/working/submission.csv (output of llm_validate.py)
- Join with /tmp/extracted.parquet to get context window
- Drop DOI rows that (1) start with typical publisher prefixes AND (2) have no data-ish words nearby
- Keep accessions untouched
"""

l = get_logger()

PAPER_PREFIXES = [
    "10.5061","10.5281","10.17632","10.1594","10.15468","10.17882","10.7937","10.7910","10.6073",
    "10.3886","10.3334","10.4121","10.5066","10.5067","10.18150","10.25377","10.25387","10.23642","10.24381","10.22033"
]

CONTEXT_RE = r"(?i)\b(data(?:set)?|repository|archive|deposited|available|supplementary|raw(?:\s+data)?|uploaded|hosted|stored|accession)\b"

def is_paper_prefix(col: str = "dataset_id") -> pl.Expr:
    expr = pl.lit(False)
    for p in PAPER_PREFIXES:
        expr = expr | pl.col(col).str.starts_with(f"{DOI_LINK}{p}")
    return expr

def main():
    sub = pl.read_csv("/kaggle/working/submission.csv")

    # Normalize columns: drop row_id if present so concat widths match
    if "row_id" in sub.columns:
        sub = sub.drop("row_id")

    # Context windows
    win = pl.read_parquet("/tmp/extracted.parquet").select("article_id", "dataset_id", "window")

    # DOI & ACC split
    doi_rows = sub.filter(is_doi_link("dataset_id")).join(win, on=["article_id", "dataset_id"], how="left")
    acc_rows = sub.filter(~is_doi_link("dataset_id"))

    keep_mask = (
        (~is_paper_prefix("dataset_id"))  # not a known paper prefix
        | doi_rows["window"].fill_null("").str.contains(CONTEXT_RE)
    )

    kept_doi = doi_rows.filter(keep_mask).select("article_id", "dataset_id", "type")
    final = pl.concat([kept_doi, acc_rows.select("article_id", "dataset_id", "type")])

    # Re-eval & save
    if not IS_KAGGLE_SUBMISSION:
        for r in evaluate(final): l.info(r)
        for r in evaluate(final, on=["article_id", "dataset_id", "type"]): l.info(r)

    final.with_row_index("row_id").write_csv("/kaggle/working/submission.csv")

if __name__ == "__main__":
    main()