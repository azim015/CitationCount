import polars as pl
from pathlib import Path
from helpers import *

l=get_logger()

def gt_dataset_id_normalization(name:str) -> pl.Expr:
    return (
        pl.when(is_doi_link(name))
        .then(pl.col(name).str.split(DOI_LINK).list.last())
        .otherwise(name)
        .str.to_lowercase()
    )

def main():
    if IS_KAGGLE_SUBMISSION:
        l.debug('skipping check_parse for submission')
        return
    df = (
        get_df('/tmp/train_parse')
        .with_columns(pl.col('text').str.replace_all('\s+', '').str.to_lowercase().alias('text'))
    )

    gt = (
        pl.read_csv(COMP_DIR/'train_labels.csv')
        .filter(pl.col('article_id').is_in(df['article_id']))
        .filter(pl.col('type')!='Missing')
        .with_columns(gt_dataset_id_normalization('dataset_id').alias('norm_id'))
    )

    l.info(f"pymupdf misses: {gt.join(df, on='article_id').with_columns(hit=pl.col('text').str.contains(pl.col('norm_id'), literal=True)).filter(~pl.col('hit')).height} dataset_ids")

if __name__=='__main__': main()