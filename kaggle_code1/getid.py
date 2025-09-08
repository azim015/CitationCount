import re
import polars as pl
from typing import Optional, Tuple

from helpers import *

COMPILED_PATTERNS = {
    'ref_header_patterns': [re.compile(r'\b(R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S|BIBLIOGRAPHY|LITERATURE CITED|WORKS CITED|CITED WORKS|ACKNOWLEDGEMENTS)\b[:\s]*', re.IGNORECASE)],    
    'citation_pattern': re.compile(r'^\s*(\[\d+\]|\(\d+\)|\d+\.|\d+\)|\d+(?=\s|$))\s*'),
    'first_citation_patterns': [
        re.compile(r'^\s*\[1\]\s*'),
        re.compile(r'^\s*\(1\)\s*'),
        re.compile(r'^\s*1\.\s*'),
        re.compile(r'^\s*1\)\s*'),
        re.compile(r'^\s*1(?=\s|$)'),
    ],
}

l = get_logger()

def find_last_reference_header(text: str, header_patterns: list[re.Pattern]) -> Optional[int]:
    last_match_idx = None
    for pattern in header_patterns:
        matches = list(pattern.finditer(text))
        if matches:
            last_match_idx = matches[-1].start()
    return last_match_idx


def find_last_first_citation(text: str) -> Optional[int]:
    lines = text.splitlines()
    last_match_line = None
    for line_num, line in enumerate(lines):
        line = line.strip()
        for pattern in COMPILED_PATTERNS['first_citation_patterns']:
            if pattern.match(line):
                next_lines = lines[line_num:line_num+3]
                if any(COMPILED_PATTERNS['citation_pattern'].match(l.strip()) for l in next_lines[1:]):
                    last_match_line = line_num
                break
    return last_match_line

def find_reference_start(text: str) -> Optional[int]:
    lines = text.splitlines()
    last_first_citation = find_last_first_citation(text)
    if last_first_citation is not None:
        return last_first_citation
    start_search_idx = int(len(lines) * 0.5)
    for i in range(start_search_idx, len(lines)):
        line = lines[i].strip()
        if COMPILED_PATTERNS['citation_pattern'].match(line):
            next_lines = lines[i:i+3]
            if sum(1 for l in next_lines if COMPILED_PATTERNS['citation_pattern'].match(l.strip())) >= 2:
                for j in range(i, max(-1, i-10), -1):
                    if not COMPILED_PATTERNS['citation_pattern'].match(lines[j].strip()):
                        return j + 1
                return max(0, i-10)
    return None

def split_text_and_references(text: str) -> Tuple[str, str]:
    header_idx = find_last_reference_header(text, COMPILED_PATTERNS['ref_header_patterns'])
    if header_idx is not None:
        header_idx2 = find_last_reference_header(text[:header_idx].strip(), COMPILED_PATTERNS['ref_header_patterns'])
        if header_idx2 is not None:
            header_idx3 = find_last_reference_header(text[:header_idx2].strip(), COMPILED_PATTERNS['ref_header_patterns'])
            if header_idx3 is not None:
                return text[:header_idx3].strip(), text[header_idx3:].strip()
            return text[:header_idx2].strip(), text[header_idx2:].strip()
        return text[:header_idx].strip(), text[header_idx:].strip()
    ref_start_line = find_reference_start(text)
    if ref_start_line is not None:
        lines = text.splitlines()
        body = '\n'.join(lines[:ref_start_line])
        refs = '\n'.join(lines[ref_start_line:])
        return body.strip(), refs.strip()
    return text.strip(), ''

def get_splits(df: pl.DataFrame) -> pl.DataFrame:
    bodies, refs = [], []
    for raw_text in df['text']:
        main, ref = split_text_and_references(raw_text)
        bodies.append(main)
        refs.append(ref)
    return df.with_columns(pl.Series('body', bodies), pl.Series('ref', refs))

def doi_gbif_ids(df):
    doi_pattern = re.compile(r'10\s*\.\s*\d{4,9}\s*/\s*\S+')
    DOI_LINK="https://doi.org/"
    records = []
    for article_id, ref_text in df.select(["article_id","ref"]).rows():
        idx = ref_text.find("GBIF Occurrence")
        if idx == -1:
            continue
        context_window = []
        context_len = 110
        rem_ref = ref_text[idx:]
        while idx != -1 :
            if len(rem_ref) < context_len:
                context_window.append(rem_ref)
                doi_matches = doi_pattern.findall(rem_ref)
            else:
                context_window.append(rem_ref[:context_len])
                doi_matches = doi_pattern.findall(rem_ref[:context_len])
    
            for match in doi_matches:
                cleaned = re.sub(r'\s+', '', match)         
                cleaned = re.sub(r'[^A-Za-z0-9]+$', '', cleaned)  
                cleaned = cleaned.lower()
                dataset_id = DOI_LINK + cleaned
                records.append((article_id, dataset_id, match))
            rem_ref = rem_ref[1:]
            idx = rem_ref.find("GBIF Occurrence")
            if idx != -1 :
                rem_ref = rem_ref[idx:]
    doi_df = pl.DataFrame(records, schema=["article_id", "dataset_id", "match"], orient="row").unique(subset=["article_id", "dataset_id"])
    doi_df= doi_df.with_columns(
        pl.col("match").map_elements(lambda x: [x] if x is not None else [], return_dtype=pl.List(pl.Utf8))
    )
    return doi_df

def tidy_extraction(df) -> pl.DataFrame:
    bad_ids = [f'{DOI_LINK}{e}' for e in ['10.5061/dryad', '10.5281/zenodo', '10.6073/pasta']]

    doi_df = (
        df.with_columns(pl.col('body').str.extract_all(r'10\s*\.\s*\d{4,9}\s*/\s*\S+').alias('match'))
          .explode('match')
          .drop_nulls('match')
          .with_columns(
              pl.col('match').str.replace_all(r'\s+', '')
                             .str.replace(r'[^A-Za-z0-9]+$', '')
                             .str.to_lowercase()
                             .alias('dataset_id')
          )
          .group_by('article_id', 'dataset_id')
          .agg('match')
          .with_columns((DOI_LINK + pl.col('dataset_id')).alias('dataset_id'))
    )

    extra_doi = doi_gbif_ids(df)
    doi_df = pl.concat([doi_df,extra_doi])

    ref_doi = (
        df.with_columns(pl.col('ref').str.extract_all(r'10\s*\.\s*\d{4,9}\s*/\s*\S+').alias('match'))
          .explode('match')
          .drop_nulls('match')
          .with_columns(
              pl.col('match').str.replace_all(r'\s+', '')
                             .str.replace(r'[^A-Za-z0-9]+$', '')
                             .str.to_lowercase()
                             .alias('dataset_id')
          )
          .group_by('article_id', 'dataset_id')
          .agg('match')
          .with_columns((DOI_LINK + pl.col('dataset_id')).alias('dataset_id'))
    )

    doi_df = pl.concat([doi_df,ref_doi])

    # REGEX_IDS = (
    #     r"(?i)\b(?:"
    #     r"CHEMBL\d+|"
    #     r"E-GEOD-\d+|E-PROT-\d+|E-MTAB-\d+|E-MEXP-\d+|EMPIAR-\d+|"
    #     r"ENSBTAG\d+|ENSOARG\d+|"
    #     r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
    #     r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|BX\d{6}|KX\d{6}|K0\d{4}|CAB\d{6}|"
    #     r"NC_\d{6}\.\d{1}|NM_\d{9}|"
    #     r"PRJNA\d+|PRJEB\d+|PRJDB\d+|PXD\d+|SAMN\d+|"
    #     r"GSE\d+|GSM\d+|GPL\d+|"
    #     r"PDB\s?[1-9][A-Z0-9]{3}|HMDB\d+|"
    #     r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+|"
    #     r"(?:SR[PRX]|STH|ERR|DRR|DRX|DRP|ERP|ERX)\d+"
    #     r")"
    # )  

    REGEX_IDS = (
        r"(?i)\b(?:"
        r"CHEMBL\d+|"
        r"E-GEOD-\d+|E-PROT-\d+|E-MTAB-\d+|E-MEXP-\d+|EMPIAR-\d+|"
        r"ENSBTAG\d+|ENSOARG\d+|"
        r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
        r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|BX\d{6}|KX\d{6}|K0\d{4}|CAB\d{6}|"
        r"NC_\d{6}\.\d{1}|NM_\d{9}|"
        r"PRJNA\d+|PRJEB\d+|PRJDB\d+|PXD\d+|SAMN\d+|"
        r"GSE\d+|GSM\d+|GPL\d+|"
        r"PDB\s?[1-9][A-Z0-9]{3}|HMDB\d+|"
        r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+|"
        r"(?:SR[PRX]|STH|ERR|DRR|DRX|DRP|ERP|ERX)\d+|"
        r"CVCL_[A-Z0-9]{4}|"
        r"[1-5]\.(?:10|20|30|40|50|60|70|80|90)\.\d{2,4}\.\d{2,4}"
        r")"
    )

    # REGEX_IDS = (
    #     r"(?i)\b(?:"
    #     r"CHEMBL\d+|"
    #     r"E-GEOD-\d+|E-PROT-\d+|EMPIAR-\d+|"
    #     r"ENSBTAG\d+|ENSOARG\d+|ENSMMUT\d+|"
    #     r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
    #     r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|KX\d{6}|K0\d{4}|"
    #     r"PRJNA\d+|PRJEB\d+|PXD\d+|SAMN\d+|"
    #     r"GSE\d+|GSM\d+|GPL\d+|"
    #     r"E-MTAB-\d+|E-MEXP-\d+|"
    #     r"PDB\s?[1-9][A-Z0-9]{3}|HMDB\d+|"
    #     r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+|"
    #     r"(?:SRR|SRX|SRP|ERR|DRR|DRX|DRP|ERP|ERX)\d+|"
    #     r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5}|CAB\d{6})|"
    #     r"CVCL_[A-Z0-9]{4}|"
    #     r"\d+\.\d+\.\d+\.\d+|"              # SCOP classification
    #     r"NM_\d+|"                           # RefSeq mRNA
    #     r"[A-Z]{2}\d{6}|"                    # GenBank 2-letter
    #     r"STH\d{5}"                          # STH IDs
    #     r")"
    # )



    
    acc_df = (
        df.with_columns(
            pl.col('text').str.extract_all(REGEX_IDS).alias('match')
        )
        .explode('match')
        .drop_nulls('match')
        .with_columns(
            pl.col('match').str.replace_all(r'\s+', '')
                           .str.replace(r'[^A-Za-z0-9]+$', '')
                           .str.replace(r'(?i)^PDB', '')
                           .alias('dataset_id')
        )
        .group_by('article_id', 'dataset_id')
        .agg('match')
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('dryad.'))
              .then(f'{DOI_LINK}10.5061/' + pl.col('dataset_id'))
              .otherwise('dataset_id')
              .alias('dataset_id')
        )
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('pasta/'))
              .then(f'{DOI_LINK}10.6073/' + pl.col('dataset_id'))
              .otherwise('dataset_id')
              .alias('dataset_id')
        )
    )

    df = pl.concat([doi_df, acc_df])

    df = (
        df.unique(['article_id', 'dataset_id'])  # CHANGED
          .filter(~pl.col('article_id').str.replace('_','/').str.contains(pl.col('dataset_id').str.split(DOI_LINK).list.last().str.escape_regex()))
          .filter(~pl.col('dataset_id').str.contains(pl.col('article_id').str.replace('_','/').str.escape_regex()))
          .filter(~pl.col('dataset_id').str.contains('figshare', literal=True))
          .filter(~pl.col('dataset_id').is_in(bad_ids))
          .filter(
              pl.when(is_doi_link('dataset_id') &
                      (pl.col('dataset_id').str.split('/').list.last().str.len_chars() < 5))
               .then(False)
               .otherwise(True)
          )
          .with_columns(pl.col('match').list.unique())
    )
    return df

def get_context_window(text: str, substring: str, window: int = 100) -> str:
    idx = text.find(substring)
    if idx == -1:
        raise ValueError
    start = max(idx - window, 0)
    end = min(idx + len(substring) + window, len(text))
    return text[start:end]

def get_window_df(text_df, ids_df):
    df = ids_df.join(text_df, on='article_id')
    windows = []
    for text, match_ids in df.select('text', 'match').rows():
        windows.append(get_context_window(text, match_ids[0]))
    return df.with_columns(pl.Series('window', windows)).select('article_id', 'dataset_id', 'window')

def write_the_match(text_df,id_df):
    df = id_df.join(text_df,on=['article_id'])
    records=[]
    for art_id,dataset_id,match_ids,text in df.select('article_id','dataset_id','match','text').rows():
        records.append({'article_id':art_id,'dataset_id':dataset_id,'match':match_ids[0],'text':text})

    pl.DataFrame(records).write_parquet('/tmp/context_data.parquet')

def main():
    text_df = get_df('/tmp/train_parse')
    df = get_splits(text_df)
    df = tidy_extraction(df)
        
    write_the_match(text_df,df)
    
    df = get_window_df(text_df, df)
    df.write_parquet('/tmp/extracted.parquet')
    df = assume_type(df)
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
    if not IS_KAGGLE_SUBMISSION:
        results = evaluate(df)
        for r in results: l.info(r)
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results: l.info(r)

if __name__=='__main__': main()