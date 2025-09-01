import logging, os, kagglehub, inspect
from pathlib import Path
import polars as pl

IS_KAGGLE_ENV = sum(['KAGGLE' in k for k in os.environ]) > 0
IS_KAGGLE_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
COMP_DIR = Path(('/kaggle/input/make-data-count-finding-data-references' if IS_KAGGLE_SUBMISSION else kagglehub.competition_download('make-data-count-finding-data-references')))
PDF_DIR = COMP_DIR / ('test' if IS_KAGGLE_SUBMISSION else 'train') / 'PDF'
WORKING_DIR = Path(('/kaggle/working/' if IS_KAGGLE_ENV else '.working/'))

DOI_LINK = 'https://doi.org/'

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper() if not IS_KAGGLE_SUBMISSION else "WARNING"
LOG_FILE_PATH = os.getenv("LOG_FILE", "logs/project.log")
LOG_DIR = Path(LOG_FILE_PATH).parent

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(levelname)s %(asctime)s  [%(filename)s:%(lineno)d - %(funcName)s()] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def get_logger(name=None):
    if name is None:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            name = "__main__"
        else:
            name = frame.f_back.f_globals.get("__name__", "__main__")

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(DEFAULT_LOG_LEVEL)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
        ch = logging.StreamHandler()
        ch.setLevel(DEFAULT_LOG_LEVEL)
        ch.setFormatter(formatter)
        fh = logging.FileHandler(LOG_FILE_PATH)
        fh.setLevel(DEFAULT_LOG_LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
        logger.propagate = False
    return logger

def is_doi_link(name: str) -> pl.Expr:
    return pl.col(name).str.starts_with(DOI_LINK).and_(
        ~pl.col(name).str.contains(r"/dl\.")
    )

def string_normalization(name: str) -> pl.Expr:
    return pl.col(name).str.normalize("NFKC").str.replace_all(r"[^\p{Ascii}]", '').str.replace_all(r"https?://zenodo\.org/record/(\d+)", r" 10.5281/zenodo.$1 ")

def get_df(parse_dir: str):
    records = []
    txt_files = list(Path(parse_dir).glob('*.txt'))
    for txt_file in txt_files:
        id_ = txt_file.stem
        with open(txt_file, 'r') as f:
            text = f.read()
        records.append({'article_id': id_, 'text': text})
    return pl.DataFrame(records).with_columns(string_normalization('text').alias('text'))

def assume_type(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.when(is_doi_link('dataset_id').or_(pl.col('dataset_id').str.starts_with('SAMN'))).then(pl.lit('Primary')).otherwise(pl.lit('Secondary')).alias('type'))
    )

def score(df, gt, on, tag='all'):
    hits = gt.join(df, on=on)
    tp = hits.height
    fp = df.height - tp
    fn = gt.height - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    return f"{tag} - f1: {f1:.4f} [{tp}/{fp}/{fn}]"

def evaluate(df, on=['article_id', 'dataset_id']):
    gt = pl.read_csv(COMP_DIR/'train_labels.csv').filter(pl.col('type')!='Missing')
    return (
        score(df, gt, on),
        score(df.filter(is_doi_link('dataset_id')), gt.filter(is_doi_link('dataset_id')), on, 'doi'),
        score(df.filter(~is_doi_link('dataset_id')), gt.filter(~is_doi_link('dataset_id')), on, 'acc'),
    )