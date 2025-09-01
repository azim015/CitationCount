import polars as pl
import os

from helpers import *

l = get_logger()

SYS_PROMPT_CLASSIFY_DOI = """
1. Priority Rules (highest → lowest)
1.1 Always classify as A (Data) if:
DOI prefix matches a known data repository:

Dryad: 10.5061

Zenodo: 10.5281

Dl: 10.15468

ICPSR: 10.3886

USGS data: 10.5066

Mendeley Data: 10.17632

Dataverse: 10.7910/DVN

OpenNeuro: 10.18112/openneuro.

PANGAEA: 10.1594/PANGAEA.


2. Classify as B (Literature) if:
DOI prefix belongs to a publisher (e.g., 10.1038, 10.1007, 10.1126, 10.1016, 10.1101, 10.1021, 10.1145, 10.1177, 10.1093, 10.1080, 10.1111, etc.).

Context indicates a journal article, book, conference paper, preprint, protocol, or method paper, without any repository/data storage signal.

Mentions only “supplementary material” or “supplementary information” without a repository.

3. Ambiguous cases
No repository prefix and no clear context → default to B.


4. Output
Only output:

A → data repository / dataset

B → literature / non-data resource

Few-shot examples

“Raw images are stored on Figshare (DOI 10.6084/m9.figshare.1234567).” → A

“Sequence reads available under BioProject accession PRJNA765432.” → A

“As described in Nature Methods (DOI 10.1038/s41592-020-0793-2).” → B

“See Supplementary Data at Zenodo (10.5281/zenodo.987654).” → A

“Method details published in J. Proteome Res. DOI: 10.1021/acs.jproteome.0c00845.” → B

“Data uploaded to Dryad (10.5061/dryad.x1y2z3).” → A

“Referenced paper: DOI 10.1101/2020.01.01.123456 (bioRxiv preprint).” → B

“Metabolomics data in MetaboLights MTBLS1234.” → A

“The MRI scans are deposited at OpenNeuro (DOI 10.18112/openneuro.ds000001.v1.0.0).” → A

“Protein structure described in Science (DOI 10.1126/science.abc1234).” → B
""".strip()

def build_df():
    df = pl.read_parquet('/tmp/extracted.parquet')
    df.filter(~is_doi_link('dataset_id')).select('article_id', 'dataset_id').write_csv('/tmp/accid_sub.csv')
    return df.filter(is_doi_link('dataset_id'))

def build_prompt(tokenizer, df):
    prompts = []
    for doi, text in df.select('dataset_id', 'window').rows():
        messages = [{'role':'system','content': SYS_PROMPT_CLASSIFY_DOI}, {'role':'user', 'content': text}]
        prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
    return df.with_columns(pl.Series('prompt', prompts))

if __name__=='__main__':
    os.environ["VLLM_USE_V1"] = "0"
    import vllm
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
    model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
    llm = vllm.LLM(model_path, quantization='awq', tensor_parallel_size=2, gpu_memory_utilization=0.9, trust_remote_code=True, dtype="half", enforce_eager=True, max_model_len=2048, disable_log_stats=True, disable_custom_all_reduce=True, enable_prefix_caching=True, task='generate')
    tokenizer = llm.get_tokenizer()
    df = build_df()
    df = build_prompt(tokenizer, df)
    prompts = df['prompt'].to_list()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])
    outputs = llm.generate(prompts, vllm.SamplingParams(seed=777, temperature=0.2, skip_special_tokens=True, max_tokens=1, logits_processors=[mclp], logprobs=len(mclp.choices)), use_tqdm=True)
    logprobs = [{lp.decoded_token: lp.logprob for lp in list(lps)} for lps in [output.outputs[0].logprobs[0].values() for output in outputs]]
    choices = [max(d, key=d.get) for d in logprobs]
    types = {'A': True, 'B': False}
    choices = [types[c] for c in choices]
    df = df.with_columns(pl.Series('type', choices))
    df.filter(pl.col('type')).select('article_id', 'dataset_id').write_csv('/tmp/doi_sub.csv')
    df = pl.concat([pl.read_csv('/tmp/doi_sub.csv'), pl.read_csv('/tmp/accid_sub.csv')])
    df = assume_type(df)
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
    if not IS_KAGGLE_SUBMISSION:
        results = evaluate(df)
        for r in results: l.info(r) 
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results: l.info(r)



    
    try:
        del llm, tokenizer
    except:
        pass
    
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()