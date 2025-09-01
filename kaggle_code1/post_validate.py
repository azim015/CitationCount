from helpers import *
import polars as pl
import os


l = get_logger()


PROMPT_CLASSIFY_CITATION_TYPE = '''
# Role & Task
You are an expert data citation analyst. Your task is to classify a given citation from a scientific paper into one of two categories: **A** (Data) or **B** (Not Data). Base your decision strictly on the provided abstract and the context of the citation.

## Instructions
1.  **Read the provided abstract** to understand the research context.
2.  **Analyze the citation context** for key linguistic cues.
3.  **Classify the citation** as either **A** or **B** based on the definitions below.
4.  **Output only a single letter: A or B.** Do not output any other text, explanation, or formatting.

## Category Definitions

### **Category A: DATA**
The citation points to a dataset. This includes:
*   **Primary Data:** Raw or processed data that the current study's authors collected, generated, or created.
*   **Secondary Data:** Data that was originally produced by other researchers but is being *used as a dataset* in the current study.
*   **Key Phrases:** "data are available at", "we collected", "we measured", "data were obtained from", "dataset", "downloaded from", "deposited in", repository names (e.g., GenBank, Zenodo, Figshare, TCIA).

### **Category B: NOT DATA**
The citation points to a traditional scholarly publication or other non-data resource. This includes:
*   Journal articles, books, conference proceedings, preprints, protocols, methods papers.
*   **Key Phrases:** "as described in", "according to", "previous study", "et al.", "paper", "article", "methodology", "was used for analysis" (without indicating data access).
*   Citations that provide background context or methodological description but do not serve as the source of the data used in the analysis.

## Input Format
You will be provided with the following three pieces of information:
Paper Abstract: {abstract}
Citation: {dataset_id}
Citation Context: {context}

## Critical Thinking Guidelines
*   A DOI or URL can point to either data (A) or a paper (B). The context determines the classification.
*   If the citation is used to describe the *source* of the data for the current study's analysis, it is likely **A**.
*   If the citation is used to provide background, justify a method, or compare results, it is likely **B** (a reference to another paper).
*   When in doubt, rely on the linguistic cues in the "Citation Context".

## Examples for Pattern Recognition

**Example 1 (Classify as A):**
*   Context: "Three out of four cohorts used in this study can be found on The Cancer Imaging Archive (TCIA)24: Canadian benchmark dataset23: https://doi.org/10.7937/K9/TCIA.2017.8oje5q00."
*   **Reasoning:** The text states cohorts are "used in this study" and provides direct repository links. This is a clear case of citing external data for use.
*   **Output:** A

**Example 2 (Classify as B):**
*   Context: "data presented here are available at the SEANOE dataportal: https://doi.org/10.17882/94052 (ZooScan dataset Grandremy et al. 2023c)"
*   **Reasoning:** The phrase "data presented here" indicates this is the authors' own data being deposited, not a citation to an external source they are using. The "(Author et al. Year)" format is a classic literature citation style.
*   **Output:** B

**Example 3 (Classify as A):**
*   Context: "GBIF occurrence data: Vulpes vulpes: https://doi.org/10.15468/dl.wgtneb (28 May 2021)."
*   **Reasoning:** Explicitly names the data source (GBIF) and provides a direct access link/DOI for the specific dataset used.
*   **Output:** A

**Example 4 (Classify as B):**
*   Context: "North American soil NCBI SRA SRP035367 Smith & Peay [36] ITS2-Soil"
*   **Reasoning:** While it mentions a data repository ID (SRP035367), it couples it with a standard literature citation "[36]". The context suggests it is referencing the *paper* by Smith & Peay that describes the data, not directly citing the dataset itself for use.
*   **Output:** B

## Ready for Input
Begin your analysis. Remember: Output only **A** or **B**.
'''

def get_context_window(text: str, substring: str, window: int = 600) -> str:
    idx = text.find(substring)
    if idx == -1:
        return "no context", "no abstraction"
    start = max(idx - window, 0)
    end = min(idx + len(substring) + window, len(text))
    return text[start:end] , text[:1000]




def find_context_win(tokenizer,df):
    text_df = pl.read_parquet('/tmp/context_data.parquet')
    # print(text_df)
    df = df.join(text_df, on=["article_id","dataset_id"], how="inner")
    df = df.drop("type")
    print(df)

    prompts = []
    
    for article_id,dataset_id,text,match in df.select(["article_id","dataset_id","text",'match']).rows():

        context, abstract = get_context_window(text,match)
        user_content = f"""
        Paper Abstract: {abstract}
        
        Citation: {dataset_id}

        
        Citation Context: {context}
        """
        messages = [
            {"role": "system", "content": PROMPT_CLASSIFY_CITATION_TYPE},
            {"role": "user", "content": user_content.strip()}
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        )
        
    return df.with_columns(pl.Series("prompt", prompts))

    

if __name__=="__main__":
    os.environ["VLLM_USE_V1"] = "0"
    MODEL_PATH = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
    import vllm
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor

    llm = vllm.LLM(
        MODEL_PATH,
        quantization='awq',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=16384,
        disable_log_stats=True, 
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
        task='generate')

    tokenizer = llm.get_tokenizer()

    df=pl.read_csv("/kaggle/working/submission.csv")
    
    if "row_id" in df.columns:
        df = df.drop("row_id")

    # print(df)

    doi_df = df.filter(is_doi_link("dataset_id"))
    acc_df = df.filter(~is_doi_link("dataset_id"))

    # print(doi_df)

    df = find_context_win(tokenizer,doi_df)

    
    
    prompts = df['prompt'].to_list()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B","C"])
    outputs = llm.generate(prompts, vllm.SamplingParams(seed=777, temperature=0.7, skip_special_tokens=True, max_tokens=1, logits_processors=[mclp], logprobs=len(mclp.choices)), use_tqdm=True)
    logprobs = [{lp.decoded_token: lp.logprob for lp in list(lps)} for lps in [output.outputs[0].logprobs[0].values() for output in outputs]]
    choices = [max(d, key=d.get) for d in logprobs]
    types = {'A': True, 'B': False}
    choices = [types[c] for c in choices]
    df = df.with_columns(pl.Series('type', choices))
    df.filter(pl.col('type')).select('article_id', 'dataset_id').write_csv('/tmp/doi_sub.csv')
    df = pl.concat([pl.read_csv('/tmp/doi_sub.csv'), pl.read_csv('/tmp/accid_sub.csv')])
    df = assume_type(df)
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
    # print(df)
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