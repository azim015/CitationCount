from helpers import *
import polars as pl
import os


l = get_logger()

PROMPT_CLASSIFY_CITATION_TYPE = '''
# Role & Task
You are an expert data citation analyst. Your task is to classify a given citation from a scientific paper into one of two categories based on the context: **A (Primary Data)** or **B (Secondary Data)**.

## Instructions
1.  **Read the provided abstract** to understand the research context.
2.  **Analyze the citation context** for key linguistic cues.
3.  **Classify the citation** as either **A** or **B** based on the definitions below.
4.  **Output only a single letter: A or B.** Do not output any other text, explanation, or formatting.

## Category Definitions

### **Category A: PRIMARY DATA**
The data was generated, collected, or created by the **authors of the current study**. This is *their* data.
*   **Key Phrases:** "we collected", "we generated", "our data", "data are available at [URL/DOI]", "data have been deposited", "this study presents", "supplementary data".

### **Category B: SECONDARY DATA**
The data was produced by **other researchers** or external sources and is being reused or analyzed by the current study's authors.
*   **Key Phrases:** "data were obtained from", "publicly available data", "previously published data", "retrieved from", "downloaded from", "[Dataset Name] dataset", "database", citing a specific external source.

## Input Format
You will be provided with the following three pieces of information:
Paper Abstract: {abstract}
Citation: {dataset_id}
Citation Context: {context}


## Decision Framework
Answer these questions based on the **Citation Context**:

1.  **Who is the source of the data?**
    *   If the context implies the **authors themselves** are the source (e.g., "we," "our"), classify as **A**.
    *   If the context names an **external source** (e.g., a repository, another study, a database), classify as **B**.

2.  **What is the action being described?**
    *   **A (Primary)** actions: *depositing, making available, presenting* their own data.
    *   **B (Secondary)** actions: *using, obtaining, accessing, downloading, analyzing* existing data from elsewhere.

## Examples for Pattern Recognition

**Example 1 (Classify as B):**
*   Context: "Three out of four cohorts **used in this study** can be found on The Cancer Imaging Archive (TCIA)24: Canadian benchmark dataset23: https://doi.org/10.7937/K9/TCIA.2017.8oje5q00."
*   **Reasoning:** The authors are describing external datasets they **used** (a Secondary action). The source is TCIA, not themselves.
*   **Output:** B

**Example 2 (Classify as A):**
*   Context: "Additional research data **supporting this publication are available** at 10.25377/sussex.21184705."
*   **Reasoning:** The authors are stating the availability of data that **supports their own publication**. The source is implied to be themselves.
*   **Output:** A

**Example 3 (Classify as B):**
*   Context: "GBIF occurrence data: Vulpes vulpes: https://doi.org/10.15468/dl.wgtneb (28 May 2021)."
*   **Reasoning:** The data is explicitly sourced from an external repository (GBIF). The authors are referring to data they reused.
*   **Output:** B

**Example 4 (Classify as A):**
*   Context: "Data referring to Barbieux et al. (2017; https://doi.org/10.17882/49388) are freely available on SEANOE."
*   **Reasoning:** This is a tricky case. The citation format "(Author et al. Year)" suggests a literature reference. However, the phrase "Data referring to" and the direct data DOI indicate the authors are citing **their own previously published dataset** (from a 2017 paper) that is now available. This is their Primary data.
*   **Output:** A

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


    doi_df = df.filter(is_doi_link("dataset_id"))
    acc_df = df.filter(~is_doi_link("dataset_id"))



    df = find_context_win(tokenizer,doi_df)

    
    
    prompts = df['prompt'].to_list()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])
    outputs = llm.generate(prompts, vllm.SamplingParams(seed=777, temperature=0.8, skip_special_tokens=True, max_tokens=1, logits_processors=[mclp], logprobs=len(mclp.choices)), use_tqdm=True)
    logprobs = [{lp.decoded_token: lp.logprob for lp in list(lps)} for lps in [output.outputs[0].logprobs[0].values() for output in outputs]]
    choices = [max(d, key=d.get) for d in logprobs]
    types = {'A':'Primary', 'B':'Secondary'}
    choices = [types[c] for c in choices]


    
    df = df.with_columns(pl.Series('type', choices))
    df.select('article_id', 'dataset_id','type').write_csv('/tmp/doi_sub.csv')

    acc_df = assume_type(acc_df)
    acc_df.select('article_id','dataset_id','type').write_csv("/tmp/accid_sub.csv")
    df = pl.concat([pl.read_csv('/tmp/doi_sub.csv'), pl.read_csv('/tmp/accid_sub.csv')])
    
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