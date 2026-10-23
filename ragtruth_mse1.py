"""
ragtruth_mse1.py

Evidence-Guided Detection and Correction of Hallucinations in LLMs (MSE-1)
Primary Dataset: RAGTruth (from official GitHub: https://github.com/ParticleMedia/RAGTruth)

Flow:
Load RAGTruth -> Dataset Inspection -> Preprocessing -> EDA -> Model Identification
"""
from typing import Any as _DocAny

import os
import json
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# STEP 1: DATA PATH SETUP
# ------------------------------------------------------------------------------
DATA_RAW = os.path.join('data', 'raw')
os.makedirs(DATA_RAW, exist_ok=True)


# ------------------------------------------------------------------------------
# STEP 2: LOAD & INSPECT RAGTRUTH DATASET
# ------------------------------------------------------------------------------
def load_and_inspect_ragtruth() -> _DocAny:
    """Load and inspect ragtruth.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters: No explicit caller arguments beyond an instance/class if applicable.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    resp_file = os.path.join(DATA_RAW, 'response.jsonl')
    src_file = os.path.join(DATA_RAW, 'source_info.jsonl')
    base_url = 'https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/'

    # Download raw files if missing
    if not os.path.exists(resp_file):
        print("Downloading response.jsonl from GitHub...")
        urllib.request.urlretrieve(base_url + 'response.jsonl', resp_file)

    if not os.path.exists(src_file):
        print("Downloading source_info.jsonl from GitHub...")
        urllib.request.urlretrieve(base_url + 'source_info.jsonl', src_file)

    # Load source_info.jsonl into dictionary
    sources = {}
    with open(src_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            sources[item['source_id']] = item

    # Load response.jsonl and combine with source context
    rows = []
    with open(resp_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            sid = item['source_id']
            src = sources.get(sid, {})
            task = src.get('task_type', '')
            sinfo = src.get('source_info', '')

            # Clean evidence extraction based on task type
            if isinstance(sinfo, str):
                evidence = sinfo
            elif isinstance(sinfo, dict):
                if task == 'QA':
                    passages = sinfo.get('passages', '')
                    evidence = ' '.join(passages) if isinstance(passages, list) else str(passages)
                elif task == 'Data2txt':
                    evidence = '; '.join([f'{k}: {v}' for k, v in sinfo.items()])
                else:
                    evidence = str(sinfo)
            else:
                evidence = str(sinfo)

            labels = item.get('labels', [])
            types = list(set([span.get('label_type', 'Unknown') for span in labels])) if labels else ['Factual']

            rows.append({
                'id': item.get('id'),
                'source_id': sid,
                'task_type': task,
                'prompt': src.get('prompt', ''),
                'evidence': evidence,
                'model': item.get('model', ''),
                'response': item.get('response', ''),
                'split': item.get('split', 'train'),
                'has_hallucination': 1 if len(labels) > 0 else 0,
                'num_hallucination_spans': len(labels),
                'hallucination_types': ', '.join(types),
                'raw_spans': json.dumps(labels)
            })

    df = pd.DataFrame(rows)
    print(f"Total RAGTruth responses loaded: {len(df)}")
    print(f"Official Splits: {dict(df['split'].value_counts())}")
    return df


# ------------------------------------------------------------------------------
# STEP 3: PREPROCESSING
# ------------------------------------------------------------------------------
def preprocess_data(df: _DocAny) -> _DocAny:
    """Preprocess data.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters:
            df: Df supplied by the caller; see the operation and call sites for its concrete format.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    print("\nMissing values check:")
    print(df.isnull().sum())

    # Clean missing text
    df['evidence'] = df['evidence'].fillna('')
    df['prompt'] = df['prompt'].fillna('')
    df['response'] = df['response'].fillna('')

    # Filter empty responses and drop exact duplicates
    df_clean = df[df['response'].str.len() > 0].copy()
    df_clean = df_clean.drop_duplicates(subset=['prompt', 'response']).copy()

    # Calculate word counts for EDA
    df_clean['response_word_count'] = df_clean['response'].apply(lambda x: len(str(x).split()))
    df_clean['evidence_word_count'] = df_clean['evidence'].apply(lambda x: len(str(x).split()))

    print(f"\nRows before preprocessing: {len(df)}")
    print(f"Rows after preprocessing : {len(df_clean)}")
    return df_clean


# ------------------------------------------------------------------------------
# STEP 4: EDA (3-4 SIMPLE PLOTS)
# ------------------------------------------------------------------------------
def run_eda(df_clean: _DocAny) -> _DocAny:
    """Run eda.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters:
            df_clean: DataFrame containing clean records for this operation.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    # Graph 1: Hallucination Label Distribution
    plt.figure(figsize=(6, 4))
    df_clean['has_hallucination'].value_counts().plot(kind='bar', color=['#2ca02c', '#d62728'])
    plt.title('RAGTruth Hallucination Label Distribution')
    plt.xlabel('Label (0 = Factual, 1 = Hallucinated)')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Factual (0)', 'Hallucinated (1)'], rotation=0)
    plt.tight_layout()
    plt.show()

    # Graph 2: Response Word Length Distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df_clean, x='response_word_count', hue='has_hallucination', kde=True, palette={0: '#2ca02c', 1: '#d62728'})
    plt.title('Response Length Distribution')
    plt.xlabel('Response Word Count')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Graph 3: Evidence Word Length Distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df_clean, x='evidence_word_count', color='#1f77b4', kde=True)
    plt.title('Evidence Length Distribution')
    plt.xlabel('Evidence Word Count')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Graph 4: Hallucination Types Distribution
    df_hal = df_clean[df_clean['has_hallucination'] == 1]
    all_types = []
    for t_str in df_hal['hallucination_types']:
        for t in t_str.split(', '):
            if t != 'Factual':
                all_types.append(t)

    plt.figure(figsize=(8, 4))
    pd.Series(all_types).value_counts().plot(kind='barh', color='#d62728')
    plt.title('Hallucination Subtypes Distribution')
    plt.xlabel('Count')
    plt.ylabel('Annotation Type')
    plt.tight_layout()
    plt.show()

    # Short Observations
    print("\n--- EDA OBSERVATIONS ---")
    print(f"Total responses: {len(df_clean)}")
    print(f"Factual: {(df_clean['has_hallucination']==0).sum()} | Hallucinated: {(df_clean['has_hallucination']==1).sum()}")
    print(f"Average Response Length: {df_clean['response_word_count'].mean():.1f} words")
    print(f"Average Evidence Length: {df_clean['evidence_word_count'].mean():.1f} words")


# ------------------------------------------------------------------------------
# STEP 5: MODEL IDENTIFICATION
# ------------------------------------------------------------------------------
def explain_models() -> _DocAny:
    """Explain models.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters: No explicit caller arguments beyond an instance/class if applicable.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    print("\n=====================================")
    print("MODEL IDENTIFICATION (CONCEPTUAL)")
    print("=====================================")
    print("1. Baseline: TF-IDF + Logistic Regression")
    print("   - Input: n-grams of Prompt + Evidence + Response")
    print("   - Output: Binary probability of hallucination")
    print("\n2. Primary Model: BERT / RoBERTa")
    print("   - Input: [CLS] Evidence [SEP] Response [SEP]")
    print("   - Output: Binary classification logits")
    print("\n3. Evidence Verification Model: DeBERTa-v3 MNLI")
    print("   - Input: Premise = Evidence, Hypothesis = Claim")
    print("   - Output: Entailment / Contradiction classification")


# ------------------------------------------------------------------------------
# MAIN FLOW
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    raw_df = load_and_inspect_ragtruth()
    processed_df = preprocess_data(raw_df)
    run_eda(processed_df)
    explain_models()

    print("\n=====================================")
    print("MSE-1 STATUS")
    print("=====================================")
    print("Problem Identification: DONE")
    print("Dataset (RAGTruth)   : DONE")
    print("Preprocessing        : DONE")
    print("EDA                  : DONE")
    print("Model Identification : DONE")
    print("=====================================")
