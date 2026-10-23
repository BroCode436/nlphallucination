"""
preprocessing.py
Preprocesses RAGTruth dataset for hallucination detection and evidence verification.

Flow:
1. Load raw RAGTruth files (source_info.jsonl and response.jsonl).
2. Clean and format evidence based on task types (QA, Data2txt, Summary).
3. Create response-level dataset (Prompt + Evidence + Response -> has_hallucination).
4. Extract claim-level (sentence) dataset (Premise: Evidence, Hypothesis: Claim -> label, subtype).
5. Ensure zero data leakage by grouping by source_id for train/val split.
6. Save processed datasets to data/processed/.
"""
from typing import Any as _DocAny

import os
import json
import re
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')


def extract_evidence(source_dict: _DocAny, task_type: _DocAny) -> _DocAny:
    """Normalize evidence field according to RAGTruth format.

    Learning notes — input/output contract:
        Parameters:
            source_dict: Source dict supplied by the caller; see the operation and call sites for its concrete format.
            task_type: Task type supplied by the caller; see the operation and call sites for its concrete format.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    sinfo = source_dict.get('source_info', '')
    if isinstance(sinfo, str):
        return sinfo.strip()
    elif isinstance(sinfo, dict):
        if task_type == 'QA':
            passages = sinfo.get('passages', '')
            if isinstance(passages, list):
                return ' '.join([p.strip() for p in passages if isinstance(p, str)])
            return str(passages).strip()
        elif task_type == 'Data2txt':
            return '; '.join([f"{k}: {v}" for k, v in sinfo.items()]).strip()
        return str(sinfo).strip()
    return str(sinfo).strip()


def split_sentences_with_spans(text: _DocAny) -> _DocAny:
    """Split text into sentences while recording character start and end spans.

    Learning notes — input/output contract:
        Parameters:
            text: Input text to normalize, split, or transform as described above.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    text_clean = text.strip()
    if not text_clean:
        return []

    # Split on sentence boundaries (., !, ?) followed by whitespace
    raw_sentences = re.split(r'(?<=[.!?])\s+', text_clean)
    spans = []
    cursor = 0
    for s in raw_sentences:
        s_str = s.strip()
        if not s_str:
            continue
        start_idx = text.find(s_str, cursor)
        if start_idx == -1:
            start_idx = cursor
        end_idx = start_idx + len(s_str)
        cursor = end_idx
        spans.append((s_str, start_idx, end_idx))
    return spans


def load_raw_ragtruth(raw_dir: _DocAny=RAW_DIR) -> _DocAny:
    """Load and merge source_info.jsonl and response.jsonl.

    Learning notes — input/output contract:
        Parameters:
            raw_dir: Raw dir supplied by the caller; see the operation and call sites for its concrete format.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    src_file = os.path.join(raw_dir, 'source_info.jsonl')
    resp_file = os.path.join(raw_dir, 'response.jsonl')

    if not os.path.exists(src_file) or not os.path.exists(resp_file):
        raise FileNotFoundError(f"Missing raw files in {raw_dir}. Ensure response.jsonl and source_info.jsonl exist.")

    print(f"Loading sources from {src_file}...")
    sources = {}
    with open(src_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            sources[item['source_id']] = item

    print(f"Loading responses from {resp_file}...")
    response_rows = []
    claim_rows = []

    with open(resp_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            sid = item['source_id']
            src = sources.get(sid, {})
            task = src.get('task_type', 'Unknown')
            evidence = extract_evidence(src, task)
            prompt = src.get('prompt', '').strip()
            response_text = item.get('response', '')
            official_split = item.get('split', 'train')
            labels = item.get('labels', [])

            # Response level
            has_hal = 1 if len(labels) > 0 else 0
            types = list(set([span.get('label_type', 'Unknown') for span in labels])) if labels else ['Factual']

            resp_record = {
                'id': item.get('id'),
                'source_id': sid,
                'task_type': task,
                'prompt': prompt,
                'evidence': evidence,
                'model': item.get('model', ''),
                'temperature': item.get('temperature', 0.0),
                'response': response_text,
                'split': official_split,
                'has_hallucination': has_hal,
                'num_hallucination_spans': len(labels),
                'hallucination_types': ', '.join(types),
                'raw_spans': json.dumps(labels)
            }
            response_rows.append(resp_record)

            # Claim level (sentence-level decomposition)
            # SCIENTIFIC NOTE ON PROJECTION NOISE:
            # Hallucination spans in RAGTruth are fine-grained character spans. Mapping a span
            # to the entire enclosing sentence labels the entire sentence as hallucinated (1).
            # This is a standard proxy in sentence-level hallucination detection, but researchers
            # must note that surrounding factual clauses within the sentence may be labeled 1.
            if response_text:
                sentences = split_sentences_with_spans(response_text)
                for s_text, s_start, s_end in sentences:
                    # Check overlap with labeled hallucination spans
                    overlapping_spans = []
                    for lb in labels:
                        lb_st = lb.get('start', 0)
                        lb_en = lb.get('end', 0)
                        # Overlap condition: not (s_end <= lb_st or s_start >= lb_en)
                        if not (s_end <= lb_st or s_start >= lb_en):
                            overlapping_spans.append(lb)

                    is_claim_hal = 1 if len(overlapping_spans) > 0 else 0
                    if is_claim_hal:
                        claim_types = sorted(set(span.get('label_type', 'Evident Baseless Info') for span in overlapping_spans))
                        c_type = claim_types[0]
                    else:
                        c_type = 'Factual'

                    claim_rows.append({
                        'response_id': item.get('id'),
                        'source_id': sid,
                        'task_type': task,
                        'evidence': evidence if evidence else '',
                        'claim': s_text,
                        'claim_start': s_start,
                        'claim_end': s_end,
                        'label': is_claim_hal,  # 0 = Supported/Factual, 1 = Hallucinated
                        'hallucination_type': c_type,
                        'official_split': official_split,
                        'has_evidence': bool(evidence and str(evidence).strip())
                    })

    df_responses = pd.DataFrame(response_rows)
    df_claims = pd.DataFrame(claim_rows)
    print(f"Total responses loaded: {len(df_responses)}")
    print(f"Total claims extracted: {len(df_claims)}")
    return df_responses, df_claims


def create_leak_free_splits(df_claims: _DocAny, val_ratio: float=0.15, seed: int=42, manifest_path: _DocAny=None, save_manifest: bool=False) -> _DocAny:
    """Split train into train and val by source_id so no claims from the same
        source text appear in both train and validation splits.
        Includes strict programmatic assertions against cross-split data leakage.
        Deterministic sorting of source_ids guarantees identical splits across environments.

    Learning notes — input/output contract:
        Parameters:
            df_claims: Claim-level table with evidence, text, labels, and source identifiers.
            val_ratio: Fraction of eligible source groups assigned to validation.
            seed: Pseudorandom seed controlling reproducible sampling or model initialization.
            manifest_path: Optional destination of the split manifest; existing data is not deleted.
            save_manifest: Whether this call should persist a manifest rather than only return splits.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    # Separate official test split
    df_test = df_claims[df_claims['official_split'] == 'test'].copy().reset_index(drop=True)
    df_train_full = df_claims[df_claims['official_split'] == 'train'].copy().reset_index(drop=True)

    # Deterministic sorting prior to shuffling ensures exact reproducibility
    unique_sources = sorted(list(df_train_full['source_id'].unique()))
    np.random.seed(seed)
    np.random.shuffle(unique_sources)

    # Handle small datasets gracefully: avoid silently empty validation sets
    if len(unique_sources) >= 2:
        n_val = max(1, int(len(unique_sources) * val_ratio))
        # Ensure at least 1 source remains in training set
        if n_val >= len(unique_sources):
            n_val = len(unique_sources) - 1
    elif len(unique_sources) == 1:
        n_val = 0  # Cannot split 1 source without leakage
    else:
        n_val = 0

    val_sources = set(unique_sources[:n_val])
    train_sources = set(unique_sources[n_val:])
    test_sources = set(df_test['source_id'].unique())

    # CRITICAL SCIENTIFIC ASSERTION: Zero source-level leakage
    train_val_overlap = train_sources.intersection(val_sources)
    train_test_overlap = train_sources.intersection(test_sources)
    val_test_overlap = val_sources.intersection(test_sources)

    assert len(train_val_overlap) == 0, f"DATA LEAKAGE DETECTED: {len(train_val_overlap)} sources overlap between Train and Val!"
    assert len(train_test_overlap) == 0, f"DATA LEAKAGE DETECTED: {len(train_test_overlap)} sources overlap between Train and Test!"
    assert len(val_test_overlap) == 0, f"DATA LEAKAGE DETECTED: {len(val_test_overlap)} sources overlap between Val and Test!"

    df_train = df_train_full[df_train_full['source_id'].isin(train_sources)].copy().reset_index(drop=True)
    df_val = df_train_full[df_train_full['source_id'].isin(val_sources)].copy().reset_index(drop=True)

    print("\n--- Claim Dataset Splits (Strict Zero Source-ID Leakage Asserted) ---")
    print(f"Train sources: {len(train_sources)}, Val sources: {len(val_sources)}, Test sources: {len(test_sources)}")
    print(f"Train claims : {len(df_train)} (Hallucinated: {(df_train['label']==1).sum()}, Factual: {(df_train['label']==0).sum()})")
    print(f"Val claims   : {len(df_val)} (Hallucinated: {(df_val['label']==1).sum()}, Factual: {(df_val['label']==0).sum()})")
    print(f"Test claims  : {len(df_test)} (Hallucinated: {(df_test['label']==1).sum()}, Factual: {(df_test['label']==0).sum()})")

    # Record data statistics and split manifest safely ONLY when requested
    if save_manifest and manifest_path:
        manifest = {
            'random_seed': seed,
            'val_ratio': val_ratio,
            'train_sources': len(train_sources),
            'val_sources': len(val_sources),
            'test_sources': len(test_sources),
            'train_claims_total': len(df_train),
            'train_claims_hallucinated': int((df_train['label']==1).sum()),
            'val_claims_total': len(df_val),
            'val_claims_hallucinated': int((df_val['label']==1).sum()),
            'test_claims_total': len(df_test),
            'test_claims_hallucinated': int((df_test['label']==1).sum()),
            'test_hallucination_prevalence_pct': round((df_test['label']==1).sum() / max(len(df_test), 1) * 100, 2),
            'source_overlap_train_val': len(train_val_overlap),
            'source_overlap_train_test': len(train_test_overlap),
            'source_overlap_val_test': len(val_test_overlap)
        }
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

    return df_train, df_val, df_test


def run_pipeline() -> _DocAny:
    """Execute preprocessing and persist processed splits.

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
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_responses, df_claims = load_raw_ragtruth()

    # Clean response text
    df_responses = df_responses[df_responses['response'].str.len() > 0].copy()
    df_responses = df_responses.drop_duplicates(subset=['prompt', 'response']).copy()

    # Save clean responses
    resp_path = os.path.join(PROCESSED_DIR, 'ragtruth_responses_clean.csv')
    df_responses.to_csv(resp_path, index=False)
    print(f"Saved cleaned responses to {resp_path} ({len(df_responses)} rows)")

    # Save leak-free claim splits
    manifest_file = os.path.join(PROCESSED_DIR, 'split_manifest.json')
    df_train, df_val, df_test = create_leak_free_splits(
        df_claims,
        manifest_path=manifest_file,
        save_manifest=True
    )
    train_path = os.path.join(PROCESSED_DIR, 'ragtruth_claims_train.csv')
    val_path = os.path.join(PROCESSED_DIR, 'ragtruth_claims_val.csv')
    test_path = os.path.join(PROCESSED_DIR, 'ragtruth_claims_test.csv')

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"Saved processed claim splits to {PROCESSED_DIR}/")
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    run_pipeline()

