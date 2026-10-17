"""
train.py
Model Training and Hyperparameter Tuning for Hallucination Detection on RAGTruth.

Contains:
1. TF-IDF + Logistic Regression (Classical Baseline) with C and n-gram tuning.
2. DistilBERT Transformer Detector (NLI Premise-Hypothesis formulation) with LR tuning.
3. Validation-driven best model selection.
4. Logging of hyperparameter comparison table.
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

# Suppress HuggingFace symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


class ClaimDataset(Dataset):
    """PyTorch Dataset for Premise (Evidence) and Hypothesis (Claim) pairs."""
    def __init__(self, df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.evidence = df['evidence'].astype(str).tolist()
        self.claims = df['claim'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ev = self.evidence[idx]
        cl = self.claims[idx]
        lbl = self.labels[idx]

        # NLI pair encoding: text_a = Evidence (Premise), text_b = Claim (Hypothesis)
        encoded = self.tokenizer(
            ev,
            cl,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(lbl, dtype=torch.long)
        }


def load_claim_data(sample_train=4000, sample_val=1000, seed=42):
    """
    Load train and val claim datasets.
    Subsample a balanced/stratified training set for computational feasibility on CPU.
    """
    train_path = os.path.join(PROCESSED_DIR, 'ragtruth_claims_train.csv')
    val_path = os.path.join(PROCESSED_DIR, 'ragtruth_claims_val.csv')

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Clean missing values
    df_train['evidence'] = df_train['evidence'].fillna('')
    df_train['claim'] = df_train['claim'].fillna('')
    df_val['evidence'] = df_val['evidence'].fillna('')
    df_val['claim'] = df_val['claim'].fillna('')

    # Stratified balance sampling for train to handle severe class imbalance
    train_hal = df_train[df_train['label'] == 1]
    train_fac = df_train[df_train['label'] == 0]

    n_half = sample_train // 2
    sampled_hal = train_hal.sample(min(n_half, len(train_hal)), random_state=seed)
    sampled_fac = train_fac.sample(min(n_half, len(train_fac)), random_state=seed)
    df_train_sub = pd.concat([sampled_hal, sampled_fac]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Stratified sample for validation
    val_hal = df_val[df_val['label'] == 1]
    val_fac = df_val[df_val['label'] == 0]
    n_val_half = sample_val // 2
    sampled_vhal = val_hal.sample(min(n_val_half, len(val_hal)), random_state=seed)
    sampled_vfac = val_fac.sample(min(n_val_half, len(val_fac)), random_state=seed)
    df_val_sub = pd.concat([sampled_vhal, sampled_vfac]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Loaded train subset: {len(df_train_sub)} claims ({df_train_sub['label'].sum()} hallucinated)")
    print(f"Loaded val subset  : {len(df_val_sub)} claims ({df_val_sub['label'].sum()} hallucinated)")

    return df_train_sub, df_val_sub


# ------------------------------------------------------------------------------
# 1. BASELINE TRAINING & TUNING (TF-IDF + LOGISTIC REGRESSION)
# ------------------------------------------------------------------------------
def tune_baseline(df_train, df_val):
    print("\n==================================================")
    print("HYPERPARAMETER TUNING: TF-IDF + LOGISTIC REGRESSION")
    print("==================================================")

    # Format text: Evidence + Claim
    X_train_text = ("Evidence: " + df_train['evidence'] + " Claim: " + df_train['claim']).tolist()
    y_train = df_train['label'].values

    X_val_text = ("Evidence: " + df_val['evidence'] + " Claim: " + df_val['claim']).tolist()
    y_val = df_val['label'].values

    tuning_records = []
    best_f1 = -1.0
    best_model = None
    best_vectorizer = None
    best_config = None

    c_values = [0.1, 1.0, 5.0, 10.0]
    ngram_ranges = [(1, 1), (1, 2)]

    for ngram in ngram_ranges:
        print(f"\nEvaluating n-gram range: {ngram}...")
        vec = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=ngram)
        X_train_vec = vec.fit_transform(X_train_text)
        X_val_vec = vec.transform(X_val_text)

        for c in c_values:
            clf = LogisticRegression(C=c, max_iter=300, random_state=42)
            clf.fit(X_train_vec, y_train)

            y_pred = clf.predict(X_val_vec)
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            rec_dict = {
                'Model': 'TF-IDF + Logistic Regression',
                'Hyperparameters': f"C={c}, ngram={ngram}",
                'C': c,
                'ngram': str(ngram),
                'Learning Rate': 'N/A',
                'Batch Size': 'N/A',
                'Epochs': 'N/A',
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1': round(f1, 4)
            }
            tuning_records.append(rec_dict)
            print(f"  Config: C={c}, ngram={ngram} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = clf
                best_vectorizer = vec
                best_config = rec_dict

    # Save best baseline
    joblib.dump(best_model, os.path.join(CHECKPOINTS_DIR, 'best_baseline_clf.joblib'))
    joblib.dump(best_vectorizer, os.path.join(CHECKPOINTS_DIR, 'best_baseline_vec.joblib'))
    print(f"\n[BEST BASELINE] {best_config['Hyperparameters']} | F1: {best_f1:.4f}")

    return tuning_records, best_model, best_vectorizer


# ------------------------------------------------------------------------------
# 2. TRANSFORMER DETECTOR TRAINING & TUNING (DISTILBERT)
# ------------------------------------------------------------------------------
def evaluate_transformer(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return acc, prec, rec, f1, np.array(all_probs), np.array(all_targets)


def tune_transformer(df_train, df_val, device='cpu'):
    print("\n==================================================")
    print("HYPERPARAMETER TUNING: DISTILBERT TRANSFORMER")
    print("==================================================")

    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = ClaimDataset(df_train, tokenizer, max_length=128)
    val_ds = ClaimDataset(df_val, tokenizer, max_length=128)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    learning_rates = [2e-5, 3e-5, 5e-5]
    batch_size = 16
    epochs = 2

    tuning_records = []
    best_overall_f1 = -1.0
    best_overall_meta = None
    best_model_dir = os.path.join(CHECKPOINTS_DIR, 'best_transformer_detector')

    for lr in learning_rates:
        print(f"\n--- Training DistilBERT with lr={lr}, batch_size={batch_size}, epochs={epochs} ---")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_epoch_f1 = -1.0
        best_epoch_num = -1
        best_epoch_weights = None
        best_epoch_metrics = None

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            t0 = time.time()

            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            acc, prec, rec, f1, _, _ = evaluate_transformer(model, val_loader, device)
            elapsed = time.time() - t0
            print(f"  Epoch {epoch}/{epochs} ({elapsed:.1f}s) - Loss: {train_loss:.4f} | Val Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

            # Track best epoch for this configuration
            if f1 > best_epoch_f1:
                best_epoch_f1 = f1
                best_epoch_num = epoch
                best_epoch_metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
                # Keep copy of best weights on CPU
                best_epoch_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Record metrics based strictly on the best epoch for this configuration
        rec_dict = {
            'Model': 'DistilBERT Cross-Encoder',
            'Hyperparameters': f"lr={lr}, bs={batch_size}, best_epoch={best_epoch_num}/{epochs}",
            'C': 'N/A',
            'ngram': 'N/A',
            'Learning Rate': lr,
            'Batch Size': batch_size,
            'Epochs': best_epoch_num,
            'Accuracy': round(best_epoch_metrics['accuracy'], 4),
            'Precision': round(best_epoch_metrics['precision'], 4),
            'Recall': round(best_epoch_metrics['recall'], 4),
            'F1': round(best_epoch_metrics['f1'], 4)
        }
        tuning_records.append(rec_dict)

        # Check if this configuration's best epoch beats overall best
        if best_epoch_f1 > best_overall_f1:
            best_overall_f1 = best_epoch_f1
            print(f"  -> New best transformer model found at epoch {best_epoch_num} (F1: {best_epoch_f1:.4f}). Saving to {best_model_dir}...")
            # Load best epoch weights back into model before saving
            model.load_state_dict({k: v.to(device) for k, v in best_epoch_weights.items()})
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            meta = {
                'best_epoch': best_epoch_num,
                'total_epochs': epochs,
                'learning_rate': lr,
                'batch_size': batch_size,
                'val_f1': best_epoch_f1,
                'val_metrics': best_epoch_metrics
            }
            with open(os.path.join(best_model_dir, 'training_meta.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)

    print(f"\n[BEST TRANSFORMER] Best F1: {best_overall_f1:.4f}")
    return tuning_records


def run_training_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution Device: {device}")

    # Load clean claim data
    df_train, df_val = load_claim_data(sample_train=3000, sample_val=1000)

    # 1. Baseline tuning
    base_records, _, _ = tune_baseline(df_train, df_val)

    # 2. Transformer tuning
    trans_records = tune_transformer(df_train, df_val, device=device)

    # Combine into final comparison table
    all_records = base_records + trans_records
    df_results = pd.DataFrame(all_records)

    csv_path = os.path.join(TABLES_DIR, 'hyperparameter_tuning.csv')
    df_results.to_csv(csv_path, index=False)

    print("\n==================================================")
    print("HYPERPARAMETER TUNING SUMMARY TABLE")
    print("==================================================")
    print(df_results[['Model', 'Hyperparameters', 'Precision', 'Recall', 'F1', 'Accuracy']].to_string(index=False))
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    run_training_pipeline()
