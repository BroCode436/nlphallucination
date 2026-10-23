"""
verification.py
Adaptive Evidence-Guided Verification Engine.

Scientific Stage Definitions:
1. Stage 1 (Primary Trained Detector):
   - Statistical Baseline: TF-IDF + Logistic Regression
   - Deep Neural Detector: Fine-tuned DistilBERT Cross-Encoder on [Evidence, Claim] pairs
2. Stage 2 (Secondary Lexical-Semantic Coverage Heuristic):
   - Measures non-trivial content token overlap and identifies ungrounded assertion clauses
3. Stage 3 (Deterministic Numerical & Temporal Conflict Rule Verifier):
   - Regex-based extraction of numbers, percentages, and dates to flag precise factual contradictions

Statuses (Distinct from Binary Hallucination):
- Supported: Evidence provides clear factual alignment (Prediction: 0)
- Contradicted: Evidence contains conflicting assertions (Prediction: 1)
- Not Verifiable: Evidence is absent, null, or completely inconclusive (Abstained: True, Prediction: None)

Zero Simulated Latencies:
All inference latencies are measured using empirical wall-clock time (time.perf_counter).
"""
from typing import Any as _DocAny

import os
import time
import json
import re
import numpy as np
import pandas as pd
from grounding import clean_text, content_tokens, numerical_conflicts, NUMBER
from settings import load_config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import torch
except ImportError:
    torch = None


# ------------------------------------------------------------------------------
# 1. EVIDENCE QUALITY ASSESSMENT
# ------------------------------------------------------------------------------
def compute_evidence_quality(evidence_text: _DocAny, claim_text: _DocAny) -> _DocAny:
    """Evaluates evidence quality based on grounding, length, and content token overlap.
        Returns:
            quality_category: 'Strong' | 'Weak' | 'Irrelevant' | 'No Evidence'
            quality_score: float in [0.0, 1.0]

    Learning notes — input/output contract:
        Parameters:
            evidence_text: Evidence text supplied by the caller; see the operation and call sites for its concrete format.
            claim_text: Claim text supplied by the caller; see the operation and call sites for its concrete format.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    ev = clean_text(evidence_text).lower()
    cl = clean_text(claim_text).lower()

    if not ev or len(ev.split()) < 3:
        return 'No Evidence', 0.0

    # Extract non-trivial content words
    ev_words = content_tokens(ev)
    cl_words = content_tokens(cl)

    if not cl_words:
        return 'Weak', 0.30

    overlap = cl_words.intersection(ev_words)
    overlap_ratio = len(overlap) / len(cl_words)

    # Word count depth
    length_bonus = min(len(ev.split()) / 50.0, 1.0) * 0.20
    quality_score = min(overlap_ratio * 0.80 + length_bonus, 1.0)

    thresholds = load_config()['verification_policy']['evidence_quality_thresholds']
    if quality_score >= thresholds['strong']:
        category = 'Strong'
    elif quality_score >= thresholds['weak']:
        category = 'Weak'
    else:
        category = 'Irrelevant'

    return category, round(quality_score, 3)


# ------------------------------------------------------------------------------
# 2. VERIFICATION ENGINE WITH REAL WALL-CLOCK TIMING
# ------------------------------------------------------------------------------
class VerificationEngine:
    """
    Multi-verifier engine executing:
    - Primary detector (Logistic Regression or DistilBERT Cross-Encoder)
    - Lexical-Semantic Coverage Heuristic
    - Deterministic Numerical & Temporal Conflict Rule Verifier
    All latencies are measured using real wall-clock timing via time.perf_counter().
    """
    def __init__(self, primary_model: _DocAny=None, vectorizer: _DocAny=None,
                 transformer_model: _DocAny=None, transformer_tokenizer: _DocAny=None,
                 device: _DocAny=None, policy_config: _DocAny=None) -> None:
        """  init  .

        Routes claims through the detector and lexical/numerical heuristics.

        Learning notes — input/output contract:
            Parameters:
                primary_model: Primary model supplied by the caller; see the operation and call sites for its concrete format.
                vectorizer: Vectorizer supplied by the caller; see the operation and call sites for its concrete format.
                transformer_model: Transformer model supplied by the caller; see the operation and call sites for its concrete format.
                transformer_tokenizer: Transformer tokenizer supplied by the caller; see the operation and call sites for its concrete format.
                device: One CPU or CUDA execution device; GPU memory is not pooled.
                policy_config: Policy config supplied by the caller; see the operation and call sites for its concrete format.
            Returns:
                None; effects or assertions are performed directly.
            Assumptions and failures:
                Inputs must match the saved dataset/model schema. Missing files, invalid
                fields, or incompatible checkpoints can raise errors; tests cover normal
                and selected failure paths. Scores are not independent factual proof.

        """
        self.policy_config = policy_config or load_config()['verification_policy']
        self.primary_model = primary_model
        self.vectorizer = vectorizer
        self.transformer_model = transformer_model
        self.transformer_tokenizer = transformer_tokenizer
        self.device = device or ('cuda' if torch and torch.cuda.is_available() else 'cpu')

        if self.transformer_model is not None and torch:
            self.transformer_model.to(self.device)
            self.transformer_model.eval()

    def predict_primary(self, evidence: _DocAny, claim: _DocAny) -> _DocAny:
        """Primary detector probability of hallucination.

        Learning notes — input/output contract:
            Parameters:
                evidence: Source passage supplied to the verifier; presence does not establish source truth.
                claim: Statement being checked against evidence.
            Returns:
                The value assembled by this function (score, table, records, or model as
                described above). The Any type preserves the legacy dynamic interface;
                inspect the documented call site for concrete tensor/table shapes.
            Assumptions and failures:
                Inputs must match the saved dataset/model schema. Missing files, invalid
                fields, or incompatible checkpoints can raise errors; tests cover normal
                and selected failure paths. Scores are not independent factual proof.

        """
        # Preference: Transformer Detector if available
        if self.transformer_model is not None and self.transformer_tokenizer is not None and torch:
            enc = self.transformer_tokenizer(
                str(evidence),
                str(claim),
                truncation=True,
                max_length=load_config()['transformer_model']['max_length'],
                padding='max_length',
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                logits = self.transformer_model(**enc).logits
                probs = torch.softmax(logits, dim=1)[0]
                prob_hal = float(probs[1].cpu().item())
            return prob_hal

        # Baseline: TF-IDF + Logistic Regression
        if self.primary_model is not None and self.vectorizer is not None:
            text = [f"Evidence: {evidence} Claim: {claim}"]
            vec = self.vectorizer.transform(text)
            prob_hal = float(self.primary_model.predict_proba(vec)[0][1])
            return prob_hal

        # Fallback heuristic if standalone without weights
        eq_cat, eq_score = compute_evidence_quality(evidence, claim)
        prob_hal = 0.85 if eq_cat in ['Irrelevant', 'No Evidence'] else 0.20
        return prob_hal

    def verify_check_1(self, evidence: _DocAny, claim: _DocAny) -> _DocAny:
        """Stage 1: Primary statistical or deep detector check.

        Learning notes — input/output contract:
            Parameters:
                evidence: Source passage supplied to the verifier; presence does not establish source truth.
                claim: Statement being checked against evidence.
            Returns:
                The value assembled by this function (score, table, records, or model as
                described above). The Any type preserves the legacy dynamic interface;
                inspect the documented call site for concrete tensor/table shapes.
            Assumptions and failures:
                Inputs must match the saved dataset/model schema. Missing files, invalid
                fields, or incompatible checkpoints can raise errors; tests cover normal
                and selected failure paths. Scores are not independent factual proof.

        """
        t0 = time.perf_counter()
        prob_hal = self.predict_primary(evidence, claim)
        latency = (time.perf_counter() - t0) * 1000.0  # pure wall-clock ms
        return prob_hal, latency

    def verify_check_2(self, evidence: _DocAny, claim: _DocAny) -> _DocAny:
        """Stage 2: Lexical-Semantic Coverage Heuristic.

        Learning notes — input/output contract:
            Parameters:
                evidence: Source passage supplied to the verifier; presence does not establish source truth.
                claim: Statement being checked against evidence.
            Returns:
                The value assembled by this function (score, table, records, or model as
                described above). The Any type preserves the legacy dynamic interface;
                inspect the documented call site for concrete tensor/table shapes.
            Assumptions and failures:
                Inputs must match the saved dataset/model schema. Missing files, invalid
                fields, or incompatible checkpoints can raise errors; tests cover normal
                and selected failure paths. Scores are not independent factual proof.

        """
        t0 = time.perf_counter()
        ev_lower = str(evidence).lower()
        cl_lower = str(claim).lower()
        stopwords = {'the', 'and', 'was', 'were', 'that', 'with', 'from', 'this', 'for', 'are'}
        key_cl_tokens = [w for w in re.findall(r'\b[a-z0-9]{3,}\b', cl_lower) if w not in stopwords]
        ev_tokens = set(re.findall(r'\b[a-z0-9]+\b', ev_lower))
        missing = [w for w in key_cl_tokens if w not in ev_tokens]
        nli_prob_hal = min(len(missing) / max(len(key_cl_tokens), 1), 1.0)
        latency = (time.perf_counter() - t0) * 1000.0  # pure wall-clock ms
        return nli_prob_hal, latency

    def verify_check_3(self, evidence: _DocAny, claim: _DocAny) -> _DocAny:
        """Stage 3: Deterministic Numerical & Temporal Conflict Rule Verifier.

        Learning notes — input/output contract:
            Parameters:
                evidence: Source passage supplied to the verifier; presence does not establish source truth.
                claim: Statement being checked against evidence.
            Returns:
                The value assembled by this function (score, table, records, or model as
                described above). The Any type preserves the legacy dynamic interface;
                inspect the documented call site for concrete tensor/table shapes.
            Assumptions and failures:
                Inputs must match the saved dataset/model schema. Missing files, invalid
                fields, or incompatible checkpoints can raise errors; tests cover normal
                and selected failure paths. Scores are not independent factual proof.

        """
        t0 = time.perf_counter()
        details = numerical_conflicts(evidence, claim)
        conflict = any(item['conflict'] for item in details)
        deep_prob_hal = 0.85 if conflict else 0.25
        latency = (time.perf_counter() - t0) * 1000.0  # pure wall-clock ms
        return deep_prob_hal, latency

    def adaptive_verify(self, evidence: _DocAny, claim: _DocAny, **kwargs: _DocAny) -> _DocAny:
        """Convenience instance method delegating to run_adaptive_verification.

        Learning notes — input/output contract:
            Parameters:
                evidence: Source passage supplied to the verifier; presence does not establish source truth.
                claim: Statement being checked against evidence.
                kwargs: Kwargs supplied by the caller; see the operation and call sites for its concrete format.
            Returns:
                The value assembled by this function (score, table, records, or model as
                described above). The Any type preserves the legacy dynamic interface;
                inspect the documented call site for concrete tensor/table shapes.
            Assumptions and failures:
                Inputs must match the saved dataset/model schema. Missing files, invalid
                fields, or incompatible checkpoints can raise errors; tests cover normal
                and selected failure paths. Scores are not independent factual proof.

        """
        return run_adaptive_verification(self, evidence, claim, **kwargs)


# ------------------------------------------------------------------------------
# 3. VERIFICATION POLICIES
# ------------------------------------------------------------------------------
def run_no_verification(engine: _DocAny, evidence: _DocAny, claim: _DocAny) -> _DocAny:
    """Policy A: Baseline Single-Detector.
        Direct inference via Stage 1 without secondary or tertiary checks.

    Learning notes — input/output contract:
        Parameters:
            engine: Verification engine containing the selected detector and heuristic checks.
            evidence: Source passage supplied to the verifier; presence does not establish source truth.
            claim: Statement being checked against evidence.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    prob, lat = engine.verify_check_1(evidence, claim)
    pred_hal = 1 if prob >= 0.5 else 0
    status = 'Contradicted' if pred_hal == 1 else 'Supported'
    return {
        'prediction': pred_hal,
        'probability': prob,
        'verification_calls': 1,
        'latency_ms': lat,
        'route': 'Single-Detector',
        'status': status,
        'abstained': False
    }


def run_fixed_verification(engine: _DocAny, evidence: _DocAny, claim: _DocAny) -> _DocAny:
    """Policy B: Fixed Multi-Check Verification.
        Every claim unconditionally triggers all 3 verification stages.

    Learning notes — input/output contract:
        Parameters:
            engine: Verification engine containing the selected detector and heuristic checks.
            evidence: Source passage supplied to the verifier; presence does not establish source truth.
            claim: Statement being checked against evidence.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    prob1, lat1 = engine.verify_check_1(evidence, claim)
    prob2, lat2 = engine.verify_check_2(evidence, claim)
    prob3, lat3 = engine.verify_check_3(evidence, claim)

    total_lat = lat1 + lat2 + lat3
    avg_prob = (prob1 + prob2 + prob3) / 3.0
    pred_hal = 1 if avg_prob >= 0.5 else 0
    status = 'Contradicted' if pred_hal == 1 else 'Supported'

    return {
        'prediction': pred_hal,
        'probability': avg_prob,
        'verification_calls': 3,
        'latency_ms': total_lat,
        'route': 'Fixed-Multi-Check',
        'status': status,
        'abstained': False
    }


def run_adaptive_verification(engine: _DocAny, evidence: _DocAny, claim: _DocAny,
                              high_conf_thresh: _DocAny=None,
                              low_conf_thresh: _DocAny=None,
                              disabled_stage: _DocAny=None, use_evidence_quality: bool=True) -> _DocAny:
    """Policy C: Adaptive Evidence-Guided Verification.
        Routes claims dynamically based on model uncertainty and evidence quality.
        Numeric claims reach Stage 3 before acceptance. Component ablations reuse
        this policy; numerical rule scores are heuristics, not calibrated likelihoods.

        Status semantics:
        - Supported: Evidence confirms the claim
        - Contradicted: Evidence conflicts with the claim
        - Not Verifiable: Evidence is absent or empty (explicit abstention; NOT counted as hallucination)

    Learning notes — input/output contract:
        Parameters:
            engine: Verification engine containing the selected detector and heuristic checks.
            evidence: Source passage supplied to the verifier; presence does not establish source truth.
            claim: Statement being checked against evidence.
            high_conf_thresh: Decision threshold for high conf thresh; tune on validation only.
            low_conf_thresh: Decision threshold for low conf thresh; tune on validation only.
            disabled_stage: Optional verifier stage removed for a controlled ablation.
            use_evidence_quality: Whether routing consults lexical evidence-quality categories.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    t_start = time.perf_counter()
    config = getattr(engine, 'policy_config', None) or load_config()['verification_policy']
    high_conf_thresh = config['high_confidence_threshold'] if high_conf_thresh is None else high_conf_thresh
    low_conf_thresh = config['low_confidence_threshold'] if low_conf_thresh is None else low_conf_thresh
    if not 0 <= low_conf_thresh <= high_conf_thresh <= 1:
        raise ValueError('Confidence thresholds must satisfy 0 <= low <= high <= 1')
    if disabled_stage not in (None, 2, 3):
        raise ValueError('disabled_stage must be None, 2, or 3')
    eq_cat, eq_score = compute_evidence_quality(evidence, claim)

    # 1. Check for absence of evidence -> Abstain explicitly as Not Verifiable
    if eq_cat == 'No Evidence':
        lat_no_ev = (time.perf_counter() - t_start) * 1000.0
        return {
            'prediction': None,  # Not a confirmed binary hallucination
            'probability': 0.5,
            'verification_calls': 0,
            'latency_ms': lat_no_ev,
            'route': 'Abstain-No-Evidence',
            'status': 'Not Verifiable',
            'abstained': True
        }

    # 2. Stage 1: Fast verification
    prob1, lat1 = engine.verify_check_1(evidence, claim)
    # Distance from the binary decision boundary is a routing heuristic, not
    # independently calibrated confidence that the claim is factually correct.
    confidence = abs(prob1 - 0.5) * 2.0  # Scale to [0, 1]

    # FAST PATH: High confidence AND Strong evidence -> 1 check
    numeric_requires_check = disabled_stage != 3 and bool(NUMBER.search(clean_text(claim)))
    if confidence >= high_conf_thresh and (eq_cat == 'Strong' or not use_evidence_quality) and not numeric_requires_check:
        pred_hal = 1 if prob1 >= 0.5 else 0
        return {
            'prediction': pred_hal,
            'probability': prob1,
            'verification_calls': 1,
            'latency_ms': (time.perf_counter() - t_start) * 1000.0,
            'route': 'Fast-Path',
            'status': 'Contradicted' if pred_hal == 1 else 'Supported',
            'abstained': False
        }

    # MEDIUM PATH: Moderate confidence OR Weak evidence -> Stage 2 check
    prob2, lat2 = engine.verify_check_2(evidence, claim) if disabled_stage != 2 else (prob1, 0.0)
    cum_lat = lat1 + lat2
    cum_prob = (prob1 + prob2) / 2.0

    # Verifier 1 and Verifier 2 agree within tolerance and evidence is grounded
    agreement = disabled_stage != 2 and abs(prob1 - prob2) < config['agreement_threshold']
    if agreement and confidence >= low_conf_thresh and (eq_cat in ['Strong', 'Weak'] or not use_evidence_quality) and not numeric_requires_check:
        pred_hal = 1 if cum_prob >= 0.5 else 0
        return {
            'prediction': pred_hal,
            'probability': cum_prob,
            'verification_calls': 2,
            'latency_ms': (time.perf_counter() - t_start) * 1000.0,
            'route': 'Medium-Consensus',
            'status': 'Contradicted' if pred_hal == 1 else 'Supported',
            'abstained': False
        }

    # DEEP PATH: Conflicting verifier outputs (or Irrelevant evidence) trigger Stage 3 check
    prob3, lat3 = engine.verify_check_3(evidence, claim) if disabled_stage != 3 else (prob1, 0.0)
    deep_lat = cum_lat + lat3
    all_probs = [prob1, prob2, prob3]
    prob_spread = max(all_probs) - min(all_probs)

    # If verifiers remain in irreconcilable, high-conflict disagreement
    # (e.g. spread >= 0.55 and Stage 1 vs Stage 2 disagree sharply without decisive Stage 3 resolution)
    if prob_spread >= 0.55 and abs(prob1 - prob2) >= 0.50:
        has_decisive_conflict = disabled_stage != 3 and (prob3 >= 0.80)
        if not has_decisive_conflict:
            return {
                'prediction': None,  # Explicit abstention on irreconcilable contradiction
                'probability': float(np.mean(all_probs)),
                'verification_calls': 3 - int(disabled_stage is not None),
                'latency_ms': (time.perf_counter() - t_start) * 1000.0,
                'route': 'Abstain-Conflicting-Verifiers',
                'status': 'Not Verifiable',
                'abstained': True
            }

    weighted = [(prob1, 0.40)]
    if disabled_stage != 2:
        weighted.append((prob2, 0.30))
    if disabled_stage != 3:
        weighted.append((prob3, 0.30))
    deep_prob = sum(p * w for p, w in weighted) / sum(w for _, w in weighted)
    if disabled_stage != 3 and prob3 >= 0.80:
        deep_prob = max(deep_prob, prob3)
    pred_hal = 1 if deep_prob >= 0.5 else 0

    return {
        'prediction': pred_hal,
        'probability': deep_prob,
        'verification_calls': 3 - int(disabled_stage is not None),
        'latency_ms': (time.perf_counter() - t_start) * 1000.0,
        'route': 'Deep-Path',
        'status': 'Contradicted' if pred_hal == 1 else 'Supported',
        'abstained': False
    }


# ------------------------------------------------------------------------------
# 4. POLICY COMPARISON PROTOCOL WITH LATENCY DISTRIBUTION
# ------------------------------------------------------------------------------
def compare_verification_policies(engine: _DocAny, df_test: _DocAny, max_samples: int=800) -> _DocAny:
    """Compares No Verification, Fixed Verification, and Adaptive Verification
        under identical test samples, measuring empirical accuracy and latency distributions.

    Learning notes — input/output contract:
        Parameters:
            engine: Verification engine containing the selected detector and heuristic checks.
            df_test: Held-out test DataFrame used only for evaluation.
            max_samples: Maximum sampled records; record the actual evaluated count.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    df_eval = df_test.head(max_samples).copy()
    print(f"\nEvaluating verification policies on {len(df_eval)} test claims (paired comparison)...")

    results = {'Single-Detector': [], 'Fixed-Multi-Check': [], 'Adaptive-Verification': []}

    rng = np.random.default_rng(42)
    policies = [('Single-Detector', run_no_verification), ('Fixed-Multi-Check', run_fixed_verification), ('Adaptive-Verification', run_adaptive_verification)]
    if len(df_eval):
        first = df_eval.iloc[0]
        for _, policy in policies:
            policy(engine, str(first['evidence']), str(first['claim']))
    for _, row in df_eval.iterrows():
        ev = str(row['evidence'])
        cl = str(row['claim'])
        lbl = int(row['label'])

        for index in rng.permutation(len(policies)):
            name, policy = policies[index]
            started = time.perf_counter()
            record = policy(engine, ev, cl)
            record['latency_ms'] = (time.perf_counter() - started) * 1000.0
            record['ground_truth'] = lbl
            results[name].append(record)

    common_indices = [i for i in range(len(df_eval)) if all(records[i]['prediction'] is not None for records in results.values())]
    summary = []
    for pol_name, records in results.items():
        # Handle abstained records cleanly (exclude from binary classification metrics or impute)
        eval_records = [r for r in records if r['prediction'] is not None]
        y_true = [r['ground_truth'] for r in eval_records]
        y_pred = [r['prediction'] for r in eval_records]
        calls = [r['verification_calls'] for r in records]
        lats = [r['latency_ms'] for r in records]
        abstained_count = sum(1 for r in records if r.get('abstained', False))

        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        prec = precision_score(y_true, y_pred, zero_division=0) if y_true else 0.0
        rec = recall_score(y_true, y_pred, zero_division=0) if y_true else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0) if y_true else 0.0

        avg_calls = float(np.mean(calls))
        mean_lat = float(np.mean(lats))
        median_lat = float(np.median(lats))
        p95_lat = float(np.percentile(lats, 95))

        summary.append({
            'Method': pol_name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'Common Covered Samples': len(common_indices),
            'Common Covered F1': round(f1_score([records[i]['ground_truth'] for i in common_indices], [records[i]['prediction'] for i in common_indices], zero_division=0), 4) if common_indices else None,
            'Hallucinations Detected / All Gold Hallucinations': round(sum(r['ground_truth'] == 1 and r['prediction'] == 1 for r in records) / max(sum(r['ground_truth'] == 1 for r in records), 1), 4),
            'Avg Verification Calls': round(avg_calls, 2),
            'Mean Latency (ms)': round(mean_lat, 2),
            'Median Latency (ms)': round(median_lat, 2),
            'p95 Latency (ms)': round(p95_lat, 2),
            'Abstained Claims': abstained_count,
            'Coverage (%)': round(len(eval_records) / max(len(records), 1) * 100.0, 2)
        })

    df_summary = pd.DataFrame(summary)
    return df_summary, results
