"""Outcome and uncertainty metrics with explicit denominators and missing labels.

The initial experiment observes final-answer exact-match loss. Unannotated
intermediate factual errors are unknown, not silently counted as zero.
"""
from __future__ import annotations

import re
import string
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def normalize_answer(text: str) -> str:
    """Apply standard lowercasing, punctuation/article removal, and whitespace cleanup.

    Args:
        text: Generated or reference short answer.
    Returns:
        Normalized string used for exact match; aliases are not invented.
    Limitations:
        Correct paraphrases or aliases can still fail exact match and need review.
    """
    lower = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', lower).split())


def exact_match(answer: str, reference: str) -> bool:
    """Compare a generated answer to a held-out reference after normalization.

    Args:
        answer: Actual generated continuation, not an assumed policy outcome.
        reference: Dataset reference used by evaluation only.
    Returns:
        True for a nonempty normalized exact match, otherwise False.
    """
    return bool(normalize_answer(reference)) and normalize_answer(answer)==normalize_answer(reference)


def prediction_metrics(truth: list[float], prediction: list[float]) -> dict[str, Any]:
    """Measure benefit-regression error without interpreting correlation as causality.

    Args:
        truth: Observed signed final-loss reductions for held-out actions.
        prediction: Pre-action estimated benefits in corresponding order.
    Returns:
        MAE=mean|y-p|, RMSE=sqrt(mean((y-p)^2)), and Spearman rank correlation.
        Correlation is null when either vector is constant or too short.
    """
    y, p = np.asarray(truth), np.asarray(prediction)
    correlation = None
    if len(y)>1 and np.std(y)>0 and np.std(p)>0:
        correlation = float(spearmanr(y, p).statistic)
    return {'n': len(y), 'mae': float(np.abs(y-p).mean()) if len(y) else None,
            'rmse': float(np.sqrt(np.mean((y-p)**2))) if len(y) else None,
            'spearman': correlation}


def summarize_policies(frame: pd.DataFrame, bootstrap: int = 500, seed: int = 42,
                       cost_lambda: float = .01) -> pd.DataFrame:
    """Aggregate matched policy outcomes and question-cluster bootstrap intervals.

    Args:
        frame: Held-out action rows with actual correct/baseline_correct/cost fields.
        bootstrap: Number of question-level bootstrap replicates.
        seed: Reproducible resampling seed.
        cost_lambda: Nonnegative penalty per logical verification call.
    Returns:
        Method/budget summaries, including paired accuracy improvement intervals.
    Method:
        Average conditions and repeated generation seeds within question first.
        Resample questions, not dependent rows, for the reported confidence limits.
    """
    if cost_lambda < 0:
        raise ValueError('Verification cost penalty cannot be negative')
    rows = []
    for (method, budget), group in frame.groupby(['method', 'budget'], sort=True):
        valid = group.dropna(subset=['correct', 'baseline_correct']).copy()
        valid['gain'] = valid['correct'].astype(float)-valid['baseline_correct'].astype(float)
        clustered = valid.groupby('question_id')['gain'].mean().to_numpy()
        rng = np.random.default_rng(seed)
        boot = [float(rng.choice(clustered, len(clustered), replace=True).mean()) for _ in range(bootstrap)] if len(clustered) else []
        total_calls = float(valid['calls'].sum())
        baseline_errors = float((1-valid['baseline_correct'].astype(float)).sum())
        gain = float(valid['gain'].sum())
        rows.append({'method': method, 'budget': int(budget), 'evaluations': len(valid),
                     'questions': int(valid['question_id'].nunique()), 'final_accuracy': float(valid['correct'].mean()),
                     'baseline_accuracy': float(valid['baseline_correct'].mean()),
                     'mean_final_node_error': float((1-valid['correct'].astype(float)).mean()),
                     'broader_downstream_error_count': None,
                     'mean_gain': float(valid['gain'].mean()), 'gain_ci_low': float(np.quantile(boot,.025)) if boot else None,
                     'gain_ci_high': float(np.quantile(boot,.975)) if boot else None,
                     'harmful_rate_per_selected_episode': float((valid.loc[valid['calls']>0,'gain']<0).mean()) if total_calls else 0.,
                     'mean_calls': float(valid['calls'].mean()), 'mean_estimated_tokens': float(valid['estimated_tokens'].mean()),
                     'gain_per_call': gain/total_calls if total_calls else None,
                     'final_node_cascade_reduction': gain/baseline_errors if baseline_errors else None,
                     'objective_loss_plus_cost': float(((1-valid['correct'].astype(float))+cost_lambda*valid['calls']).mean()),
                     'seed_accuracy_std': float(valid.groupby('generation_seed')['correct'].mean().std(ddof=0))})
    return pd.DataFrame(rows)
