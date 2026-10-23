"""Probability calibration and source-cluster uncertainty for detector evaluation.
"""
from typing import Any as _DocAny
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def probability_metrics(labels: _DocAny, probabilities: _DocAny, bins: int=10) -> _DocAny:
    """Probability metrics.

    Measures probability quality and source-cluster uncertainty.

    Learning notes — input/output contract:
        Parameters:
            labels: Reference correctness labels; evaluation targets, not serving-time features.
            probabilities: Predicted positive-class scores; calibration must be evaluated separately.
            bins: Number of score intervals used to summarize calibration.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(labels) or labels.shape != probabilities.shape:
        raise ValueError('Labels and probabilities must be nonempty aligned vectors')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Expected binary labels and finite probabilities in [0, 1]')
    if bins < 1:
        raise ValueError('bins must be positive')
    membership = np.minimum((probabilities * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = membership == index
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return {'Brier Score': float(brier_score_loss(labels, probabilities)),
            'Expected Calibration Error': float(ece),
            'Average Precision': float(average_precision_score(labels, probabilities)) if labels.sum() else None,
            'ROC AUC': float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None}


def source_bootstrap_f1(labels: _DocAny, predictions: _DocAny, source_ids: _DocAny, repeats: int=200, seed: int=42) -> _DocAny:
    """Resample sources, keeping correlated claims together; percentile 95% CI.

    Learning notes — input/output contract:
        Parameters:
            labels: Reference correctness labels; evaluation targets, not serving-time features.
            predictions: Model/heuristic decisions whose quality is being measured.
            source_ids: Source-group identifiers used to preserve dependence during resampling.
            repeats: Number of bootstrap resamples; more resamples reduce Monte Carlo noise.
            seed: Pseudorandom seed controlling reproducible sampling or model initialization.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    labels, predictions, source_ids = map(np.asarray, (labels, predictions, source_ids))
    if not len(labels) or labels.ndim != 1 or labels.shape != predictions.shape or labels.shape != source_ids.shape:
        raise ValueError('Expected nonempty aligned vectors')
    if repeats < 2:
        raise ValueError('At least two bootstrap replicates are required')
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError('Expected binary labels and predictions')
    sources = np.unique(source_ids)
    groups = [np.flatnonzero(source_ids == source) for source in sources]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        scores.append(f1_score(labels[indices], predictions[indices], zero_division=0))
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return {'F1 Source Bootstrap 95% Lower': float(lower),
            'F1 Source Bootstrap 95% Upper': float(upper),
            'Bootstrap Sources': len(sources), 'Bootstrap Replicates': repeats}
