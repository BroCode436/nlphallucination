"""Validation-tuned benefit boosting with question-balanced regression weights.

Extends the original CHC regressor rather than introducing a large architecture.
The central richer/final-only comparison uses this identical tuning procedure.
Neutral targets remain; weights balance questions, not desired benefit signs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
from collections import Counter
import json, joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from ..models import training_graph
from ..graph import FEATURE_NAMES


def flatten_targets(
    records: list[dict[str, Any]], columns: list[int]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract masked actual labels after label-free features, retaining question IDs.

    Args:
        records: One predeclared train or validation partition with named targets.
        columns: Numerical feature columns for a matched retrained ablation.
    Returns:
        [A,F] inputs, [A] signed utility targets, and [A] question grouping IDs.
        Missing targets are excluded by the existing graph mask, never filled.
    """
    xs = []
    ys = []
    ids = []
    for record in records:
        x, edges, y, mask = training_graph(record)
        if mask.any():
            xs.append(x[mask][:, columns])
            ys.append(y[mask])
            ids.extend([record["question_id"]] * int(mask.sum()))
    if not xs:
        raise ValueError("No measurable targets in this partition")
    return np.concatenate(xs), np.concatenate(ys), ids


def fit_boosting(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    path: Path,
    columns: list[int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit a small predeclared grid and choose mean question-validation MSE.

    Training weights give each question equal total weight regardless of missing
    labels or repeated seed rows. The grid varies leaf count, minimum leaf size
    and L2 regularization. No final-test features/labels enter fitting or tuning.
    Best model and feature columns use the existing TabularPolicy file contract.
    Failure cases include empty measurable partitions or incompatible state data.
    """
    columns = list(range(len(FEATURE_NAMES))) if columns is None else columns
    x, y, ids = flatten_targets(train, columns)
    vx, vy, vids = flatten_targets(validation, columns)
    counts = Counter(ids)
    weights = np.array([1 / counts[q] for q in ids])
    weights *= len(weights) / weights.sum()
    groups = {q: np.array([i for i, v in enumerate(vids) if v == q]) for q in sorted(set(vids))}
    best = None
    trials = []
    for leaves in (3, 7, 15):
        for minimum in (3, 10):
            for l2 in (1.0, 5.0, 20.0):
                model = HistGradientBoostingRegressor(
                    max_iter=80,
                    max_leaf_nodes=leaves,
                    min_samples_leaf=minimum,
                    l2_regularization=l2,
                    random_state=seed,
                    early_stopping=False,
                )
                model.fit(x, y, sample_weight=weights)
                error = (model.predict(vx) - vy) ** 2
                score = float(np.mean([error[index].mean() for index in groups.values()]))
                params = {
                    "max_leaf_nodes": leaves,
                    "min_samples_leaf": minimum,
                    "l2_regularization": l2,
                    "max_iter": 80,
                }
                trials.append(dict(params, validation_question_mse=score))
                if best is None or score < best[0]:
                    best = (score, model, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best[1],
            "columns": columns,
            "feature_names": FEATURE_NAMES,
            "seed": seed,
            "target": "Explicit V2 utility; scope set by caller",
            "question_balanced_weights": True,
            "parameters": best[2],
        },
        path,
    )
    report = {
        "validation_question_mse": best[0],
        "selected_parameters": best[2],
        "train_candidates": len(y),
        "train_questions": len(counts),
        "target_counts": {str(v): int((y == v).sum()) for v in np.unique(y)},
        "trials": trials,
    }
    path.with_suffix(".tuning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
