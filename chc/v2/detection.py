"""Matched natural-distribution detector evaluation with validation-only tuning.

Separates evidence alignment, token length, and model changes. Binary RAGTruth
labels include unsupported hallucinations, so NLI contradiction alone may not
cover them; both contradiction and non-entailment scores are reported honestly.
"""

from __future__ import annotations
import gc, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
)
from ..detector import ExistingDetector
from .evidence import align_evidence


def reliability(y: np.ndarray, p: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    """Return ten-bin expected calibration error and empirical reliability points.

    Every score enters exactly one half-open bin, including p=1 in the last bin.
    Empty bins are omitted. ECE depends on binning and is not sufficient alone
    to establish useful or trustworthy probability calibration.
    """
    bins = []
    ece = 0.0
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        if mask.any():
            n = int(mask.sum())
            confidence = float(p[mask].mean())
            positive = float(y[mask].mean())
            bins.append({"n": n, "mean_probability": confidence, "positive_fraction": positive})
            ece += n / len(y) * abs(confidence - positive)
    return ece, bins


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate detection metrics on the original sampled class distribution.

    PR-AUC is average precision (step-weighted PR area), not trapezoidal ROC AUC.
    The threshold must be supplied from a validation decision, never test labels.
    """
    prediction = p >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, average="binary", zero_division=0
    )
    ece, bins = reliability(y, p)
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10_bins": ece,
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "threshold": threshold,
        "reliability_bins": bins,
    }


def evaluate_detection(
    project: Path, judge: Any, directory: Path, count: int = 1024
) -> dict[str, Any]:
    """Evaluate fixed variants and fit Platt calibration on validation rows only.

    Samples are uniform with seed 42; positives are not enriched. Test IDs/source
    IDs are audited against validation. Original 256 sample is reproduced as a
    separately marked comparison, not conflated with the larger sample.
    No detector is fine-tuned without a separate supervised experiment.
    """
    directory.mkdir(parents=True, exist_ok=True)
    val = (
        pd.read_csv(project / "data/processed/ragtruth_claims_val.csv")
        .sample(1024, random_state=42)
        .reset_index(drop=True)
    )
    test_all = pd.read_csv(project / "data/processed/ragtruth_claims_test.csv")
    test = test_all.sample(min(count, len(test_all)), random_state=42).reset_index(drop=True)
    # The first 256 rows of this sampled ordering match the historical sample.
    frames = {"validation": val, "test": test}
    scores = {}
    lengths = {}
    detector = ExistingDetector(project, max_length=128)
    for split, frame in frames.items():
        claims = frame.claim.fillna("").astype(str).tolist()
        evidence = frame.evidence.fillna("").astype(str).tolist()
        aligned = [align_evidence(c, e) for c, e in zip(claims, evidence)]
        scores[split] = {}
        lengths[split] = {}
        for name, passages, limit in [
            ("distilbert_full128", evidence, 128),
            ("distilbert_aligned128", aligned, 128),
            ("distilbert_aligned384", aligned, 384),
        ]:
            detector.max_length = limit
            scores[split][name] = np.asarray(detector.predict_pairs(claims, passages))
            tokenlength = [
                len(detector.tokenizer.encode(e, c, truncation=False))
                for c, e in zip(claims, passages)
            ]
            lengths[split][name] = {
                "truncated_pairs": sum(n > limit for n in tokenlength),
                "mean_full_pair_tokens": float(np.mean(tokenlength)),
                "limit": limit,
            }
        nli = judge.probabilities(list(zip(claims, evidence)))
        scores[split]["nli_contradiction"] = np.asarray([r["contradiction"] for r in nli])
        scores[split]["nli_non_entailment"] = np.asarray([1 - r["entailment"] for r in nli])
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    validation_y = val.label.to_numpy(dtype=int)
    test_y = test.label.to_numpy(dtype=int)
    results = []
    predictions = []
    calibrators = {}
    for name, raw in scores["validation"].items():
        for calibrated in (False, True):
            # Platt scaling on log-odds permits prior correction while regularizing
            # a small validation sample; no final test targets enter fitting.
            def logodds(p: np.ndarray) -> np.ndarray:
                """Map bounded probabilities to finite logistic-regression inputs."""
                p = np.clip(p, 1e-5, 1 - 1e-5)
                return np.log(p / (1 - p)).reshape(-1, 1)

            if calibrated:
                cal = LogisticRegression(C=1.0, random_state=42).fit(logodds(raw), validation_y)
                vp = cal.predict_proba(logodds(raw))[:, 1]
                tp = cal.predict_proba(logodds(scores["test"][name]))[:, 1]
                calibrators[name] = {
                    "coefficient": float(cal.coef_[0, 0]),
                    "intercept": float(cal.intercept_[0]),
                    "fit_split": "validation_only",
                }
            else:
                vp = raw
                tp = scores["test"][name]
            thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 99), 0.5])
            best = max(
                thresholds,
                key=lambda t: (binary_metrics(validation_y, vp, float(t))["f1"], float(t)),
            )
            tag = name + ("_platt" if calibrated else "_raw")
            for label, y, p in [
                ("test_1024", test_y, tp),
                ("historical_matched_256", test_y[:256], tp[:256]),
            ]:
                results.append(
                    dict(
                        method=tag,
                        evaluation=label,
                        **binary_metrics(y, p, float(best)),
                        validation_f1=binary_metrics(validation_y, vp, float(best))["f1"],
                    )
                )
            predictions += [
                {
                    "method": tag,
                    "index": int(i),
                    "label": int(y),
                    "score": float(p),
                    "source_id": str(test.iloc[i].source_id),
                }
                for i, (y, p) in enumerate(zip(test_y, tp))
            ]
    for label, y, p in [
        ("test_1024", test_y, scores["test"]["distilbert_full128"]),
        ("historical_matched_256", test_y[:256], scores["test"]["distilbert_full128"][:256]),
    ]:
        results.append(
            dict(
                method="distilbert_original_threshold_0.5",
                evaluation=label,
                **binary_metrics(y, p, 0.5),
            )
        )
    audit = {}
    allframes = {
        s: pd.read_csv(project / f"data/processed/ragtruth_claims_{s}.csv")
        for s in ("train", "val", "test")
    }
    for s, f in allframes.items():
        audit[s] = {
            "rows": len(f),
            "positives": int(f.label.sum()),
            "duplicate_claim_evidence_rows": int(f.duplicated(["claim", "evidence"]).sum()),
            "sources": int(f.source_id.nunique()),
            "missing_evidence": int(f.evidence.isna().sum()),
        }
    audit["source_overlap"] = {
        f"{a}/{b}": len(set(allframes[a].source_id) & set(allframes[b].source_id))
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    }
    report = {
        "results": results,
        "truncation": lengths,
        "calibrators": calibrators,
        "audit": audit,
        "selection": "All thresholds and calibration fitted on validation only; test not used for model choice",
        "sampling": "Uniform seed42 natural rows; not positive-enriched",
        "label_scope": "Original binary RAGTruth hallucination labels; NLI neutral is not proof of falsity",
    }
    (directory / "detector_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(predictions).to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("reliability_bins", "confusion_matrix")}
            for r in results
        ]
    ).to_csv(directory / "metrics.csv", index=False)
    return report


class CalibratedDetector:
    """Validation-selected existing detector with explicit domain-limited calibration.

    Args:
        project: Root holding the unchanged DistilBERT checkpoint.
        report: Completed detector report with validation scores and Platt weights.
    Attributes:
        selected: Fixed choice made by validation F1, never by test performance.
        detector: The retained local detector, with the selected input condition.
    """

    def __init__(self, project: Path, report: Path) -> None:
        """Select an eligible DistilBERT variant from validation measurements only.

        The model is not fine-tuned here. Missing calibration metadata fails
        explicitly. RAGTruth calibration is reported as out-of-domain on HotpotQA.
        """
        metadata = json.loads(report.read_text(encoding="utf-8"))
        candidates = [
            r
            for r in metadata["results"]
            if r["evaluation"] == "test_1024"
            and r["method"].startswith("distilbert_")
            and "validation_f1" in r
        ]
        self.selected = max(candidates, key=lambda r: r["validation_f1"])
        self.name = self.selected["method"]
        self.variant = self.name.rsplit("_", 1)[0]
        self.cal = (
            metadata["calibrators"].get(self.variant) if self.name.endswith("_platt") else None
        )
        self.detector = ExistingDetector(project, max_length=384 if "384" in self.name else 128)

    def predict(
        self,
        claims: list[str],
        evidence: list[str],
        evidence_ids: list[list[str]] | None = None,
        domain: str = "HotpotQA",
    ) -> list[dict[str, Any]]:
        """Return stable binary suspicion scores and actual evidence references.

        Classification means suspected hallucination, not certified contradiction.
        Empty evidence abstains. A calibrated number is only exposed as calibrated
        for RAGTruth; transferred calibrated scores remain marked domain-shifted.
        """
        aligned = (
            [align_evidence(c, e) for c, e in zip(claims, evidence)]
            if "aligned" in self.name
            else evidence
        )
        raw = np.asarray(self.detector.predict_pairs(claims, aligned))
        p = raw.copy()
        if self.cal:
            clipped = np.clip(p, 1e-5, 1 - 1e-5)
            z = self.cal["coefficient"] * np.log(clipped / (1 - clipped)) + self.cal["intercept"]
            p = 1 / (1 + np.exp(-z))
        results = []
        for i, (c, e, s, r) in enumerate(zip(claims, aligned, p, raw)):
            status = (
                "insufficient_evidence"
                if not e.strip()
                else (
                    "hallucination_suspected"
                    if s >= self.selected["threshold"]
                    else "supported_prediction"
                )
            )
            results.append(
                {
                    "hallucination_score": float(s),
                    "raw_score": float(r),
                    "predicted_classification": status,
                    "calibrated_probability": (
                        float(s) if self.cal and domain == "RAGTruth" else None
                    ),
                    "calibration_status": "Validation-only RAGTruth Platt; domain shift on "
                    + domain,
                    "uncertainty": float(1 - abs(2 * s - 1)),
                    "threshold": self.selected["threshold"],
                    "evidence_ids": evidence_ids[i] if evidence_ids else [],
                    "evidence": e,
                    "model_id": self.name,
                }
            )
        return results
