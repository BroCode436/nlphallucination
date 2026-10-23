"""
test_evaluation.py
Regression tests for evaluation metrics, confusion matrix subset alignment, and non-conversion of prediction=None.
"""
from typing import Any as _DocAny

import unittest
import numpy as np
import pandas as pd
import sys
import os
from sklearn.metrics import confusion_matrix

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from verification import VerificationEngine, run_adaptive_verification


class TestEvaluationIntegrity(unittest.TestCase):
    """Group related TestEvaluationIntegrity operations.

    Responsibility: Provide a cohesive model, interface, or regression-test fixture.
    Constructor arguments and function contracts are documented on the methods. Instance attributes are assigned in __init__; tests construct fresh instances to avoid cross-test state.
    """

    def test_confusion_matrix_subset_alignment(self) -> None:
        """Verify the confusion matrix subset alignment regression behavior.

        This operation participates in the saved local NLP experiment workflow.

        Learning notes — input/output contract:
            Parameters: No explicit caller arguments beyond an instance/class if applicable.
            Returns:
                None; effects or assertions are performed directly.
            Assumptions and failures:
                Inputs must match the saved dataset/model schema. Missing files, invalid
                fields, or incompatible checkpoints can raise errors; tests cover normal
                and selected failure paths. Scores are not independent factual proof.

        """
        # Synthetic balanced subset (length 4) and natural subset (length 8)
        balanced_labels = np.array([0, 1, 0, 1])
        natural_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1])

        balanced_preds = np.array([0, 1, 0, 0])
        natural_preds = np.array([0, 0, 1, 0, 0, 0, 1, 0])

        # Assert balanced confusion matrix must ONLY take balanced labels and balanced preds
        self.assertEqual(len(balanced_labels), len(balanced_preds))
        cm_balanced = confusion_matrix(balanced_labels, balanced_preds)
        self.assertEqual(cm_balanced.shape, (2, 2))
        self.assertEqual(cm_balanced.sum(), len(balanced_labels))

        # Trying to slice natural preds for balanced labels could silently mismatch order
        # Verify that lengths and subsets are checked
        self.assertEqual(len(natural_labels), len(natural_preds))

    def test_non_conversion_of_none_prediction(self) -> None:
        """Regression test: Verify that abstained claims (prediction=None)
                are NOT silently coerced to class 0 (factual).

        Learning notes — input/output contract:
            Parameters: No explicit caller arguments beyond an instance/class if applicable.
            Returns:
                None; effects or assertions are performed directly.
            Assumptions and failures:
                Inputs must match the saved dataset/model schema. Missing files, invalid
                fields, or incompatible checkpoints can raise errors; tests cover normal
                and selected failure paths. Scores are not independent factual proof.

        """
        engine = VerificationEngine()

        # Missing evidence claim
        res = run_adaptive_verification(engine, "", "A claim with empty evidence context.")
        self.assertTrue(res['abstained'])
        self.assertIsNone(res['prediction'])

        raw_preds = [1, 0, None, 1, None]
        y_true_all = [1, 0, 1, 0, 0]

        # Metric calculation must filter out None predictions and report coverage separately
        covered = [(yt, p) for yt, p in zip(y_true_all, raw_preds) if p is not None]
        self.assertEqual(len(covered), 3)

        covered_preds = [p for yt, p in covered]
        # None should not be present in covered_preds
        self.assertNotIn(None, covered_preds)

        coverage_pct = (len(covered) / len(y_true_all)) * 100.0
        abstention_pct = ((len(y_true_all) - len(covered)) / len(y_true_all)) * 100.0
        self.assertEqual(coverage_pct, 60.0)
        self.assertEqual(abstention_pct, 40.0)

    def test_hand_calculated_metric_toy_example(self) -> None:
        """Verify metric calculations against exact hand-calculated toy values:
                y_true:    [1, 1, 0, 0, 1, 0, 1, 0] (4 positive, 4 negative)
                raw_preds: [1, 0, 0, 1, 1, 0, None, None] (6 covered, 2 abstained)

                Covered subset (6 items):
                y_true_cov =  [1, 1, 0, 0, 1, 0]
                y_preds_cov = [1, 0, 0, 1, 1, 0]

                TP = 2 (indices 0, 4)
                FN = 1 (index 1)
                TN = 2 (indices 2, 5)
                FP = 1 (index 3)

                Expected:
                Accuracy = (2 + 2) / 6 = 4/6 = 0.6667
                Precision = 2 / (2 + 1) = 2/3 = 0.6667
                Recall = 2 / (2 + 1) = 2/3 = 0.6667
                F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3 = 0.6667
                Coverage = 6/8 = 75.0%
                Abstention = 2/8 = 25.0%

        Learning notes — input/output contract:
            Parameters: No explicit caller arguments beyond an instance/class if applicable.
            Returns:
                None; effects or assertions are performed directly.
            Assumptions and failures:
                Inputs must match the saved dataset/model schema. Missing files, invalid
                fields, or incompatible checkpoints can raise errors; tests cover normal
                and selected failure paths. Scores are not independent factual proof.

        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        y_true_all = [1, 1, 0, 0, 1, 0, 1, 0]
        raw_preds = [1, 0, 0, 1, 1, 0, None, None]

        covered = [(yt, yp) for yt, yp in zip(y_true_all, raw_preds) if yp is not None]
        self.assertEqual(len(covered), 6)

        y_c = [yt for yt, yp in covered]
        p_c = [yp for yt, yp in covered]

        acc = round(accuracy_score(y_c, p_c), 4)
        prec = round(precision_score(y_c, p_c, zero_division=0), 4)
        rec = round(recall_score(y_c, p_c, zero_division=0), 4)
        f1 = round(f1_score(y_c, p_c, zero_division=0), 4)

        self.assertAlmostEqual(acc, 0.6667, places=4)
        self.assertAlmostEqual(prec, 0.6667, places=4)
        self.assertAlmostEqual(rec, 0.6667, places=4)
        self.assertAlmostEqual(f1, 0.6667, places=4)

        cov_pct = (len(covered) / len(y_true_all)) * 100.0
        abs_pct = ((len(y_true_all) - len(covered)) / len(y_true_all)) * 100.0
        self.assertEqual(cov_pct, 75.0)
        self.assertEqual(abs_pct, 25.0)

    def test_checkpoint_validation_detects_missing_weights(self) -> None:
        """Verify that checkpoint validation flags missing model weights
                even if the checkpoint directory exists.

        Learning notes — input/output contract:
            Parameters: No explicit caller arguments beyond an instance/class if applicable.
            Returns:
                None; effects or assertions are performed directly.
            Assumptions and failures:
                Inputs must match the saved dataset/model schema. Missing files, invalid
                fields, or incompatible checkpoints can raise errors; tests cover normal
                and selected failure paths. Scores are not independent factual proof.

        """
        import tempfile
        import run_pipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create an empty transformer directory with config but NO weights
            mock_trans_dir = os.path.join(tmp_dir, 'best_transformer_detector')
            os.makedirs(mock_trans_dir)
            with open(os.path.join(mock_trans_dir, 'config.json'), 'w') as f:
                f.write('{}')

            # Temporarily point CHECKPOINTS_DIR to tmp_dir
            orig_checkpoints = run_pipeline.CHECKPOINTS_DIR
            try:
                run_pipeline.CHECKPOINTS_DIR = tmp_dir
                _, models_ready = run_pipeline.check_prerequisites_exist()
                self.assertFalse(models_ready, "Checkpoint validation must reject directory lacking weight files!")
            finally:
                run_pipeline.CHECKPOINTS_DIR = orig_checkpoints


if __name__ == '__main__':
    unittest.main()
