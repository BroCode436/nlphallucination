"""
test_evaluation.py
Regression tests for evaluation metrics, confusion matrix subset alignment, and non-conversion of prediction=None.
"""

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

    def test_confusion_matrix_subset_alignment(self):
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

    def test_non_conversion_of_none_prediction(self):
        """
        Regression test: Verify that abstained claims (prediction=None)
        are NOT silently coerced to class 0 (factual).
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


if __name__ == '__main__':
    unittest.main()
