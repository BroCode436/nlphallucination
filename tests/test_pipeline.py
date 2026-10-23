"""
test_pipeline.py
Integration smoke test verifying the end-to-end pipeline components:
Data Preprocessing -> Baseline/Verification Engine -> Adaptive Routing -> Correction -> Metric Reporting.
"""
from typing import Any as _DocAny

import unittest
import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from preprocessing import split_sentences_with_spans, create_leak_free_splits
from verification import VerificationEngine, run_adaptive_verification, compare_verification_policies
from correction import EvidenceGuidedCorrector, evaluate_correction_pipeline


class TestIntegrationPipeline(unittest.TestCase):
    """Group related TestIntegrationPipeline operations.

    Responsibility: Provide a cohesive model, interface, or regression-test fixture.
    Constructor arguments and function contracts are documented on the methods. Instance attributes are assigned in __init__; tests construct fresh instances to avoid cross-test state.
    """

    def test_end_to_end_mini_batch_flow(self) -> None:
        """Verify the end to end mini batch flow regression behavior.

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
        # 1. Mini dataset
        mock_claims = [
            {'source_id': 's1', 'official_split': 'train', 'claim': 'Apollo 11 landed in 1969.', 'evidence': 'Apollo 11 landed on the Moon in July 1969.', 'label': 0},
            {'source_id': 's2', 'official_split': 'train', 'claim': 'Apollo 11 landed in 1985.', 'evidence': 'Apollo 11 landed on the Moon in July 1969.', 'label': 1},
            {'source_id': 's3', 'official_split': 'test',  'claim': 'Oxygen has 8 protons.', 'evidence': 'Oxygen atomic number is 8.', 'label': 0},
            {'source_id': 's4', 'official_split': 'test',  'claim': 'Oxygen has 12 protons.', 'evidence': 'Oxygen atomic number is 8.', 'label': 1},
            {'source_id': 's5', 'official_split': 'test',  'claim': 'Unverifiable statement.', 'evidence': '', 'label': 1}
        ]
        df_claims = pd.DataFrame(mock_claims)

        # 2. Split assertions
        df_train, df_val, df_test = create_leak_free_splits(df_claims, val_ratio=0.5, seed=42)
        self.assertEqual(len(df_test), 3)

        # 3. Verification Engine
        engine = VerificationEngine()
        df_summary, raw_results = compare_verification_policies(engine, df_test, max_samples=3)
        self.assertEqual(len(df_summary), 3)
        self.assertIn('Method', df_summary.columns)

        # 4. Correction and Post-Verification
        corrector = EvidenceGuidedCorrector(verification_engine=engine)
        test_records = df_test[['claim', 'evidence', 'label']].to_dict('records')
        corr_metrics, detailed = evaluate_correction_pipeline(corrector, test_records)

        self.assertIn('Correction Success Rate (CSR %)', corr_metrics)
        self.assertIn('Abstention Rate (AR %)', corr_metrics)


if __name__ == '__main__':
    unittest.main()
