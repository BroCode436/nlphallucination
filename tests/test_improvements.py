"""Local NLP project module: test improvements.

See docs/CODE_WALKTHROUGH.md for dependencies and architecture.
"""
from typing import Any as _DocAny
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from grounding import numerical_conflicts
from verification import VerificationEngine, compute_evidence_quality, run_adaptive_verification, compare_verification_policies
from correction import EvidenceGuidedCorrector
from reliability import probability_metrics, source_bootstrap_f1
from review_corrections import score_reviews
from audit_dataset import audit_splits
import pandas as pd


class ImprovementTests(unittest.TestCase):
    """Group related ImprovementTests operations.

    Responsibility: Provide a cohesive model, interface, or regression-test fixture.
    Constructor arguments and function contracts are documented on the methods. Instance attributes are assigned in __init__; tests construct fresh instances to avoid cross-test state.
    """
    def test_missing_values_abstain(self) -> None:
        """Verify the missing values abstain regression behavior.

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
        for evidence in (None, float('nan'), 'null', ''):
            self.assertEqual(compute_evidence_quality(evidence, 'Mars is red.')[0], 'No Evidence')

    def test_whole_token_coverage(self) -> None:
        """Verify the whole token coverage regression behavior.

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
        score, _ = VerificationEngine().verify_check_2('The cathedral is large.', 'The cat is large.')
        self.assertGreater(score, 0)

    def test_number_from_wrong_entity_does_not_support_claim(self) -> None:
        """Verify the number from wrong entity does not support claim regression behavior.

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
        checks = numerical_conflicts('Alice scored 10 points; Bob scored 20 points.', 'Alice scored 20 points.')
        self.assertTrue(checks[0]['conflict'])

    def test_equivalent_number_formats(self) -> None:
        """Verify the equivalent number formats regression behavior.

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
        checks = numerical_conflicts('Revenue reached 1,000 dollars.', 'Revenue reached 1000.0 dollars.')
        self.assertTrue(checks[0]['supported'])

    def test_ambiguous_number_is_not_definite_conflict(self) -> None:
        """Verify the ambiguous number is not definite conflict regression behavior.

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
        checks = numerical_conflicts('Revenue was 10 dollars; Revenue was 20 dollars.', 'Revenue was 20 dollars.')
        self.assertTrue(checks[0]['ambiguous'])
        self.assertFalse(checks[0]['conflict'])

    def test_numeric_claim_cannot_bypass_rule(self) -> None:
        """Verify the numeric claim cannot bypass rule regression behavior.

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
        engine = VerificationEngine()
        with patch.object(engine, 'predict_primary', return_value=0.01):
            result = run_adaptive_verification(engine, 'Alice scored 10 points; Bob scored 20 points.', 'Alice scored 20 points.')
        self.assertEqual(result['verification_calls'], 3)
        self.assertEqual(result['prediction'], 1)
        self.assertGreaterEqual(result['probability'], 0.5)

    def test_rejected_proposal_preserves_original(self) -> None:
        """Verify the rejected proposal preserves original regression behavior.

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
        engine = VerificationEngine()
        corrector = EvidenceGuidedCorrector(engine)
        with patch.object(corrector, 'correct_claim', return_value=('Revenue was 99 million.', False, 'test')):
            result = corrector.correct_and_reverify('Revenue was 90 million.', 'Revenue was 50 million.')
        self.assertTrue(result['proposal_rejected'])
        self.assertFalse(result['was_edited'])
        self.assertEqual(result['corrected_claim'], 'Revenue was 90 million.')

    def test_all_repeated_numbers_repaired(self) -> None:
        """Verify the all repeated numbers repaired regression behavior.

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
        corrected, abstained, _ = EvidenceGuidedCorrector().correct_claim('Revenue was 90 million; revenue was 90 million.', 'Revenue was 50 million.')
        self.assertFalse(abstained)
        self.assertNotIn('90', corrected)

    def test_calibration_and_cluster_intervals(self) -> None:
        """Verify the calibration and cluster intervals regression behavior.

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
        metrics = probability_metrics([0, 1], [0., 1.])
        self.assertEqual(metrics['Brier Score'], 0)
        self.assertEqual(metrics['Expected Calibration Error'], 0)
        interval = source_bootstrap_f1([0, 1], [0, 1], ['a', 'b'], repeats=20)
        self.assertLessEqual(interval['F1 Source Bootstrap 95% Lower'], interval['F1 Source Bootstrap 95% Upper'])
        self.assertEqual(interval, source_bootstrap_f1([0, 1], [0, 1], ['a', 'b'], repeats=20))

    def test_incomplete_review_cannot_produce_safety_metric(self) -> None:
        """Verify the incomplete review cannot produce safety metric regression behavior.

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
        with self.assertRaises(ValueError):
            score_reviews([{'ground_truth': '0'}])

    def test_independent_review_metrics(self) -> None:
        """Verify the independent review metrics regression behavior.

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
        rows = [{'ground_truth': '1', 'original_claim': 'bad', 'corrected_claim': 'good', 'reviewer': 'r1', 'independent_post_label': '0'},
                {'ground_truth': '0', 'original_claim': 'good', 'corrected_claim': 'bad', 'reviewer': 'r1', 'independent_post_label': '1'}]
        metrics = score_reviews(rows)
        self.assertEqual(metrics['Independent CSR (%)'], 100)
        self.assertEqual(metrics['Independent CIHR (%)'], 100)

    def test_ablation_call_counts_and_common_subset_metrics(self) -> None:
        """Verify the ablation call counts and common subset metrics regression behavior.

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
        engine = VerificationEngine()
        result = run_adaptive_verification(engine, 'Revenue was 10 dollars.', 'Revenue was 20 dollars.', disabled_stage=2)
        self.assertEqual(result['verification_calls'], 2)
        frame = pd.DataFrame([{'evidence': '', 'claim': 'A claim.', 'label': 1},
                              {'evidence': 'Mars is a red planet.', 'claim': 'Mars is red.', 'label': 0}])
        summary, _ = compare_verification_policies(engine, frame)
        self.assertEqual(summary['Common Covered Samples'].nunique(), 1)
        self.assertEqual(summary['Common Covered Samples'].iloc[0], 1)

    def test_duplicate_evidence_detected_under_different_source_ids(self) -> None:
        """Verify the duplicate evidence detected under different source ids regression behavior.

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
        frames = {'train': pd.DataFrame([{'source_id': 'a', 'claim': 'x', 'evidence': 'Same evidence.', 'label': 0}]),
                  'test': pd.DataFrame([{'source_id': 'b', 'claim': 'y', 'evidence': '  same   evidence. ', 'label': 1}])}
        report = audit_splits(frames)
        self.assertTrue(report['passed'])
        self.assertEqual(report['source_overlaps']['train/test'], 0)
        self.assertEqual(report['evidence_fingerprint_overlaps']['train/test'], 1)
        self.assertTrue(report['evidence_overlap_requires_review'])

    def test_configured_threshold_changes_route(self) -> None:
        """Verify the configured threshold changes route regression behavior.

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
        engine = VerificationEngine()
        with patch.object(engine, 'predict_primary', return_value=0.05):
            fast = run_adaptive_verification(engine, 'Mars is a red planet in our solar system.', 'Mars is a red planet.')
            cautious = run_adaptive_verification(engine, 'Mars is a red planet in our solar system.', 'Mars is a red planet.', high_conf_thresh=0.99)
        self.assertEqual(fast['verification_calls'], 1)
        self.assertGreater(cautious['verification_calls'], 1)


if __name__ == '__main__':
    unittest.main()
