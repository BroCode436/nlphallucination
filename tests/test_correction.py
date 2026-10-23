"""
test_correction.py
Unit tests for evidence-guided targeted claim correction, post-correction verification, and metrics.
"""
from typing import Any as _DocAny

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from verification import VerificationEngine
from correction import EvidenceGuidedCorrector, evaluate_correction_pipeline


class TestCorrection(unittest.TestCase):
    """Group related TestCorrection operations.

    Responsibility: Provide a cohesive model, interface, or regression-test fixture.
    Constructor arguments and function contracts are documented on the methods. Instance attributes are assigned in __init__; tests construct fresh instances to avoid cross-test state.
    """

    def setUp(self) -> None:
        """Setup.

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
        self.engine = VerificationEngine()
        self.corrector = EvidenceGuidedCorrector(verification_engine=self.engine)

    def test_numerical_contradiction_correction(self) -> None:
        """Verify the numerical contradiction correction regression behavior.

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
        ev = "The observatory was founded in 1928 on the summit of the mountain."
        cl = "The observatory was founded in 1985 on the summit of the mountain."

        corrected, abstained, reason = self.corrector.correct_claim(cl, ev)
        self.assertFalse(abstained)
        self.assertIn("1928", corrected)
        self.assertNotIn("1985", corrected)

    def test_abstention_on_empty_evidence(self) -> None:
        """Verify the abstention on empty evidence regression behavior.

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
        cl = "The treatment cured 90 percent of patients."
        corrected, abstained, reason = self.corrector.correct_claim(cl, "")
        self.assertTrue(abstained)
        self.assertEqual(corrected, cl)

    def test_preservation_of_factual_claims(self) -> None:
        """Verify the preservation of factual claims regression behavior.

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
        ev = "The telescope was deployed in 2021 into the L2 orbit."
        cl = "The telescope was deployed in 2021 into the L2 orbit."
        res = self.corrector.correct_and_reverify(cl, ev, is_detected_hallucination=False)
        self.assertFalse(res['was_edited'])
        self.assertEqual(res['post_verification_status'], 'Accepted-Unmodified')
        # An unmodified detector prediction is not an independent factuality label.
        self.assertIsNone(res['is_hallucinated_post'])

    def test_correction_metrics_computation(self) -> None:
        """Verify the correction metrics computation regression behavior.

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
        test_records = [
            {'claim': 'Revenue was 90 million.', 'evidence': 'Revenue was 50 million.', 'label': 1, 'prediction': 1},
            {'claim': 'Mars is red.', 'evidence': 'Mars is red due to iron oxide.', 'label': 0, 'prediction': 0},
            {'claim': 'Water freezes at 100 degrees.', 'evidence': '', 'label': 1, 'prediction': 1}
        ]
        metrics, detailed = evaluate_correction_pipeline(self.corrector, test_records)
        self.assertIn('Correction Success Rate (CSR %)', metrics)
        self.assertIn('Residual Hallucination Rate (RHR %)', metrics)
        self.assertIn('Correct Content Preservation Rate (CCPR %)', metrics)
    def test_context_aligned_numerical_correction_with_multiple_evidence_numbers(self) -> None:
        """Verify the context aligned numerical correction with multiple evidence numbers regression behavior.

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
        # Evidence has multiple numbers (11, 1969, 3)
        ev = "Mission Apollo 11 launched into orbit in July 1969 carrying 3 astronauts."
        cl = "The mission launched into orbit in July 1985."

        corrected, abstained, reason = self.corrector.correct_claim(cl, ev)
        self.assertFalse(abstained)
        self.assertIn("1969", corrected)
        self.assertNotIn("1985", corrected)
        self.assertNotIn("11", corrected)  # Must not blindly substitute 11 for 1985!

    def test_ambiguous_numerical_correction_abstains(self) -> None:
        """Verify the ambiguous numerical correction abstains regression behavior.

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
        # Evidence has numbers without any contextual grounding to the claim number
        ev = "The recipe requires 4 eggs and 2 cups of sugar."
        cl = "The temperature reached 98 degrees."

        corrected, abstained, reason = self.corrector.correct_claim(cl, ev)
        self.assertTrue(abstained, "Must abstain when numerical alignment is ambiguous!")
        self.assertEqual(corrected, cl, "Unverified claim must remain unmodified upon abstention!")

    def test_omitted_prediction_without_engine_raises_keyerror(self) -> None:
        """Verify the omitted prediction without engine raises keyerror regression behavior.

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
        # Corrector without verification engine cannot predict dynamically;
        # missing 'prediction' must strictly raise an error rather than falling back to gold label.
        corrector_no_engine = EvidenceGuidedCorrector(verification_engine=None)
        test_records = [
            {'claim': 'Revenue was 90 million.', 'evidence': 'Revenue was 50 million.', 'label': 1}
        ]
        with self.assertRaises(KeyError):
            evaluate_correction_pipeline(corrector_no_engine, test_records)

    def test_gold_label_cannot_control_correction_decision(self) -> None:
        """Verify the gold label cannot control correction decision regression behavior.

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
        # Even if gold label is 1 (Hallucinated), detector prediction=0 must keep claim unmodified
        test_records = [
            {'claim': 'The observatory was founded in 1985.', 'evidence': 'The observatory was founded in 1928.', 'label': 1, 'prediction': 0}
        ]
        metrics, detailed = evaluate_correction_pipeline(self.corrector, test_records)
        self.assertFalse(detailed[0]['was_edited'], "Prediction=0 must prevent editing even if gold label=1!")
        self.assertEqual(detailed[0]['post_verification_status'], 'Accepted-Unmodified')

        # Even if gold label is 0 (Factual), detector prediction=1 must trigger targeted edit attempt
        test_records_fp = [
            {'claim': 'The observatory was founded in 1985.', 'evidence': 'The observatory was founded in 1928.', 'label': 0, 'prediction': 1}
        ]
        metrics_fp, detailed_fp = evaluate_correction_pipeline(self.corrector, test_records_fp)
        self.assertTrue(detailed_fp[0]['was_edited'], "Prediction=1 must trigger edit attempt even if gold label=0!")


if __name__ == '__main__':
    unittest.main()

