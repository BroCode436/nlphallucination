"""
test_correction.py
Unit tests for evidence-guided targeted claim correction, post-correction verification, and metrics.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from verification import VerificationEngine
from correction import EvidenceGuidedCorrector, evaluate_correction_pipeline


class TestCorrection(unittest.TestCase):

    def setUp(self):
        self.engine = VerificationEngine()
        self.corrector = EvidenceGuidedCorrector(verification_engine=self.engine)

    def test_numerical_contradiction_correction(self):
        ev = "The observatory was founded in 1928 on the summit of the mountain."
        cl = "The observatory was founded in 1985 on the summit of the mountain."
        
        corrected, abstained, reason = self.corrector.correct_claim(cl, ev)
        self.assertFalse(abstained)
        self.assertIn("1928", corrected)
        self.assertNotIn("1985", corrected)

    def test_abstention_on_empty_evidence(self):
        cl = "The treatment cured 90 percent of patients."
        corrected, abstained, reason = self.corrector.correct_claim(cl, "")
        self.assertTrue(abstained)
        self.assertEqual(corrected, cl)

    def test_preservation_of_factual_claims(self):
        ev = "The telescope was deployed in 2021 into the L2 orbit."
        cl = "The telescope was deployed in 2021 into the L2 orbit."
        res = self.corrector.correct_and_reverify(cl, ev, is_detected_hallucination=False)
        self.assertFalse(res['was_edited'])
        self.assertEqual(res['post_verification_status'], 'Accepted-Unmodified')
        self.assertEqual(res['is_hallucinated_post'], 0)

    def test_correction_metrics_computation(self):
        test_records = [
            {'claim': 'Revenue was 90 million.', 'evidence': 'Revenue was 50 million.', 'label': 1, 'prediction': 1},
            {'claim': 'Mars is red.', 'evidence': 'Mars is red due to iron oxide.', 'label': 0, 'prediction': 0},
            {'claim': 'Water freezes at 100 degrees.', 'evidence': '', 'label': 1, 'prediction': 1}
        ]
        metrics, detailed = evaluate_correction_pipeline(self.corrector, test_records)
        self.assertIn('Correction Success Rate (CSR %)', metrics)
        self.assertIn('Residual Hallucination Rate (RHR %)', metrics)
        self.assertIn('Correct Content Preservation Rate (CCPR %)', metrics)
    def test_context_aligned_numerical_correction_with_multiple_evidence_numbers(self):
        # Evidence has multiple numbers (11, 1969, 3)
        ev = "Mission Apollo 11 launched into orbit in July 1969 carrying 3 astronauts."
        cl = "The mission launched into orbit in July 1985."

        corrected, abstained, reason = self.corrector.correct_claim(cl, ev)
        self.assertFalse(abstained)
        self.assertIn("1969", corrected)
        self.assertNotIn("1985", corrected)
        self.assertNotIn("11", corrected)  # Must not blindly substitute 11 for 1985!

    def test_ambiguous_numerical_correction_abstains(self):
        # Evidence has numbers without any contextual grounding to the claim number
        ev = "The recipe requires 4 eggs and 2 cups of sugar."
        cl = "The temperature reached 98 degrees."

        corrected, abstained, reason = self.corrector.correct_claim(cl, ev)
        self.assertTrue(abstained, "Must abstain when numerical alignment is ambiguous!")
        self.assertEqual(corrected, cl, "Unverified claim must remain unmodified upon abstention!")

    def test_omitted_prediction_without_engine_raises_keyerror(self):
        # Corrector without verification engine cannot predict dynamically;
        # missing 'prediction' must strictly raise an error rather than falling back to gold label.
        corrector_no_engine = EvidenceGuidedCorrector(verification_engine=None)
        test_records = [
            {'claim': 'Revenue was 90 million.', 'evidence': 'Revenue was 50 million.', 'label': 1}
        ]
        with self.assertRaises(KeyError):
            evaluate_correction_pipeline(corrector_no_engine, test_records)

    def test_gold_label_cannot_control_correction_decision(self):
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

