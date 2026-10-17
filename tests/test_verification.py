"""
test_verification.py
Unit tests for evidence quality assessment, verification engine stages, status assignment, and adaptive routing.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from verification import (
    compute_evidence_quality,
    VerificationEngine,
    run_no_verification,
    run_fixed_verification,
    run_adaptive_verification
)


class TestVerification(unittest.TestCase):

    def setUp(self):
        # Lightweight standalone engine (no heavy model download needed for unit tests)
        self.engine = VerificationEngine()

    def test_evidence_quality_categories(self):
        # 1. No Evidence
        cat, score = compute_evidence_quality("", "The vaccine was approved in 2021.")
        self.assertEqual(cat, 'No Evidence')
        self.assertEqual(score, 0.0)

        # 2. Strong Evidence
        ev_strong = "The vaccine was developed by Pfizer and approved in December 2021 following clinical trials."
        cl_strong = "The vaccine was approved in 2021."
        cat, score = compute_evidence_quality(ev_strong, cl_strong)
        self.assertEqual(cat, 'Strong')
        self.assertGreaterEqual(score, 0.55)

        # 3. Irrelevant Evidence
        ev_irrel = "The international chess championship tournament was held in London with grandmasters."
        cl_irrel = "Photosynthesis requires sunlight and chlorophyll in plant chloroplasts."
        cat, score = compute_evidence_quality(ev_irrel, cl_irrel)
        self.assertIn(cat, ['Irrelevant', 'Weak'])
        self.assertLess(score, 0.55)

    def test_missing_evidence_explicit_abstention(self):
        res = run_adaptive_verification(self.engine, "", "A medical claim without evidence.")
        self.assertTrue(res['abstained'])
        self.assertEqual(res['status'], 'Not Verifiable')
        self.assertIsNone(res['prediction'])
        self.assertEqual(res['verification_calls'], 0)

    def test_numerical_conflict_detection(self):
        ev = "The company reported total quarterly revenue of 45 million dollars."
        cl_conflict = "The company reported revenue of 85 million dollars."
        cl_matching = "The company reported revenue of 45 million dollars."

        prob_conflict, lat1 = self.engine.verify_check_3(ev, cl_conflict)
        prob_matching, lat2 = self.engine.verify_check_3(ev, cl_matching)

        self.assertGreater(prob_conflict, prob_matching)
        self.assertGreaterEqual(prob_conflict, 0.80)
        self.assertLessEqual(prob_matching, 0.30)

    def test_wall_clock_latency_is_non_negative_and_real(self):
        ev = "Mars is the fourth planet from the Sun."
        cl = "Mars orbits the Sun as the fourth planet."
        prob, lat = self.engine.verify_check_2(ev, cl)
        self.assertGreaterEqual(lat, 0.0)
        self.assertLess(lat, 1000.0)  # should execute in under 1 second

    def test_conflicting_verifiers_trigger_deep_verification_and_abstention(self):
        # Create a mock engine with contradictory verifier outputs
        class MockConflictingEngine:
            def verify_check_1(self, ev, cl):
                return 0.70, 1.0  # Stage 1 says moderate hallucination (conf = 0.40, avoids fast path)
            def verify_check_2(self, ev, cl):
                return 0.10, 1.0  # Stage 2 says strong factual (disagreement = 0.60)
            def verify_check_3(self, ev, cl):
                return 0.25, 1.0  # Stage 3 has no numerical conflict

        mock_engine = MockConflictingEngine()
        ev = "The space shuttle was launched in Florida on a sunny morning."
        cl = "The shuttle launched in Florida."

        res = run_adaptive_verification(mock_engine, ev, cl)
        # Must have triggered all 3 checks due to disagreement
        self.assertEqual(res['verification_calls'], 3)
        # Due to extreme irreconcilable contradiction (0.90 vs 0.10, spread 0.80), must abstain explicitly!
        self.assertTrue(res['abstained'])
        self.assertIsNone(res['prediction'])
        self.assertEqual(res['status'], 'Not Verifiable')
        self.assertEqual(res['route'], 'Abstain-Conflicting-Verifiers')


if __name__ == '__main__':
    unittest.main()
