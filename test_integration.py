"""Run from installed project root: python -m unittest test_integration -v.
"""
from typing import Any as _DocAny
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from correction import EvidenceGuidedCorrector

class IntegrationTests(unittest.TestCase):
    """Group related IntegrationTests operations.

    Responsibility: Provide a cohesive model, interface, or regression-test fixture.
    Constructor arguments and function contracts are documented on the methods. Instance attributes are assigned in __init__; tests construct fresh instances to avoid cross-test state.
    """
    def test_primary_rejection_cannot_be_averaged_away(self) -> None:
        """Verify the primary rejection cannot be averaged away regression behavior.

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
        engine=Mock()
        engine.verify_check_1.return_value=(.9,0)
        engine.verify_check_2.return_value=(0.,0)
        engine.verify_check_3.return_value=(0.,0)
        guard=Mock();guard.assess.return_value={'accepted':True}
        corrector=EvidenceGuidedCorrector(engine,guard)
        corrector.correct_claim=Mock(return_value=('Alice scored 10 points.',False,'test'))
        result=corrector.correct_and_reverify('Alice scored 30 points.','Alice scored 10 points.')
        self.assertTrue(result['proposal_rejected'])
        self.assertEqual(result['corrected_claim'],'Alice scored 30 points.')
    def test_supported_wrong_entity_is_rejected(self) -> None:
        """Verify the supported wrong entity is rejected regression behavior.

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
        corrector=EvidenceGuidedCorrector()
        self.assertIsNotNone(corrector.validity_model)
        corrector.correct_claim=Mock(return_value=('Bob scored 20 points.',False,'test'))
        result=corrector.correct_and_reverify('Alice scored 30 points.','Alice scored 10 points. Bob scored 20 points.')
        self.assertTrue(result['proposal_rejected'])
        self.assertFalse(result['correction_validity']['accepted'])
    def test_model_is_loaded_by_default(self) -> None:
        """Verify the model is loaded by default regression behavior.

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
        corrector=EvidenceGuidedCorrector()
        self.assertIsNotNone(corrector.validity_model)
        result=corrector.correct_and_reverify('Alice scored 30 points.','Alice scored 10 points.')
        self.assertTrue(result['was_edited'])
        self.assertTrue(result['correction_validity']['accepted'])

if __name__=='__main__':unittest.main()
