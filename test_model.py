"""Local NLP project module: test model.

See docs/CODE_WALKTHROUGH.md for dependencies and architecture.
"""
from typing import Any as _DocAny
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from correction_validity import CorrectionValidityModel, numbers
from app import verify_payload

class ModelTests(unittest.TestCase):
    """Group related ModelTests operations.

    Responsibility: Provide a cohesive model, interface, or regression-test fixture.
    Constructor arguments and function contracts are documented on the methods. Instance attributes are assigned in __init__; tests construct fresh instances to avoid cross-test state.
    """
    @classmethod
    def setUpClass(cls):cls.model=CorrectionValidityModel()
    def test_original_swap(self) -> None:
        """Verify the original swap regression behavior.

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
        ev='Atlas Museum opened in 1998. Beacon Museum opened in 2005.'
        self.assertTrue(self.model.assess(ev,'Atlas Museum opened in 2001.','Atlas Museum opened in 1998.')['accepted'])
        self.assertFalse(self.model.assess(ev,'Beacon Museum opened in 2001.','Atlas Museum opened in 1998.')['accepted'])
    def test_deletion_rejected_and_rollback(self) -> None:
        """Verify the deletion rejected and rollback regression behavior.

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
        original='Atlas Museum opened in 2001 and is located in Delhi.'
        result=self.model.assess('Atlas Museum opened in 1998 and is located in Delhi.',original,'Atlas Museum opened in 1998.')
        self.assertFalse(result['accepted']);self.assertEqual(result['final_claim'],original)
    def test_signed_decimal_and_unicode(self) -> None:
        """Verify the signed decimal and unicode regression behavior.

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
        self.assertEqual(numbers('१० पदक'),{'10'})
        self.assertNotEqual(numbers('-5 degrees'),numbers('5 degrees'))
        self.assertEqual(numbers('1,000.00 kg'),{'1000'})
        self.assertFalse(self.model.assess('The mass is 1.5 kg.','The mass is 1.6 kg.','The mass is 1.6 kg.')['accepted'])
    def test_missing_inputs(self) -> None:
        """Verify the missing inputs regression behavior.

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
        self.assertFalse(self.model.assess('','Alice scored 20.','Alice scored 10.')['accepted'])
    def test_negation_evidence(self) -> None:
        """Verify the negation evidence regression behavior.

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
        self.assertFalse(self.model.assess('The medicine does not cure the disease.','The medicine cures the disease.','The medicine does cure the disease.')['accepted'])
    def test_relation_roles_require_review(self) -> None:
        """Verify the relation roles require review regression behavior.

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
        self.assertFalse(self.model.assess('Alice hired Bob.','Alice hired Bob.','Bob hired Alice.')['accepted'])
    def test_no_forced_edit(self) -> None:
        """Verify the no forced edit regression behavior.

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
        self.assertEqual(self.model.propose('Alice scored 10 points.','Alice scored 10 points.')['status'],'unchanged')
    def test_numeric_repair(self) -> None:
        """Verify the numeric repair regression behavior.

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
        result=self.model.propose('Alice scored 10 points. Bob scored 20 points.','Alice scored 30 points.')
        self.assertEqual(result['status'],'repaired');self.assertEqual(result['final_claim'],'Alice scored 10 points.')
    def test_source_validation(self) -> None:
        """Verify the source validation regression behavior.

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
        with self.assertRaises(ValueError):verify_payload(self.model,{'original':'Claim','sources':[{'text':'Evidence','url':'javascript:alert(1)'}]})
    def test_source_provenance(self) -> None:
        """Verify the source provenance regression behavior.

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
        result=verify_payload(self.model,{'original':'Alice scored 30 points.','sources':[{'text':'Alice scored 10 points.','url':'https://example.org/source'}]})
        self.assertEqual(result['sources'][0]['url'],'https://example.org/source')
        self.assertIn('not fetched',result['sources'][0]['provenance'])

if __name__=='__main__':unittest.main()
