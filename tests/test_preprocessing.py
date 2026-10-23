"""
test_preprocessing.py
Unit tests for data preprocessing, split construction, leakage prevention, and claim alignment.
"""
from typing import Any as _DocAny

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from preprocessing import split_sentences_with_spans, create_leak_free_splits


class TestPreprocessing(unittest.TestCase):
    """Group related TestPreprocessing operations.

    Responsibility: Provide a cohesive model, interface, or regression-test fixture.
    Constructor arguments and function contracts are documented on the methods. Instance attributes are assigned in __init__; tests construct fresh instances to avoid cross-test state.
    """

    def test_sentence_splitting_and_spans(self) -> None:
        """Verify the sentence splitting and spans regression behavior.

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
        text = "Deep learning models can hallucinate. Verification against evidence is essential!"
        spans = split_sentences_with_spans(text)
        self.assertEqual(len(spans), 2)

        sent1, start1, end1 = spans[0]
        self.assertEqual(sent1, "Deep learning models can hallucinate.")
        self.assertEqual(text[start1:end1], sent1)

        sent2, start2, end2 = spans[1]
        self.assertEqual(sent2, "Verification against evidence is essential!")
        self.assertEqual(text[start2:end2], sent2)

    def test_source_id_leakage_assertion(self) -> None:
        """Verify the source id leakage assertion regression behavior.

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
        # Create synthetic dataset with overlapping source_ids
        records = [
            {'source_id': 'src_1', 'official_split': 'train', 'label': 0, 'claim': 'Claim 1', 'evidence': 'Ev 1'},
            {'source_id': 'src_2', 'official_split': 'train', 'label': 1, 'claim': 'Claim 2', 'evidence': 'Ev 2'},
            {'source_id': 'src_3', 'official_split': 'train', 'label': 0, 'claim': 'Claim 3', 'evidence': 'Ev 3'},
            {'source_id': 'src_4', 'official_split': 'test',  'label': 1, 'claim': 'Claim 4', 'evidence': 'Ev 4'},
            {'source_id': 'src_5', 'official_split': 'test',  'label': 0, 'claim': 'Claim 5', 'evidence': 'Ev 5'}
        ]
        df = pd.DataFrame(records)
        df_train, df_val, df_test = create_leak_free_splits(df, val_ratio=0.33, seed=42)

        train_src = set(df_train['source_id'].unique())
        val_src = set(df_val['source_id'].unique())
        test_src = set(df_test['source_id'].unique())

        # Assert zero leakage
        self.assertEqual(len(train_src.intersection(val_src)), 0)
        self.assertEqual(len(train_src.intersection(test_src)), 0)
        self.assertEqual(len(val_src.intersection(test_src)), 0)

    def test_empty_string_handling(self) -> None:
        """Verify the empty string handling regression behavior.

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
        spans = split_sentences_with_spans("")
        self.assertEqual(spans, [])
        spans_whitespace = split_sentences_with_spans("    \n\t  ")
        self.assertEqual(spans_whitespace, [])

    def test_tiny_split_sizes_creates_non_empty_val(self) -> None:
        """Verify the tiny split sizes creates non empty val regression behavior.

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
        # Even with only 2 sources in train, val split must NOT be empty!
        records = [
            {'source_id': 'src_1', 'official_split': 'train', 'label': 0, 'claim': 'Claim 1', 'evidence': 'Ev 1'},
            {'source_id': 'src_2', 'official_split': 'train', 'label': 1, 'claim': 'Claim 2', 'evidence': 'Ev 2'},
            {'source_id': 'src_3', 'official_split': 'test',  'label': 0, 'claim': 'Claim 3', 'evidence': 'Ev 3'}
        ]
        df = pd.DataFrame(records)
        df_train, df_val, df_test = create_leak_free_splits(df, val_ratio=0.15, seed=42)

        self.assertGreater(len(df_val), 0, "Validation set must not be empty when >= 2 training sources exist!")
        self.assertGreater(len(df_train), 0, "Train set must remain non-empty!")
        self.assertEqual(len(set(df_train['source_id']).intersection(set(df_val['source_id']))), 0)

    def test_single_source_handling(self) -> None:
        """Verify the single source handling regression behavior.

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
        records = [
            {'source_id': 'src_1', 'official_split': 'train', 'label': 0, 'claim': 'Claim 1', 'evidence': 'Ev 1'},
            {'source_id': 'src_2', 'official_split': 'test',  'label': 0, 'claim': 'Claim 2', 'evidence': 'Ev 2'}
        ]
        df = pd.DataFrame(records)
        # Should not crash on single training source
        df_train, df_val, df_test = create_leak_free_splits(df, val_ratio=0.15, seed=42)
        self.assertEqual(len(df_train), 1)
        self.assertEqual(len(df_val), 0)

    def test_split_generation_does_not_overwrite_production_manifest(self) -> None:
        """Verify the split generation does not overwrite production manifest regression behavior.

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
        import tempfile
        prod_manifest_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'split_manifest.json')
        initial_mtime = os.path.getmtime(prod_manifest_path) if os.path.exists(prod_manifest_path) else None

        records = [
            {'source_id': 'src_1', 'official_split': 'train', 'label': 0, 'claim': 'Claim 1', 'evidence': 'Ev 1'},
            {'source_id': 'src_2', 'official_split': 'train', 'label': 1, 'claim': 'Claim 2', 'evidence': 'Ev 2'},
            {'source_id': 'src_3', 'official_split': 'test',  'label': 0, 'claim': 'Claim 3', 'evidence': 'Ev 3'}
        ]
        df = pd.DataFrame(records)
        # Calling create_leak_free_splits without save_manifest must not touch production manifest!
        create_leak_free_splits(df, val_ratio=0.15, seed=42)

        if initial_mtime is not None:
            new_mtime = os.path.getmtime(prod_manifest_path)
            self.assertEqual(initial_mtime, new_mtime, "Production manifest was overwritten by unit test!")

    def test_deterministic_splits(self) -> None:
        """Verify the deterministic splits regression behavior.

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
        records = [
            {'source_id': f'src_{i}', 'official_split': 'train', 'label': i % 2, 'claim': f'Claim {i}', 'evidence': f'Ev {i}'}
            for i in range(10)
        ] + [
            {'source_id': f'src_test_{j}', 'official_split': 'test', 'label': j % 2, 'claim': f'Claim T{j}', 'evidence': f'Ev T{j}'}
            for j in range(4)
        ]
        df = pd.DataFrame(records)
        train1, val1, test1 = create_leak_free_splits(df, val_ratio=0.3, seed=42)
        train2, val2, test2 = create_leak_free_splits(df, val_ratio=0.3, seed=42)

        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_isolated_manifest_saving_in_temp_directory(self) -> None:
        """Verify the isolated manifest saving in temp directory regression behavior.

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
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_manifest = os.path.join(tmp_dir, 'test_manifest.json')
            records = [
                {'source_id': 's1', 'official_split': 'train', 'label': 0, 'claim': 'C1', 'evidence': 'E1'},
                {'source_id': 's2', 'official_split': 'train', 'label': 1, 'claim': 'C2', 'evidence': 'E2'},
                {'source_id': 's3', 'official_split': 'test',  'label': 0, 'claim': 'C3', 'evidence': 'E3'}
            ]
            df = pd.DataFrame(records)
            create_leak_free_splits(df, val_ratio=0.5, seed=42, manifest_path=tmp_manifest, save_manifest=True)
            self.assertTrue(os.path.exists(tmp_manifest), "Manifest was not saved to injected temporary path!")
            with open(tmp_manifest, 'r') as f:
                data = json.load(f)
            self.assertEqual(data['train_sources'], 1)
            self.assertEqual(data['val_sources'], 1)


if __name__ == '__main__':
    unittest.main()
