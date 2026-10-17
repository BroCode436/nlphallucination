"""
test_preprocessing.py
Unit tests for data preprocessing, split construction, leakage prevention, and claim alignment.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from preprocessing import split_sentences_with_spans, create_leak_free_splits


class TestPreprocessing(unittest.TestCase):

    def test_sentence_splitting_and_spans(self):
        text = "Deep learning models can hallucinate. Verification against evidence is essential!"
        spans = split_sentences_with_spans(text)
        self.assertEqual(len(spans), 2)
        
        sent1, start1, end1 = spans[0]
        self.assertEqual(sent1, "Deep learning models can hallucinate.")
        self.assertEqual(text[start1:end1], sent1)

        sent2, start2, end2 = spans[1]
        self.assertEqual(sent2, "Verification against evidence is essential!")
        self.assertEqual(text[start2:end2], sent2)

    def test_source_id_leakage_assertion(self):
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

    def test_empty_string_handling(self):
        spans = split_sentences_with_spans("")
        self.assertEqual(spans, [])
        spans_whitespace = split_sentences_with_spans("    \n\t  ")
        self.assertEqual(spans_whitespace, [])

    def test_tiny_split_sizes_creates_non_empty_val(self):
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

    def test_single_source_handling(self):
        records = [
            {'source_id': 'src_1', 'official_split': 'train', 'label': 0, 'claim': 'Claim 1', 'evidence': 'Ev 1'},
            {'source_id': 'src_2', 'official_split': 'test',  'label': 0, 'claim': 'Claim 2', 'evidence': 'Ev 2'}
        ]
        df = pd.DataFrame(records)
        # Should not crash on single training source
        df_train, df_val, df_test = create_leak_free_splits(df, val_ratio=0.15, seed=42)
        self.assertEqual(len(df_train), 1)
        self.assertEqual(len(df_val), 0)


if __name__ == '__main__':
    unittest.main()
