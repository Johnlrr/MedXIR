import unittest
import pandas as pd
from scripts.check_passage_pilot import validate


class PilotValidation(unittest.TestCase):
    def setUp(self):
        self.candidates = pd.DataFrame([dict(query_id=1, doc_id=5, doc_ids=[5, 6],
            text_sha256='abc', seed='abc-0', start=10, end=13, chunk_text='abc', passage_tokens=2)])
        self.scores = self.candidates.assign(bge=.2, qwen=.3)
        self.report = dict(complete=True, pairs=1, n_queries=1, passage_tokens=768,
                           fixed_qwen_suffix_lost=0, truncations=dict(bge=0, qwen=0))

    def test_valid_and_corrupted_outputs(self):
        self.assertTrue(validate(self.candidates, self.scores, self.report)['valid'])
        for column, value in [('bge', float('nan')), ('qwen', 1.1), ('end', 14), ('chunk_text', 'xyz')]:
            corrupted = self.scores.copy()
            corrupted.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises((ValueError, AssertionError)):
                validate(self.candidates, corrupted, self.report)

    def test_invalid_fanout_even_when_both_files_match(self):
        candidates = self.candidates.copy()
        candidates.at[0, 'doc_ids'] = [6]
        with self.assertRaisesRegex(ValueError, 'fanout'):
            validate(candidates, candidates.assign(bge=.2, qwen=.3), self.report)


if __name__ == '__main__':
    unittest.main()
