import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import title_search as ts


class Helpers(unittest.TestCase):
    def test_rrf_order_rewards_both_lists(self):
        self.assertEqual(ts.rrf_order([['a', 'b', 'c'], ['b', 'd']], k=1), ['b', 'a', 'd', 'c'])

    def test_shown_text_like_doc_text(self):
        self.assertEqual(ts.shown_text('Title. Lead', 0, 'Title. Lead'), 'Title. Lead')
        self.assertEqual(ts.shown_text('Title. Lead', 3, 'answer', lead_chars=6), 'Title.\n...\nanswer')

    def test_cut_score(self):
        self.assertEqual(ts.cut_score([0.1, 0.9, 0.5], cut=2), 0.5)
        self.assertEqual(ts.cut_score([0.1], cut=2), float('-inf'))

    def test_load_titles_one_row_per_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chunks' / 'vi' / 'part-000.parquet'
            path.parent.mkdir(parents=True)
            pd.DataFrame(dict(text_sha256=['a', 'a', 'b', 'c'], rep_doc_id=[1, 1, 2, 3], order=[0, 1, 0, 0],
                              title=['Sỏi thận là gì?', 'Sỏi thận là gì?', '', None])).to_parquet(path)
            titles = ts.load_titles([str(path)])
        self.assertEqual(titles.to_dict('records'), [dict(text_sha256='a', rep_doc_id=1, lang='vi', title='Sỏi thận là gì?')])


if __name__ == '__main__':
    unittest.main()
