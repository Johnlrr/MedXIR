import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import bm25_words as bw


def segment(text):
    """Stand-in for pyvi: joins the two known compounds."""
    return text.replace('sỏi thận', 'sỏi_thận').replace('triệu chứng', 'triệu_chứng')


class QueryWords(unittest.TestCase):
    def test_words_keep_repeats_and_drop_punctuation(self):
        self.assertEqual(bw.query_words('Triệu chứng sỏi thận? Sỏi thận!', segment),
                         ['triệu chứng', 'sỏi thận', 'sỏi thận'])

    def test_count_terms_matches_syllable_runs(self):
        bw.VOCAB.clear()
        bw.VOCAB.update({'sỏi thận': 0, 'đau': 1})
        bw.PREFIX.clear()
        bw.PREFIX.add(('sỏi', 'thận'))
        counts, length = bw.count_terms('Đau do sỏi thận, sỏi ở thận', 4)
        self.assertEqual(dict(counts), {0: 1, 1: 1})
        self.assertEqual(length, 7)


class EndToEnd(unittest.TestCase):
    def test_small_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'in', Path(tmp) / 'out'
            (root / 'chunks' / 'vi').mkdir(parents=True)
            out.mkdir()
            texts = {'a' * 64: ['sỏi thận gây đau lưng', 'điều trị sỏi thận bằng thuốc', 'kết luận'],
                     'b' * 64: ['đau đầu và chóng mặt'], 'c' * 64: ['triệu chứng của sỏi thận là đau']}
            rows, doc = [], 10
            for sha, chunks in texts.items():
                start = 0
                for order, text in enumerate(chunks):
                    rows.append(dict(chunk_id=f'{sha[:16]}-{order}', text_sha256=sha, rep_doc_id=doc, order=order,
                                     start=start, end=start + len(text), chunk_text=text))
                    start += len(text)
                doc += 1
            pd.DataFrame(rows).to_parquet(root / 'chunks' / 'vi' / 'part-000.parquet')
            pd.DataFrame(dict(text_sha256=['a' * 64], doc_ids=[[10, 99]], fanout_safe=[True])).to_parquet(
                root / 'hash_groups.parquet')
            pd.DataFrame(dict(id=[1, 2], query=['Triệu chứng sỏi thận?', 'chóng mặt'])).to_parquet(root / 'query.parquet')
            bw.INPUT, bw.OUT = str(root), out
            bw.TOP_K, bw.N_DOCS = 3, (2,)
            bw.main(segment, workers=1)
            with zipfile.ZipFile(out / 'medxir-bm25w-d2-submission.zip') as archive:
                sub = json.loads(archive.read(archive.namelist()[0]))
            first = sub[0]
            self.assertEqual(first['relevant_docs'][:1], [12])            # text c: both query words
            self.assertIn(10, first['relevant_docs'])
            self.assertIn(99, first['relevant_docs'])                    # fan-out document of text a
            a_window = [c['chunk_text'] for c in first['relevant_chunks'] if c['doc_id'] == 10][0]
            self.assertIn('sỏi thận gây đau lưng', a_window)              # best chunk 0 + neighbour 1
            self.assertEqual(sub[1]['relevant_docs'], [11])
            with np.load(out / 'bm25w-texts-k500.npz', allow_pickle=True) as texts_npz:
                self.assertEqual(texts_npz['text_sha256'][0, 0], 'c' * 64)


if __name__ == '__main__':
    unittest.main()
