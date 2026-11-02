import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import expand_chunks as ec


def make_rows(spec):
    """spec: chunk id -> (lang, n_bytes); every chunk its own text and document (doc = index)."""
    rows, by_pos = {}, {}
    for i, (cid, (lang, size)) in enumerate(spec.items()):
        rows[cid] = dict(sha=f's{i}', doc=i, order=0, start=0, end=size, text='x' * size, lang=lang)
        by_pos[(f's{i}', 0)] = cid
    return rows, by_pos


class ChunkQuota(unittest.TestCase):
    def setUp(self):
        self.rows, self.by_pos = make_rows({f'v{i}': ('vi', 10) for i in range(8)} | {'z0': ('zh', 10), 'e0': ('en', 10)})
        self.ranked = [f'v{i}' for i in range(8)] + ['z0', 'e0']
        self.docs_of = lambda cid: [self.rows[cid]['doc']]

    def test_no_quota_takes_rank_order(self):
        chosen = ec.build_query(self.ranked, 0, 50, self.rows, self.by_pos, self.docs_of)
        self.assertEqual(chosen, ['v0', 'v1', 'v2', 'v3', 'v4'])

    def test_quota_reserves_zh_en_bytes(self):
        chosen = ec.build_query(self.ranked, 0, 50, self.rows, self.by_pos, self.docs_of, quota=0.4)
        self.assertEqual(chosen, ['v0', 'v1', 'v2', 'z0', 'e0'])

    def test_quota_unused_share_goes_back_to_vi(self):
        ranked = [f'v{i}' for i in range(8)] + ['z0']
        chosen = ec.build_query(ranked, 0, 50, self.rows, self.by_pos, self.docs_of, quota=0.4)
        self.assertEqual(sorted(chosen), sorted(['v0', 'v1', 'v2', 'z0', 'v3']))


class DocList(unittest.TestCase):
    def setUp(self):
        self.lang_of = {1: 'vi', 2: 'vi', 3: 'vi', 4: 'vi', 5: 'zh', 6: 'en', 7: 'vi'}
        self.ranked = ['c1', 'c5', 'c2', 'c3', 'c4', 'c6', 'c7']
        self.docs_of = lambda cid: [int(cid[1:])]

    def test_unchanged_by_default(self):
        self.assertEqual(ec.doc_list([1, 2, 3, 4], self.ranked, self.docs_of, self.lang_of), [1, 2, 3, 4])

    def test_extend_with_pool_order(self):
        self.assertEqual(ec.doc_list([1, 2], self.ranked, self.docs_of, self.lang_of, n_docs=4), [1, 2, 5, 3])

    def test_doc_quota_swaps_lowest_vi(self):
        docs = ec.doc_list([1, 2, 3, 4], self.ranked, self.docs_of, self.lang_of, doc_quota=0.5)
        self.assertEqual(docs, [1, 2, 5, 6])

    def test_doc_quota_already_met(self):
        self.assertEqual(ec.doc_list([1, 5, 6, 2], self.ranked, self.docs_of, self.lang_of, doc_quota=0.5), [1, 5, 6, 2])


class DocRerank(unittest.TestCase):
    def test_no_scores_keeps_order(self):
        self.assertEqual(ec.fuse_docs([3, 1, 2], {'m': {}}, {'m': 1.0}), [3, 1, 2])

    def test_reranker_lifts_a_document_over_the_cut(self):
        candidates = list(range(10))
        scores = {'m': {d: (1.0 if d == 9 else 0.5 - d / 100) for d in candidates}}
        fused = ec.fuse_docs(candidates, scores, {'m': 1.0}, k=1)
        self.assertIn(9, fused[:5])

    def test_rerank_only_ignores_base_order(self):
        scores = {'m': {1: 0.1, 2: 0.9, 3: 0.5}}
        self.assertEqual(ec.fuse_docs([1, 2, 3], scores, {'m': 1.0}, w_base=0.0), [2, 3, 1])

    def test_best_chunk_and_text(self):
        rows = {'a-0': dict(sha='a', order=0, text='Title. Lead text'), 'a-3': dict(sha='a', order=3, text='answer'),
                'a-5': dict(sha='a', order=5, text='more'), 'b-0': dict(sha='b', order=0, text='other')}
        by_text = {'a': ['a-0', 'a-3', 'a-5'], 'b': ['b-0']}
        docs = {'a-3': [7], 'b-0': [8], 'a-5': [7], 'a-0': [7]}
        best = ec.best_chunks(['a-5', 'b-0', 'a-3', 'a-0'], docs.get)
        self.assertEqual(best, {7: ['a-5', 'a-3', 'a-0'], 8: ['b-0']})
        self.assertEqual(ec.doc_text(best[7], rows, by_text, lead_chars=6), 'Title.\n...\nmore')
        self.assertEqual(ec.doc_text(best[7], rows, by_text, lead_chars=6, n_chunks=2), 'Title.\n...\nanswer\n...\nmore')
        self.assertEqual(ec.doc_text(best[8], rows, by_text), 'other')

    def test_offset_weakens_the_base_term(self):
        scores = {'m': {1: 0.1, 2: 0.9}}
        self.assertEqual(ec.fuse_docs([1, 2], scores, {'m': 1.0}, w_base=2.0, k=1), [1, 2])
        self.assertEqual(ec.fuse_docs([1, 2], scores, {'m': 1.0}, w_base=2.0, k=1, offset=100), [2, 1])


class Windows(unittest.TestCase):
    def setUp(self):
        # Text a: chunks a-0..a-5 (doc 1), text b: b-0..b-2 (doc 2), text c: c-0 (doc 3); 10 bytes each, no overlap.
        self.rows, self.by_pos = {}, {}
        for sha, n, doc in (('a', 6, 1), ('b', 3, 2), ('c', 1, 3)):
            for o in range(n):
                cid = f'{sha}-{o}'
                self.rows[cid] = dict(sha=sha, doc=doc, order=o, start=10 * o, end=10 * o + 10, text=f'{cid:<10}')
                self.by_pos[(sha, o)] = cid
        self.docs_of = lambda cid: [self.rows[cid]['doc']]

    def build(self, order, docs, n_cover, budget):
        return [c for c, _ in ec.build_windows(order, docs, n_cover, 1, budget, self.rows, self.by_pos, self.docs_of)]

    def test_window_is_the_chunk_and_its_neighbours(self):
        self.assertEqual(ec.window_ids('a-3', 1, self.rows, self.by_pos), ['a-2', 'a-3', 'a-4'])
        self.assertEqual(ec.window_ids('a-0', 1, self.rows, self.by_pos), ['a-0', 'a-1'])

    def test_one_window_per_text_before_second_windows(self):
        self.assertEqual(self.build(['a-1', 'a-4', 'b-1', 'c-0'], [1, 2, 3], 0, 1000), ['a-1', 'b-1', 'c-0', 'a-4'])

    def test_centre_inside_a_window_is_skipped(self):
        self.assertEqual(self.build(['a-1', 'a-2', 'a-4'], [1], 0, 1000), ['a-1', 'a-4'])

    def test_top_documents_first(self):
        # Budget for two 3-chunk windows: doc 2 (listed first) gets its window before the second text in score order.
        self.assertEqual(self.build(['a-1', 'c-0', 'b-1'], [2, 1, 3], 1, 60), ['b-1', 'a-1'])

    def test_blended_order(self):
        order = ec.blended_order(['x', 'y', 'z'], [[1.0, 0.5, 0.0], [0.0, 0.2, 1.0]], [0.25, 0.75])
        self.assertEqual(order, ['z', 'y', 'x'])   # 0.75, 0.275, 0.25

    def test_with_bm25w_appends_outside_chunks(self):
        ids, bge, rrf, bm = ec.with_bm25w(['x', 'y'], [0.9, 0.1], [0.02, 0.01], (['y', 'w'], [5.0, 3.0]))
        self.assertEqual(ids, ['x', 'y', 'w'])
        self.assertEqual(bge, [0.9, 0.1, 0.0])
        self.assertEqual(rrf, [0.02, 0.01, 0.01])
        self.assertEqual(bm, [0.0, 5.0, 3.0])

    def test_conf_doc_count_clips(self):
        scores = {'m': {d: d / 10 for d in range(10)}}
        self.assertEqual(ec.conf_doc_count(scores, 'm', 0.45, 2, 8), 5)
        self.assertEqual(ec.conf_doc_count(scores, 'm', 0.95, 2, 8), 2)
        self.assertEqual(ec.conf_doc_count(scores, 'm', 0.0, 2, 8), 8)


class Variants(unittest.TestCase):
    def test_only_names_exist(self):
        self.assertTrue(set(ec.ONLY) <= set(ec.VARIANTS))


if __name__ == '__main__':
    unittest.main()
