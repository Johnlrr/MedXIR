import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import build_hybrid_notebook
import hybrid_retrieval as hr

LONG_QUERY = ('Em bị viêm xung huyết 1/3 dưới thực quản, viêm dạ dày trợt xung huyết mạn tính, sẹo loét hành tá tràng. '
              'Cho em hỏi các bệnh này của em có nặng và có thể khỏi được không. Em cảm ơn bác sĩ.')


class QueryVariants(unittest.TestCase):
    def test_chitchat_removed_medical_words_kept(self):
        cleaned = hr.clean_query('Chào bác sĩ, cho em hỏi viêm dạ dày có lây không? Em cảm ơn bác sĩ ạ.')
        self.assertNotIn('cho em hỏi', cleaned.lower())
        self.assertNotIn('cảm ơn', cleaned.lower())
        self.assertIn('viêm dạ dày có lây không', cleaned)

    def test_short_query_not_split(self):
        self.assertEqual(hr.sub_queries('Môi nổi mụn mủ vàng là dấu hiệu của bệnh lý gì'), [])

    def test_long_query_split_into_sentences_and_clauses(self):
        subs = hr.sub_queries(LONG_QUERY, max_subq=10)
        self.assertIn('viêm dạ dày trợt xung huyết mạn tính', subs)
        self.assertIn('sẹo loét hành tá tràng', subs)
        self.assertTrue(all(len(s.split()) >= 3 for s in subs))
        self.assertEqual(len(subs), len({s.lower() for s in subs}))
        self.assertLessEqual(len(hr.sub_queries(LONG_QUERY, max_subq=2)), 2)


class Fusion(unittest.TestCase):
    def test_single_list_keeps_order(self):
        ids = ['c3', 'c1', 'c2']
        self.assertEqual([cid for cid, _ in hr.rrf([(ids, 1.0)])], ids)

    def test_agreement_beats_single_list(self):
        fused = hr.rrf([(['a', 'b'], 1.0), (['b', 'c'], 0.5)])
        self.assertEqual(fused[0][0], 'b')


class TranslationLists(unittest.TestCase):
    def test_translation_lists_weighted_only_when_on(self):
        results = {'m': dict(full_ids=[['a']], sub_lists=[[['b']]], trans_lists=[[['c'], ['d']]], prf_ids=None)}
        off = hr.model_candidate_lists(results, {'m': 0.5}, 0, 0.5, 0.5)
        on = hr.model_candidate_lists(results, {'m': 0.5}, 0, 0.5, 0.5, 0.5)
        self.assertEqual(off, [(['a'], 0.5), (['b'], 0.25)])
        self.assertEqual(on, off + [(['c'], 0.25), (['d'], 0.25)])

    def test_results_without_translation_lists(self):
        results = {'m': dict(full_ids=[['a']], sub_lists=[[]], prf_ids=None)}
        self.assertEqual(hr.model_candidate_lists(results, {'m': 1.0}, 0, 0.5, 0.5, 0.5), [(['a'], 1.0)])


class Documents(unittest.TestCase):
    # Doc A: four good chunks; doc B: one slightly better lucky chunk.
    SCORED = [('b1', 0.72), ('a1', 0.71), ('a2', 0.70), ('a3', 0.69), ('a4', 0.66)]

    @staticmethod
    def docs_of(cid):
        return [cid[0].upper()]

    def test_max_and_top3(self):
        self.assertEqual(hr.doc_ranking(self.SCORED, self.docs_of, 'max'), ['B', 'A'])
        self.assertEqual(hr.doc_ranking(self.SCORED, self.docs_of, 'top3'), ['A', 'B'])
        with self.assertRaises(ValueError):
            hr.doc_ranking(self.SCORED, self.docs_of, 'mean')

    def test_assemble_matches_old_dense_lists(self):
        # Old: docs in order of their best chunk (first appearance), chunk docs appended when missing.
        fanout = {'x': [1, 2], 'y': [3], 'z': [1], 'w': [4]}
        scored = [(c, s) for c, s in zip(['x', 'y', 'z', 'w'], [0.9, 0.8, 0.7, 0.6])]
        docs, chunks = hr.assemble(scored, [], fanout.get, n_docs=2, n_chunks=2, agg='max')
        self.assertEqual(docs, [1, 2, 3])  # 3 appended: chunk 'y' is submitted but outside the top-2 documents
        self.assertEqual(chunks, [(1, 'x'), (2, 'x'), (3, 'y')])

    def test_fill_tops_up_documents(self):
        docs, _ = hr.assemble([('a1', 1.0)], ['b1', 'c1', 'a2'], self.docs_of, n_docs=3, n_chunks=1)
        self.assertEqual(docs, ['A', 'B', 'C'])


class Notebook(unittest.TestCase):
    def test_inactive_bm25_does_not_change_dense_only_scoring_mode(self):
        import numpy as np
        env = dict(np=np, rrf=hr.rrf, USE_QUERY_SPLIT=False, USE_PRF=False,
                   USE_BM25=True, BM25_ACTIVE=False, USE_RERANK=False, QUERY_TEXT=['q'],
                   DENSE_FULL_IDS=np.array([['a', 'b']]), SUB_LISTS=[[]], PRF_IDS=None,
                   BM25_IDS=[[]], RRF_K=60, W_SUB=0.5, W_PRF=0.5, W_BM25=0.3, POOL_MAX=600, USE_TRANSLATION=False, W_TRANS=0.5)
        env.update(ACTIVE_MODEL_WEIGHTS={'qwen4b': 1.0}, model_candidate_lists=hr.model_candidate_lists,
                   MODEL_RESULTS={'qwen4b': dict(full_ids=[['a','b']], sub_lists=[[]], prf_ids=None)})
        exec(build_hybrid_notebook.FUSE, env)
        self.assertTrue(env['ONLY_DENSE'])
        self.assertEqual([cid for cid, _ in env['POOLS'][0]], ['a', 'b'])

    def test_weighted_models_and_bm25_are_each_counted_once(self):
        import numpy as np
        results = {'qwen4b': dict(full_ids=[['a','b']], sub_lists=[[]], prf_ids=None),
                   'qwen8b': dict(full_ids=[['b','a']], sub_lists=[[]], prf_ids=None)}
        env = dict(np=np, rrf=hr.rrf, model_candidate_lists=hr.model_candidate_lists,
                   ACTIVE_MODEL_WEIGHTS={'qwen4b': .25, 'qwen8b': .75}, MODEL_RESULTS=results,
                   USE_QUERY_SPLIT=False, USE_PRF=False, BM25_ACTIVE=True, USE_RERANK=False,
                   QUERY_TEXT=['q'], BM25_IDS=[['a']], RRF_K=60, W_SUB=.5, W_PRF=.5, W_BM25=.3, POOL_MAX=600, USE_TRANSLATION=False, W_TRANS=0.5)
        exec(build_hybrid_notebook.FUSE, env)
        scores = dict(env['FUSED'][0])
        self.assertAlmostEqual(scores['a'], .25/61 + .75/62 + .3/61)
        self.assertAlmostEqual(scores['b'], .25/62 + .75/61)
        self.assertFalse(env['ONLY_DENSE'])

    def test_model_weights_change_plain_fusion_without_auxiliary_lists(self):
        import numpy as np
        results = {'qwen4b': dict(full_ids=[['a','b']], sub_lists=[[]], prf_ids=None),
                   'qwen8b': dict(full_ids=[['b','a']], sub_lists=[[]], prf_ids=None)}
        for weights, expected in [({'qwen4b':.1,'qwen8b':.9},'b'),({'qwen4b':.9,'qwen8b':.1},'a')]:
            env=dict(np=np,rrf=hr.rrf,model_candidate_lists=hr.model_candidate_lists,
                     ACTIVE_MODEL_WEIGHTS=weights,MODEL_RESULTS=results,USE_QUERY_SPLIT=False,
                     USE_PRF=False,BM25_ACTIVE=False,USE_RERANK=False,QUERY_TEXT=['q'],
                     BM25_IDS=[[]],RRF_K=60,W_SUB=.5,W_PRF=.5,W_BM25=.3,POOL_MAX=600, USE_TRANSLATION=False, W_TRANS=0.5)
            exec(build_hybrid_notebook.FUSE,env)
            self.assertFalse(env['ONLY_DENSE'])
            self.assertEqual(env['FUSED'][0][0][0],expected)

    def test_every_code_cell_parses_and_config_is_first(self):
        nb = build_hybrid_notebook.notebook(build_hybrid_notebook.CELLS)
        code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        for source in code:
            ast.parse(source)
        self.assertIn('USE_QUERY_SPLIT', code[0])
        for switch in ['USE_CUSTOM_INSTRUCT', 'USE_PRF', 'USE_BM25', 'USE_RERANK', 'DOC_AGG']:
            self.assertIn(switch, code[0])
        json.dumps(nb)


if __name__ == '__main__':
    unittest.main()
