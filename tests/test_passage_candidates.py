import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from passage_candidates import prepare_query


class CharacterTokenizer:
    def __call__(self,text,**kwargs):return dict(offset_mapping=[(i,i+1) for i in range(len(text))])
    def encode(self,text,**kwargs):return list(text)


class CandidateWindows(unittest.TestCase):
    def test_exact_source_and_fanout_across_distant_gap(self):
        source='abc defghijk  nopqrstuvwx'
        rows={f's-{i}':dict(sha='s',doc=9,order=i,start=a,end=b,text=source[a:b])
              for i,(a,b) in enumerate([(0,3),(4,10),(8,12),(14,20),(18,24)])}
        result,stats=prepare_query(1,['s-1','s-4'],[9],rows,lambda cid:[9,10],CharacterTokenizer(),passage_tokens=20)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['chunk_text'],source[4:12])
        for r in result:
            self.assertEqual(r['chunk_text'],source[r['start']:r['end']])
            self.assertEqual(r['doc_ids'],[9,10])
        self.assertEqual(stats['context_extended'],2)
        self.assertEqual(len({r['passage_id'] for r in result}),2)

    def test_candidate_identity_changes_with_query(self):
        rows={'s-0':dict(sha='s',doc=9,order=0,start=0,end=3,text='abc')}
        args=(['s-0'],[9],rows,lambda cid:[9],CharacterTokenizer())
        a,_=prepare_query(1,*args);b,_=prepare_query(2,*args)
        self.assertNotEqual(a[0]['passage_id'],b[0]['passage_id'])


if __name__=='__main__':unittest.main()
