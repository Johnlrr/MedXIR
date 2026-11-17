from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_passage_corpus import build_source_index,query_context


class SourceIndex(unittest.TestCase):
    def test_only_requested_texts_and_safe_fanout(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);corpus=root/'corpus';out=root/'output';out.mkdir()
            parts=corpus/'chunks/vi';parts.mkdir(parents=True)
            rows=[]
            for letter,doc,safe in [('a',1,True),('b',2,False),('c',3,True)]:
                rows.append(dict(chunk_id=letter*16+'-0',text_sha256=letter*64,rep_doc_id=doc,
                                 order=0,start=0,end=3,chunk_text='abc'))
            pd.DataFrame(rows).to_parquet(parts/'part-000.parquet',index=False)
            pd.DataFrame([dict(text_sha256=letter*64,doc_ids=[doc,doc+10],fanout_safe=safe)
                          for letter,doc,safe in [('a',1,True),('b',2,False),('c',3,True)]]).to_parquet(corpus/'hash_groups.parquet',index=False)
            db,info=build_source_index(corpus,out,{'a'*16,'b'*16})
            try:
                self.assertEqual(info['indexed_chunks'],2)
                indexed,docs=query_context(db,['a'*16+'-0','b'*16+'-0'])
                self.assertEqual(len(indexed),2)
                self.assertEqual(docs('a'*16+'-0'),[1,11])
                self.assertEqual(docs('b'*16+'-0'),[2])
                with self.assertRaisesRegex(ValueError,'missing'):query_context(db,['c'*16+'-0'])
            finally:db.close()


if __name__=='__main__':unittest.main()
