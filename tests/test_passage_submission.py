import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from assemble_passage_submission import choose_query,assemble,sha


def row(pid,doc,score,start=0,text='a'*100,source=None,tokens=50):
    return dict(passage_id=pid,query_id=1,doc_id=doc,doc_ids=[doc],text_sha256=source or str(doc),
                seed_rank=0,start=start,end=start+len(text),chunk_text=text,passage_tokens=tokens,score=score)


class PassageSubmission(unittest.TestCase):
    def test_cover_head_then_another_region_and_reject_tiny_stub(self):
        frame=pd.DataFrame([row('a',1,.9),row('a2',1,.8,90,'b'*100),row('b',2,.7),row('stub',3,.99,text='abc',tokens=3)])
        docs,chunks=choose_query(frame,n_chunks=2,cover_head=2)
        self.assertEqual(docs,[1,2])
        self.assertEqual([c['doc_id'] for c in chunks],[1,2])
        _,chunks=choose_query(frame,n_chunks=3,cover_head=2)
        self.assertEqual([c['doc_id'] for c in chunks],[1,2,1])

    def test_fanout_and_near_duplicate(self):
        a=row('a',1,.9);a['doc_ids']=[1,7]
        frame=pd.DataFrame([a,row('a2',1,.8,10,'a'*100)])
        docs,chunks=choose_query(frame)
        self.assertEqual(docs,[1,7])
        self.assertEqual(len(chunks),2)
        self.assertEqual({c['doc_id'] for c in chunks},{1,7})

    def test_invalid_score(self):
        with self.assertRaises(ValueError):choose_query(pd.DataFrame([row('a',1,float('nan'))]))

    def test_zip_roundtrip_and_source_hash_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);croot=root/'candidates';sroot=root/'scores';croot.mkdir();sroot.mkdir()
            data=pd.DataFrame([row('a',1,.9)]);data.loc[1]=row('b',2,.8);data.loc[1,'query_id']=2
            source=croot/'part.parquet';data.drop(columns='score').to_parquet(source,index=False)
            scores=sroot/'scores.parquet';data[['passage_id','score']].to_parquet(scores,index=False)
            manifest=dict(complete=True,spec=dict(query_ids=[1,2]),parts=[dict(path=source.name,sha256=sha(source),pairs=2,query_ids=[1,2])])
            mp=croot/'passage_candidates_manifest.json';mp.write_text(json.dumps(manifest))
            sm=dict(complete=True,spec=dict(candidate_manifest_sha256=sha(mp)),parts=[dict(candidate_path=source.name,path=scores.name,sha256=sha(scores))])
            (sroot/'passage_scores_bge_manifest.json').write_text(json.dumps(sm))
            result=assemble(croot,sroot,'bge')
            self.assertEqual(result['queries'],2)
            with zipfile.ZipFile(sroot/'medxir-passage-bge-d200.zip') as archive:
                self.assertEqual(archive.infolist()[0].compress_type,zipfile.ZIP_DEFLATED)
                output=json.loads(archive.read('submission.json'))
            self.assertEqual([e['id'] for e in output],[1,2])
            self.assertEqual(output[0]['relevant_chunks'][0]['chunk_text'],'a'*100)
            original_hash=result['zip_sha256']
            bz=assemble(croot,sroot,'bge',compression='bzip2')
            with zipfile.ZipFile(sroot/'medxir-passage-bge-d200-bz.zip') as archive:
                self.assertEqual(archive.infolist()[0].compress_type,zipfile.ZIP_BZIP2)
                self.assertEqual(json.loads(archive.read('submission.json')),output)
            self.assertEqual(bz['compression'],'bzip2')
            self.assertEqual(sha(sroot/'medxir-passage-bge-d200.zip'),original_hash)
            self.assertTrue((sroot/'passage_bge-bz_submission_manifest.json').exists())
            source.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'Corrupted'):assemble(croot,sroot,'bge')


if __name__=='__main__':unittest.main()
