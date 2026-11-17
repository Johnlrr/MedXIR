"""Kaggle CPU diagnostic: measure truncation and candidate coverage without labels."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import expand_chunks as ex


def summary(xs):
    a=np.asarray(xs,dtype=float)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.quantile(a,.5)),
                p90=float(np.quantile(a,.9)), maximum=float(a.max())) if len(a) else {}


def main():
    start=time.monotonic()
    out=Path('/kaggle/working')
    pool_path=ex.one_match('pool.parquet')
    pool=pd.read_parquet(pool_path)
    qids=sorted(pool.query_id.unique(),key=lambda q:hashlib.sha256(str(q).encode()).hexdigest())[:64]
    sample=pool[pool.query_id.isin(qids)]
    sample_path=out/'audit_pool.parquet';sample.to_parquet(sample_path,index=False)
    base=[e for e in ex.load_submission() if e['id'] in set(qids)]
    ctx=ex.load_inputs(pool_path=sample_path,base=base)
    queries=dict(pd.read_parquet(ex.one_match('query.parquet'),columns=['id','query']).itertuples(index=False,name=None))
    tokenizer_path=ex.one_match('bge-reranker-v2-m3/config.json').parent
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    result={'sample_qids':list(map(int,qids)), 'pool_rows':len(pool), 'pool_queries':int(pool.query_id.nunique()),
            'pool_sha256':hashlib.sha256(pool_path.read_bytes()).hexdigest(),
            'candidate_texts':len(ctx['by_text']), 'tokenizer':str(tokenizer_path),
            'note':'No gold labels; truncation and coverage diagnostics are not retrieval metrics.'}
    lengths={k:[] for k in ['query','chunk_pair','doc_pair','two_chunk_pair','three_chunk_pair']}
    coverage=[]; losses=[]; spans=[]
    for qi,entry in enumerate(base):
        qid=entry['id'];query=str(queries[qid]);ranked=ctx['ranked'][qid]
        best=ex.best_chunks(ranked,ctx['docs_of'])
        docs=ex.doc_list(entry['relevant_docs'],ranked,ctx['docs_of'],ctx['lang_of'],n_docs=10**6)[:400]
        seen=set()
        coverage.append({'qid':qid,'pool_texts':len({ctx['rows'][c]['sha'] for c in ranked}),
                         'candidate_docs':len(docs),'no_chunk':sum(d not in best for d in docs)})
        qtokens=tok.encode(query,add_special_tokens=False);lengths['query'].append(len(qtokens))
        for doc in docs:
            if doc not in best:continue
            cid=best[doc][0];row=ctx['rows'][cid];sha=row['sha']
            if sha in seen:continue
            seen.add(sha)
            ids=ex.window_ids(cid,1,ctx['rows'],ctx['by_pos'])
            # contiguous two-chunk candidate containing seed; prefer following neighbour.
            orders={ctx['rows'][c]['order']:c for c in ids}
            nxt=orders.get(row['order']+1);prev=orders.get(row['order']-1)
            two=[cid,nxt] if nxt else ([prev,cid] if prev else [cid])
            texts={'chunk_pair':row['text'],
                   'doc_pair':ex.doc_text(best[doc],ctx['rows'],ctx['by_text'],400,1),
                   'two_chunk_pair':ex.merge_run([ctx['rows'][c] for c in two]),
                   'three_chunk_pair':ex.merge_run([ctx['rows'][c] for c in ids])}
            for key,text in texts.items():
                lengths[key].append(len(tok(query,text,add_special_tokens=True,truncation=False)['input_ids']))
            lead=ctx['rows'][ctx['by_text'][sha][0]]
            lead_tokens=len(tok.encode(lead['text'][:400],add_special_tokens=False)) if lead['order']!=row['order'] else 0
            losses.append(dict(lang=row['lang'],seed_order=row['order'],
                               lead_tokens=lead_tokens,seed_tokens=len(tok.encode(row['text'],add_special_tokens=False)),
                               query_tokens=len(qtokens)))
            if qi<4 and len(spans)<40:
                spans.append(dict(query_id=int(qid),doc_id=int(doc),chunk_id=cid,query=query,
                                  source_chunk=row['text'],rerank_doc=texts['doc_pair']))
        print('audit',qi+1,len(base),flush=True)
    result['lengths']={k:summary(v) for k,v in lengths.items()}
    result['truncated_fraction']={k:{str(limit):float(np.mean(np.asarray(v)>limit)) for limit in [512,1024,2048]}
                                  for k,v in lengths.items() if k!='query'}
    result['candidate_coverage']=coverage
    result['seed_not_at_document_start_fraction']=float(np.mean([r['seed_order']>0 for r in losses]))
    result['lead_tokens']=summary([r['lead_tokens'] for r in losses if r['lead_tokens']])
    result['by_language']={lang:{'pairs':sum(r['lang']==lang for r in losses),
                                'seed_tokens':summary([r['seed_tokens'] for r in losses if r['lang']==lang])}
                           for lang in ['vi','en','zh']}
    result.update(complete=True,elapsed_seconds=time.monotonic()-start)
    (out/'retrieval_input_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'retrieval_input_samples.json').write_text(json.dumps(spans,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result['truncated_fraction']),flush=True)


if __name__=='__main__':main()
