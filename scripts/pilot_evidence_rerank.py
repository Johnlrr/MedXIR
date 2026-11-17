"""Bounded Kaggle GPU experiment; score source passages, never submit to leaderboard."""
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

import expand_chunks as ex
from evidence_spans import join_overlapping,centered_span,contiguous_around_anchor
from reranker_inputs import qwen_reranker_inputs,QWEN_RERANK_PREFIX,QWEN_RERANK_SUFFIX

N_QUERIES=16
DOC_DEPTH=400
PASSAGE_TOKENS=768
PAIR_TOKENS=2048
INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query'


def main():
    started=time.monotonic();out=Path('/kaggle/working')
    gpu=torch.cuda.get_device_name(0);assert '6000' in gpu,gpu
    print('GPU:',gpu,flush=True)
    pool_path=ex.one_match('pool.parquet');pool=pd.read_parquet(pool_path)
    qids=sorted(pool.query_id.unique(),key=lambda q:hashlib.sha256(str(q).encode()).hexdigest())[:N_QUERIES]
    pool_sample=out/'pilot_pool.parquet';pool[pool.query_id.isin(qids)].to_parquet(pool_sample,index=False)
    base=[e for e in ex.load_submission() if e['id'] in set(qids)]
    ctx=ex.load_inputs(pool_path=pool_sample,base=base)
    queries=dict(pd.read_parquet(ex.one_match('query.parquet'),columns=['id','query']).itertuples(index=False,name=None))
    paths={key:ex.one_match(name+'/config.json').parent for key,name in
           [('bge','bge-reranker-v2-m3'),('qwen','Qwen3-Reranker-4B')]}
    tokenizer=AutoTokenizer.from_pretrained(paths['bge'],local_files_only=True)
    pairs=[];metadata=[];legacy_docs=[];skips=[];seed_only=0;component_recovered=0
    for entry in base:
        qid=int(entry['id']);query=str(queries[qid]);ranked=ctx['ranked'][qid]
        seed_rank={c:i for i,c in enumerate(ranked)}
        best=ex.best_chunks(ranked,ctx['docs_of'])
        candidates=ex.doc_list(entry['relevant_docs'],ranked,ctx['docs_of'],ctx['lang_of'],n_docs=10**6)[:DOC_DEPTH]
        done=set()
        for doc in candidates:
            if doc not in best:continue
            sha=ctx['rows'][best[doc][0]]['sha']
            if sha in done:continue
            done.add(sha)
            selected=[]
            # At most two distinct seeds; no per-query hyperparameters or labels.
            for cid in best[doc]:
                if all(abs(ctx['rows'][cid]['order']-ctx['rows'][c]['order'])>1 for c in selected):
                    selected.append(cid)
                if len(selected)==2:break
            used_spans=set()
            for cid in selected:
                row=ctx['rows'][cid]
                ids=ex.window_ids(cid,2,ctx['rows'],ctx['by_pos'])
                try:source_start,source=join_overlapping([ctx['rows'][c] for c in ids])
                except ValueError as err:
                    if 'Unknown source gap' not in str(err):raise
                    skips.append(dict(qid=qid,chunk_id=cid,reason=str(err)))
                    source_start,source=contiguous_around_anchor([ctx['rows'][c] for c in ids],row)
                    component_recovered+=int(len(source)>len(row['text']))
                seed_only+=int(source_start==row['start'] and source==row['text'])
                offsets=tokenizer(source,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
                a,b,text=centered_span(source,offsets,row['start']-source_start,row['end']-source_start,PASSAGE_TOKENS)
                # Re-tokenize exact final slice, rather than assume token counts are additive.
                tokens=len(tokenizer.encode(text,add_special_tokens=False))
                if tokens>PASSAGE_TOKENS:
                    a,b,text=centered_span(source,offsets,row['start']-source_start,row['end']-source_start,PASSAGE_TOKENS-8)
                    tokens=len(tokenizer.encode(text,add_special_tokens=False))
                assert tokens<=PASSAGE_TOKENS+2 and text==source[a:b]
                key=(source_start+a,source_start+b)
                if key in used_spans:continue
                used_spans.add(key)
                pairs.append((query,text))
                metadata.append(dict(query_id=qid,doc_id=int(doc),text_sha256=sha,seed=cid,seed_rank=seed_rank[cid],
                                     doc_ids=list(map(int,ctx['docs_of'](cid))),start=key[0],end=key[1],
                                     chunk_text=text,passage_tokens=tokens))
            legacy_docs.append((query,ex.doc_text(best[doc],ctx['rows'],ctx['by_text'],400,1)))
        print('prepared',qid,'pairs',len(pairs),flush=True)
    frame=pd.DataFrame(metadata)
    assert len(frame) and frame.query_id.nunique()==N_QUERIES
    frame.to_parquet(out/'passage_candidates.parquet',index=False)
    report=dict(complete=False,gpu=gpu,n_queries=N_QUERIES,pairs=len(pairs),doc_depth=DOC_DEPTH,
                passage_tokens=PASSAGE_TOKENS,pair_tokens=PAIR_TOKENS,skipped_source_gaps=skips,
                pool_sha256=hashlib.sha256(pool_path.read_bytes()).hexdigest(),
                score_quality='UNMEASURED: no gold labels',timings={},truncations={},
                seed_only_windows=seed_only,disconnected_windows_recovered=component_recovered,
                source_gap_policy='maximal contiguous component around seed; never invent gap text')
    order=np.argsort([len(d) for q,d in pairs],kind='stable')
    for key in ['bge','qwen']:
        t0=time.monotonic();path=paths[key]
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left' if key=='qwen' else 'right')
        if key=='qwen':
            model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,local_files_only=True,
                                                     attn_implementation='sdpa').cuda().eval()
            yes,no=tok.convert_tokens_to_ids('yes'),tok.convert_tokens_to_ids('no')
            old_pairs=legacy_docs[:128]
            legacy_texts=[QWEN_RERANK_PREFIX+f'<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {d}'+QWEN_RERANK_SUFFIX for q,d in old_pairs]
            old_ids=tok(legacy_texts,padding=False,truncation=True,max_length=512)['input_ids']
            suffix=tok.encode(QWEN_RERANK_SUFFIX,add_special_tokens=False)
            report['legacy_qwen_suffix_lost']=sum(row[-len(suffix):]!=suffix for row in old_ids)
            report['legacy_qwen_checked']=len(old_ids)
            fixed=qwen_reranker_inputs(tok,[q for q,d in old_pairs],[d for q,d in old_pairs],INSTRUCTION,512,return_tensors=None)
            assert all(row[-len(suffix):]==suffix for row in fixed['input_ids'])
            report['fixed_qwen_suffix_lost']=0
        else:
            model=AutoModelForSequenceClassification.from_pretrained(path,torch_dtype=torch.bfloat16,local_files_only=True,
                                                                    attn_implementation='sdpa').cuda().eval()
        scores=np.empty(len(pairs),dtype=np.float32);batch=16 if key=='qwen' else 32
        truncated=0
        with torch.inference_mode():
            for start in range(0,len(order),batch):
                ix=order[start:start+batch];qs=[pairs[i][0] for i in ix];ds=[pairs[i][1] for i in ix]
                if key=='qwen':
                    body_texts=[f'<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {d}' for q,d in zip(qs,ds)]
                    raw_ids=tok(body_texts,add_special_tokens=False,truncation=False)['input_ids']
                    overhead=len(tok.encode(QWEN_RERANK_PREFIX,add_special_tokens=False))+len(tok.encode(QWEN_RERANK_SUFFIX,add_special_tokens=False))
                    truncated+=sum(len(ids)+overhead>PAIR_TOKENS for ids in raw_ids)
                    enc=qwen_reranker_inputs(tok,qs,ds,INSTRUCTION,PAIR_TOKENS).to('cuda')
                    logits=model(**enc,logits_to_keep=1).logits[:,-1,:]
                    result=torch.softmax(torch.stack([logits[:,no],logits[:,yes]],dim=1).float(),dim=1)[:,1]
                else:
                    enc=tok(qs,ds,padding=True,truncation=False,return_tensors='pt')
                    assert enc['input_ids'].shape[1]<=PAIR_TOKENS,'BGE pair too long'
                    result=torch.sigmoid(model(**enc.to('cuda')).logits.view(-1).float())
                scores[ix]=result.cpu().numpy()
                if start//batch%20==0:
                    rate=(start+len(ix))/max(time.monotonic()-t0,.01)
                    print(key,start+len(ix),'/',len(pairs),'pairs/s',round(rate,2),flush=True)
        assert np.isfinite(scores).all()
        frame[key]=scores
        report['truncations'][key]=truncated
        report['timings'][key]=dict(seconds=time.monotonic()-t0,pairs_per_second=len(pairs)/(time.monotonic()-t0))
        frame.to_parquet(out/'passage_scores.parquet',index=False)
        del model,tok;gc.collect();torch.cuda.empty_cache()
    # Compact review artifact; model agreement is a diagnostic, not a gold metric.
    review=[]
    for qid,g in frame.groupby('query_id'):
        for key in ['bge','qwen']:
            for rank,(_,r) in enumerate(g.sort_values(key,ascending=False).head(5).iterrows()):
                review.append(dict(query_id=int(qid),query=str(queries[qid]),model=key,rank=rank,
                                   doc_id=int(r.doc_id),score=float(r[key]),text=r.chunk_text))
    (out/'passage_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf8')
    report.update(complete=True,elapsed_seconds=time.monotonic()-started,
                  full_run_pair_estimate=len(pairs)*1200/N_QUERIES)
    (out/'passage_pilot_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report['timings']),flush=True)


if __name__=='__main__':main()
