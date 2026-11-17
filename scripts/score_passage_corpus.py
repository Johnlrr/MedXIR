"""Kaggle GPU: score prepared parts with durable, hash-bound checkpoints."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification,AutoModelForCausalLM

from reranker_inputs import qwen_reranker_inputs,QWEN_RERANK_PREFIX,QWEN_RERANK_SUFFIX
from assemble_passage_submission import assemble

INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query'
PAIR_TOKENS=4096


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def one(root,pattern):
    hits=sorted(root.rglob(pattern))
    if len(hits)!=1:raise ValueError(f'{pattern}: expected one input, found {len(hits)}')
    return hits[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['bge','qwen'],required=True)
    parser.add_argument('--resume-root',type=Path)
    args=parser.parse_args();key=args.model
    root=Path('/kaggle/input');out=Path('/kaggle/working');started=time.monotonic()
    gpu=torch.cuda.get_device_name(0)
    if '6000' not in gpu:raise ValueError('Expected RTX PRO6000, found '+gpu)
    print('GPU:',gpu,flush=True)
    manifest_path=one(root,'passage_candidates_manifest.json');candidate_root=manifest_path.parent
    candidates=json.loads(manifest_path.read_text())
    if not candidates['complete'] or candidates['queries']!=1200:raise ValueError('Incomplete full candidates')
    qpath=candidate_root/'passage_queries.parquet'
    if sha(qpath)!=candidates['query_file_sha256']:raise ValueError('Corrupted queries')
    queries=dict(pd.read_parquet(qpath,columns=['id','query']).itertuples(index=False,name=None))
    name='bge-reranker-v2-m3' if key=='bge' else 'Qwen3-Reranker-4B'
    model_path=one(root,name+'/config.json').parent
    spec=dict(model=key,candidate_manifest_sha256=sha(manifest_path),model_config_sha256=sha(model_path/'config.json'),
        implementation_sha256=sha(Path(__file__)),input_helper_sha256=sha(Path(__file__).with_name('reranker_inputs.py')),
        max_pair_tokens=PAIR_TOKENS,instruction=INSTRUCTION,precision='bfloat16',attention='sdpa')
    report=dict(complete=False,gpu=gpu,spec=spec,parts=[],pairs=0,truncations=0)
    report_path=out/f'passage_scores_{key}_manifest.json';destination=out/f'passage_scores_{key}';destination.mkdir(exist_ok=True)
    if args.resume_root:
        previous=json.loads((args.resume_root/report_path.name).read_text())
        if previous['spec']!=spec:raise ValueError('Resume scoring contract changed')
        for part in previous['parts']:
            source=args.resume_root/part['path']
            if sha(source)!=part['sha256']:raise ValueError('Corrupted resume score')
            target=out/part['path'];target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source,target);report['parts'].append(part)
    done={p['candidate_path']:p for p in report['parts']}
    def checkpoint():
        report['pairs']=sum(p['pairs'] for p in report['parts'])
        report['elapsed_seconds']=time.monotonic()-started
        temp=report_path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(report_path)
    checkpoint()
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True,padding_side='left' if key=='qwen' else 'right')
    cls=AutoModelForSequenceClassification if key=='bge' else AutoModelForCausalLM
    model=cls.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').cuda().eval()
    if key=='qwen':
        yes,no=tok.convert_tokens_to_ids('yes'),tok.convert_tokens_to_ids('no')
        if yes==no or yes==tok.unk_token_id or no==tok.unk_token_id:raise ValueError('Invalid yes/no token IDs')
        overhead=len(tok.encode(QWEN_RERANK_PREFIX,add_special_tokens=False))+len(tok.encode(QWEN_RERANK_SUFFIX,add_special_tokens=False))
    batch_size=32 if key=='bge' else 16
    for part in candidates['parts']:
        if part['path'] in done:continue
        part_started=time.monotonic();source=candidate_root/part['path']
        if sha(source)!=part['sha256']:raise ValueError('Corrupted candidate part')
        frame=pd.read_parquet(source)
        if len(frame)!=part['pairs'] or frame.passage_id.duplicated().any():raise ValueError('Bad part identity')
        texts=frame.chunk_text.tolist();qs=[str(queries[int(q)]) for q in frame.query_id]
        order=np.argsort([len(t) for t in texts],kind='stable');values=np.empty(len(frame),dtype=np.float32)
        with torch.inference_mode():
            for start in range(0,len(order),batch_size):
                ix=order[start:start+batch_size];q=[qs[i] for i in ix];d=[texts[i] for i in ix]
                if key=='bge':
                    encoded=tok(q,d,padding=True,truncation=False,return_tensors='pt')
                    if encoded['input_ids'].shape[1]>PAIR_TOKENS:raise ValueError('BGE over context budget; no silent truncation')
                    result=torch.sigmoid(model(**encoded.to('cuda')).logits.view(-1).float())
                else:
                    bodies=[f'<Instruct>: {INSTRUCTION}\n<Query>: {a}\n<Document>: {b}' for a,b in zip(q,d)]
                    lengths=tok(bodies,add_special_tokens=False,truncation=False)['input_ids']
                    if any(len(ids)+overhead>PAIR_TOKENS for ids in lengths):raise ValueError('Qwen over context budget; no silent truncation')
                    encoded=qwen_reranker_inputs(tok,q,d,INSTRUCTION,PAIR_TOKENS).to('cuda')
                    logits=model(**encoded,logits_to_keep=1).logits[:,-1,:]
                    result=torch.softmax(torch.stack([logits[:,no],logits[:,yes]],dim=1).float(),dim=1)[:,1]
                values[ix]=result.cpu().numpy()
                if start//batch_size%100==0:
                    rate=(start+len(ix))/max(time.monotonic()-part_started,.01)
                    print(key,source.name,start+len(ix),'/',len(order),'pairs/s',round(rate,1),flush=True)
        if not np.isfinite(values).all():raise ValueError('Nonfinite model score')
        target=destination/source.name
        pd.DataFrame(dict(passage_id=frame.passage_id,score=values)).to_parquet(target,index=False,compression='zstd')
        report['parts'].append(dict(candidate_path=part['path'],candidate_sha256=part['sha256'],
            path=str(target.relative_to(out)),sha256=sha(target),pairs=len(frame),seconds=time.monotonic()-part_started))
        checkpoint();print('checkpoint',key,len(report['parts']),'/',len(candidates['parts']),report['pairs'],flush=True)
    del model,tok;gc.collect();torch.cuda.empty_cache()
    report['complete']=True;checkpoint()
    if report['pairs']!=candidates['pairs']:raise ValueError('Scoring coverage mismatch')
    result=assemble(candidate_root,out,key)
    print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
