"""Kaggle CPU: stream candidate sources into SQLite, then emit resumable query parts.

RAM estimate: <2 GB for Arrow batches, pool, query lists and per-query text; no
whole-corpus Python dictionary. SQLite is temporary disk state, not an output.
"""
import argparse
from collections import Counter
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import time
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from audit_submission_coverage import iter_queries
from passage_candidates import prepare_query


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def one(root,pattern):
    hits=sorted(root.rglob(pattern))
    if len(hits)!=1:raise ValueError(f'{pattern}: expected one input, got {len(hits)}')
    return hits[0]


def base_documents(root):
    hits=list(root.rglob('medxir-hybrid-submission.json'))
    output={}
    def consume(stream):
        for entry in iter_queries(stream):
            qid=int(entry['id'])
            if qid in output:raise ValueError('Duplicate base query')
            output[qid]=list(map(int,entry['relevant_docs']))
    if len(hits)==1:
        with hits[0].open(encoding='utf-8-sig') as stream:consume(stream)
    elif not hits:
        with zipfile.ZipFile(one(root,'medxir-hybrid-submission.zip')) as archive:
            names=[n for n in archive.namelist() if not n.endswith('/')]
            if len(names)!=1:raise ValueError('Ambiguous base ZIP')
            with archive.open(names[0]) as raw,io.TextIOWrapper(raw,encoding='utf-8-sig') as stream:consume(stream)
    else:raise ValueError('Ambiguous base JSON')
    return output


def build_source_index(root,out,needed):
    db_path=out/'source_cache.sqlite'
    # Rebuild after a crash during ingest, never reuse a partial index.
    if db_path.exists():db_path.unlink()
    db=sqlite3.connect(db_path)
    db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF')
    db.execute('PRAGMA cache_size=-65536')
    db.execute('CREATE TABLE chunks (cid TEXT PRIMARY KEY, prefix TEXT, sha TEXT, doc INTEGER, ordinal INTEGER, start INTEGER, stop INTEGER, text TEXT, lang TEXT)')
    db.execute('CREATE TABLE groups (sha TEXT PRIMARY KEY, docs TEXT)')
    value_set=pa.array(sorted(needed));files=[];kept=0;source_bytes=0
    columns=['chunk_id','text_sha256','rep_doc_id','order','start','end','chunk_text']
    paths=sorted(root.glob('**/chunks/*/part-*.parquet'))
    if not paths:raise ValueError('No chunk corpus parts')
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096,columns=columns):
            table=pa.Table.from_batches([batch])
            prefix=pc.utf8_slice_codeunits(table['text_sha256'],0,16)
            table=table.filter(pc.is_in(prefix,value_set=value_set))
            values=[]
            for r in table.to_pylist():
                if r['end']-r['start']!=len(r['chunk_text']):raise ValueError('Bad corpus source offsets')
                if r['chunk_id'].rsplit('-',1)[0]!=r['text_sha256'][:16]:raise ValueError('Bad chunk identity')
                values.append((r['chunk_id'],r['text_sha256'][:16],r['text_sha256'],int(r['rep_doc_id']),
                    int(r['order']),int(r['start']),int(r['end']),r['chunk_text'],path.parent.name))
                source_bytes+=len(r['chunk_text'].encode())
            db.executemany('INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)',values);kept+=len(values)
        db.commit()
        files.append(dict(path=str(path.relative_to(root)),rows=pq.ParquetFile(path).metadata.num_rows,sha256=sha(path)))
        print('indexed',path.parent.name,path.name,'kept',kept,flush=True)
    db.execute('CREATE INDEX by_prefix ON chunks(prefix)')
    if db.execute('SELECT prefix FROM chunks GROUP BY prefix HAVING COUNT(DISTINCT sha)>1 LIMIT 1').fetchone():
        raise ValueError('SHA prefix collision')
    for batch in pq.ParquetFile(one(root,'hash_groups.parquet')).iter_batches(batch_size=4096,
            columns=['text_sha256','doc_ids','fanout_safe']):
        records=[]
        for r in batch.to_pylist():
            if r['text_sha256'][:16] in needed and r['fanout_safe']:
                ids=list(map(int,r['doc_ids']))
                if len(ids)!=len(set(ids)):raise ValueError('Duplicate fanout ID')
                records.append((r['text_sha256'],json.dumps(ids)))
        db.executemany('INSERT INTO groups VALUES (?,?)',records)
    db.commit()
    return db,dict(indexed_chunks=kept,source_text_bytes=source_bytes,source_files=files)


def query_context(db,ranked):
    prefixes=list(dict.fromkeys(c.rsplit('-',1)[0] for c in ranked))
    marks=','.join('?' for _ in prefixes)
    rows={cid:dict(sha=s,doc=d,order=o,start=a,end=b,text=t,lang=l)
          for cid,p,s,d,o,a,b,t,l in db.execute(f'SELECT * FROM chunks WHERE prefix IN ({marks})',prefixes)}
    if set(ranked)-set(rows):raise ValueError('Pool chunk missing in source corpus')
    shas=list(dict.fromkeys(r['sha'] for r in rows.values()))
    groups={s:json.loads(ds) for s,ds in db.execute('SELECT * FROM groups WHERE sha IN ('+','.join('?' for _ in shas)+')',shas)}
    def docs_of(cid):return groups.get(rows[cid]['sha']) or [rows[cid]['doc']]
    return rows,docs_of


def main():
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working'))
    parser.add_argument('--resume-root',type=Path)
    parser.add_argument('--max-queries',type=int,default=1200)
    args=parser.parse_args();root=args.input;out=args.output;out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();pool_path=one(root,'pool.parquet')
    pool=pd.read_parquet(pool_path,columns=['query_id','chunk_id','final_rank']).sort_values(['query_id','final_rank'])
    query_path=one(root,'query.parquet')
    queries=pd.read_parquet(query_path,columns=['id','query'])
    qids=sorted(map(int,pool.query_id.unique()),key=lambda q:hashlib.sha256(str(q).encode()).hexdigest())[:args.max_queries]
    if len(qids)!=args.max_queries:raise ValueError('Wrong number of pool queries')
    if queries.id.duplicated().any() or not set(qids)<=set(queries.id):raise ValueError('Query input mismatch')
    if args.max_queries==1200 and set(qids)!=set(range(1,1201)):raise ValueError('Unexpected full query IDs')
    base=base_documents(root)
    if not set(qids)<=set(base):raise ValueError('Base submission missing queries')
    pool=pool[pool.query_id.isin(qids)]
    ranked={int(q):g.chunk_id.tolist() for q,g in pool.groupby('query_id')}
    for q,ids in ranked.items():
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate pool candidate')
    spec=dict(pool_sha256=sha(pool_path),query_sha256=sha(query_path),doc_depth=400,
              passage_tokens=768,seeds_per_text=2,neighbour_radius=2,
              source_policy='maximal contiguous component; exact substring',query_ids=qids,
              implementation_sha256=sha(Path(__file__).with_name('passage_candidates.py')),
              span_implementation_sha256=sha(Path(__file__).with_name('evidence_spans.py')),
              tokenizer_config_sha256=sha(one(root,'bge-reranker-v2-m3/config.json')))
    parts_dir=out/'passage_candidates';parts_dir.mkdir(exist_ok=True)
    manifest=dict(complete=False,spec=spec,parts=[],pairs=0,queries=0)
    manifest_path=out/'passage_candidates_manifest.json'
    if args.resume_root:
        old=json.loads((args.resume_root/'passage_candidates_manifest.json').read_text())
        if old['spec']!=spec:raise ValueError('Resume contract changed')
        for part in old['parts']:
            source=args.resume_root/part['path']
            if sha(source)!=part['sha256']:raise ValueError('Resume part corrupted')
            target=out/part['path'];target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source,target);manifest['parts'].append(part)
    done={q for part in manifest['parts'] for q in part['query_ids']}
    needed={c.rsplit('-',1)[0] for q in qids if q not in done for c in ranked[q]}
    tokenizer=AutoTokenizer.from_pretrained(one(root,'bge-reranker-v2-m3/config.json').parent,local_files_only=True)
    db,index_info=build_source_index(root,out,needed) if needed else (None,{})
    manifest['source_index']=index_info;totals=Counter();buffer=[];buffer_qids=[]
    def checkpoint():
        manifest['pairs']=sum(p['pairs'] for p in manifest['parts'])
        manifest['queries']=sum(len(p['query_ids']) for p in manifest['parts'])
        manifest['elapsed_seconds']=time.monotonic()-started
        temp=manifest_path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2));temp.replace(manifest_path)
    def flush():
        if not buffer:return
        path=parts_dir/f'part-{len(manifest["parts"]):03d}.parquet'
        frame=pd.DataFrame(buffer)
        if frame.passage_id.duplicated().any():raise ValueError('Duplicate passage ID')
        frame.to_parquet(path,index=False,compression='zstd')
        manifest['parts'].append(dict(path=str(path.relative_to(out)),sha256=sha(path),
                                     pairs=len(frame),query_ids=list(buffer_qids)))
        buffer.clear();buffer_qids.clear();checkpoint()
    checkpoint()
    for qid in qids:
        if qid in done:continue
        rows,docs_of=query_context(db,ranked[qid])
        current,stats=prepare_query(qid,ranked[qid],base[qid],rows,docs_of,tokenizer)
        totals.update(stats);buffer.extend(current);buffer_qids.append(qid)
        print('prepared',qid,'pairs',len(current),'context_extended',stats['context_extended'],flush=True)
        del rows,docs_of,current
        if len(buffer_qids)>=32:flush();gc.collect()
    flush()
    if db:db.close();(out/'source_cache.sqlite').unlink()
    queries[queries.id.isin(qids)].to_parquet(out/'passage_queries.parquet',index=False)
    manifest.update(complete=True,diagnostics=dict(totals),query_file_sha256=sha(out/'passage_queries.parquet'))
    checkpoint()
    assert manifest['queries']==len(qids)
    print('COMPLETE',manifest['queries'],manifest['pairs'],'seconds',round(manifest['elapsed_seconds']),flush=True)


if __name__=='__main__':main()
