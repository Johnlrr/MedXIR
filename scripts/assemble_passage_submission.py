"""Select scored, source-exact passages and stream a DEFLATE submission ZIP."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def choose_query(frame,n_docs=200,n_chunks=100,cover_head=50,min_tokens=40):
    """Max-passage doc ranking, cover the head, then score-order remaining windows.

    Parameters are fixed across queries. No score-to-probability calibration.
    """
    if frame.passage_id.duplicated().any():raise ValueError('Duplicate passage ID')
    values=frame['score'].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('Invalid scores')
    ranked=frame[frame.passage_tokens>=min_tokens].sort_values(
        ['score','seed_rank','passage_id'],ascending=[False,True,True],kind='stable')
    records=ranked.to_dict('records');best={}
    for row in records:
        if row['end']-row['start']!=len(row['chunk_text']):raise ValueError('Invalid passage offsets')
        for doc in row['doc_ids']:best.setdefault(int(doc),row)
    docs=list(best)[:n_docs];docset=set(docs);doc_rank={d:i for i,d in enumerate(docs)}
    chosen=[];used=set();windows={};entries=set()
    def add(row):
        if row['passage_id'] in used or len(chosen)>=n_chunks:return
        start,end=int(row['start']),int(row['end']);key=row['text_sha256']
        for a,b in windows.get(key,[]):
            overlap=max(0,min(end,b)-max(start,a))
            if overlap/min(end-start,b-a)>=.8:return
        ids=sorted((int(d) for d in row['doc_ids'] if int(d) in docset),key=doc_rank.get)
        added=False
        for doc in ids:
            identity=(doc,row['chunk_text'])
            if identity in entries:continue
            if len(chosen)>=n_chunks:break
            entries.add(identity);chosen.append(dict(doc_id=doc,chunk_text=row['chunk_text']));added=True
        if added:
            used.add(row['passage_id']);windows.setdefault(key,[]).append((start,end))
    for doc in docs[:cover_head]:add(best[doc])
    for row in records:add(row)
    if not docs or not chosen:raise ValueError('No usable documents/passages')
    return docs,chosen


def assemble(candidate_root,score_root,model,n_docs=200,start_chunks=100,max_bytes=99_000_000,
             compression='deflate'):
    if compression not in ('deflate','bzip2'):
        raise ValueError('Unsupported ZIP compression')
    zip_method=zipfile.ZIP_DEFLATED if compression=='deflate' else zipfile.ZIP_BZIP2
    suffix='' if compression=='deflate' else '-bz'
    candidate_root,score_root=map(Path,(candidate_root,score_root))
    manifest_path=candidate_root/'passage_candidates_manifest.json'
    candidates=json.loads(manifest_path.read_text())
    score_manifest=json.loads((score_root/f'passage_scores_{model}_manifest.json').read_text())
    if not candidates['complete'] or not score_manifest['complete']:raise ValueError('Incomplete inputs')
    if score_manifest['spec']['candidate_manifest_sha256']!=sha(manifest_path):raise ValueError('Wrong candidate version')
    parts={p['candidate_path']:p for p in score_manifest['parts']}
    expected=set(candidates['spec']['query_ids']);counts=start_chunks
    staging=score_root/f'_submission_{model}';staging.mkdir(exist_ok=True)
    zip_path=score_root/f'medxir-passage-{model}-d{n_docs}{suffix}.zip'
    while counts>=20:
        written=set();doc_count=chunk_count=covered=0
        for part in candidates['parts']:
            source=candidate_root/part['path'];scored_part=parts[part['path']];scores=score_root/scored_part['path']
            if sha(source)!=part['sha256'] or sha(scores)!=scored_part['sha256']:raise ValueError('Corrupted input part')
            frame=pd.read_parquet(source).merge(pd.read_parquet(scores),on='passage_id',validate='one_to_one')
            if len(frame)!=part['pairs'] or set(frame.query_id)!=set(part['query_ids']):raise ValueError('Part coverage mismatch')
            for qid,group in frame.groupby('query_id',sort=False):
                qid=int(qid)
                if qid in written:raise ValueError('Duplicate output query')
                docs,chunks=choose_query(group,n_docs,counts)
                entry=dict(id=qid,relevant_docs=docs,relevant_chunks=chunks)
                (staging/f'{qid}.json').write_text(json.dumps(entry,ensure_ascii=False,separators=(',',':')),encoding='utf8')
                written.add(qid);doc_count+=len(docs);chunk_count+=len(chunks);covered+=len({c['doc_id'] for c in chunks})
        if written!=expected:raise ValueError('Submission query mismatch')
        with zipfile.ZipFile(zip_path,'w',compression=zip_method,compresslevel=9) as archive:
            with archive.open('submission.json','w') as handle:
                handle.write(b'[')
                for i,qid in enumerate(sorted(written)):
                    if i:handle.write(b',')
                    with (staging/f'{qid}.json').open('rb') as item:shutil.copyfileobj(item,handle)
                handle.write(b']')
        print('ZIP',model,'cap',counts,'MB',round(zip_path.stat().st_size/1e6,2),flush=True)
        if zip_path.stat().st_size<max_bytes:break
        counts=min(counts-1,int(counts*.95))
    else:raise ValueError('Cannot fit submission under byte limit')
    with zipfile.ZipFile(zip_path) as archive:
        infos=archive.infolist()
        assert len(infos)==1 and infos[0].compress_type==zip_method
        if archive.testzip() is not None:raise ValueError('ZIP CRC failure')
    for path in staging.glob('*.json'):path.unlink()
    staging.rmdir()
    result=dict(complete=True,model=model,compression=compression,queries=len(written),zip_bytes=zip_path.stat().st_size,zip_sha256=sha(zip_path),
        candidate_manifest_sha256=sha(manifest_path),score_manifest_sha256=sha(score_root/f'passage_scores_{model}_manifest.json'),
        selection=dict(doc_depth=n_docs,chunk_cap=counts,cover_head=50,min_passage_tokens=40,near_overlap_threshold=.8),
        docs_per_query=doc_count/len(written),chunks_per_query=chunk_count/len(written),covered_docs_per_query=covered/len(written),
        source_fidelity='Text copied unchanged from hashed candidates; source offsets asserted during generation.',
        relevance_metrics='UNMEASURED: no gold labels')
    (score_root/f'passage_{model}{suffix}_submission_manifest.json').write_text(json.dumps(result,indent=2))
    return result
