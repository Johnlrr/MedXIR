"""Stream already-ranked submissions to test fixed document cutoffs.

Preserves every submitted chunk. Refuses a cutoff which would remove any chunk's
document. No corpus/model work and no leaderboard calls.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from audit_submission_coverage import iter_queries


def file_sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--depths',type=int,nargs='+',default=[160,120])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if any(d<1 for d in args.depths):raise ValueError('Document cutoff must be positive')
    report=dict(source=str(args.source.resolve()),source_sha256=file_sha(args.source),variants={})
    for depth in args.depths:
        target=args.output/f'medxir-passage-bge-d{depth}-samechunks.zip'
        if target.exists():raise FileExistsError(target)
        temporary=target.with_suffix('.tmp');seen=set();doc_count=chunk_count=0
        old_chunks=hashlib.sha256();new_chunks=hashlib.sha256();changed=0
        with zipfile.ZipFile(args.source) as source,zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as dest:
            members=[m for m in source.infolist() if not m.is_dir()]
            if len(members)!=1 or not members[0].filename.endswith('.json'):raise ValueError('Expected one JSON')
            with source.open(members[0]) as raw,io.TextIOWrapper(raw,encoding='utf-8-sig') as stream,dest.open('submission.json','w') as output:
                output.write(b'[')
                for entry in iter_queries(stream):
                    qid=int(entry['id'])
                    if qid in seen:raise ValueError('Duplicate query')
                    original=json.dumps([qid,entry['relevant_chunks']],ensure_ascii=False,separators=(',',':')).encode()
                    old_chunks.update(original)
                    docs=entry['relevant_docs'][:depth];allowed={str(d) for d in docs}
                    if any(str(c['doc_id']) not in allowed for c in entry['relevant_chunks']):
                        raise ValueError(f'Cutoff {depth} removes a submitted chunk document in query {qid}')
                    changed+=int(docs!=entry['relevant_docs']);entry['relevant_docs']=docs
                    new_chunks.update(json.dumps([qid,entry['relevant_chunks']],ensure_ascii=False,separators=(',',':')).encode())
                    if seen:output.write(b',')
                    output.write(json.dumps(entry,ensure_ascii=False,separators=(',',':')).encode())
                    seen.add(qid);doc_count+=len(docs);chunk_count+=len(entry['relevant_chunks'])
                output.write(b']')
        if seen!=set(range(1,1201)) or old_chunks.digest()!=new_chunks.digest():raise ValueError('Query/chunk preservation failed')
        if temporary.stat().st_size>=99_000_000:raise ValueError('ZIP exceeds byte limit')
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:raise ValueError('ZIP CRC error')
        temporary.replace(target)
        report['variants'][str(depth)]=dict(path=str(target.resolve()),sha256=file_sha(target),zip_bytes=target.stat().st_size,
            queries=len(seen),queries_with_changed_docs=changed,docs_per_query=doc_count/len(seen),
            chunks_per_query=chunk_count/len(seen),chunk_payload_preserved=True,chunk_payload_sha256=new_chunks.hexdigest())
        (args.output/'doc_depth_variants_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        print('READY',target.name,report['variants'][str(depth)],flush=True)


if __name__=='__main__':main()
