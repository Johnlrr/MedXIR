"""Stream submission ZIPs to measure structure/coverage, never estimate gold metrics.

Memory is bounded by one query and compact numeric summaries; no corpus is loaded.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile


def iter_queries(stream, block=65536):
    decoder = json.JSONDecoder()
    buf = ''
    eof = False
    started = False
    need_separator = False
    while True:
        if not eof:
            more = stream.read(block)
            eof = not more
            buf += more
        buf = buf.lstrip()
        if not started:
            if not buf and not eof:
                continue
            if not buf.startswith('['):
                raise ValueError('Expected a JSON array')
            buf = buf[1:].lstrip()
            started = True
        while buf:
            if buf[0] == ']':
                if (buf[1:] + stream.read()).strip():
                    raise ValueError('Trailing JSON data')
                return
            if need_separator:
                if buf[0] != ',':
                    raise ValueError('Missing query separator')
                buf = buf[1:].lstrip()
                need_separator = False
                if not buf:
                    break
            try:
                item, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                break
            if not isinstance(item, dict):
                raise ValueError('Expected query object')
            yield item
            buf = buf[end:].lstrip()
            need_separator = True
        if eof:
            raise ValueError('Truncated JSON array')


def describe(values):
    values = sorted(values)
    if not values:
        return {}
    return dict(mean=round(statistics.mean(values), 3),
                p10=values[int((len(values)-1)*.1)],
                median=statistics.median(values), p90=values[int((len(values)-1)*.9)], max=values[-1])


def audit(path):
    stats = {k: [] for k in ['docs', 'chunks', 'covered_docs', 'repeat_windows', 'chars_per_query',
                            'chars_per_chunk', 'words_per_chunk', 'outside_doc_list', 'duplicate_entries']}
    bands = [(0,20),(20,50),(50,100),(100,150),(150,200)]
    covered, totals = Counter(), Counter()
    qids, lengths = [], Counter()
    doc_digest, chunk_digest = hashlib.sha256(), hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        members = [i for i in archive.infolist() if not i.is_dir()]
        if len(members) != 1 or not members[0].filename.endswith('.json'):
            raise ValueError(f'{path}: expected one JSON member')
        info = members[0]
        with archive.open(info) as raw, io.TextIOWrapper(raw, encoding='utf-8-sig') as stream:
            for entry in iter_queries(stream):
                qid = int(entry['id'])
                docs = list(map(str, entry['relevant_docs']))
                chunks = entry['relevant_chunks']
                cdocs = {str(c['doc_id']) for c in chunks}
                texts = [c['chunk_text'] for c in chunks]
                qids.append(qid)
                vals = dict(docs=len(docs), chunks=len(chunks), covered_docs=len(cdocs),
                            repeat_windows=len(chunks)-len(cdocs), chars_per_query=sum(map(len,texts)),
                            outside_doc_list=len(cdocs-set(docs)),
                            duplicate_entries=len(chunks)-len({(str(c['doc_id']),c['chunk_text']) for c in chunks}))
                for k,v in vals.items(): stats[k].append(v)
                stats['chars_per_chunk'].extend(map(len,texts))
                stats['words_per_chunk'].extend(len(t.split()) for t in texts)
                for t in texts:
                    lengths['lt500'] += len(t)<500
                    lengths['gt6000'] += len(t)>6000
                for lo,hi in bands:
                    key=f'{lo+1}-{hi}'
                    covered[key] += sum(d in cdocs for d in docs[lo:hi])
                    totals[key] += len(docs[lo:hi])
                doc_digest.update(json.dumps([qid,sorted(set(docs))],separators=(',',':')).encode())
                chunk_digest.update(json.dumps([qid,sorted((str(c['doc_id']),c['chunk_text']) for c in chunks)],
                                               ensure_ascii=False,separators=(',',':')).encode())
    return dict(path=str(path), zip_bytes=path.stat().st_size, json_bytes=info.file_size,
                compression=info.compress_type, queries=len(qids), unique_queries=len(set(qids)),
                qids=qids, stats={k:describe(v) for k,v in stats.items()},
                length_counts=dict(lengths), doc_set_digest=doc_digest.hexdigest(),
                chunk_set_digest=chunk_digest.hexdigest(),
                covered_fraction_by_doc_rank={k:round(covered[k]/totals[k],5) if totals[k] else None for k in totals})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args=p.parse_args()
    out=[]
    for path in args.paths:
        result=audit(path)
        out.append(result)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
        print(path.name, result['queries'], {k:result['stats'][k]['mean'] for k in ['docs','chunks','covered_docs']},
              result['covered_fraction_by_doc_rank'], flush=True)


if __name__ == '__main__':
    main()
