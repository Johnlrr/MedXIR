"""Build source-exact windows for one query, with bounded per-query state."""
import hashlib
import json

from evidence_spans import contiguous_around_anchor, centered_span


def prepare_query(qid, ranked, base_docs, rows, docs_of, tokenizer,
                  doc_depth=400, passage_tokens=768):
    best={}
    for cid in ranked:
        for doc in docs_of(cid):best.setdefault(int(doc),[]).append(cid)
    candidates=list(dict.fromkeys(map(int,base_docs)))
    listed=set(candidates)
    candidates += [d for d in best if d not in listed]
    candidates=candidates[:doc_depth]
    by_pos={(r['sha'],r['order']):cid for cid,r in rows.items()}
    ranks={cid:i for i,cid in enumerate(ranked)}
    seen=set();output=[];stats=dict(seed_only=0,context_extended=0,windows=0)
    for doc in candidates:
        if doc not in best:continue
        sha=rows[best[doc][0]]['sha']
        if sha in seen:continue
        seen.add(sha);seeds=[];spans=set()
        for cid in best[doc]:
            if all(abs(rows[cid]['order']-rows[c]['order'])>1 for c in seeds):seeds.append(cid)
            if len(seeds)==2:break
        for cid in seeds:
            row=rows[cid]
            neighbours=[rows[by_pos[(sha,o)]] for o in range(row['order']-2,row['order']+3)
                        if (sha,o) in by_pos]
            source_start,source=contiguous_around_anchor(neighbours,row)
            offsets=tokenizer(source,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
            a,b,text=centered_span(source,offsets,row['start']-source_start,row['end']-source_start,passage_tokens)
            tokens=len(tokenizer.encode(text,add_special_tokens=False))
            if tokens>passage_tokens:
                a,b,text=centered_span(source,offsets,row['start']-source_start,row['end']-source_start,passage_tokens-8)
                tokens=len(tokenizer.encode(text,add_special_tokens=False))
            assert tokens<=passage_tokens+2 and text==source[a:b]
            start,end=source_start+a,source_start+b
            if (start,end) in spans:continue
            spans.add((start,end))
            stats['windows']+=1
            stats['seed_only']+=int(source_start==row['start'] and source==row['text'])
            stats['context_extended']+=int(len(source)>len(row['text']))
            text_hash=hashlib.sha256(text.encode()).hexdigest()
            pid=hashlib.sha256(json.dumps([int(qid),sha,start,end,text_hash],separators=(',',':')).encode()).hexdigest()
            output.append(dict(passage_id=pid,query_id=int(qid),doc_id=int(doc),
                text_sha256=sha,seed=cid,seed_rank=ranks[cid],doc_ids=list(map(int,docs_of(cid))),
                start=start,end=end,chunk_text=text,passage_tokens=tokens))
    if not output:raise ValueError(f'No passages for query {qid}')
    return output,stats
