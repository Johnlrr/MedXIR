"""Title search: documents whose article title matches the query, scored by the document reranker.

Many Vietnamese health pages are Q&A articles titled with the question itself; chunk search compares the query
with body text only. Here every text's title (chunk corpus column `title`) is embedded with Qwen3-Embedding-8B
and searched with the query and its title-style rewrite (translate_queries.py task 'vq', optional). Texts in the
TITLE_K best of either list (RRF) whose documents are not yet among the candidates of doc_rank.parquet (the
bge-scored document candidates of the current best pipeline) are scored like those candidates:
bge-reranker-v2-m3 on (query, first LEAD_CHARS characters of the text + one chunk), the best of its first
CHUNKS_PER_TEXT chunks. A new document whose score beats the 200th best existing score would enter a bge-only
top-200 list; the manifest counts them.

RTX Pro 6000, Internet OFF. Inputs under INPUT_ROOT: chunk corpus v2 (chunks/*/part-*.parquet, hash_groups.parquet),
query.parquet, doc_rank.parquet, packaged Qwen3-Embedding-8B and bge-reranker-v2-m3, query_rewrites.json (optional).
Outputs in OUT_DIR: title_candidates.parquet, title_manifest.json.
"""
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TASK = 'Given a Vietnamese patient question, retrieve health articles whose title asks or answers the same question'
TITLE_K = 150           # titles per query text (query, rewrite)
NEW_MAX = 100           # new texts scored per query
CHUNKS_PER_TEXT = 6     # chunks tried as the shown chunk of a new text
LEAD_CHARS = 400
CUT = 200               # document list length of the pipeline
RRF_K = 60
INPUT = os.environ.get('INPUT_ROOT', '/kaggle/input')
OUT = Path(os.environ.get('OUT_DIR', '/kaggle/working'))


def note(message, started=[time.perf_counter()]):
    print(f'{time.perf_counter() - started[0]:8.1f}s  {message}', flush=True)


def one_match(pattern):
    hits = sorted(glob.glob(f'{INPUT}/**/{pattern}', recursive=True))
    assert len(hits) == 1, f'{pattern}: expected one match, found {hits[:10]}'
    return Path(hits[0])


def rrf_order(lists, k=RRF_K):
    """Items of several ranked lists, best summed 1 / (k + rank) first."""
    score = {}
    for items in lists:
        for rank, item in enumerate(items):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda item: -score[item])


def shown_text(lead, chunk_order, chunk_text, lead_chars=LEAD_CHARS):
    """Text the document reranker sees (as expand_chunks.doc_text): the lead, then the chunk unless it is the lead."""
    return chunk_text if chunk_order == 0 else lead[:lead_chars] + '\n...\n' + chunk_text


def cut_score(scores, cut=CUT):
    """The cut-th best score (the score a new document must beat to enter a top-cut list); -inf if fewer."""
    ordered = sorted(scores, reverse=True)
    return ordered[cut - 1] if len(ordered) >= cut else float('-inf')


def load_titles(parts):
    """One row per text: sha, representative doc, language, title (order-0 chunk rows, non-empty titles)."""
    frames = []
    for path in parts:
        table = pq.read_table(path, columns=['text_sha256', 'rep_doc_id', 'order', 'title'])
        table = table.filter(pc.equal(table['order'], 0))
        frame = table.to_pandas()
        frame['lang'] = Path(path).parent.name
        frames.append(frame[['text_sha256', 'rep_doc_id', 'lang', 'title']])
    titles = pd.concat(frames, ignore_index=True)
    titles['title'] = titles['title'].fillna('').astype(str).str.strip()
    return titles[titles['title'].str.len() >= 5].drop_duplicates('text_sha256').reset_index(drop=True)


def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    os.environ['HF_HUB_OFFLINE'] = '1'
    parts = sorted(glob.glob(f'{INPUT}/**/chunks/*/part-*.parquet', recursive=True))
    queries = pd.read_parquet(one_match('query.parquet'), columns=['id', 'query'])
    qids, qtext = [int(q) for q in queries['id']], queries['query'].astype(str).tolist()
    rewrites = {}
    hits = sorted(glob.glob(f'{INPUT}/**/query_rewrites.json', recursive=True))
    if hits:
        rewrites = {int(x['id']): x['vq'] for x in json.loads(Path(hits[0]).read_text(encoding='utf-8'))['items'] if x.get('vq')}
    note(f'{len(qids)} queries, {len(rewrites)} rewrites')

    titles = load_titles(parts)
    note(f'{len(titles):,} titles; by language {titles["lang"].value_counts().to_dict()}')

    # Embed titles (documents: no instruction) and queries (instruction), Qwen3-Embedding-8B last-token pooling.
    path = one_match('Qwen3-Embedding-8B/config.json').parent
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, padding_side='left')
    model = AutoModel.from_pretrained(str(path), torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                      local_files_only=True).cuda().eval()

    @torch.inference_mode()
    def encode(texts, batch, max_length):
        ids = tok(texts, truncation=True, max_length=max_length)['input_ids']
        order = np.argsort([-len(row) for row in ids], kind='stable')
        out = torch.empty((len(texts), model.config.hidden_size), dtype=torch.float16, device='cuda')
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            enc = tok.pad({'input_ids': [ids[i] for i in index]}, return_tensors='pt').to('cuda')
            hidden = model(**enc, use_cache=False).last_hidden_state
            out[torch.from_numpy(index).cuda()] = F.normalize(hidden[:, -1].float(), dim=1).half()   # left padding
        return out

    title_vec = encode(titles['title'].tolist(), 256, 64)
    note('titles encoded')
    variants = [(qi, f'Instruct: {TASK}\nQuery:{t}') for qi, t in enumerate(qtext)]
    variants += [(qi, f'Instruct: {TASK}\nQuery:{rewrites[q]}') for qi, q in enumerate(qids) if q in rewrites]
    query_vec = encode([t for _, t in variants], 32, 512)
    del model
    torch.cuda.empty_cache()
    with torch.inference_mode():
        top = torch.cat([torch.topk(query_vec[s:s + 256] @ title_vec.T, TITLE_K, dim=1).indices
                         for s in range(0, len(variants), 256)]).cpu().numpy()
    lists = [[] for _ in qids]
    for (qi, _), row in zip(variants, top):
        lists[qi].append(titles['text_sha256'].values[row].tolist())
    ranked = [rrf_order(x) for x in lists]
    note('title search done')

    groups = pd.read_parquet(one_match('hash_groups.parquet'), columns=['text_sha256', 'doc_ids', 'fanout_safe'])
    fan = {s: [int(d) for d in ids] for s, ids, ok in zip(groups['text_sha256'], groups['doc_ids'], groups['fanout_safe']) if bool(ok)}
    text_of_doc = {int(d): s for s, ids in zip(groups['text_sha256'], groups['doc_ids']) for d in ids}
    del groups
    rep = dict(zip(titles['text_sha256'], titles['rep_doc_id'].astype(int)))
    text_of_doc.update({d: s for s, d in rep.items() if d not in text_of_doc})

    # Title match of every candidate text (pool chunks, doc_rank documents, title-search texts) for reranking:
    # cosine of the query (cos_q) and of its rewrite (cos_r, NaN without one) with the text's title.
    row_of = {s: i for i, s in enumerate(titles['text_sha256'])}
    by_prefix = {s[:16]: s for s in titles['text_sha256']}
    cands = {q: set() for q in qids}
    for q, g in pd.read_parquet(one_match('doc_rank.parquet'), columns=['query_id', 'doc_id']).groupby('query_id'):
        cands[int(q)].update(text_of_doc[int(d)] for d in g['doc_id'] if int(d) in text_of_doc)
    for path in glob.glob(f'{INPUT}/**/pool*.parquet', recursive=True):
        for q, g in pd.read_parquet(path, columns=['query_id', 'chunk_id']).groupby('query_id'):
            cands[int(q)].update(by_prefix[c.rsplit('-', 1)[0]] for c in g['chunk_id'] if c.rsplit('-', 1)[0] in by_prefix)
    for qi, q in enumerate(qids):
        cands[q].update(ranked[qi][:TITLE_K])
    rewrite_row = {qi: n for n, (qi, _) in enumerate(variants) if n >= len(qtext)}
    score_rows = []
    with torch.inference_mode():
        for qi, q in enumerate(qids):
            shas = sorted(s for s in cands[q] if s in row_of)
            index = torch.as_tensor([row_of[s] for s in shas], device='cuda')
            sub = title_vec.index_select(0, index)
            cos_q = (sub @ query_vec[qi]).float().cpu().numpy()
            cos_r = ((sub @ query_vec[rewrite_row[qi]]).float().cpu().numpy() if qi in rewrite_row
                     else np.full(len(shas), np.nan, dtype=np.float32))
            score_rows.append(pd.DataFrame(dict(query_id=q, text_sha256=shas, cos_q=cos_q, cos_r=cos_r)))
    title_scores = pd.concat(score_rows, ignore_index=True)
    title_scores.to_parquet(OUT / 'title_scores.parquet', index=False)
    note(f'title scores: {len(title_scores):,} (query, text) rows')
    del title_vec, query_vec
    torch.cuda.empty_cache()
    title_of = dict(zip(titles['text_sha256'], titles['title']))
    lang_of = dict(zip(titles['text_sha256'], titles['lang']))

    def docs_of(sha):
        return fan.get(sha) or [rep[sha]]

    doc_rank = pd.read_parquet(one_match('doc_rank.parquet'))
    column = 'bge-reranker-v2-m3'
    existing = {int(q): set(map(int, g['doc_id'])) for q, g in doc_rank.groupby('query_id')}
    threshold = {int(q): cut_score(g[column].dropna().tolist()) for q, g in doc_rank.groupby('query_id')}
    new = []   # (qi, sha, title rank)
    for qi, (qid, shas) in enumerate(zip(qids, ranked)):
        taken = 0
        for rank, sha in enumerate(shas):
            if taken == NEW_MAX:
                break
            if not any(d in existing.get(qid, ()) for d in docs_of(sha)):
                new.append((qi, sha, rank))
                taken += 1
    need = sorted({sha for _, sha, _ in new})
    note(f'{len(new):,} new (query, text) pairs over {len(need):,} texts')

    # Chunks of the new texts (first CHUNKS_PER_TEXT), read with a sha-prefix filter.
    prefixes = pa.array(sorted({s[:16] for s in need}))
    chunks = {}
    for path in parts:
        table = pq.read_table(path, columns=['chunk_id', 'text_sha256', 'order', 'chunk_text'])
        table = table.filter(pc.and_(pc.is_in(pc.utf8_slice_codeunits(table['text_sha256'], 0, 16), value_set=prefixes),
                                     pc.less(table['order'], CHUNKS_PER_TEXT)))
        for cid, sha, order, text in zip(*(table[c].to_pylist() for c in ('chunk_id', 'text_sha256', 'order', 'chunk_text'))):
            chunks.setdefault(sha, []).append((int(order), cid, text))
    for sha in chunks:
        chunks[sha].sort()
    note(f'chunks read for {len(chunks):,} texts')

    pairs, owner = [], []   # owner: (index in new, chunk id)
    for n, (qi, sha, _) in enumerate(new):
        rows = chunks.get(sha, [])
        if not rows or rows[0][0] != 0:
            continue
        lead = rows[0][2]
        for order, cid, text in rows:
            pairs.append((qtext[qi], shown_text(lead, order, text)))
            owner.append((n, cid))
    path = one_match('bge-reranker-v2-m3/config.json').parent
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(path), torch_dtype=torch.bfloat16,
                                                               local_files_only=True).cuda().eval()
    values = np.empty(len(pairs), dtype=np.float32)
    order = np.argsort([len(d) for _, d in pairs], kind='stable')
    with torch.inference_mode():
        for start in range(0, len(order), 64):
            index = order[start:start + 64]
            enc = tok([pairs[i][0] for i in index], [pairs[i][1] for i in index], padding=True, truncation=True,
                      max_length=512, return_tensors='pt').to('cuda')
            values[index] = torch.sigmoid(model(**enc).logits.view(-1).float()).cpu().numpy()
            if start // 64 % 2000 == 0:
                note(f'bge {start + len(index):,}/{len(order):,}')
    note('bge done')
    best = {}
    for (n, cid), v in zip(owner, values):
        if n not in best or v > best[n][0]:
            best[n] = (float(v), cid)

    rows = []
    for n, (qi, sha, rank) in enumerate(new):
        score, cid = best.get(n, (float('nan'), None))
        qid = qids[qi]
        rows.append(dict(query_id=qid, text_sha256=sha, doc_ids=docs_of(sha), title=title_of[sha], lang=lang_of[sha],
                         title_rank=rank, bge=score, best_chunk_id=cid, threshold=threshold.get(qid, float('-inf'))))
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / 'title_candidates.parquet', index=False)
    above = frame[frame['bge'] > frame['threshold']]
    per_query = above.groupby('query_id').size().reindex(qids, fill_value=0)
    manifest = dict(titles=len(titles), rewrites=len(rewrites), new_pairs=len(new), bge_pairs=len(pairs),
                    new_above_cut_per_query=dict(mean=round(float(per_query.mean()), 2),
                                                 p50=float(per_query.median()), p90=float(per_query.quantile(0.9))),
                    above_cut_by_lang=above['lang'].value_counts().to_dict(),
                    above_cut_title_rank_p50=float(above['title_rank'].median()) if len(above) else None)
    (OUT / 'title_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    note(json.dumps(manifest, ensure_ascii=False))
    for _, r in above.sort_values('bge', ascending=False).head(5).iterrows():
        print(r['query_id'], round(r['bge'], 3), r['title'][:100], flush=True)
