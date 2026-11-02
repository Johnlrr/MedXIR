"""Rebuild the chunk lists of the hybrid submission with neighbouring chunks merged into longer spans.

A reference passage counts as found when a submitted text of the same document covers >= 40% of it (token
LCS). Chunk recall (0.18) is far below document recall (0.47): the right documents are found, the passages
are not. Here the top-ranked chunks bring their neighbours (order - 1, order + 1) along, and contiguous
chunks of one text are merged into one span (the 50-word overlaps are dropped once). The byte budget per
query stays that of the hybrid submission, so lower-ranked chunks make room for the neighbours.

The document lists are kept as submitted; documents of new spans are appended (as in assemble()).
Optional per variant: n_docs extends the document list with the next pool documents (best chunk first);
quota reserves that share of the chunk bytes for ZH/EN texts; doc_quota swaps the lowest VI documents of the
list for the best ZH/EN pool documents (both lost on the leaderboard, 29/10); doc_rr re-orders the
candidate documents by weighted RRF of their submitted rank and document-reranker scores (doc_rank.parquet,
written by the doc-rerank notebook, kaggle_doc_rerank.ipynb), so the reranker decides which make the cut.
conf_gamma moves chunk bytes between queries, total unchanged: a query's confidence is its number of pool
chunks with rerank score >= conf_t; confident queries lose budget (their lowest-ranked chunks go) and the
budget goes to queries with few strong chunks, which reach deeper into their pool.
Runs on a Kaggle CPU session (Internet OFF is fine). Inputs:
  medxir-chunk-corpus      chunks/*/part-*.parquet, hash_groups.parquet
  medxir-hybrid-retrieval  pool.parquet, medxir-hybrid-submission.zip   (notebook output or uploaded dataset)
  doc_rank.parquet         only for doc_rr variants (in OUT_DIR or an input)
Output: one ZIP per variant in /kaggle/working (or OUT_DIR), plus expand_manifest.json.
"""
import glob
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# nb: number of top chunks whose neighbours are added (0 = only merge, no new neighbours);
#     'whole' = probe: full texts of the top documents instead of ranked chunks.
# n_docs: document list length (0 = as submitted). quota / doc_quota: min ZH/EN share of chunk bytes / documents.
# doc_rr: {score column of doc_rank.parquet: RRF weight}; w_base: RRF weight of the submitted document rank;
# keep_top: that many top candidates kept as submitted, the rest of the list re-ordered.
# conf_gamma: budget weight (confidence / median confidence) ** -gamma, clipped to CONF_CLIP, renormalised.
# min_chars: spans shorter than this are not submitted. A span matches when it covers 40% of a reference
# passage, so a 5-character ZH entry ('食管炎') can only match an equally short passage; such spans were 11% of
# the entries of the 0.2464 ZIP (< 200 chars) for 0.35% of its text.
# min_score: ranked chunks with a rerank score below it are not taken (nor their neighbours); the bytes a query
# leaves unused go to the queries that used their whole budget, by the same confidence weights. bge scores are
# high: at 0.2 the cut removes ~46 chunks in 16% of the queries (the unsure ones) and none elsewhere.
# zip: 'bzip2' writes the ZIP with bzip2 instead of deflate: the 0.2477 JSON is 86.5 MB deflated, 56.5 MB with
# bzip2 (LZMA 54.4 MB, 10x slower), so ~1.5x more text fits under the 100 MB limit. budget_scale replaces
# BUDGET_SCALE; the size loop shrinks it until the ZIP fits. Under the limit chunk recall grows with the bytes
# (each gold document holds ~1 reference passage, so more bytes = more documents reached).
# cover (first, last): after the nb top chunks (with neighbours), the best pool chunk of every listed document
# at ranks first..last that has none yet, then the rest of the pool. 30/10 (Qwen-8B, 0.2509): ranks 100-200 of
# the document list got chunk text for 10% of the documents, ranks 50-100 for 48%, while ~15% of them are
# relevant and a relevant document holds ~1 reference passage.
# win: every entry is a window of the chunk +- win neighbours of the same text (the entry shape of the 0.2755
# ensemble ZIP: no tiny spans, no 5+ chunk merges); a centre inside a chosen window is skipped. Centres in the
# order of w_bge * bge + (1 - w_bge) * rrf, both min-max per query (0.2755 used bge 0.25 over its fused list).
# cover_top: first the best window of each of the top cover_top listed documents, then one window per text in
# score order, then second windows. A relevant document holds ~1 reference passage, so a second window of one
# text rarely adds recall, while ~23% of the top-20 documents of the 0.2755 ZIP had no window.
# doc_conf (t, lo, hi): per query, the documents with doc-reranker score >= t, at least lo and at most hi. A
# document pays off in F2 when P(relevant) > ~F2 / 5 (~0.065 at Doc F2 0.32); bge scores scaled to the 0.161 Doc P
# of the 0.2539 ZIP (mean bge of its top 200 = 0.514) put that at t = 0.208 (~189 documents on average).
# bm25w (needs bm25_words.py outputs attached): doc_rr column 'bm25w' = the word-BM25 rank of the best
# BM25W_TEXTS texts (their documents), fused by RRF with bge; w_bm25w adds the word-BM25 chunk score (best
# BM25W_CHUNKS chunks, also those outside the pool) to the window order. 30/10: 48% of the word-BM25 top 40
# documents are in none of our 206-document lists.
VARIANTS = {
    'nb0': dict(nb=0), 'nb10': dict(nb=10), 'nb30': dict(nb=30), 'nb60': dict(nb=60), 'wholedoc': dict(nb='whole'),
    'nb60-d300': dict(nb=60, n_docs=300),
    'nb60-q20': dict(nb=60, quota=0.2),
    'nb60-q20-dq20': dict(nb=60, quota=0.2, doc_quota=0.2),
    'nb60-drr-bge': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=1.0),
    'nb60-drr-bge-only': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0),
    # Qwen3-Reranker-4B scores only the zone around the cut (ranks 120-360): the top 120 stay as submitted.
    'nb60-drr-qwen4b': dict(nb=60, doc_rr={'Qwen3-Reranker-4B': 1.0}, w_base=1.0, keep_top=120),
    'nb60-drr-qwen4b-only': dict(nb=60, doc_rr={'Qwen3-Reranker-4B': 1.0}, w_base=0.0, keep_top=120),
    'nb60-drr-both': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0, 'Qwen3-Reranker-4B': 1.0}, w_base=1.0,
                          keep_top=120),
    # Bge-only document list, chunk bytes moved between queries. 30/10 on v2 (bge-only 0.2464): glow (to unsure
    # queries) 0.2352, ghigh (to confident queries) 0.2477; macro F2 rewards the queries that already score.
    'nb60-drr-bge-only-glow': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0, conf_gamma=0.5, conf_t=0.5),
    'nb60-drr-bge-only-ghigh': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0, conf_gamma=-0.5, conf_t=0.5),
    'nb60-drr-bge-only-ghigh1': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0, conf_gamma=-1.0,
                                     conf_t=0.5),
    # Precision: no tiny spans; 'nball' gives every ranked chunk its neighbours (no single chunks filling the
    # budget), the entry shape of the x1-c70 submission (Chunk P 0.208 vs 0.155).
    'nb60-drr-bge-only-ghigh1-min200': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0, conf_gamma=-1.0,
                                            conf_t=0.5, min_chars=200),
    'nb60-drr-bge-only-ghigh-min200-s20': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0,
                                               conf_gamma=-0.5, conf_t=0.5, min_chars=200, min_score=0.2),
    'nb60-drr-bge-only-ghigh1-min200-s20': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0,
                                                conf_gamma=-1.0, conf_t=0.5, min_chars=200, min_score=0.2),
    'nb60-drr-bge-only-ghigh-bz': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0, conf_gamma=-0.5,
                                       conf_t=0.5, zip='bzip2', budget_scale=1.6),
    'nb60-drr-bge-only-ghigh-cover': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0, conf_gamma=-0.5,
                                          conf_t=0.5, cover=(20, 200)),
    'nb30-drr-bge-only-ghigh-cover': dict(nb=30, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0, conf_gamma=-0.5,
                                          conf_t=0.5, cover=(20, 200)),
    'nb60-drr-bge-only-ghigh-min200': dict(nb=60, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0, conf_gamma=-0.5,
                                           conf_t=0.5, min_chars=200),
    'nball-drr-bge-only-ghigh-min200': dict(nb=10 ** 6, doc_rr={'bge-reranker-v2-m3': 1.0}, w_base=0.0,
                                            conf_gamma=-0.5, conf_t=0.5, min_chars=200),
    'win1-cov50-drr-bge-only': dict(nb=0, win=1, w_bge=0.25, cover_top=50, doc_rr={'bge-reranker-v2-m3': 1.0},
                                    w_base=0.0, budget_scale=1.15),
    'win1-cov50-drr-bge-conf': dict(nb=0, win=1, w_bge=0.25, cover_top=50, doc_rr={'bge-reranker-v2-m3': 1.0},
                                    w_base=0.0, budget_scale=1.15, doc_conf=(0.208, 80, 300)),
    # Ensemble pool of the 0.2916 ZIP (POOL_FILE=pool-bgefull.parquet: final_score = bge on all 500 pool chunks,
    # rrf = ensemble score), windows in its order (ensemble 0.75, bge 0.25), BZIP2 with the budget loop filling
    # ~99 MB. cov0: one window per text before second windows; cov50: the top 50 listed documents first.
    'ensbge-win1-cov50-bz': dict(nb=0, win=1, w_bge=0.25, cover_top=50, doc_rr={'bge-reranker-v2-m3': 1.0},
                                 w_base=0.0, zip='bzip2', budget_scale=2.0),
    'ensbge-win1-cov0-bz': dict(nb=0, win=1, w_bge=0.25, cover_top=0, doc_rr={'bge-reranker-v2-m3': 1.0},
                                w_base=0.0, zip='bzip2', budget_scale=2.0),
    'win1-cov50-bm25w-drr': dict(nb=0, win=1, w_bge=0.25, w_bm25w=0.25, cover_top=50,
                                 doc_rr={'bge-reranker-v2-m3': 1.0, 'bm25w': 1.0}, w_base=0.0, budget_scale=1.15),
}
ONLY = ['nb60-d300']   # variants to build this run; [] = all
BUDGET_SCALE = 1.08   # 87.5 MB hybrid ZIP * 1.08 stays below the 100 MB limit
MAX_ZIP_MB = 99.0
CONF_CLIP = (0.5, 2.0)
BM25W_CHUNKS, BM25W_TEXTS = 200, 100   # word-BM25 candidates read per query
RRF_K = 60
INPUT = os.environ.get('INPUT_ROOT', '/kaggle/input')
OUT = Path(os.environ.get('OUT_DIR', '/kaggle/working'))


def one_match(pattern):
    hits = sorted(p for p in glob.glob(f'{INPUT}/**/{pattern}', recursive=True))
    if len(hits) != 1:
        for root in sorted(glob.glob(f'{INPUT}/*/*/*')) or sorted(glob.glob(f'{INPUT}/*')):
            print('input:', root, sorted(os.listdir(root))[:8] if os.path.isdir(root) else '')
    assert len(hits) == 1, f'{pattern}: expected one match, found {hits[:10]}'
    return Path(hits[0])


def load_submission():
    # An uploaded dataset is unzipped by Kaggle: accept the JSON too.
    hits = sorted(glob.glob(f'{INPUT}/**/medxir-hybrid-submission.json', recursive=True))
    if hits:
        return json.loads(Path(hits[0]).read_text(encoding='utf-8'))
    with zipfile.ZipFile(one_match('medxir-hybrid-submission.zip')) as archive:
        return json.loads(archive.read(archive.namelist()[0]))


def merge_run(rows):
    """One text out of chunks of one document text with consecutive orders (rows sorted by order)."""
    text, end = rows[0]['text'], rows[0]['end']
    for row in rows[1:]:
        text += row['text'][max(0, end - row['start']):]
        end = row['end']
    return text


def build_query(ranked, n_neighbour, budget, rows, by_pos, docs_of, quota=0.0, scores=None, min_score=0.0):
    """Selected chunk ids for one query: ranked chunks (+ neighbours of the top ones) until the budget is used.

    quota > 0: a first pass caps VI chunks at (1 - quota) of the budget so ZH/EN chunks further down the pool
    get the rest; a second pass fills what ZH/EN could not use.
    """
    chosen, used, vi_used = [], 0, 0
    for vi_cap in ([budget * (1 - quota)] if quota else []) + [budget]:
        for rank, cid in enumerate(ranked):
            if min_score and scores[rank] < min_score:
                break   # pool is in score order
            group = [cid]
            if rank < n_neighbour:
                sha, order = rows[cid]['sha'], rows[cid]['order']
                group += [by_pos[(sha, o)] for o in (order - 1, order + 1) if (sha, o) in by_pos]
            for c in group:
                if c in chosen:
                    continue
                cost = len(rows[c]['text'].encode('utf-8')) * len(docs_of(c))
                vi = rows[c].get('lang', 'vi') == 'vi'
                if used + cost > budget or (vi and vi_used + cost > vi_cap):
                    continue  # a long or wide-fanout chunk does not fit; smaller ones further down may
                chosen.append(c)
                used += cost
                vi_used += cost if vi else 0
    return chosen


def doc_list(docs, ranked, docs_of, lang_of, n_docs=0, doc_quota=0.0):
    """Submitted documents with the ZH/EN quota applied, then extended to n_docs with pool documents."""
    docs = list(docs)
    pool_docs = list(dict.fromkeys(d for cid in ranked for d in docs_of(cid)))
    if doc_quota:
        listed = set(docs)
        extra = [d for d in pool_docs if d not in listed and lang_of.get(d, 'vi') != 'vi']
        need = math.ceil(doc_quota * len(docs)) - sum(lang_of.get(d, 'vi') != 'vi' for d in docs)
        vi_slots = [i for i, d in enumerate(docs) if lang_of.get(d, 'vi') == 'vi']
        slots = vi_slots[len(vi_slots) - min(max(need, 0), len(extra), len(vi_slots)):]
        for i, doc in zip(slots, extra):   # lowest VI slots, best ZH/EN document first
            docs[i] = doc
    if n_docs > len(docs):
        listed = set(docs)
        docs += [d for d in pool_docs if d not in listed][:n_docs - len(docs)]
    return docs


def best_chunks(ranked, docs_of):
    """{doc: its pool chunks, best first}. Documents filled from beyond the reranked pool have none."""
    best = {}
    for cid in ranked:
        for doc in docs_of(cid):
            best.setdefault(doc, []).append(cid)
    return best


def doc_text(best, rows, by_text, lead_chars=400, n_chunks=1):
    """Text a document reranker sees: the start of the document text (title / lead) + its best pool chunks.

    best: the document's pool chunks, best first; the n_chunks best are shown in text order.
    """
    shown = sorted(best[:n_chunks], key=lambda c: rows[c]['order'])
    parts = [rows[c]['text'] for c in shown]
    lead = rows[by_text[rows[shown[0]]['sha']][0]]
    if lead['order'] != rows[shown[0]]['order']:
        parts.insert(0, lead['text'][:lead_chars])
    return '\n...\n'.join(parts)


def fuse_docs(candidates, scores, weights, w_base=1.0, k=RRF_K, offset=0):
    """Candidates (submitted order) re-ordered by weighted RRF of their position and reranker scores.

    scores: {column: {doc: score}}; a document without a score in a column gets no term from it.
    offset: submitted rank of the first candidate (when the documents above it are kept fixed).
    """
    fused = {doc: w_base / (k + offset + rank) for rank, doc in enumerate(candidates)}
    for column, weight in weights.items():
        scored = sorted((d for d in candidates if d in scores[column]), key=lambda d: -scores[column][d])
        for rank, doc in enumerate(scored):
            fused[doc] += weight / (k + rank)
    return sorted(candidates, key=lambda d: -fused[d])


def spans(chosen, rows):
    """Merge chosen chunks of the same text with consecutive orders. Returns [(first chunk id, text)] in rank order."""
    by_sha = {}
    for cid in chosen:
        by_sha.setdefault(rows[cid]['sha'], []).append(cid)
    first_rank = {cid: i for i, cid in enumerate(chosen)}
    out = []
    for cids in by_sha.values():
        cids = sorted(cids, key=lambda c: rows[c]['order'])
        run = [cids[0]]
        for c in cids[1:]:
            if rows[c]['order'] == rows[run[-1]]['order'] + 1:
                run.append(c)
            else:
                out.append(run)
                run = [c]
        out.append(run)
    out.sort(key=lambda run: min(first_rank[c] for c in run))
    return [(run[0], merge_run([rows[c] for c in run])) for run in out]


def build_whole_docs(ranked, docs, budget, rows, by_text, docs_of):
    """Probe: every chunk of the submitted documents, best document first, until the budget is used.

    Tells whether chunk recall is lost inside the right documents (recall jumps) or by missing documents.
    """
    text_of = {}
    for cid in ranked:
        for doc in docs_of(cid):
            text_of.setdefault(doc, rows[cid]['sha'])
    chosen, used = [], 0
    for doc in docs:
        if doc not in text_of:
            continue
        for c in by_text[text_of[doc]]:
            cost = len(rows[c]['text'].encode('utf-8')) * len(docs_of(c))
            if used + cost > budget:
                return chosen
            if c not in chosen:
                chosen.append(c)
                used += cost
    return chosen


def load_inputs(pool_path=None, base=None, chunk_files=None):
    """Pool ranking, base submission, every chunk of the candidate texts and the document groups.

    The hybrid notebook passes its own pool.parquet, in-memory submission and validated chunk parts;
    otherwise they are found under INPUT.
    """
    pool_path = pool_path or one_match(os.environ.get('POOL_FILE', 'pool.parquet'))
    columns = ['query_id', 'chunk_id', 'final_rank', 'final_score']
    pool = pd.read_parquet(pool_path, columns=columns + (['rrf'] if 'rrf' in pq.read_schema(pool_path).names else []))
    pool = pool.sort_values(['query_id', 'final_rank'])
    ranked = {qid: g['chunk_id'].tolist() for qid, g in pool.groupby('query_id', sort=False)}
    score = {qid: g['final_score'].tolist() for qid, g in pool.groupby('query_id', sort=False)}
    rrf = {qid: g['rrf'].tolist() for qid, g in pool.groupby('query_id', sort=False)} if 'rrf' in pool else None
    base = load_submission() if base is None else base
    needed_sha_prefix = {cid.rsplit('-', 1)[0] for ids in ranked.values() for cid in ids}
    bm25w = load_bm25w()
    if bm25w:
        needed_sha_prefix |= {cid.rsplit('-', 1)[0] for ids, _ in bm25w['chunks'].values() for cid in ids}
        needed_sha_prefix |= {sha[:16] for shas, _ in bm25w['texts'].values() for sha in shas}

    # Every chunk of the candidate texts: the neighbours are rows of the same text_sha256.
    rows, by_pos = {}, {}
    chunk_files = chunk_files or sorted(glob.glob(f'{INPUT}/**/chunks/*/part-*.parquet', recursive=True))
    assert chunk_files, 'no chunk parts: attach medxir-chunk-corpus'
    columns = ['chunk_id', 'text_sha256', 'rep_doc_id', 'order', 'start', 'end', 'chunk_text']
    for path in chunk_files:
        table = pq.read_table(path, columns=columns)
        prefix = pc.utf8_slice_codeunits(table['text_sha256'], 0, 16)
        table = table.filter(pc.is_in(prefix, value_set=pa.array(sorted(needed_sha_prefix))))
        lang = Path(path).parent.name   # chunks/<lang>/part-*.parquet
        for cid, sha, doc, order, start, end, text in zip(*(table[c].to_pylist() for c in columns)):
            rows[cid] = dict(sha=sha, doc=int(doc), order=int(order), start=int(start), end=int(end), text=text,
                             lang=lang)
            by_pos[(sha, int(order))] = cid
        print(Path(path).parent.name, Path(path).name, f'{len(rows):,} rows kept', flush=True)
    missing = {cid for ids in ranked.values() for cid in ids} - set(rows)
    assert not missing, f'{len(missing)} pool chunks not in the chunk corpus (chunk id != sha prefix?), e.g. {sorted(missing)[:3]}'

    groups = pd.read_parquet(one_match('hash_groups.parquet'), columns=['text_sha256', 'doc_ids', 'fanout_safe'])
    group = {sha: [int(d) for d in ids] for sha, ids, safe in
             zip(groups['text_sha256'], groups['doc_ids'], groups['fanout_safe']) if bool(safe)}
    del groups

    def docs_of(cid):
        return group.get(rows[cid]['sha']) or [rows[cid]['doc']]

    by_text = {}
    for cid in sorted(rows, key=lambda c: rows[c]['order']):
        by_text.setdefault(rows[cid]['sha'], []).append(cid)

    def text_docs(sha):
        return group.get(sha) or ([rows[by_text[sha][0]]['doc']] if sha in by_text else [])

    if bm25w:   # word-BM25 chunks whose rows were not found (other corpus parts) are dropped
        for q, (ids, scores) in bm25w['chunks'].items():
            keep = [(c, s) for c, s in zip(ids, scores) if c in rows]
            bm25w['chunks'][q] = ([c for c, _ in keep], [s for _, s in keep])
    return dict(ranked=ranked, score=score, rrf=rrf, bm25w=bm25w, text_docs=text_docs,
                base=base, rows=rows, by_pos=by_pos, by_text=by_text, docs_of=docs_of,
                lang_of={doc: row['lang'] for cid, row in rows.items() for doc in docs_of(cid)})


def load_bm25w(n_chunks=None, n_texts=None):
    """Word-BM25 candidates of bm25_words.py, {'chunks': {qid: (ids, scores)}, 'texts': {qid: (shas, scores)}}, or
    None when bm25w-chunks-k500.npz is not attached. Only positive scores, the BM25W_CHUNKS / BM25W_TEXTS best."""
    if not glob.glob(f'{INPUT}/**/bm25w-chunks-k500.npz', recursive=True):
        return None
    out = {}
    for key, name, column, n in (('chunks', 'bm25w-chunks-k500.npz', 'chunk_ids', n_chunks or BM25W_CHUNKS),
                                 ('texts', 'bm25w-texts-k500.npz', 'text_sha256', n_texts or BM25W_TEXTS)):
        with np.load(one_match(name), allow_pickle=True) as z:
            out[key] = {int(q): ([str(x) for x, s in zip(ids[:n], sc[:n]) if s > 0], [float(s) for s in sc[:n] if s > 0])
                        for q, ids, sc in zip(z['query_ids'], z[column], z['scores'])}
    return out


def load_doc_scores():
    """{query_id: {column: {doc: score}}} from doc_rank.parquet (OUT_DIR first, then the inputs)."""
    path = OUT / 'doc_rank.parquet'
    frame = pd.read_parquet(path if path.exists() else one_match('doc_rank.parquet'))
    columns = [c for c in frame.columns if c not in ('query_id', 'doc_id', 'base_rank')]
    out = {}
    for qid, g in frame.groupby('query_id'):
        out[int(qid)] = {c: {int(d): float(s) for d, s in zip(g['doc_id'], g[c]) if s == s} for c in columns}
    return out


def confidence_budget(budget, score, gamma, threshold):
    """Per-query budgets re-weighted by (confidence / median) ** -gamma, the total kept."""
    conf = {qid: sum(s >= threshold for s in score.get(qid, [])) + 1 for qid in budget}
    median = sorted(conf.values())[len(conf) // 2]
    weight = {qid: min(max((c / median) ** -gamma, CONF_CLIP[0]), CONF_CLIP[1]) for qid, c in conf.items()}
    norm = sum(budget.values()) / sum(budget[q] * weight[q] for q in budget)
    return {qid: budget[qid] * weight[qid] * norm for qid in budget}


def cover_order(ranked, n_head, docs, first, last, docs_of):
    """Pool order with the best chunk of each listed document at ranks first..last moved right after the top n_head
    chunks, unless a top chunk already belongs to it."""
    head = ranked[:n_head]
    seen = {d for c in head for d in docs_of(c)}
    best = {}
    for cid in ranked[n_head:]:
        for doc in docs_of(cid):
            best.setdefault(doc, cid)
    promoted = list(dict.fromkeys(best[d] for d in docs[first:last] if d in best and d not in seen))
    moved = set(promoted)
    return head + promoted + [c for c in ranked[n_head:] if c not in moved]


def blended_order(ranked, columns, weights):
    """Chunks sorted by sum(weight * column), each column min-max scaled within the query."""
    def scaled(values):
        lo, hi = min(values), max(values)
        return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in values]
    blend = [sum(w * v for w, v in zip(weights, row)) for row in zip(*(scaled(c) for c in columns))]
    return [ranked[i] for i in sorted(range(len(ranked)), key=lambda i: -blend[i])]


def with_bm25w(ranked, bge, rrf, bm25w):
    """Pool chunks + the word-BM25 chunks outside the pool, with (bge, rrf, bm25w) columns; absent scores are the
    column minimum (0 for bge and BM25, the pool minimum for rrf)."""
    ids, scores = bm25w
    by_id = dict(zip(ids, scores))
    extra = [c for c in ids if c not in set(ranked)]
    out = list(ranked) + extra
    rrf_floor = min(rrf) if rrf else 0.0
    return (out, list(bge) + [0.0] * len(extra), list(rrf) + [rrf_floor] * len(extra),
            [by_id.get(c, 0.0) for c in out])


def window_ids(cid, context, rows, by_pos):
    """Chunk ids of the window order - context .. order + context of the chunk's text, text order."""
    sha, order = rows[cid]['sha'], rows[cid]['order']
    return [by_pos[(sha, o)] for o in range(order - context, order + context + 1) if (sha, o) in by_pos]


def build_windows(order, docs, n_cover, context, budget, rows, by_pos, docs_of):
    """[(centre, window chunk ids)] for one query until the budget is used.

    Passes: the best chunk of each of the top n_cover documents, then the chunks in order with one window per
    text, then the chunks in order again (second windows). A centre inside a chosen window is skipped.
    """
    best = {}
    for cid in order:
        for doc in docs_of(cid):
            best.setdefault(doc, cid)
    head = list(dict.fromkeys(best[d] for d in docs[:n_cover] if d in best))
    chosen, covered, used = [], {}, 0
    for one_per_text, candidates in ((True, head), (True, order), (False, order)):
        for cid in candidates:
            sha, o = rows[cid]['sha'], rows[cid]['order']
            if o in covered.get(sha, ()) or (one_per_text and sha in covered):
                continue
            ids = window_ids(cid, context, rows, by_pos)
            cost = len(merge_run([rows[c] for c in ids]).encode('utf-8')) * len(docs_of(cid))
            if used + cost > budget:
                continue   # a long or wide-fanout window does not fit; smaller ones further down may
            chosen.append((cid, ids))
            used += cost
            covered.setdefault(sha, set()).update(range(o - context, o + context + 1))
    return chosen


def conf_doc_count(q_scores, column, threshold, lo, hi):
    """Documents with reranker score >= threshold, clipped to [lo, hi]."""
    return min(max(sum(s >= threshold for s in q_scores.get(column, {}).values()), lo), hi)


def chosen_bytes(chosen, rows, docs_of):
    return sum(len(rows[c]['text'].encode('utf-8')) * len(docs_of(c)) for c in chosen)


def score_cut_budget(budget, ctx, n_neighbour, min_score):
    """Budgets after the score cut: what cut queries leave unused goes to the queries the cut did not stop,
    in proportion to their budgets."""
    used, full = {}, []
    for qid, b in budget.items():
        args = (ctx['ranked'][qid], n_neighbour, b, ctx['rows'], ctx['by_pos'], ctx['docs_of'])
        cut = chosen_bytes(build_query(*args, scores=ctx['score'][qid], min_score=min_score), ctx['rows'], ctx['docs_of'])
        if cut < chosen_bytes(build_query(*args), ctx['rows'], ctx['docs_of']):
            used[qid] = cut
        else:
            full.append(qid)
    freed = sum(budget[q] - used[q] for q in used)
    grow = 1 + freed / max(sum(budget[q] for q in full), 1)
    print(f'score cut {min_score}: {len(used)} queries cut, {freed / 1e6:.1f} MB moved to {len(full)} queries', flush=True)
    return {qid: used[qid] if qid in used else budget[qid] * grow for qid in budget}


def build_variants(ctx, only=None, doc_scores=None):
    ranked, base, rows, by_pos, by_text, docs_of, lang_of = (
        ctx[k] for k in ('ranked', 'base', 'rows', 'by_pos', 'by_text', 'docs_of', 'lang_of'))
    base_bytes = {e['id']: sum(len(c['chunk_text'].encode('utf-8')) for c in e['relevant_chunks'])
                  for e in base}
    only = ONLY if only is None else only
    manifest = {'budget_scale': BUDGET_SCALE, 'variants': {}}
    for name, spec in VARIANTS.items():
        if only and name not in only:
            continue
        if spec.get('win') is not None and ctx.get('rrf') is None:
            print(name, 'skipped: pool.parquet has no rrf column', flush=True)
            continue
        uses_bm25w = 'bm25w' in spec.get('doc_rr', {}) or spec.get('w_bm25w')
        if uses_bm25w and not ctx.get('bm25w'):
            print(name, 'skipped: bm25w-chunks-k500.npz not attached', flush=True)
            continue
        if spec.get('doc_rr'):
            doc_scores = doc_scores or load_doc_scores()
            some = next(iter(doc_scores.values()))
            absent = [c for c in spec['doc_rr'] if c not in some and c != 'bm25w']
            if absent:
                print(name, 'skipped: no doc_rank column', absent, flush=True)
                continue
        n_neighbour = spec['nb']
        budget = {qid: b * spec.get('budget_scale', BUDGET_SCALE) for qid, b in base_bytes.items()}
        q_budget = (confidence_budget(budget, ctx['score'], spec['conf_gamma'], spec.get('conf_t', 0.5))
                    if spec.get('conf_gamma') else budget)
        if spec.get('min_score'):
            q_budget = score_cut_budget(q_budget, ctx, n_neighbour, spec['min_score'])
        scale = 1.0
        while True:
            submission, n_spans, n_chunks, n_docs, other_docs, other_bytes, all_bytes, moved = [], 0, 0, 0, 0, 0, 0, 0
            dropped = 0
            for entry in base:
                qid = entry['id']
                if spec.get('doc_rr'):
                    q_scores = doc_scores.get(qid, {})
                    if 'bm25w' in spec['doc_rr']:
                        q_scores = dict(q_scores, bm25w={})
                        for sha, s in zip(*ctx['bm25w']['texts'].get(qid, ([], []))):
                            for doc in ctx['text_docs'](sha):
                                q_scores['bm25w'].setdefault(doc, s)
                    candidates = list(dict.fromkeys(
                        doc_list(entry['relevant_docs'], ranked[qid], docs_of, lang_of, n_docs=10 ** 6)
                        + [d for c in spec['doc_rr'] for d in q_scores.get(c, {})]))
                    keep = spec.get('n_docs') or len(entry['relevant_docs'])
                    if spec.get('doc_conf'):
                        keep = conf_doc_count(q_scores, next(iter(spec['doc_rr'])), *spec['doc_conf'])
                    fixed = candidates[:spec.get('keep_top', 0)]
                    docs = fixed + fuse_docs(candidates[len(fixed):], q_scores, spec['doc_rr'], spec.get('w_base', 1.0),
                                             offset=len(fixed))[:keep - len(fixed)]
                    moved += len(set(docs) - set(entry['relevant_docs']))
                else:
                    docs = doc_list(entry['relevant_docs'], ranked[qid], docs_of, lang_of,
                                    spec.get('n_docs', 0), spec.get('doc_quota', 0.0))
                windows = None
                if spec.get('win') is not None:
                    if spec.get('w_bm25w'):
                        ids, bge, rrf, bm = with_bm25w(ranked[qid], ctx['score'][qid], ctx['rrf'][qid],
                                                       ctx['bm25w']['chunks'].get(qid, ([], [])))
                        weights = [spec['w_bge'], 1 - spec['w_bge'] - spec['w_bm25w'], spec['w_bm25w']]
                        order = blended_order(ids, [bge, rrf, bm], weights)
                    else:
                        order = blended_order(ranked[qid], [ctx['score'][qid], ctx['rrf'][qid]],
                                              [spec['w_bge'], 1 - spec['w_bge']])
                    windows = build_windows(order, docs, spec.get('cover_top', 0), spec['win'], q_budget[qid] * scale,
                                            rows, by_pos, docs_of)
                    chosen = list(dict.fromkeys(c for _, ids in windows for c in ids))
                elif n_neighbour == 'whole':
                    chosen = build_whole_docs(ranked[qid], entry['relevant_docs'], q_budget[qid] * scale,
                                              rows, by_text, docs_of)
                elif spec.get('cover'):
                    order = cover_order(ranked[qid], n_neighbour, docs, *spec['cover'], docs_of)
                    chosen = build_query(order, n_neighbour, q_budget[qid] * scale, rows, by_pos, docs_of)
                else:
                    chosen = build_query(ranked[qid], n_neighbour, q_budget[qid] * scale, rows, by_pos, docs_of,
                                         spec.get('quota', 0.0), ctx['score'][qid], spec.get('min_score', 0.0))
                for c in chosen:
                    size = len(rows[c]['text'].encode('utf-8'))
                    all_bytes += size
                    other_bytes += size if rows[c]['lang'] != 'vi' else 0
                listed = set(docs)
                chunks = []
                entries = ([(cid, merge_run([rows[c] for c in ids])) for cid, ids in windows] if windows is not None
                           else spans(chosen, rows))
                for cid, text in entries:
                    if len(text) < spec.get('min_chars', 0):
                        dropped += 1
                        continue
                    for doc in docs_of(cid):
                        chunks.append(dict(doc_id=doc, chunk_text=text, chunk_order=rows[cid]['order']))
                        if doc not in listed:
                            listed.add(doc)
                            docs.append(doc)
                n_spans += len(chunks)
                n_docs += len(docs)
                other_docs += sum(lang_of.get(d, 'vi') != 'vi' for d in docs) / max(len(docs), 1)
                n_chunks += len(chosen)
                submission.append(dict(id=qid, relevant_docs=docs, relevant_chunks=chunks))
            zip_path = OUT / f'medxir-expand-{name}-submission.zip'
            json_name = zip_path.name.replace('.zip', '.json')
            method = zipfile.ZIP_BZIP2 if spec.get('zip') == 'bzip2' else zipfile.ZIP_DEFLATED
            with zipfile.ZipFile(zip_path, 'w', compression=method, compresslevel=9) as archive:
                archive.writestr(json_name, json.dumps(submission, ensure_ascii=False, separators=(',', ':')))
            mb = zip_path.stat().st_size / 1e6
            if mb <= MAX_ZIP_MB:
                break
            scale *= 0.95 * MAX_ZIP_MB / mb
        stats = dict(spec, zip_mb=round(mb, 1), budget_scale_applied=round(scale, 3),
                     chunks_per_query=round(n_chunks / len(base), 1), entries_per_query=round(n_spans / len(base), 1),
                     docs_per_query=round(n_docs / len(base), 1), short_dropped_per_query=round(dropped / len(base), 1),
                     zh_en_doc_share=round(other_docs / len(base), 3),
                     zh_en_chunk_byte_share=round(other_bytes / max(all_bytes, 1), 3))
        if spec.get('doc_rr'):
            stats['new_docs_per_query'] = round(moved / len(base), 1)   # documents not in the submitted list
        manifest['variants'][name] = stats
        print(name, stats, flush=True)
    (OUT / 'expand_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    build_variants(load_inputs())


if __name__ == '__main__':
    main()
