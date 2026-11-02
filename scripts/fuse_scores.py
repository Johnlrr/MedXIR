"""Pick the submitted chunks from the dense top-500 pool by a combined dense + BM25 score, then write submissions.

Unlike fuse_dense_bm25.py (rank fusion, keeps dense order for the first chunks) this scores every pooled chunk:
per query, dense cosine and BM25 are each min-max scaled over that query's own list, a chunk missing from one
list takes that list's minimum (0), and score = (1 - w) * dense + w * bm25.
Each --extra list (e.g. the Qwen3-0.6B candidates) adds its own weighted term; the 4B weight is 1 minus the rest. The pool is the dense top-500;
BM25-only chunks join when --pool union.

Chunk list: the top C chunks by combined score, except that at most --max-per-doc chunks of one text (one
text_sha256, i.e. one document before hash-group fan-out) are taken before every other text has had its turn;
the deferred chunks fill the list only when the pool runs out. 0 keeps the plain score order.

--sibling-boost gamma goes the other way: chunks of texts that have several strong chunks move up, so the
list concentrates on fewer, stronger documents (diversifying with --max-per-doc 1 scored lower on 27/10).

Document list: a text's score is its best chunk score plus --doc-beta times the sum of its 2nd and 3rd best
chunk scores (0 = the max-chunk ranking of retrieve_local_4b.py), and the N best texts' doc_ids are listed.

--split K submits each chosen chunk as K consecutive verbatim pieces of about equal word count (same text
bytes, K times the entries). Under the 27/10 metric a match also needs LCS >= 40% of the predicted chunk, so
pieces shorter than our 350-word chunks match short reference chunks that a whole chunk would miss.

--context K submits each chosen chunk as a verbatim window with K neighbouring chunks of the same text on each
side (chunks carry character offsets, so overlaps are removed). A chunk already inside a higher-ranked window
of the same text is skipped, so the byte budget is not spent on repeated text.

--corpus v2 reads the three corpus v2 addition parts and the v2 hash groups as well (candidate lists from
retrieve_additions.py, BM25 from --bm25). --bge-pool adds the hybrid run's bge scores (pool.parquet final_score) as one
more list. --export-pool N writes this run's top-N pool and submission in the layout kaggle_doc_rerank.ipynb reads;
--doc-rank then re-orders the submitted documents by that notebook's bge scores (its 'bge-only' rule: candidate
documents sorted by bge, the same count kept, then every submitted chunk's document added).

Inputs:  artifacts/submission/candidates-4b-k500.npz, artifacts/bm25/vi-candidates-k500.npz
Outputs: artifacts/submission/medxir-<tag>-d<N>-c<C>-submission.zip, plus a printed set-difference vs dense.

    uv run python scripts/fuse_scores.py --tag sc03u-m1 --w-bm25 0.3 --pool union --max-per-doc 1
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import retrieve_local_4b as dense

ROOT = dense.ROOT


def minmax(scores):
    low, high = scores.min(), scores.max()
    return (scores - low) / (high - low) if high > low else np.zeros_like(scores)


def combine(lists, pool, depth):
    """lists: [(ids, scores, weight)], the first being the dense 4B list. Returns (queries, depth) chunk ids
    sorted by the weighted sum of per-query min-max scores, and those scores."""
    n = len(lists[0][0])
    out_ids = np.empty((n, depth), dtype=object)
    out_scores = np.zeros((n, depth), dtype=np.float32)
    for qi in range(n):
        scaled = [(dict(zip(ids[qi], minmax(scores[qi]))), w) for ids, scores, w in lists]
        keys = list(scaled[0][0]) if pool == 'dense' else list(set().union(*(s for s, _ in scaled)))
        score = np.array([sum(w * s.get(k, 0.0) for s, w in scaled) for k in keys], dtype=np.float32)
        order = np.argsort(-score, kind='stable')[:depth]
        out_ids[qi] = [keys[i] for i in order]
        out_scores[qi] = score[order]
    return out_ids, out_scores


def text_key(cid):
    return cid.rsplit('-', 1)[0]


def boost_siblings(ids, scores, gamma):
    """Rescore each chunk as score + gamma * (sum of the 3 best chunk scores of its text), so chunks of texts with
    several strong chunks move up. Returns re-sorted (ids, scores)."""
    out_ids, out_scores = np.empty_like(ids), np.zeros_like(scores)
    for qi in range(len(ids)):
        top = {}
        for cid, sc in zip(ids[qi], scores[qi]):  # rows are sorted descending
            top.setdefault(text_key(cid), []).append(sc)
        text_score = {k: sum(v[:3]) for k, v in top.items()}
        new = np.array([sc + gamma * text_score[text_key(cid)] for cid, sc in zip(ids[qi], scores[qi])], dtype=np.float32)
        order = np.argsort(-new, kind='stable')
        out_ids[qi], out_scores[qi] = ids[qi][order], new[order]
    return out_ids, out_scores


def chunk_order(ids, max_per_doc):
    """Per query, the pool reordered so no text gets more than max_per_doc chunks before the others are used."""
    if max_per_doc <= 0:
        return ids
    out = np.empty_like(ids)
    for qi, row in enumerate(ids):
        taken, first, later = {}, [], []
        for cid in row:
            key = text_key(cid)
            taken[key] = taken.get(key, 0) + 1
            (first if taken[key] <= max_per_doc else later).append(cid)
        out[qi] = first + later
    return out


def doc_order(ids, scores, beta):
    """Per query, one chunk id per text, texts sorted by best score + beta * (2nd + 3rd best)."""
    out = []
    for row, row_scores in zip(ids, scores):
        best, extra, rep = {}, {}, {}
        for cid, s in zip(row, row_scores):
            key = text_key(cid)
            if key not in best:
                best[key], extra[key], rep[key] = s, [], cid
            elif len(extra[key]) < 2:
                extra[key].append(s)
        out.append([rep[k] for k in sorted(best, key=lambda k: -(best[k] + beta * sum(extra[k])))])
    return out


def pieces(text, k):
    """text cut into k verbatim pieces of about equal word count (character count for zh text without spaces)."""
    if k <= 1:
        return [text]
    starts = [m.start() for m in re.finditer(r'\S+', text)]
    if len(starts) < 20 * k:  # zh: no word spaces, cut by characters
        starts = list(range(len(text)))
    bounds = sorted({starts[len(starts) * i // k] for i in range(1, k)} - {starts[0]}) if starts else []
    cuts = [0] + bounds + [len(text)]
    return [text[a:b].strip() for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]


def select_windows(row, count, context):
    """Chunk ids in score order whose window (order +- context) does not hold an already chosen chunk's order."""
    chosen, covered = [], {}
    for cid in row:
        key, order = cid.rsplit('-', 1)
        order = int(order)
        if order in covered.setdefault(key, set()):
            continue  # already inside the window of a higher-ranked chunk of the same text
        chosen.append(cid)
        covered[key].update(range(order - context, order + context + 1))
        if len(chosen) == count:
            break
    return chosen


def load_spans(chunk_files, ids):
    """chunk_id -> (start, end, chunk_text) for the given ids; start/end are character offsets in the text."""
    spans = {}
    for path in chunk_files:
        table = pq.read_table(path, columns=['chunk_id', 'start', 'end', 'chunk_text'])
        keep = pc.is_in(table['chunk_id'], value_set=pa.array(sorted(ids)))
        for row in table.filter(keep).to_pylist():
            spans[row['chunk_id']] = (row['start'], row['end'], row['chunk_text'])
    return spans


def window_text(cid, context, spans):
    """Verbatim text from chunk order - context to order + context of the same text, overlaps removed.
    Neighbours that do not exist (text start or end) are left out; a 1-2 character whitespace gap between
    two chunks without overlap becomes a newline."""
    key, order = cid.rsplit('-', 1)
    parts = [spans[f'{key}-{o}'] for o in range(int(order) - context, int(order) + context + 1) if f'{key}-{o}' in spans]
    text, last_end = parts[0][2], parts[0][1]
    for start, end, chunk in parts[1:]:
        text += chunk[last_end - start:] if start < last_end else '\n' + chunk
        last_end = end
    return text


def bge_doc_order(doc_rank, column, extra=None, k=60):
    """{query_id: candidate documents sorted by the reranker score, unscored ones last in candidate order}.

    extra: (frame with query_id, doc_id, score; weight). Documents it scores get weight / (k + rank among them) on top
    of 1 / (k + bge rank) (RRF); the rest keep the bge term only.
    """
    frame = pd.read_parquet(doc_rank, columns=['query_id', 'doc_id', 'base_rank', column])
    frame = frame[frame['base_rank'] >= 0].sort_values(['query_id', 'base_rank'])
    other = {} if extra is None else {int(q): g for q, g in extra[0].groupby('query_id')}
    out = {}
    for qid, g in frame.groupby('query_id'):
        scored = g[g[column].notna()].sort_values(column, ascending=False, kind='stable')
        docs = [int(d) for d in scored['doc_id']]
        if extra is not None and int(qid) in other:
            fused = {d: 1 / (k + r) for r, d in enumerate(docs)}
            e = other[int(qid)].sort_values('score', ascending=False, kind='stable')
            for r, d in enumerate(e['doc_id'].astype(int)):
                if d in fused:
                    fused[d] += extra[1] / (k + r)
            docs = sorted(docs, key=lambda d: -fused[d])
        out[int(qid)] = docs + [int(d) for d in g.loc[g[column].isna(), 'doc_id']]
    return out


def extra_doc_scores(spec):
    """'PARQUET[,PARQUET...]:COLUMN:WEIGHT' (sharded doc_rank.parquet files) -> (frame query_id, doc_id, score; weight)."""
    paths, column, weight = spec.rsplit(':', 2)
    frame = pd.concat([pd.read_parquet(x, columns=['query_id', 'doc_id', column]) for x in paths.split(',')])
    frame = frame.dropna().rename(columns={column: 'score'})
    assert not frame.duplicated(['query_id', 'doc_id']).any()
    return frame, float(weight)


def pool_list(path):
    """A hybrid pool.parquet (final_score = reranker score) as a ragged candidate list in query.parquet order."""
    pool = pd.read_parquet(path, columns=['query_id', 'chunk_id', 'final_score'])
    groups = {int(q): g for q, g in pool.groupby('query_id')}
    queries = pd.read_parquet(ROOT / 'dataset' / 'query.parquet', columns=['id'])['id'].tolist()
    ids = np.empty(len(queries), dtype=object)
    scores = np.empty(len(queries), dtype=object)
    for qi, qid in enumerate(queries):
        g = groups[int(qid)]
        ids[qi], scores[qi] = g['chunk_id'].to_numpy(), g['final_score'].to_numpy(dtype=np.float32)
    return ids, scores


def build_submission(queries, chunk_ids, doc_reps, records, lookup, doc_count, chunk_count, split=1,
                     windows=None, context=0, spans=None, doc_override=None):
    submission = []
    for qi, query_id in enumerate(queries['id'].tolist()):
        chunks = []
        centers = windows[qi][:chunk_count] if windows is not None else chunk_ids[qi, :chunk_count]
        for cid in centers:
            record = records[cid]
            texts = [window_text(cid, context, spans)] if windows is not None else pieces(record['chunk_text'], split)
            for doc_id in dense.docs_of(record, lookup):
                for text in texts:
                    chunks.append(dict(doc_id=doc_id, chunk_text=text, chunk_order=record['chunk_order']))
        relevant_docs = dense.ranked_docs(doc_reps[qi], records, lookup)[:doc_count]
        if doc_override is not None:
            relevant_docs = doc_override[int(query_id)]
        # Same rule as retrieve_local_4b.build_submission: every submitted chunk's document is listed.
        listed = set(relevant_docs)
        for chunk in chunks:
            if chunk['doc_id'] not in listed:
                listed.add(chunk['doc_id'])
                relevant_docs.append(chunk['doc_id'])
        submission.append(dict(id=int(query_id), relevant_docs=relevant_docs, relevant_chunks=chunks))
    return submission


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--w-bm25', type=float, default=0.2)
    parser.add_argument('--extra', action='append', default=[], metavar='NPZ:WEIGHT',
                        help='another candidate list in the BM25 format (e.g. candidates-06b-k500.npz:0.2); repeatable')
    parser.add_argument('--lang-file', action='append', default=[], metavar='NPZ:BONUS',
                        help='per-language dense candidates (retrieve_lang_4b.py) merged into the dense list after '
                             'adding BONUS to their cosine scores; repeatable')
    parser.add_argument('--icd', type=float, default=0.0,
                        help='weight of the ICD-10 chapter agreement (scripts/icd_feature.py) added to every pooled chunk')
    parser.add_argument('--pool', choices=['dense', 'union'], default='dense')
    parser.add_argument('--depth', type=int, default=500)
    parser.add_argument('--max-per-doc', type=int, default=0, help='chunks per text before others get a turn; 0 = off')
    parser.add_argument('--sibling-boost', type=float, default=0.0,
                        help='gamma: add gamma * (3 best chunk scores of the same text) to every chunk score')
    parser.add_argument('--context', type=int, default=0,
                        help='submit each chosen chunk joined with this many neighbouring chunks on each side')
    parser.add_argument('--split', type=int, default=1, help='submit each chunk as this many verbatim pieces')
    parser.add_argument('--doc-beta', type=float, default=0.0, help='weight of the 2nd + 3rd best chunk in a doc score')
    parser.add_argument('--doc-counts', default='200')
    parser.add_argument('--chunk-counts', default='150')
    parser.add_argument('--corpus', choices=['v1', 'v2'], default='v1')
    parser.add_argument('--bm25', type=Path, default=ROOT / 'artifacts' / 'bm25' / 'vi-candidates-k500.npz')
    parser.add_argument('--bge-pool', metavar='PARQUET:WEIGHT', default=None,
                        help='reranker scores of a hybrid pool.parquet as one more list')
    parser.add_argument('--export-pool', type=int, default=0,
                        help='write the top-N fused pool + the submission JSON for kaggle_doc_rerank.ipynb')
    parser.add_argument('--doc-rank', type=Path, default=None,
                        help="doc_rank.parquet of kaggle_doc_rerank.ipynb run on this run's --export-pool output")
    parser.add_argument('--doc-rank-column', default='bge-reranker-v2-m3')
    parser.add_argument('--doc-prior-bge-only', action='store_true',
                        help='--doc-prior uses the --doc-rank scores only, not --doc-rank-extra (31/10: LLM in the prior '
                             'cost 0.0018 Chunk F2 while LLM in the document list gained 0.0039 Doc F2)')
    parser.add_argument('--doc-rank-extra', default=None, metavar='PARQUET[,PARQUET]:COLUMN:WEIGHT',
                        help='second document reranker (e.g. LLM shards), RRF-fused with --doc-rank and averaged into '
                             '--doc-prior')
    parser.add_argument('--doc-prior', type=float, default=0.0,
                        help='add this times the best --doc-rank score of a chunk\'s documents to its score before '
                             'choosing chunks/windows')
    parser.add_argument('--zip', choices=['deflate', 'bzip2', 'lzma'], default='deflate', help='ZIP compression method')
    parser.add_argument('--dry-run', action='store_true', help='print the set difference only, write nothing')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts' / 'submission')
    args = parser.parse_args()
    kaggle = ROOT / 'artifacts' / 'kaggle'
    vector_paths = dense.part_paths(kaggle)
    queries = pd.read_parquet(ROOT / 'dataset' / 'query.parquet', columns=['id'])
    cached = np.load(args.out / 'candidates-4b-k500.npz')
    assert cached['keys'].tolist() == [p.stem for p in vector_paths]
    dense_ids = dense.chunk_ids(cached['parts'], cached['rows'], vector_paths)
    bm25 = np.load(args.bm25, allow_pickle=True)
    assert bm25['query_ids'].tolist() == queries['id'].tolist()
    lists = [(bm25['chunk_ids'], bm25['scores'], args.w_bm25)]
    for spec in args.extra:
        path, weight = spec.rsplit(':', 1)
        extra = np.load(path, allow_pickle=True)
        assert extra['query_ids'].tolist() == queries['id'].tolist(), path
        lists.append((extra['chunk_ids'], extra['scores'], float(weight)))
    if args.bge_pool:
        path, weight = args.bge_pool.rsplit(':', 1)
        lists.append((*pool_list(path), float(weight)))
    w_dense = 1 - sum(w for _, _, w in lists)  # weights sum to 1
    assert w_dense > -1e-9, 'extra weights add up to more than 1'  # 0 keeps 4B chunks in the pool, unscored
    merged_ids, merged_scores = dense_ids, cached['scores']
    for spec in args.lang_file:
        path, bonus = spec.rsplit(':', 1)
        lang_list = np.load(path, allow_pickle=True)
        assert lang_list['query_ids'].tolist() == queries['id'].tolist(), path
        merged_ids = np.concatenate([merged_ids, lang_list['chunk_ids']], axis=1)
        merged_scores = np.concatenate([merged_scores, lang_list['scores'] + float(bonus)], axis=1)
    order = np.argsort(-merged_scores, axis=1, kind='stable')
    merged_ids, merged_scores = np.take_along_axis(merged_ids, order, axis=1), np.take_along_axis(merged_scores, order, axis=1)
    ids, scores = combine([(merged_ids, merged_scores, w_dense)] + lists, args.pool, args.depth)
    if args.icd:
        from icd_feature import Agreement
        scores = scores + args.icd * Agreement()(ids)
        order = np.argsort(-scores, axis=1, kind='stable')
        ids, scores = np.take_along_axis(ids, order, axis=1), np.take_along_axis(scores, order, axis=1)
    if args.sibling_boost:
        ids, scores = boost_siblings(ids, scores, args.sibling_boost)
    chunk_ids = chunk_order(ids, args.max_per_doc)
    doc_reps = doc_order(ids, scores, args.doc_beta)

    chunk_counts = [int(x) for x in args.chunk_counts.split(',')]
    doc_counts = [int(x) for x in args.doc_counts.split(',')]
    for c in chunk_counts:
        kept = np.mean([len(set(chunk_ids[i, :c]) & set(dense_ids[i, :c])) for i in range(len(ids))])
        texts = np.mean([len({text_key(cid) for cid in chunk_ids[i, :c]}) for i in range(len(ids))])
        print(json.dumps(dict(tag=args.tag, w_bm25=args.w_bm25, extra=args.extra, icd=args.icd, pool=args.pool, max_per_doc=args.max_per_doc, chunks=c,
                              mean_chunks_shared_with_dense_topc=round(float(kept), 1),
                              mean_distinct_texts=round(float(texts), 1))), flush=True)
    for n in doc_counts:
        dense_texts = [list(dict.fromkeys(map(text_key, row)))[:n] for row in dense_ids]
        kept = np.mean([len({text_key(c) for c in doc_reps[i][:n]} & set(dense_texts[i])) for i in range(len(ids))])
        print(json.dumps(dict(doc_beta=args.doc_beta, docs=n, mean_texts_shared_with_dense=round(float(kept), 1))), flush=True)
    if args.dry_run:
        return
    wanted = set(ids.ravel().tolist())
    chunk_files = sorted((kaggle / 'chunk-corpus-v1' / 'chunks').glob('*/part-*.parquet'))
    hash_path = kaggle / 'chunk-corpus-v1' / 'hash_groups.parquet'
    if args.corpus == 'v2':  # v1 parts unchanged + vi-part-007, en-part-001, zh-part-001; hash groups v1 + additions
        chunk_files += sorted((kaggle / 'chunk-corpus-v2' / 'chunks').glob('*/part-*.parquet'))
        hash_path = kaggle / 'chunk-corpus-v2' / 'hash_groups.parquet'
        assert len(chunk_files) == 12, chunk_files
    records = dense.chunk_records(chunk_files, wanted, set(chunk_ids[:, :max(chunk_counts)].ravel().tolist()))
    assert not wanted - set(records)
    lookup = dense.group_lookup(hash_path)
    if args.doc_prior:  # chunk score + lambda * best bge document score of its documents: P(chunk) ~ P(doc) P(chunk|doc)
        assert args.doc_rank, '--doc-prior needs --doc-rank'
        frame = pd.read_parquet(args.doc_rank, columns=['query_id', 'doc_id', args.doc_rank_column]).dropna()
        doc_score = {int(q): dict(zip(g['doc_id'].astype(int), g[args.doc_rank_column])) for q, g in frame.groupby('query_id')}
        if args.doc_rank_extra and not args.doc_prior_bge_only:  # documents the second reranker scored: weighted mean of the two scores
            extra, w = extra_doc_scores(args.doc_rank_extra)
            for q, d, sc in zip(extra['query_id'].astype(int), extra['doc_id'].astype(int), extra['score']):
                if d in doc_score.get(q, {}):
                    doc_score[q][d] = (doc_score[q][d] + w * sc) / (1 + w)
        before = [select_windows(row, max(chunk_counts), max(args.context, 0)) for row in chunk_ids]
        for qi, qid in enumerate(queries['id'].tolist()):
            ds = doc_score.get(int(qid), {})
            prior = np.array([max((ds.get(d, 0.0) for d in dense.docs_of(records[c], lookup)), default=0.0)
                              for c in ids[qi]], dtype=np.float32)
            new = scores[qi] + args.doc_prior * prior
            order = np.argsort(-new, kind='stable')
            ids[qi], scores[qi] = ids[qi][order], new[order]
        chunk_ids = chunk_order(ids, args.max_per_doc)
        after = [select_windows(row, max(chunk_counts), max(args.context, 0)) for row in chunk_ids]
        changed = np.mean([1 - len(set(a) & set(b)) / len(b) for a, b in zip(after, before)])
        print(json.dumps(dict(doc_prior=args.doc_prior, window_centres_changed=round(float(changed), 3))), flush=True)
    windows = spans = None
    if args.context:
        windows = [select_windows(row, max(chunk_counts), args.context) for row in chunk_ids]
        centers = {cid for row in windows for cid in row}
        needed = {f"{cid.rsplit('-', 1)[0]}-{int(cid.rsplit('-', 1)[1]) + d}"
                  for cid in centers for d in range(-args.context, args.context + 1)}
        spans = load_spans(chunk_files, needed)
        assert centers <= set(spans)
    corpus_ids = set(map(int, pd.read_parquet(ROOT / 'dataset' / 'links_corpus_vi_en_zh.parquet', columns=['id'])['id']))
    doc_override = None
    if args.doc_rank:
        order = bge_doc_order(args.doc_rank, args.doc_rank_column,
                              extra_doc_scores(args.doc_rank_extra) if args.doc_rank_extra else None)
        base = build_submission(queries, chunk_ids, doc_reps, records, lookup, doc_counts[0], chunk_counts[0],
                                args.split, windows, args.context, spans)
        # The notebook's candidates start with the submitted list: keep as many documents as it had.
        doc_override = {e['id']: order[e['id']][:len(e['relevant_docs'])] for e in base}
        moved = np.mean([len(set(doc_override[e['id']]) - set(e['relevant_docs'])) for e in base])
        print(json.dumps(dict(doc_rank=str(args.doc_rank), new_docs_per_query=round(float(moved), 1))), flush=True)
    for n in doc_counts:
        for c in chunk_counts:
            submission = build_submission(queries, chunk_ids, doc_reps, records, lookup, n, c, args.split,
                                          windows, args.context, spans, doc_override)
            dense.validate(submission, queries, corpus_ids)
            if args.export_pool:  # kaggle_doc_rerank.ipynb inputs: pool.parquet + medxir-hybrid-submission.json
                folder = args.out / f'docrerank-{args.tag}'
                folder.mkdir(exist_ok=True)
                rows = [(int(qid), cid, rank, float(sc), 0.0, -1, -1)
                        for qid, row, row_scores in zip(queries['id'], ids, scores)
                        for rank, (cid, sc) in enumerate(zip(row[:args.export_pool], row_scores[:args.export_pool]))]
                pd.DataFrame(rows, columns=['query_id', 'chunk_id', 'final_rank', 'final_score', 'rrf', 'dense_rank',
                                            'bm25_rank']).to_parquet(folder / 'pool.parquet', index=False,
                                                                     compression='zstd')
                (folder / 'medxir-hybrid-submission.json').write_text(
                    json.dumps(submission, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
                print('doc rerank inputs:', folder, flush=True)
            name = (f'medxir-{args.tag}-d{n}-c{c}' + (f'-s{args.split}' if args.split > 1 else '')
                    + (f'-x{args.context}' if args.context else '') + {'deflate': '', 'bzip2': '-bz', 'lzma': '-lz'}[args.zip]
                    + '-submission')
            print(f'd{n}-c{c}', dense.write_submission(submission, args.out, name, args.zip), flush=True)


if __name__ == '__main__':
    main()
