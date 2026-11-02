"""Exact dense retrieval on this machine (CPU) over the stored Qwen3-Embedding-4B vectors, then the submissions.

Used when the Kaggle search kernel cannot read the vector datasets (it dies on the first full read).
The query vectors come from the Kaggle encode kernel (notebooks/kaggle_encode_queries_4b.ipynb).

Inputs (local, read-only):
  artifacts/kaggle/query-vectors-4b/query_vectors.npy, query_ids.json   encode kernel output
  artifacts/kaggle/vectors-qwen3-4b-s0..s2/vectors/<part>.npy           float16, unit norm
  artifacts/kaggle/vectors-qwen3-4b-s0..s2/vectors/<part>-ids.parquet   chunk_id per row
  artifacts/kaggle/chunk-corpus-v1/chunks/<lang>/part-NNN.parquet       chunk text
  artifacts/kaggle/chunk-corpus-v1/hash_groups.parquet                  duplicate-text groups
  dataset/query.parquet, dataset/links_corpus_vi_en_zh.parquet

The vector .npy files were deleted locally on 25/10 (the ids parquet files stay). Candidates are cached in
artifacts/submission/candidates-4b-k500.npz; a new search needs the vectors again, from the Kaggle datasets
boinhbo/medxir-vectors-qwen3-4b-s0..s2 (13 GB, `kaggle datasets download`).

Output (artifacts/submission/):
  candidates-4b-k<K>.npz                       cached top-K chunk candidates per query (reused on rerun)
  medxir-qwen3-4b-d<N>-submission.json, .zip   one submission per document count N
  retrieval_manifest.json

Method: each part is scored by an exact float32 matrix product, its top-K is kept, and the per-part
candidates are merged into the global top-K. The chunk list of every submission is the top TOP_CHUNKS
chunks. The document list is scored separately: a document's score is the best score of its candidate
chunks, and the N best documents are listed. Document and chunk metrics are computed independently, so N
trades document precision for document recall without touching the chunk list.

    uv run python scripts/retrieve_local_4b.py --doc-counts 20,30,50,100
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
TOP_CHUNKS = 20
QUERY_BATCH = 200


def part_paths(kaggle):
    """The nine vector parts, named by their ids files so the list works without the .npy files."""
    return sorted((path.with_name(path.name.removesuffix('-ids.parquet') + '.npy') for s in range(3)
                   for path in (kaggle / f'vectors-qwen3-4b-s{s}' / 'vectors').glob('*-ids.parquet')),
                  key=lambda path: path.stem)


def search(query_vectors, vector_paths, top_k):
    """Return (scores, part_index, rows), each shaped (queries, top_k), sorted by score descending."""
    n = len(query_vectors)
    all_scores, all_parts, all_rows = [], [], []
    for part_index, path in enumerate(vector_paths):
        vectors = np.load(path).astype(np.float32)
        assert vectors.shape[1] == query_vectors.shape[1] and np.isfinite(vectors).all(), path.stem
        k = min(top_k, len(vectors))
        part_scores = np.empty((n, k), dtype=np.float32)
        part_rows = np.empty((n, k), dtype=np.int64)
        for start in range(0, n, QUERY_BATCH):
            scores = query_vectors[start:start + QUERY_BATCH] @ vectors.T
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            part_scores[start:start + QUERY_BATCH] = np.take_along_axis(scores, top, axis=1)
            part_rows[start:start + QUERY_BATCH] = top
        all_scores.append(part_scores)
        all_rows.append(part_rows)
        all_parts.append(np.full((n, k), part_index, dtype=np.int16))
        print(path.stem, f'{len(vectors):,} rows', flush=True)
        del vectors
    scores, parts, rows = (np.concatenate(a, axis=1) for a in (all_scores, all_parts, all_rows))
    order = np.argsort(-scores, axis=1, kind='stable')[:, :top_k]
    return tuple(np.take_along_axis(a, order, axis=1) for a in (scores, parts, rows))


def load_or_search(cache_path, query_vectors, vector_paths, top_k):
    keys = [path.stem for path in vector_paths]
    if cache_path.exists():
        cached = np.load(cache_path)
        assert cached['keys'].tolist() == keys and cached['scores'].shape == (len(query_vectors), top_k)
        print('candidates from cache', cache_path.name, flush=True)
        return cached['scores'], cached['parts'], cached['rows']
    scores, parts, rows = search(query_vectors, vector_paths, top_k)
    np.savez(cache_path, scores=scores, parts=parts, rows=rows, keys=np.array(keys))
    return scores, parts, rows


def chunk_ids(parts, rows, vector_paths):
    """(queries, top_k) array of chunk_id, read from each part's ids file next to its .npy."""
    ids = np.empty(parts.shape, dtype=object)
    for part_index, path in enumerate(vector_paths):
        mask = parts == part_index
        if mask.any():
            column = np.array(pq.read_table(path.with_name(f'{path.stem}-ids.parquet'),
                                            columns=['chunk_id']).column('chunk_id').to_pylist(), dtype=object)
            ids[mask] = column[rows[mask]]
    return ids


def chunk_records(chunk_files, wanted_ids, text_ids):
    """chunk_id -> text_sha256, rep_doc_id, order; chunk_text only for text_ids (the submitted chunks)."""
    records = {}
    for path in chunk_files:
        table = pq.read_table(path, columns=['chunk_id', 'text_sha256', 'rep_doc_id', 'order', 'chunk_text'])
        frame = table.select(['chunk_id', 'text_sha256', 'rep_doc_id', 'order']).to_pandas()
        keep = frame['chunk_id'].isin(wanted_ids).to_numpy()
        for i in np.flatnonzero(keep):
            cid = frame['chunk_id'].iat[i]
            records[cid] = dict(
                text_sha256=frame['text_sha256'].iat[i],
                rep_doc_id=int(frame['rep_doc_id'].iat[i]),
                chunk_order=int(frame['order'].iat[i]),
                chunk_text=table.column('chunk_text')[int(i)].as_py() if cid in text_ids else None,
            )
        del table, frame
    return records


def group_lookup(hash_path):
    groups = pd.read_parquet(hash_path, columns=['text_sha256', 'doc_ids', 'fanout_safe'])
    return {
        sha: ([int(doc) for doc in doc_ids] if bool(safe) else None)
        for sha, doc_ids, safe in zip(groups['text_sha256'], groups['doc_ids'], groups['fanout_safe'])
    }


def docs_of(record, lookup):
    return lookup.get(record['text_sha256']) or [record['rep_doc_id']]


def ranked_docs(query_ids, records, lookup):
    """Documents in order of their best candidate chunk (candidates are already sorted by score)."""
    ranked, seen = [], set()
    for cid in query_ids:
        for doc_id in docs_of(records[cid], lookup):
            if doc_id not in seen:
                seen.add(doc_id)
                ranked.append(doc_id)
    return ranked


def build_submission(queries, ids, records, lookup, doc_count, chunk_count=TOP_CHUNKS):
    submission = []
    for qi, query_id in enumerate(queries['id'].tolist()):
        chunks = []
        for cid in ids[qi, :chunk_count]:
            record = records[cid]
            for doc_id in docs_of(record, lookup):
                chunks.append(dict(doc_id=doc_id, chunk_text=record['chunk_text'], chunk_order=record['chunk_order']))
        relevant_docs = ranked_docs(ids[qi], records, lookup)[:doc_count]
        # Doc scores are the max over chunks, so the submitted chunks' documents rank first; keep them listed
        # even when one top chunk fans out to more documents than doc_count.
        listed = set(relevant_docs)
        for chunk in chunks:
            if chunk['doc_id'] not in listed:
                listed.add(chunk['doc_id'])
                relevant_docs.append(chunk['doc_id'])
        submission.append(dict(id=int(query_id), relevant_docs=relevant_docs, relevant_chunks=chunks))
    return submission


def validate(submission, queries, corpus_ids):
    assert len(submission) == 1200 and {row['id'] for row in submission} == set(map(int, queries['id'].tolist()))
    for entry in submission:
        assert entry['relevant_chunks'], entry['id']
        assert set(entry['relevant_docs']).issubset(corpus_ids), entry['id']
        assert len(set(entry['relevant_docs'])) == len(entry['relevant_docs']), entry['id']
        listed = set(entry['relevant_docs'])
        assert all(chunk['doc_id'] in listed and chunk['chunk_text'].strip() for chunk in entry['relevant_chunks'])


def write_submission(submission, out, name, method='deflate'):
    """method 'bzip2' packs ~1.55x the text of deflate into the same ZIP size (30/10 test on the 0.2477 JSON)."""
    json_path = out / f'{name}.json'
    zip_path = out / f'{name}.zip'
    json_path.write_text(json.dumps(submission, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    compression = {'deflate': zipfile.ZIP_DEFLATED, 'bzip2': zipfile.ZIP_BZIP2, 'lzma': zipfile.ZIP_LZMA}[method]
    with zipfile.ZipFile(zip_path, 'w', compression=compression, compresslevel=9) as archive:
        archive.write(json_path, arcname=json_path.name)
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == [json_path.name]
    docs = [len(entry['relevant_docs']) for entry in submission]
    chunks = [len(entry['relevant_chunks']) for entry in submission]
    json_path.unlink()  # the ZIP is what gets submitted; saves disk
    return dict(chunks_per_query_median=float(np.median(chunks)), zip=zip_path.name, zip_bytes=zip_path.stat().st_size,
                docs_per_query_min=min(docs), docs_per_query_median=float(np.median(docs)), docs_per_query_max=max(docs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts' / 'submission')
    parser.add_argument('--top-k', type=int, default=500, help='candidate chunks kept per query for document ranking')
    parser.add_argument('--doc-counts', default='20,30,50,100', help='comma list of N, one submission per N')
    parser.add_argument('--chunk-counts', default='20', help='comma list of submitted chunks per query')
    args = parser.parse_args()
    doc_counts = [int(n) for n in args.doc_counts.split(',')]
    chunk_counts = [int(n) for n in args.chunk_counts.split(',')]
    started = time.perf_counter()
    kaggle = ROOT / 'artifacts' / 'kaggle'
    vector_paths = part_paths(kaggle)
    assert len(vector_paths) == 9, 'expected 9 vector parts'

    query_vectors = np.load(kaggle / 'query-vectors-4b' / 'query_vectors.npy')
    queries = pd.read_parquet(ROOT / 'dataset' / 'query.parquet', columns=['id'])
    assert json.loads((kaggle / 'query-vectors-4b' / 'query_ids.json').read_text()) == queries['id'].tolist()
    assert query_vectors.shape == (1200, 2560) and np.isfinite(query_vectors).all()

    args.out.mkdir(parents=True, exist_ok=True)
    scores, parts, rows = load_or_search(args.out / f'candidates-4b-k{args.top_k}.npz', query_vectors, vector_paths, args.top_k)
    print('search done', round(time.perf_counter() - started, 1), 's', flush=True)
    ids = chunk_ids(parts, rows, vector_paths)
    chunk_files = sorted((kaggle / 'chunk-corpus-v1' / 'chunks').glob('*/part-*.parquet'))
    wanted_ids = set(ids.ravel().tolist())
    records = chunk_records(chunk_files, wanted_ids, set(ids[:, :max(chunk_counts)].ravel().tolist()))
    assert not wanted_ids - set(records), 'candidate chunks without source text'
    lookup = group_lookup(kaggle / 'chunk-corpus-v1' / 'hash_groups.parquet')
    corpus_ids = set(map(int, pd.read_parquet(ROOT / 'dataset' / 'links_corpus_vi_en_zh.parquet', columns=['id'])['id']))

    variants = {}
    for doc_count in doc_counts:
        for chunk_count in chunk_counts:
            submission = build_submission(queries, ids, records, lookup, doc_count, chunk_count)
            validate(submission, queries, corpus_ids)
            name = f'd{doc_count}' + ('' if chunk_count == TOP_CHUNKS else f'-c{chunk_count}')
            variants[name] = write_submission(submission, args.out, f'medxir-qwen3-4b-{name}-submission')
            print(name, variants[name], flush=True)
    manifest = dict(
        model='Qwen/Qwen3-Embedding-4B', search='exact float32 matmul over stored float16 vectors, local CPU',
        query_count=len(queries), top_chunks=TOP_CHUNKS, candidate_top_k=args.top_k,
        doc_ranking='max candidate chunk score per document, hash-group fan-out when fanout_safe',
        variants=variants,
        elapsed_seconds=round(time.perf_counter() - started, 1),
    )
    (args.out / 'retrieval_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
