"""BM25 over Vietnamese words, the recipe of the organizers' 'BM25 Baseline' as far as it is published.

The organizers' BM25 Baseline (leaderboard, 30/09) has Doc P 0.88 at ~35 documents per query; our syllable +
bigram BM25 (bm25_vi.py, stopwords and chit-chat dropped) reaches ~0.1-0.2. The mentor slides (12/10, p.21)
show BM25 on word tokens ('triệu_chứng | sỏi_thận'), and the scorer normalises with NFKC + lowercase. So here:

- query words from pyvi (ViTokenizer), punctuation dropped, nothing else removed; repeated words count again;
- a document side word is the same syllable sequence: a k-syllable query word matches each occurrence of its
  k syllables in a row (the corpus is not segmented: 3 GB of text through pyvi would take hours);
- BM25 k1 1.5, b 0.75 (rank_bm25 / bm25s defaults), IDF log(1 + (N - df + 0.5) / (df + 0.5));
- two units: chunks (our 350/50 chunks) and texts (all chunks of a text summed; the 50-word overlaps count twice).

Inputs (under INPUT_ROOT, default /kaggle/input): chunks/vi/part-*.parquet + hash_groups.parquet of a chunk corpus,
query.parquet. Outputs in OUT_DIR: bm25w-chunks-k500.npz, bm25w-texts-k500.npz, one diagnostic submission ZIP
per N_DOCS (the top N texts' documents, each with its best chunk +- 1 neighbour) and bm25w_manifest.json.
"""
import glob
import json
import os
import re
import time
import unicodedata
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

K1, B = 1.5, 0.75
TOP_K = 500
N_DOCS = (30, 40)          # diagnostic submissions: top N texts (their documents), one window each
CONTEXT = 1
WORD = re.compile(r'\w+')
INPUT = os.environ.get('INPUT_ROOT', '/kaggle/input')
OUT = Path(os.environ.get('OUT_DIR', '/kaggle/working'))

VOCAB, PREFIX = {}, set()


def normalize(text):
    return unicodedata.normalize('NFKC', text).lower()


def query_words(text, segment):
    """Words of a query as syllable strings ('sỏi thận'), in order, repeats kept."""
    words = []
    for token in segment(normalize(text)).split():
        syllables = WORD.findall(token.replace('_', ' '))
        if syllables:
            words.append(' '.join(syllables))
    return words


def count_terms(text, max_n):
    """(term id -> tf, length in syllables) for the vocabulary words in one text."""
    toks = WORD.findall(normalize(text))
    counts = Counter(VOCAB[t] for t in toks if t in VOCAB)
    for i in range(len(toks) - 1):
        if (toks[i], toks[i + 1]) in PREFIX:
            for n in range(2, max_n + 1):
                tid = VOCAB.get(' '.join(toks[i:i + n]))
                if tid is not None:
                    counts[tid] += 1
    return counts, len(toks)


def set_vocab(words):
    """Vocabulary and two-syllable prefixes of the multi-syllable words (also the Pool initializer)."""
    VOCAB.clear()
    VOCAB.update({w: i for i, w in enumerate(words)})
    PREFIX.clear()
    PREFIX.update(tuple(w.split()[:2]) for w in words if ' ' in w)


def index_part(args):
    path, base, max_n = args
    rows, terms, tfs, lengths = [], [], [], []
    offset = base
    for batch in pq.ParquetFile(path).iter_batches(batch_size=5000, columns=['chunk_text']):
        for text in batch.column('chunk_text').to_pylist():
            counts, length = count_terms(text, max_n)
            lengths.append(length)
            for tid, tf in counts.items():
                rows.append(offset)
                terms.append(tid)
                tfs.append(min(tf, 255))
            offset += 1
    return (np.array(rows, dtype=np.int32), np.array(terms, dtype=np.int32), np.array(tfs, dtype=np.float32),
            np.array(lengths, dtype=np.float32))


def bm25_matrix(rows, terms, tfs, lengths, n_rows, vocab_size):
    """Rows x vocabulary matrix of BM25 term weights."""
    df = np.bincount(terms, minlength=vocab_size)
    idf = np.log1p((n_rows - df + 0.5) / (df + 0.5)).astype(np.float32)
    norm = K1 * (1 - B + B * lengths / lengths.mean())
    weights = tfs * (K1 + 1) / (tfs + norm[rows]) * idf[terms]
    return sp.csr_matrix((weights, (rows, terms)), shape=(n_rows, vocab_size))


def top_k(matrix, q, k, block=200000):
    """Per query (columns of q): the k best rows and their scores."""
    n_queries = q.shape[1]
    best_s = np.zeros((n_queries, 0), dtype=np.float32)
    best_i = np.zeros((n_queries, 0), dtype=np.int64)
    for start in range(0, matrix.shape[0], block):
        scores = (matrix[start:start + block] @ q).toarray().T
        kk = min(k, scores.shape[1])
        part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        best_s = np.concatenate([best_s, np.take_along_axis(scores, part, axis=1)], axis=1)
        best_i = np.concatenate([best_i, part + start], axis=1)
        order = np.argsort(-best_s, axis=1)[:, :k]
        best_s, best_i = np.take_along_axis(best_s, order, axis=1), np.take_along_axis(best_i, order, axis=1)
    return best_s, best_i


def one_match(pattern):
    hits = sorted(glob.glob(f'{INPUT}/**/{pattern}', recursive=True))
    assert len(hits) == 1, f'{pattern}: expected one match, found {hits[:10]}'
    return Path(hits[0])


def window_texts(needed, parts):
    """{chunk id: (start, end, text)} for the needed chunk ids."""
    out = {}
    for path in parts:
        table = pq.read_table(path, columns=['chunk_id', 'start', 'end', 'chunk_text'])
        table = table.filter(pc.is_in(table['chunk_id'], value_set=pa.array(sorted(needed))))
        for cid, start, end, text in zip(*(table[c].to_pylist() for c in ('chunk_id', 'start', 'end', 'chunk_text'))):
            out[cid] = (int(start), int(end), text)
    return out


def merge(parts):
    """One verbatim text from consecutive chunks (start, end, text) of one text, overlaps dropped."""
    text, end = parts[0][2], parts[0][1]
    for start, stop, chunk in parts[1:]:
        text += chunk[max(0, end - start):]
        end = stop
    return text


def main(segment, workers=4, max_n=4):
    started = time.perf_counter()
    queries = pd.read_parquet(one_match('query.parquet'), columns=['id', 'query'])
    per_query = [query_words(t, segment) for t in queries['query']]
    vocab = sorted({w for words in per_query for w in words})
    set_vocab(vocab)
    multi = sum(' ' in w for w in vocab)
    print(f'{len(queries)} queries, {len(vocab):,} words ({multi:,} multi-syllable); e.g. {per_query[0]}', flush=True)

    parts = sorted(glob.glob(f'{INPUT}/**/chunks/vi/part-*.parquet', recursive=True))
    assert parts, 'attach a chunk corpus'
    meta, jobs, base = [], [], 0
    for path in parts:
        table = pq.read_table(path, columns=['chunk_id', 'text_sha256', 'rep_doc_id', 'order'])
        meta.append(table.to_pandas())
        jobs.append((path, base, max_n))
        base += table.num_rows
    meta = pd.concat(meta, ignore_index=True)
    with Pool(min(workers, len(jobs)), initializer=set_vocab, initargs=(vocab,)) as pool:
        results = []
        for n, result in enumerate(pool.imap(index_part, jobs), 1):
            results.append(result)
            print(f'indexed part {n}/{len(jobs)}, {time.perf_counter() - started:.0f}s', flush=True)
    rows, terms, tfs, lengths = (np.concatenate([r[i] for r in results]) for i in range(4))
    del results
    n_chunks = len(lengths)
    print(f'{n_chunks:,} VI chunks, {len(rows):,} postings', flush=True)

    q_rows, q_cols, q_vals = [], [], []
    for qi, words in enumerate(per_query):
        for w, c in Counter(words).items():
            q_rows.append(VOCAB[w]), q_cols.append(qi), q_vals.append(float(c))
    q = sp.csr_matrix((q_vals, (q_rows, q_cols)), shape=(len(vocab), len(queries)), dtype=np.float32)

    # Chunk unit.
    chunk_m = bm25_matrix(rows, terms, tfs, lengths, n_chunks, len(vocab))
    c_scores, c_idx = top_k(chunk_m, q, TOP_K)
    chunk_ids = meta['chunk_id'].to_numpy(dtype=object)
    np.savez(OUT / 'bm25w-chunks-k500.npz', scores=c_scores, chunk_ids=chunk_ids[c_idx], query_ids=queries['id'].to_numpy())
    print(f'chunk BM25 done, {time.perf_counter() - started:.0f}s', flush=True)

    # Text unit: tf and length summed over the chunks of a text.
    sha_codes, shas = pd.factorize(meta['text_sha256'])
    t_rows = sha_codes[rows].astype(np.int64)
    tf_text = sp.csr_matrix((tfs, (t_rows, terms)), shape=(len(shas), len(vocab)))   # duplicates summed
    tf_text.sum_duplicates()
    coo = tf_text.tocoo()
    len_text = np.bincount(sha_codes, weights=lengths, minlength=len(shas)).astype(np.float32)
    text_m = bm25_matrix(coo.row, coo.col, coo.data.astype(np.float32), len_text, len(shas), len(vocab))
    t_scores, t_idx = top_k(text_m, q, TOP_K)
    np.savez(OUT / 'bm25w-texts-k500.npz', scores=t_scores, text_sha256=np.asarray(shas, dtype=object)[t_idx],
             query_ids=queries['id'].to_numpy())
    print(f'text BM25 done, {time.perf_counter() - started:.0f}s', flush=True)

    # Diagnostic submissions: top N texts -> their documents, each with its best chunk +- CONTEXT.
    groups = pd.read_parquet(one_match('hash_groups.parquet'), columns=['text_sha256', 'doc_ids', 'fanout_safe'])
    fan = {s: [int(d) for d in ids] for s, ids, ok in zip(groups['text_sha256'], groups['doc_ids'], groups['fanout_safe'])
           if bool(ok)}
    rep = dict(zip(meta['text_sha256'], meta['rep_doc_id'].astype(int)))
    chunks_of = meta.groupby(sha_codes).indices   # text code -> chunk row indices
    by_pos = {(s, int(o)): cid for s, o, cid in zip(meta['text_sha256'], meta['order'], meta['chunk_id'])}
    n_max = max(N_DOCS)
    picks = []   # per query: [(sha, best chunk id)]
    for qi in range(len(queries)):
        qcol = q[:, qi]
        row = []
        for code, score in zip(t_idx[qi, :n_max], t_scores[qi, :n_max]):
            if score <= 0:
                break
            members = chunks_of[code]
            best = members[int(np.argmax((chunk_m[members] @ qcol).toarray().ravel()))]
            row.append((shas[code], chunk_ids[best]))
        picks.append(row)
    needed = set()
    for row in picks:
        for sha, cid in row:
            o = int(cid.rsplit('-', 1)[1])
            needed.update(by_pos[(sha, k)] for k in range(o - CONTEXT, o + CONTEXT + 1) if (sha, k) in by_pos)
    texts = window_texts(needed, parts)
    manifest = dict(k1=K1, b=B, max_n=max_n, vocab=len(vocab), multi_syllable=multi, vi_chunks=n_chunks,
                    texts=len(shas), seconds=round(time.perf_counter() - started, 1), submissions={})
    for n_docs in N_DOCS:
        submission = []
        for qid, row in zip(queries['id'].tolist(), picks):
            docs, chunks = [], []
            for sha, cid in row[:n_docs]:
                o = int(cid.rsplit('-', 1)[1])
                ids = [by_pos[(sha, k)] for k in range(o - CONTEXT, o + CONTEXT + 1) if (sha, k) in by_pos]
                text = merge([texts[c] for c in ids])
                for doc in fan.get(sha) or [rep[sha]]:
                    docs.append(doc)
                    chunks.append(dict(doc_id=doc, chunk_text=text, chunk_order=o))
            submission.append(dict(id=int(qid), relevant_docs=list(dict.fromkeys(docs)), relevant_chunks=chunks))
        path = OUT / f'medxir-bm25w-d{n_docs}-submission.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr(path.name.replace('.zip', '.json'), json.dumps(submission, ensure_ascii=False,
                                                                             separators=(',', ':')))
        manifest['submissions'][path.name] = dict(
            zip_mb=round(path.stat().st_size / 1e6, 1),
            docs_per_query=round(np.mean([len(e['relevant_docs']) for e in submission]), 1),
            chunks_per_query=round(np.mean([len(e['relevant_chunks']) for e in submission]), 1))
    (OUT / 'bm25w_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
