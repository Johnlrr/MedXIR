"""Rerank the submitted windows of the 0.298 pipeline with a title-match prior (and windows for title-found documents).

The 0.2965 windows (kept in the 0.298 ZIP) were chosen by a score per centre chunk: ensemble 0.75 + bge 0.25 (both
min-max per query) + 0.3 x a document score ('dp03', Bảo's code not on the remote). Step 1 rebuilds that choice
under several definitions of the document score and reports how many of the submitted centres each recovers, so
the variants below start from the definition that matches. Step 2 adds w_title x title match (cosine of the query or
its title-style rewrite with the text's title, title_search.py's title_scores.parquet, min-max per query) and the
best chunk of every title-found document in the list (title_candidates.parquet) as extra centres; windows are then
chosen in score order (centre +- 1, centres inside a chosen window skipped), as many per query as the 0.298 ZIP has.
Document lists stay those of the 0.298 ZIP; documents of new windows are appended. LZMA ZIP.

Kaggle CPU. Inputs under INPUT_ROOT: chunk corpus v2, pool-bgefull.parquet + doc_rank.parquet
(ntklinhfitus/medxir-ens-v2-bgefull), the 0.298 submission (JSON), title_candidates.parquet (v1 run),
title_scores.parquet (v2 run). Outputs in OUT_DIR: window_rerank_manifest.json and one ZIP per variant.
"""
import glob
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import expand_chunks as ec

INPUT = os.environ.get('INPUT_ROOT', '/kaggle/input')
OUT = Path(os.environ.get('OUT_DIR', '/kaggle/working'))
W_ENS, W_BGE, W_DOC = 0.75, 0.25, 0.3
DOC_SCORES = ('bge_raw', 'bge_minmax', 'chunk_top3_minmax')   # definitions tried for the dp03 document score
VARIANTS = {'title10': 0.10, 'title20': 0.20}                  # name -> w_title


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    lo, hi = np.nanmin(values), np.nanmax(values)
    return np.zeros_like(values) if not hi > lo else (values - lo) / (hi - lo)


def select(order, count, rows, by_pos):
    """Centres in score order, skipping centres inside a chosen window (context 1), until count."""
    chosen, covered = [], {}
    for cid in order:
        sha, o = rows[cid]['sha'], rows[cid]['order']
        if o in covered.get(sha, ()):
            continue
        chosen.append(cid)
        covered.setdefault(sha, set()).update((o - 1, o, o + 1))
        if len(chosen) == count:
            break
    return chosen


def doc_score_of(kind, ids, centre, docs_of, bge_doc):
    """Per chunk: the score of its document under definition kind (max over the chunk's fan-out documents)."""
    if kind.startswith('bge'):
        per_doc = {d: s for d, s in bge_doc.items()}
        if kind == 'bge_minmax' and per_doc:
            keys = list(per_doc)
            per_doc = dict(zip(keys, minmax([per_doc[k] for k in keys])))
    else:   # best + 2nd + 3rd centre score of the document's chunks, min-max
        top = {}
        for cid, s in zip(ids, centre):
            for d in docs_of(cid):
                top.setdefault(d, []).append(s)
        keys = list(top)
        per_doc = dict(zip(keys, minmax([sum(sorted(top[k], reverse=True)[:3]) for k in keys])))
    return np.array([max((per_doc.get(d, 0.0) for d in docs_of(cid)), default=0.0) for cid in ids])


def main():
    ctx = ec.load_inputs()
    rows, by_pos, docs_of = ctx['rows'], ctx['by_pos'], ctx['docs_of']
    doc_scores = ec.load_doc_scores()
    title = pd.read_parquet(ec.one_match('title_scores.parquet'))
    title['match'] = title[['cos_q', 'cos_r']].max(axis=1)
    title_of = {int(q): dict(zip(g['text_sha256'], g['match'])) for q, g in title.groupby('query_id')}
    new = pd.read_parquet(ec.one_match('title_candidates.parquet'))
    base = {e['id']: e for e in ctx['base']}

    # Centres of the submitted windows: (doc, chunk order) -> pool chunk id.
    manifest = dict(replication={}, variants={})
    actual, counts = {}, {}
    for qid, entry in base.items():
        key = {}
        for cid in ctx['ranked'][qid]:
            for d in docs_of(cid):
                key[(d, rows[cid]['order'])] = cid
        cents = list(dict.fromkeys(key.get((c['doc_id'], c['chunk_order'])) for c in entry['relevant_chunks']))
        actual[qid] = [c for c in cents if c]
        counts[qid] = len(cents)
    manifest['centres_found'] = round(float(np.mean([len(actual[q]) / max(counts[q], 1) for q in base])), 4)

    def centre_scores(qid, kind, ids=None, extra=None):
        ids = list(ctx['ranked'][qid]) if ids is None else ids
        ens = minmax(ctx['rrf'][qid] + ([0.0] * len(extra) if extra else []))
        bge = minmax(ctx['score'][qid] + (extra or []))
        centre = W_ENS * ens + W_BGE * bge
        return ids, centre + W_DOC * doc_score_of(kind, ids, centre, docs_of, doc_scores.get(qid, {}).get('bge-reranker-v2-m3', {}))

    for kind in DOC_SCORES + ('none',):
        overlap = []
        for qid in base:
            if kind == 'none':
                ids = ctx['ranked'][qid]
                score = W_ENS * minmax(ctx['rrf'][qid]) + W_BGE * minmax(ctx['score'][qid])
            else:
                ids, score = centre_scores(qid, kind)
            order = [ids[i] for i in np.argsort(-score, kind='stable')]
            got = set(select(order, counts[qid], rows, by_pos))
            overlap.append(len(got & set(actual[qid])) / max(len(actual[qid]), 1))
        manifest['replication'][kind] = round(float(np.mean(overlap)), 4)
        print('replication', kind, manifest['replication'][kind], flush=True)
    best_kind = max(DOC_SCORES, key=lambda k: manifest['replication'][k])
    manifest['doc_score'] = best_kind

    for name, w_title in [('rep', 0.0)] + list(VARIANTS.items()):
        submission, n_new_windows = [], 0
        for qid, entry in base.items():
            ids = list(ctx['ranked'][qid])
            extra_ids, extra_bge = [], []
            if name != 'rep':   # best chunk of each title-found document of the list, as an extra centre
                listed = set(entry['relevant_docs'])
                for r in new[new['query_id'] == qid].itertuples():
                    if r.best_chunk_id and r.best_chunk_id in rows and r.best_chunk_id not in set(ids) \
                            and any(d in listed for d in r.doc_ids):
                        extra_ids.append(r.best_chunk_id)
                        extra_bge.append(float(r.bge))
            ids, score = centre_scores(qid, best_kind, ids + extra_ids, extra_bge or None)
            if w_title:
                t = title_of.get(qid, {})
                score = score + w_title * minmax([t.get(rows[c]['sha'], np.nan) for c in ids]).clip(0) \
                    if t else score
            order = [ids[i] for i in np.argsort(-np.nan_to_num(score, nan=-1.0), kind='stable')]
            centres = select(order, counts[qid], rows, by_pos)
            n_new_windows += sum(c in set(extra_ids) for c in centres)
            docs, chunks = list(entry['relevant_docs']), []
            listed = set(docs)
            for cid in centres:
                text = ec.merge_run([rows[c] for c in ec.window_ids(cid, 1, rows, by_pos)])
                for d in docs_of(cid):
                    chunks.append(dict(doc_id=d, chunk_text=text, chunk_order=rows[cid]['order']))
                    if d not in listed:
                        listed.add(d)
                        docs.append(d)
            submission.append(dict(id=qid, relevant_docs=docs, relevant_chunks=chunks))
        path = OUT / f'medxir-298-wrr-{name}-submission.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_LZMA) as archive:
            archive.writestr(path.name.replace('.zip', '.json'), json.dumps(submission, ensure_ascii=False, separators=(',', ':')))
        overlap = np.mean([len({(c['doc_id'], c['chunk_order']) for c in s['relevant_chunks']}
                               & {(c['doc_id'], c['chunk_order']) for c in base[s['id']]['relevant_chunks']})
                           / max(len(base[s['id']]['relevant_chunks']), 1) for s in submission])
        manifest['variants'][name] = dict(w_title=w_title, zip_mb=round(path.stat().st_size / 1e6, 1),
                                          entries_per_query=round(np.mean([len(s['relevant_chunks']) for s in submission]), 1),
                                          new_doc_windows_per_query=round(n_new_windows / len(submission), 2),
                                          entries_same_as_0298=round(float(overlap), 4))
        print(name, manifest['variants'][name], flush=True)
    (OUT / 'window_rerank_manifest.json').write_text(json.dumps(manifest, indent=2))
