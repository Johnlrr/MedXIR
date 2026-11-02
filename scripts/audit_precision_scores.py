"""Match submitted window centres to exact pool scores using hash-group metadata.

Scans metadata only; never reads corpus text or runs a model. Scores describe the
centre chunk, not the complete three-chunk window. Unmatched centres stay unknown.
"""
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from audit_submission_coverage import iter_queries
from build_precision_controls import digest


def normalize(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite score')
    low, high = values.min(), values.max()
    return (values-low)/(high-low) if high > low else np.zeros_like(values)


def run(source, ensemble, bge, groups, output):
    output.mkdir(parents=True, exist_ok=True)
    a = pd.read_parquet(ensemble, columns=['query_id', 'chunk_id', 'final_score'])
    b = pd.read_parquet(bge, columns=['query_id', 'chunk_id', 'final_score'])
    keys = ['query_id', 'chunk_id']
    joined = a.merge(b, on=keys, how='outer', suffixes=('_ensemble', '_bge'), indicator=True, validate='one_to_one')
    if not (joined['_merge'] == 'both').all():
        raise ValueError('Pool ID mismatch')
    if set(joined.query_id) != set(range(1, 1201)) or not (joined.groupby('query_id').size() == 500).all():
        raise ValueError('Expected 1200 x 500')
    del a, b
    prefixes = set(joined.chunk_id.str.rsplit('-', n=1).str[0])
    prefix_array = pa.array(sorted(prefixes))
    doc_to_prefix = {}
    for batch in pq.ParquetFile(groups).iter_batches(batch_size=8192, columns=['text_sha256', 'rep_doc_id', 'doc_ids', 'fanout_safe']):
        table = pa.Table.from_batches([batch])
        wanted = pc.is_in(pc.utf8_slice_codeunits(table['text_sha256'], 0, 16), value_set=prefix_array)
        for row in table.filter(wanted).to_pylist():
            ids = row['doc_ids'] if row['fanout_safe'] else [row['rep_doc_id']]
            for doc in ids:
                key = str(doc); prefix = row['text_sha256'][:16]
                if key in doc_to_prefix and doc_to_prefix[key] != prefix:
                    raise ValueError('Ambiguous doc mapping')
                doc_to_prefix[key] = prefix
    pools = {}
    for qid, frame in joined.groupby('query_id'):
        combined = .75*normalize(frame.final_score_ensemble) + .25*normalize(frame.final_score_bge)
        pools[int(qid)] = {cid: (float(bge_score), float(score)) for cid, bge_score, score in zip(frame.chunk_id, frame.final_score_bge, combined)}
    del joined
    report = dict(source_sha256=digest(source), ensemble_sha256=digest(ensemble), bge_sha256=digest(bge),
                  groups_sha256=digest(groups), weights=dict(ensemble=.75, bge=.25), matched=0, unmatched=0,
                  adjacent_score_inversions=0, adjacent_comparisons=0,
                  limitation='Scores belong to centre chunks, not complete windows; no gold labels.')
    records = []
    with zipfile.ZipFile(source) as archive:
        with io.TextIOWrapper(archive.open(archive.namelist()[0]), encoding='utf8') as stream:
            for entry in iter_queries(stream):
                qid = int(entry['id']); previous = None
                for index, chunk in enumerate(entry['relevant_chunks']):
                    prefix = doc_to_prefix.get(str(chunk['doc_id']))
                    cid = f"{prefix}-{chunk.get('chunk_order')}" if prefix else None
                    score = pools[qid].get(cid)
                    item = dict(query_id=qid, entry_index=index, doc_id=chunk['doc_id'], chunk_id=cid,
                                bge_score=score[0] if score else None, blended_score=score[1] if score else None)
                    records.append(item)
                    report['matched' if score else 'unmatched'] += 1
                    if score is not None:
                        if previous is not None:
                            report['adjacent_comparisons'] += 1
                            report['adjacent_score_inversions'] += int(score[1] > previous + 1e-7)
                        previous = score[1]
                if qid % 200 == 0:
                    print('Mapped query', qid, flush=True)
    pd.DataFrame(records).to_parquet(output/'submitted_centre_scores.parquet', index=False)
    (output/'centre_score_audit.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--ensemble', type=Path, required=True)
    p.add_argument('--bge', type=Path, required=True)
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.ensemble, a.bge, a.groups, a.output)
