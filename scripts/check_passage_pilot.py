"""Validate small pilot outputs; agreement is diagnostic, never a relevance metric."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate(candidates, scored, report):
    if report.get('complete') is not True:
        raise ValueError('Pilot is incomplete')
    required = ['query_id', 'doc_id', 'doc_ids', 'text_sha256', 'seed',
                'start', 'end', 'chunk_text', 'passage_tokens']
    for column in required:
        if column not in candidates or column not in scored:
            raise ValueError(f'Missing candidate column: {column}')
    if len(scored) != report['pairs'] or not len(scored):
        raise ValueError('Pair count differs from report')
    if scored.query_id.nunique() != report['n_queries']:
        raise ValueError('Query count differs from report')
    # Scores must correspond to precisely the saved candidates, including order.
    pd.testing.assert_frame_equal(candidates, scored[candidates.columns], check_exact=True)
    keys = ['query_id', 'text_sha256', 'start', 'end']
    if scored.duplicated(keys).any():
        raise ValueError('Duplicate source passage for the same query')
    for row in scored.itertuples(index=False):
        if row.start < 0 or row.end <= row.start or row.end-row.start != len(row.chunk_text):
            raise ValueError('Invalid source offsets or text length')
        ids = list(map(int, row.doc_ids))
        if not ids or row.doc_id not in ids or len(ids) != len(set(ids)):
            raise ValueError('Invalid document fanout')
        if not 0 < row.passage_tokens <= report['passage_tokens'] + 2:
            raise ValueError('Passage exceeds configured token budget')
    if report.get('fixed_qwen_suffix_lost') != 0:
        raise ValueError('Fixed Qwen prompt lost its suffix')
    for model in ['bge', 'qwen']:
        values = scored[model].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f'Invalid {model} scores')
    overlap = []
    for _, group in scored.groupby('query_id'):
        # Compare unique source texts, without overweighting duplicate doc IDs.
        docs = group.groupby('text_sha256')[['bge', 'qwen']].max()
        k = min(20, len(docs))
        left = set(docs.sort_values('bge', ascending=False, kind='stable').head(k).index)
        right = set(docs.sort_values('qwen', ascending=False, kind='stable').head(k).index)
        overlap.append(len(left & right)/k)
    return dict(valid=True, pairs=len(scored), queries=int(scored.query_id.nunique()),
                mean_top20_source_agreement=float(np.mean(overlap)),
                truncations=report['truncations'],
                limitation='No relevance labels. Offset lengths checked; source-byte fidelity is asserted during generation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    report = json.loads((root/'passage_pilot_report.json').read_text(encoding='utf8'))
    result = validate(pd.read_parquet(root/'passage_candidates.parquet'),
                      pd.read_parquet(root/'passage_scores.parquet'), report)
    (root/'passage_validation.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
