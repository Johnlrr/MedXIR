"""Compare rankings on identical candidates. These diagnostics are not gold metrics."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def diagnostics(frame):
    required={'query_id','text_sha256','seed_rank','passage_tokens','chunk_text','bge','qwen'}
    if not required<=set(frame):raise ValueError('Missing comparison columns')
    for model in ['bge','qwen']:
        if not np.isfinite(frame[model].to_numpy(dtype=float)).all():raise ValueError('Invalid scores')
    eligible=frame[frame.passage_tokens>=40]
    result=dict(queries=int(frame.query_id.nunique()),pairs=len(frame),eligible_pairs=len(eligible),
                min_passage_tokens=40,models={},overlap={},limitation='No gold labels; score values are not comparable probabilities.')
    ranked={}
    for model in ['bge','qwen']:
        first=[];raw_first=[];cut_ties=[];cjk=0;top20_count=0;groups={}
        for qid,g in frame.groupby('query_id'):
            raw_first.append(g.sort_values([model,'seed_rank'],ascending=[False,True],kind='stable').iloc[0].passage_tokens)
            clean=g[g.passage_tokens>=40].sort_values([model,'seed_rank'],ascending=[False,True],kind='stable')
            if clean.empty:raise ValueError('Query has no eligible passage')
            docs=clean.drop_duplicates('text_sha256');groups[int(qid)]=docs
            first.append(int(docs.iloc[0].passage_tokens));head=docs.head(20)
            threshold=head.iloc[-1][model]
            cut_ties.append(int((docs[model]==threshold).sum()))
            for text in head.chunk_text:
                letters=sum(c.isalpha() for c in text)
                chinese=len(re.findall(r'[\u3400-\u9fff]',text))
                cjk+=int(chinese>=20 and chinese/max(letters,1)>=.2)
                top20_count+=1
        ranked[model]=groups
        result['models'][model]=dict(top1_stub_queries=int(sum(n<40 for n in raw_first)),
            median_top1_bge_tokens=float(np.median(first)),top20_cjk_dominant_fraction=cjk/max(top20_count,1),
            cjk_definition='>=20 CJK characters and >=20% of alphabetic characters; heuristic, not corpus language metadata',
            mean_number_tied_at_top20_boundary=float(np.mean(cut_ties)))
    for depth in [1,20,80,200]:
        overlaps=[]
        for qid in ranked['bge']:
            left=set(ranked['bge'][qid].head(depth).text_sha256)
            right=set(ranked['qwen'][qid].head(depth).text_sha256)
            overlaps.append(len(left&right)/max(min(len(left),len(right)),1))
        result['overlap'][str(depth)]=float(np.mean(overlaps))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=diagnostics(pd.read_parquet(args.scores))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
