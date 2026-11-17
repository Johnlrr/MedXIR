"""Generate src/notebooks/kaggle_window_rerank_s<i>.ipynb: Qwen3-Reranker-8B scores candidate chunk windows.

The 0.2984 submission picks its 131 windows per query by the fused chunk score of the window's centre chunk. This
kernel scores the window text itself (the thing that is submitted) for the 180 candidate windows per query that
fuse_scores.py --export-windows writes; fuse_scores.py --window-scores then re-picks 131 of them.

Input: one dataset with windows.parquet (query_id, center_id, window_rank, fused_score, text, query) and a packaged
Qwen3-Reranker-8B (kernel source). Output: window_scores.parquet (query_id, center_id, score). Queries are sharded
(QUERY_SHARD = (i, n): queries i, i+n, ... in sorted id order) so several accounts run in parallel.
Prompt and truncation follow the Qwen3-Reranker reference: only the window text is cut, the assistant suffix is kept.

    uv run python scripts/build_window_rerank_notebook.py --shard 0 3
"""
import argparse
import json
from pathlib import Path

from build_retrieval_notebook import CODE_HELPERS, CODE_REPORT, guarded, notebook

CONFIG = """
MODEL_DIR = 'Qwen3-Reranker-8B'
QUERY_SHARD = @SHARD@     # (i, n): this run scores queries i, i+n, ... of the sorted query ids
MAX_TOKENS = 1280         # windows are ~680 words (p90 ~1,000): long ones lose their end, never the prompt suffix
BATCH = 8
INSTRUCT = 'Given a Vietnamese medical question, judge whether the passage answers the question'
"""

SETUP = """
import glob
import subprocess

import numpy as np
import pandas as pd
import torch


def one_match(pattern, description):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))
    assert len(hits) == 1, f'{description}: expected one match, found {hits[:10]}'
    return Path(hits[0])


MODEL_PATH = one_match(f'{MODEL_DIR}/config.json', 'packaged reranker').parent
windows = pd.read_parquet(one_match('windows.parquet', 'candidate windows'))
qids = sorted(windows['query_id'].unique())
mine = set(qids[QUERY_SHARD[0]::QUERY_SHARD[1]])
windows = windows[windows['query_id'].isin(mine)].reset_index(drop=True)
print(subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv'],
                     capture_output=True, text=True).stdout)
gpu_name = torch.cuda.get_device_name(0)
assert '6000' in gpu_name, f'Expected RTX Pro 6000, got {gpu_name!r}'
note(f'GPU {gpu_name}; {len(mine)} queries, {len(windows):,} windows')
"""

RERANK = """
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), padding_side='left')
model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), torch_dtype=torch.bfloat16).cuda().eval()
yes, no = tok.convert_tokens_to_ids('yes'), tok.convert_tokens_to_ids('no')
prefix = ('<|im_start|>system\\nJudge whether the Document meets the requirements based on the Query and the Instruct '
          'provided. Note that the answer can only be "yes" or "no".<|im_end|>\\n<|im_start|>user\\n')
suffix_ids = tok('<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n', add_special_tokens=False)['input_ids']
heads = tok([f'{prefix}<Instruct>: {INSTRUCT}\\n<Query>: {q}\\n<Document>: ' for q in windows['query']],
            add_special_tokens=False)['input_ids']
bodies = tok(windows['text'].tolist(), add_special_tokens=False)['input_ids']
seqs = [h + b[:max(MAX_TOKENS - len(h) - len(suffix_ids), 0)] + suffix_ids for h, b in zip(heads, bodies)]
note(f'tokens per pair: median {int(np.median([len(x) for x in seqs]))}; cut windows '
     f'{np.mean([len(h) + len(b) + len(suffix_ids) > MAX_TOKENS for h, b in zip(heads, bodies)]):.1%}')
order = np.argsort([len(x) for x in seqs], kind='stable')
scores = np.empty(len(seqs), dtype=np.float32)
t0, next_report = time.perf_counter(), 0.05
with torch.inference_mode():
    for start in range(0, len(order), BATCH):
        index = order[start:start + BATCH]
        enc = tok.pad({'input_ids': [seqs[i] for i in index]}, padding=True, return_tensors='pt').to('cuda')
        hidden = model.model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).last_hidden_state
        logits = model.lm_head(hidden[:, -1, :])
        scores[index] = torch.softmax(torch.stack([logits[:, no], logits[:, yes]], dim=1).float(), dim=1)[:, 1].cpu().numpy()
        done = (start + len(index)) / len(order)
        if done >= next_report:
            rate = (start + len(index)) / (time.perf_counter() - t0)
            note(f'{done:.0%} | {rate:.1f} pairs/s | ETA {(len(order) - start) / rate / 60:.0f} min')
            next_report += 0.05
assert np.isfinite(scores).all()
windows[['query_id', 'center_id']].assign(score=scores).to_parquet(OUT / 'window_scores.parquet', index=False)
note(f'window_scores.parquet: {len(scores):,} rows; score median {float(np.median(scores)):.3f}')
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shard', type=int, nargs=2, default=(0, 1), metavar=('I', 'N'))
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    cells = [('md', '# MedXIR: Qwen3-Reranker-8B over candidate chunk windows\n\nGenerated by '
                    '`scripts/build_window_rerank_notebook.py`. RTX Pro 6000, Internet OFF.'),
             ('code', CONFIG.replace('@SHARD@', repr(tuple(args.shard)))),
             ('code', CODE_HELPERS),
             ('code', guarded('setup', SETUP)),
             ('code', guarded('rerank', RERANK)),
             ('code', CODE_REPORT)]
    output = args.output or Path(f'src/notebooks/kaggle_window_rerank_s{args.shard[0]}.ipynb')
    output.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{output}: {len(cells)} cells')


if __name__ == '__main__':
    main()
