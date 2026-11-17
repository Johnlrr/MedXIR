"""Translate the 1,200 Vietnamese queries to English and Simplified Chinese with an open instruction LLM, or
rewrite them as short Vietnamese title-style questions (task 'vq', for title search).

The translations are extra query variants for the hybrid notebook (USE_TRANSLATION): dense search in the
document's own language, and same-language reranking of EN/ZH chunks (bge-reranker-v2-m3 is weaker across
languages). Greedy decoding, one prompt per (query, language), medical terms / numbers / units kept, no answer.

Kaggle GPU job with Internet ON (the model is downloaded from Hugging Face). Input: query.parquet under
INPUT_ROOT. Output: OUT_DIR/query_translations.json = {model, revision, items: [{id, en, zh}]}.
"""
import glob
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
BATCH = 32
MAX_NEW_TOKENS = 512
INPUT = os.environ.get('INPUT_ROOT', '/kaggle/input')
OUT = Path(os.environ.get('OUT_DIR', '/kaggle/working'))
TRANSLATE = ('Translate the following Vietnamese medical question into {language}. Keep every disease, symptom, drug '
             'name, test name, number and unit exact. Do not answer the question and do not add explanations. '
             'Output only the translation.\n\n{query}')
# 'vq': the query as one short Vietnamese question in the style of a health Q&A article title (the criteria of the
# 'question' variant of build_variants_notebook.py), for title search.
REWRITE = ('Viết lại câu hỏi sau thành MỘT câu hỏi tiếng Việt có dấu, đúng chính tả, tối đa 30 từ, giống tiêu đề '
           'một bài hỏi đáp sức khỏe. Giữ triệu chứng, vị trí, thời gian, tuổi, giới tính, bệnh đã được chẩn đoán, '
           'thuốc và xét nghiệm đã nêu. Dùng thuật ngữ y khoa chuẩn thay cho cách nói thông thường. Bỏ lời chào, '
           'lời cảm ơn và phần kể chuyện không cần cho chuyên môn. Không thêm thông tin mới, không trả lời. '
           'Chỉ in ra câu hỏi.\n\n{query}')
TASKS = {   # output key -> (system message, user prompt)
    'en': ('You are a professional medical translator.', TRANSLATE.replace('{language}', 'English')),
    'zh': ('You are a professional medical translator.', TRANSLATE.replace('{language}', 'Simplified Chinese')),
    'vq': ('Bạn là bác sĩ biên tập câu hỏi y khoa.', REWRITE),
}


def prompts(tok, texts, task):
    system, prompt = TASKS[task]
    return [tok.apply_chat_template([{'role': 'system', 'content': system},
                                     {'role': 'user', 'content': prompt.format(query=t)}],
                                    tokenize=False, add_generation_prompt=True) for t in texts]


def clean(text):
    text = text.strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in '"\'“”':
        text = text[1:-1].strip()
    return text


def main(tasks=('en', 'zh'), out_name='query_translations.json'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started = time.perf_counter()
    hits = sorted(glob.glob(f'{INPUT}/**/query.parquet', recursive=True))
    assert len(hits) == 1, hits
    queries = pd.read_parquet(hits[0], columns=['id', 'query'])
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map='auto').eval()
    texts = queries['query'].astype(str).tolist()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # similar lengths per batch
    out = {code: [None] * len(texts) for code in tasks}
    for code in tasks:
        for start in range(0, len(order), BATCH):
            index = order[start:start + BATCH]
            enc = tok(prompts(tok, [texts[i] for i in index], code), return_tensors='pt', padding=True).to(model.device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
            for i, ids in zip(index, gen[:, enc['input_ids'].shape[1]:]):
                out[code][i] = clean(tok.decode(ids, skip_special_tokens=True))
            print(f'{code} {start + len(index)}/{len(order)} {time.perf_counter() - started:.0f}s', flush=True)
    items = [dict(id=int(qid), vi=t, **{code: out[code][i] for code in tasks})
             for i, (qid, t) in enumerate(zip(queries['id'], texts))]
    empty = {code: sum(not item[code] for item in items) for code in tasks}
    revision = getattr(model.config, '_commit_hash', None)
    (OUT / out_name).write_text(json.dumps(dict(model=MODEL, revision=revision, items=items),
                                                            ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(model=MODEL, revision=revision, queries=len(items), empty=empty,
                          seconds=round(time.perf_counter() - started, 1)), ensure_ascii=False), flush=True)
    for item in items[:3]:
        print(json.dumps(item, ensure_ascii=False), flush=True)
