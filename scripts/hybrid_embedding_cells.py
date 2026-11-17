"""Embedding/search cells used by build_hybrid_notebook.py. Edit here and regenerate."""

ENCODE = r'''
started = time.perf_counter()
QUERY_VECTORS = {}
for model_key in ACTIVE_MODEL_WEIGHTS:
    spec = EMBEDDING_SPECS[model_key]
    metadata = dict(model=spec['model'], model_key=model_key, task=TASK, variants=VARIANTS,
                    query_ids=QUERY_IDS, model_revision=MODEL_REVISIONS[model_key],
                    query_format='Instruct: {task}\nQuery:{text}', max_length=1024)
    if ENCODE_HERE:
        from transformers import AutoModel, AutoTokenizer
        import torch.nn.functional as F
        os.environ['HF_HUB_OFFLINE'] = '1'
        tok = AutoTokenizer.from_pretrained(str(MODEL_PATHS[model_key]), local_files_only=True)
        tok.padding_side = 'left'
        model = AutoModel.from_pretrained(str(MODEL_PATHS[model_key]), torch_dtype=torch.bfloat16,
                                         attn_implementation='sdpa', local_files_only=True).cuda().eval()
        assert model.config.hidden_size == spec['dim']
        note(f'{model_key}: query encoder loaded')

        @torch.inference_mode()
        def encode(texts, batch=16):
            tokenized = tok(texts, truncation=True, max_length=1024)['input_ids']
            order = np.argsort([-len(row) for row in tokenized], kind='stable')
            vectors = np.empty((len(texts), spec['dim']), dtype=np.float32)
            for start in range(0, len(order), batch):
                index = order[start:start + batch]
                enc = tok.pad({'input_ids': [tokenized[i] for i in index]}, return_tensors='pt').to('cuda')
                hidden = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], use_cache=False).last_hidden_state
                mask = enc['attention_mask']
                if mask[:, -1].sum() == mask.shape[0]:
                    pooled = hidden[:, -1]
                else:
                    lengths = mask.sum(dim=1) - 1
                    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
                vectors[index] = F.normalize(pooled.float(), p=2, dim=1).cpu().numpy()
            return vectors

        vectors = encode([f'Instruct: {TASK}\nQuery:{v["text"]}' for v in VARIANTS])
        del model, tok
        torch.cuda.empty_cache()
    else:
        cache_hits = sorted(glob.glob(f'/kaggle/input/**/query_variants_{model_key}.json', recursive=True))
        legacy = False
        if not cache_hits and model_key == 'qwen4b':
            cache_path = one_match('/kaggle/input/**/query_variants.json', 'legacy 4B query variants')
            vector_path = cache_path.with_name('query_variant_vectors.npy')
            legacy = True
        else:
            assert len(cache_hits) == 1, f'{model_key}: attach exactly one matching query variant cache, found {cache_hits}'
            cache_path = Path(cache_hits[0])
            vector_path = cache_path.with_name(f'query_variant_vectors_{model_key}.npy')
        saved = json.loads(cache_path.read_text(encoding='utf-8'))
        vectors = np.load(vector_path, allow_pickle=False)
        validate_query_cache(saved, vectors, model_key, VARIANTS, TASK, QUERY_IDS, legacy=legacy,
                             revision=MODEL_REVISIONS[model_key])
    validate_query_cache(metadata, vectors, model_key, VARIANTS, TASK, QUERY_IDS,
                         revision=MODEL_REVISIONS[model_key])
    QUERY_VECTORS[model_key] = vectors
    np.save(OUT / f'query_variant_vectors_{model_key}.npy', vectors)
    (OUT / f'query_variants_{model_key}.json').write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
    if list(ACTIVE_MODEL_WEIGHTS) == ['qwen4b']:
        np.save(OUT / 'query_variant_vectors.npy', vectors)  # legacy output name
    note(f'{model_key}: {len(vectors):,} query variants ready, dim={spec["dim"]}')
TIMINGS['encode_s'] = round(time.perf_counter() - started, 1)
'''

LOAD_VECTORS = r'''
# Load/search/release one embedding model at a time, so fusion does not require all three matrices resident.
READ_ROWS = 32768
ALL_IDS = None
TIMINGS['load_vectors_s'] = 0.0


def load_vector_parts(model_key):
    global ALL_IDS
    started = time.perf_counter()
    paths = MODEL_VECTOR_PATHS[model_key]
    dim = EMBEDDING_SPECS[model_key]['dim']
    parts, ids = [], []
    read_ids = ALL_IDS is None
    for key in sorted(paths):
        t0 = time.perf_counter()
        array = np.load(paths[key], mmap_mode='r', allow_pickle=False)
        assert array.dtype == np.float16 and array.shape[1] == dim
        gpu = torch.empty(array.shape, dtype=torch.float16, device='cuda')
        for start in range(0, array.shape[0], READ_ROWS):
            gpu[start:start + READ_ROWS] = torch.from_numpy(np.array(array[start:start + READ_ROWS])).cuda()
        del array
        if read_ids:
            ids_path = Path(paths[key]).with_name(f'{key}-ids.parquet')
            ids.append(pq.read_table(ids_path, columns=['chunk_id'])['chunk_id'].to_numpy())
        parts.append(gpu)
        note(f'{model_key}/{key}: {gpu.shape[0]:,} rows on GPU in {time.perf_counter() - t0:.0f}s')
    offsets = np.cumsum([0] + [p.shape[0] for p in parts])
    if read_ids:
        ALL_IDS = np.concatenate(ids)
    assert len(ALL_IDS) == offsets[-1]
    TIMINGS['load_vectors_s'] += round(time.perf_counter() - started, 1)
    return parts, offsets
'''

SEARCH = r'''
torch.backends.cuda.matmul.allow_tf32 = SEARCH_TF32
ROW_BLOCK, QUERY_BATCH = 262144, 1024


@torch.inference_mode()
def search(query_vectors, k):
    k = min(k, len(ALL_IDS))
    assert k > 0
    q = torch.from_numpy(np.ascontiguousarray(query_vectors, dtype=np.float32)).cuda()
    best_s = torch.full((len(q), k), -float('inf'), device='cuda')
    best_i = torch.full((len(q), k), -1, dtype=torch.int64, device='cuda')
    for part, offset in zip(PARTS, OFFSETS[:-1]):
        for row in range(0, part.shape[0], ROW_BLOCK):
            block = part[row:row + ROW_BLOCK].float()
            for qs in range(0, len(q), QUERY_BATCH):
                scores = q[qs:qs + QUERY_BATCH] @ block.T
                top_s, top_i = torch.topk(scores, min(k, scores.shape[1]), dim=1)
                cat_s = torch.cat([best_s[qs:qs + QUERY_BATCH], top_s], dim=1)
                cat_i = torch.cat([best_i[qs:qs + QUERY_BATCH], top_i + int(offset) + row], dim=1)
                keep_s, order = torch.topk(cat_s, k, dim=1)
                best_s[qs:qs + QUERY_BATCH] = keep_s
                best_i[qs:qs + QUERY_BATCH] = torch.gather(cat_i, 1, order)
            del block
    result_s, result_i = best_s.cpu().numpy(), best_i.cpu().numpy()
    assert (result_i >= 0).all() and np.isfinite(result_s).all()
    return result_s, result_i


def gather_vectors(rows):
    flat = np.asarray(rows).ravel()
    part_of = np.searchsorted(OFFSETS, flat, side='right') - 1
    out = torch.empty((len(flat), DIM), dtype=torch.float32, device='cuda')
    for p in np.unique(part_of):
        mask = np.flatnonzero(part_of == p)
        local = torch.from_numpy(flat[mask] - OFFSETS[p]).cuda()
        out[torch.from_numpy(mask).cuda()] = PARTS[p].index_select(0, local).float()
    return out.view(*np.shape(rows), DIM)


started = time.perf_counter()
kinds = np.array([v['kind'] for v in VARIANTS])
owner = np.array([v['qi'] for v in VARIANTS])
FULL = np.flatnonzero(kinds == 'full')
assert (owner[FULL] == np.arange(len(QUERY_TEXT))).all()
MODEL_RESULTS = {}
for model_key in ACTIVE_MODEL_WEIGHTS:
    t0 = time.perf_counter()
    DIM = EMBEDDING_SPECS[model_key]['dim']
    PARTS, OFFSETS = load_vector_parts(model_key)
    variant_vectors = QUERY_VECTORS[model_key]
    d_scores, d_rows = search(variant_vectors, DENSE_K)
    subs = [[] for _ in QUERY_TEXT]
    for i in np.flatnonzero(kinds == 'sub'):
        subs[owner[i]].append(ALL_IDS[d_rows[i, :SUB_K]])
    trans = [[] for _ in QUERY_TEXT]   # EN / ZH translations of the query (USE_TRANSLATION)
    for i in np.flatnonzero(kinds == 'trans'):
        trans[owner[i]].append(ALL_IDS[d_rows[i, :TRANS_K]])
    prf_ids = None
    if USE_PRF:
        with torch.inference_mode():
            feedback = gather_vectors(d_rows[FULL, :PRF_K]).mean(dim=1)
            q = torch.from_numpy(variant_vectors[FULL]).cuda()
            prf = torch.nn.functional.normalize((1 - PRF_ALPHA) * q + PRF_ALPHA * feedback, p=2, dim=1).cpu().numpy()
        _, prf_rows = search(prf, PRF_KEEP)
        prf_ids = ALL_IDS[prf_rows]
        del feedback, q, prf, prf_rows
    MODEL_RESULTS[model_key] = dict(full_ids=ALL_IDS[d_rows[FULL]], full_scores=d_scores[FULL],
                                   sub_lists=subs, trans_lists=trans, prf_ids=prf_ids)
    del PARTS, variant_vectors, d_scores, d_rows
    torch.cuda.empty_cache()
    note(f'{model_key}: full/sub/PRF retrieval complete in {time.perf_counter() - t0:.0f}s; GPU vectors released')

if len(ACTIVE_MODEL_WEIGHTS) == 1:
    result = MODEL_RESULTS[next(iter(ACTIVE_MODEL_WEIGHTS))]
    DENSE_FULL_IDS, DENSE_FULL_SCORES = result['full_ids'], result['full_scores']
else:
    # Reference dense ranking for overlap/export; final hybrid fusion adds every model's lists once.
    dense_fused = [rrf([(list(MODEL_RESULTS[key]['full_ids'][qi]), weight)
                        for key, weight in ACTIVE_MODEL_WEIGHTS.items()], RRF_K)[:DENSE_K]
                   for qi in range(len(QUERY_TEXT))]
    DENSE_FULL_IDS = np.array([[cid for cid, _ in row] for row in dense_fused], dtype=object)
    DENSE_FULL_SCORES = np.array([[score for _, score in row] for row in dense_fused], dtype=np.float32)
TIMINGS['search_s'] = round(time.perf_counter() - started - TIMINGS['load_vectors_s'], 1)
'''
