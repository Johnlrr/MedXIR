"""Qwen reranker inputs: preserve the classification suffix when truncating.

Matches the official Qwen3-Reranker Transformers recipe: reserve prompt tokens,
truncate the body, append the untouched prefix/suffix, and left-pad the batch.
"""

QWEN_RERANK_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
                     'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
QWEN_RERANK_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def qwen_reranker_inputs(tokenizer, queries, documents, instruction, max_length, return_tensors='pt'):
    if len(queries) != len(documents) or not queries:
        raise ValueError('Expected equal nonempty query/document batches')
    if tokenizer.padding_side != 'left':
        raise ValueError('Qwen last-position scoring requires left padding')
    prefix = tokenizer.encode(QWEN_RERANK_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(QWEN_RERANK_SUFFIX, add_special_tokens=False)
    available = max_length - len(prefix) - len(suffix)
    if available < 1:
        raise ValueError('Token limit leaves no room for query/document body')
    bodies = [f'<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}' for q,d in zip(queries,documents)]
    encoded = tokenizer(bodies, padding=False, truncation=True, add_special_tokens=False,
                        return_attention_mask=False, max_length=available)
    ids = [prefix + body + suffix for body in encoded['input_ids']]
    assert all(len(row) <= max_length and row[-len(suffix):] == suffix for row in ids)
    return tokenizer.pad({'input_ids': ids}, padding=True, return_tensors=return_tensors)
