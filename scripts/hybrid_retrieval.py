"""CPU-side logic of the hybrid retrieval notebook (scripts/build_hybrid_notebook.py copies this file into it).

Query variants: rule-based cleanup of patient chit-chat and a split of long queries into sentences and
comma clauses; no LLM. Every variant is searched on its own and the lists are fused with weighted RRF.
Document ranking: a document scores the max, or the sum of its 3 best chunk scores (DOC_AGG). Scores
only depend on the submitted sets, so the ranking matters through what falls inside N_DOCS / N_CHUNKS.

With one dense list, DOC_AGG='max' and no reranker, assemble() gives exactly the old d200-c150 lists
(scripts/retrieve_local_4b.py): docs in order of their best chunk, chunk docs appended when missing.
"""
import re
import unicodedata

CHITCHAT = re.compile(
    r'(?i)\b(?:'
    r'(?:xin )?chào (?:bác s[ĩỹi]|bs)[^,.?!]*[,.!]?'
    r'|(?:cho|xin cho) (?:em|tôi|mình|cháu|con|chị|anh) (?:được )?hỏi(?: (?:bác s[ĩỹi]|bs))?(?: là)?'
    r'|(?:em |tôi |mình )?xin hỏi'
    r'|bác s[ĩỹi] ơi'
    r'|(?:em |tôi |mình |xin )?(?:chân thành )?cảm ơn(?: (?:bác s[ĩỹi]|bs))?(?: nhiều)?(?: ạ)?'
    r'|mong (?:bác s[ĩỹi]|bs)[^.?!]*(?:tư vấn|giải đáp|trả lời|giúp)[^.?!]*'
    r'|nhờ (?:bác s[ĩỹi]|bs)[^.?!]*(?:tư vấn|giải đáp)[^.?!]*'
    r')')
SENTENCE = re.compile(r'(?<=[.?!;…])\s+|\n+')


def clean_query(text):
    """Drop greetings, thanks and 'cho em hỏi'-style openers; keep every medical word."""
    text = CHITCHAT.sub(' ', unicodedata.normalize('NFC', text))
    return re.sub(r'\s+', ' ', text).strip(' ,.')


def sub_queries(query, split_min_words=20, min_sub_words=4, max_subq=4):
    """Sentences of a long query, plus the comma clauses of a sentence when every clause has 3+ words.

    Short queries (most are one sentence of ~18 words) return []: they are searched whole only.
    """
    if len(query.split()) < split_min_words:
        return []
    found = []
    for sentence in SENTENCE.split(clean_query(query)):
        sentence = sentence.strip(' ,.;')
        if len(sentence.split()) < min_sub_words:
            continue
        found.append(sentence)
        clauses = [part.strip() for part in sentence.split(',')]
        if len(clauses) >= 2 and all(len(part.split()) >= 3 for part in clauses):
            found.extend(clauses)
    seen, out = {query.lower().strip()}, []
    for text in found:
        if text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out[:max_subq]


def rrf(lists, k=60):
    """Weighted reciprocal-rank fusion. lists: [(ranked ids, weight)]. Returns [(id, score)], best first.

    Ties keep the order of first appearance, so a single list comes back in its own order.
    """
    score, first = {}, {}
    for ids, weight in lists:
        for rank, cid in enumerate(ids):
            score[cid] = score.get(cid, 0.0) + weight / (k + rank + 1)
            first.setdefault(cid, len(first))
    return sorted(score.items(), key=lambda item: (-item[1], first[item[0]]))


def model_candidate_lists(results, weights, qi, sub_weight, prf_weight, trans_weight=0.0):
    """One RRF stage: model weights multiply full/sub/PRF/translation lists; BM25 is added once by caller."""
    lists = []
    for key, weight in weights.items():
        result = results[key]
        lists.append((list(result['full_ids'][qi]), weight))
        lists.extend((list(ids), weight * sub_weight) for ids in result['sub_lists'][qi])
        if trans_weight and result.get('trans_lists'):
            lists.extend((list(ids), weight * trans_weight) for ids in result['trans_lists'][qi])
        if result['prf_ids'] is not None:
            lists.append((list(result['prf_ids'][qi]), weight * prf_weight))
    return lists


def doc_ranking(scored, docs_of, agg='max'):
    """Documents of the scored chunks ([(chunk id, score)], best first), best document first.

    agg='max': a document scores its best chunk (the old ranking). agg='top3': the sum of its 3 best chunks,
    so a document with several matching chunks beats one with a single lucky match.
    """
    per_doc, first = {}, {}
    for cid, score in scored:
        for doc_id in docs_of(cid):
            per_doc.setdefault(doc_id, []).append(score)
            first.setdefault(doc_id, len(first))
    if agg == 'max':
        value = {doc: max(scores) for doc, scores in per_doc.items()}
    elif agg == 'top3':
        value = {doc: sum(sorted(scores, reverse=True)[:3]) for doc, scores in per_doc.items()}
    else:
        raise ValueError(f'unknown DOC_AGG {agg!r}')
    return sorted(per_doc, key=lambda doc: (-value[doc], first[doc]))


def assemble(scored, fill_ids, docs_of, n_docs, n_chunks, agg='max'):
    """One query's (relevant_docs, [(doc_id, chunk id)]).

    scored: [(chunk id, score)] best first, the chunks with a final score (the reranked pool, or all fused
    candidates). fill_ids: further candidate chunk ids in fused order, used only to top the document list
    up to n_docs. Chunk documents missing from the list are appended, as in the earlier submissions.
    """
    docs = doc_ranking(scored, docs_of, agg)[:n_docs]
    listed = set(docs)
    for cid in fill_ids:
        if len(docs) >= n_docs:
            break
        for doc_id in docs_of(cid):
            if doc_id not in listed and len(docs) < n_docs:
                listed.add(doc_id)
                docs.append(doc_id)
    chunks = []
    for cid, _ in scored[:n_chunks]:
        for doc_id in docs_of(cid):
            chunks.append((doc_id, cid))
            if doc_id not in listed:
                listed.add(doc_id)
                docs.append(doc_id)
    return docs, chunks
