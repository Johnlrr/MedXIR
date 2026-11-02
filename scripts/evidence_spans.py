"""Contiguous source passages for reranking; offsets always refer to original chunk text.

These helpers do not change the frozen chunk corpus or embeddings.
"""
from bisect import bisect_left, bisect_right
import re


def join_overlapping(rows):
    """Return (original start, source text); reject unknown gaps/inconsistent overlaps."""
    if not rows:
        raise ValueError('No source rows')
    rows=sorted(rows,key=lambda r:r['start'])
    start=rows[0]['start'];text=rows[0]['text'];end=rows[0]['end']
    sha=rows[0]['sha']
    for row in rows:
        if row['sha']!=sha or row['end']-row['start']!=len(row['text']):
            raise ValueError('Invalid source identity or offsets')
    for row in rows[1:]:
        if row['start']>end:
            raise ValueError('Unknown source gap; cannot invent separator text')
        overlap=min(end,row['end'])-row['start']
        if text[row['start']-start:row['start']-start+overlap]!=row['text'][:overlap]:
            raise ValueError('Source chunks disagree in overlapping region')
        if row['end']>end:
            text+=row['text'][end-row['start']:];end=row['end']
    return start,text


def contiguous_around_anchor(rows, anchor):
    """Keep the maximal known contiguous component containing the seed.

    A distant gap must not discard the seed's valid left/right neighbours.
    Unknown characters across a gap are never reconstructed or guessed.
    """
    if not rows or not any(row == anchor for row in rows):
        raise ValueError('Anchor must be one of the source rows')
    components=[]
    for row in sorted(rows,key=lambda r:r['start']):
        if row['sha'] != anchor['sha']:
            raise ValueError('Invalid source identity')
        if not components or row['start'] > max(r['end'] for r in components[-1]):
            components.append([row])
        else:
            components[-1].append(row)
    # Validate all components, including those outside the selected region.
    joined=[join_overlapping(component) for component in components]
    for start,text in joined:
        if start <= anchor['start'] and start+len(text) >= anchor['end']:
            if text[anchor['start']-start:anchor['end']-start] != anchor['text']:
                raise ValueError('Anchor disagrees with source component')
            return start,text
    raise ValueError('Anchor not covered by source component')


def centered_span(text, offsets, anchor_start, anchor_end, max_tokens):
    """A token-budgeted, contiguous source slice around a retrieved seed's centre.

Use tokenizer offset_mapping, not decoded tokens (which can rewrite whitespace).
Snap inward to sentence boundaries only within 10% of the token budget.
"""
    if not text or not offsets or max_tokens<1 or not 0<=anchor_start<anchor_end<=len(text):
        raise ValueError('Invalid source/anchor/token budget')
    offsets=[(int(a),int(b)) for a,b in offsets if b>a]
    if not offsets:raise ValueError('No nonempty token offsets')
    if len(offsets)<=max_tokens:return 0,len(text),text
    midpoint=(anchor_start+anchor_end)//2
    centre=max(0,bisect_right([a for a,_ in offsets],midpoint)-1)
    lo=max(0,min(centre-max_tokens//2,len(offsets)-max_tokens));hi=lo+max_tokens
    begin=offsets[lo][0];end=offsets[hi-1][1]
    margin=max(1,max_tokens//10)
    left_limit=offsets[min(lo+margin,hi-1)][0]
    right_limit=offsets[max(lo,hi-margin)][0]
    boundaries=[m.end() for m in re.finditer(r'(?:[.!?。！？](?:\s+|(?=[\u3400-\u9fff]))|\n+)',text[begin:end])]
    left=[begin+x for x in boundaries if begin+x<=left_limit and begin+x<midpoint]
    right=[begin+x for x in boundaries if begin+x>=right_limit and begin+x>midpoint]
    if left:begin=left[0]
    if right:end=right[-1]
    assert 0<=begin<end<=len(text) and begin<=midpoint<=end
    return begin,end,text[begin:end]
