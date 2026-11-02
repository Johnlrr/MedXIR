import unittest
from scripts.reranker_inputs import qwen_reranker_inputs, QWEN_RERANK_PREFIX, QWEN_RERANK_SUFFIX


class CharacterTokenizer:
    padding_side = 'left'

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, texts, max_length, **kwargs):
        return {'input_ids':[self.encode(t)[:max_length] for t in texts]}

    def pad(self, encoded, **kwargs):
        n=max(map(len,encoded['input_ids']))
        return {'input_ids':[[0]*(n-len(row))+row for row in encoded['input_ids']]}


class PromptTruncation(unittest.TestCase):
    def test_long_body_keeps_suffix_and_length(self):
        tok=CharacterTokenizer()
        rows=qwen_reranker_inputs(tok,['query','query'],['x'*4000,'short'],'task',512)['input_ids']
        suffix=tok.encode(QWEN_RERANK_SUFFIX)
        self.assertEqual(len(rows[0]),512)
        for row in rows:self.assertEqual(row[-len(suffix):],suffix)
        self.assertEqual(rows[1][0],0)
        legacy=tok.encode(QWEN_RERANK_PREFIX+'x'*4000+QWEN_RERANK_SUFFIX)[:512]
        self.assertNotEqual(legacy[-len(suffix):],suffix)

    def test_short_body_is_exact_prompt(self):
        tok=CharacterTokenizer()
        actual=qwen_reranker_inputs(tok,['q'],['d'],'task',1024)['input_ids'][0]
        expected=tok.encode(QWEN_RERANK_PREFIX+'<Instruct>: task\n<Query>: q\n<Document>: d'+QWEN_RERANK_SUFFIX)
        self.assertEqual(actual,expected)

    def test_reject_unsafe_batch_or_cap(self):
        tok=CharacterTokenizer()
        with self.assertRaises(ValueError):qwen_reranker_inputs(tok,['q'],[],'i',512)
        with self.assertRaises(ValueError):qwen_reranker_inputs(tok,['q'],['d'],'i',2)
        tok.padding_side='right'
        with self.assertRaises(ValueError):qwen_reranker_inputs(tok,['q'],['d'],'i',512)


if __name__=='__main__':unittest.main()
