import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_precision_controls import select, export, digest
from audit_precision_scores import normalize


class PrecisionControls(unittest.TestCase):
    def test_normalization_and_invalid_scores(self):
        self.assertEqual(normalize([2, 4, 6]).tolist(), [0, .5, 1])
        self.assertEqual(normalize([2, 2]).tolist(), [0, 0])
        with self.assertRaises(ValueError):
            normalize([0, float('nan')])

    def test_protect_chunks_and_preserve_source(self):
        entry = dict(id=1, relevant_docs=[1, 2, 3], relevant_chunks=[dict(doc_id=3, chunk_text='abc')])
        self.assertEqual(select(entry, 1)['relevant_docs'], [1, 3])
        self.assertEqual(entry['relevant_docs'], [1, 2, 3])

    def test_keep_fanout_beyond_cutoff(self):
        entry = dict(id=1, relevant_docs=[1, 2, 3], relevant_chunks=[
            dict(doc_id=1, chunk_text='same'), dict(doc_id=2, chunk_text='other'), dict(doc_id=3, chunk_text='same')])
        result = select(entry, chunk_cap=1)
        self.assertEqual([c['doc_id'] for c in result['relevant_chunks']], [1, 3])
        self.assertEqual(result['relevant_docs'], entry['relevant_docs'])

    def test_reject_orphan(self):
        with self.assertRaises(ValueError):
            select(dict(id=1, relevant_docs=[1], relevant_chunks=[dict(doc_id=2, chunk_text='x')]))

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source.zip'
            entry = dict(id=1, relevant_docs=list(range(180)), relevant_chunks=[dict(doc_id=i, chunk_text=f'Text {i}') for i in range(100)])
            with zipfile.ZipFile(source, 'w') as z:
                z.writestr('submission.json', json.dumps([entry]))
            before = digest(source)
            result = export(source, root/'out', expected_ids=[1])
            self.assertEqual(before, digest(source))
            self.assertEqual(result['variants']['d150-protected']['docs_per_query'], 150)
            self.assertEqual(result['variants']['c80-prefix']['chunks_per_query'], 80)
            for item in result['variants'].values():
                with zipfile.ZipFile(item['path']) as z:
                    self.assertEqual(z.infolist()[0].compress_type, zipfile.ZIP_BZIP2)
            with self.assertRaises(FileExistsError):
                export(source, root/'out', expected_ids=[1])
