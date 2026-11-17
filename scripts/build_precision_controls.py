"""Bounded-memory controls from a scored ZIP; selection order is NOT a score.

Preserve text and fanout groups. No gold metrics are estimated. Document controls
protect all retained chunk document IDs, so their actual size can exceed the cap.
"""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import zipfile

from audit_submission_coverage import iter_queries


SPECS = {'d150-protected': (150, None), 'c80-prefix': (None, 80)}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def select(entry, doc_cap=None, chunk_cap=None):
    docs = entry['relevant_docs']
    chunks = entry['relevant_chunks']
    keys = [str(d) for d in docs]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate document ID')
    if any(str(c['doc_id']) not in set(keys) for c in chunks):
        raise ValueError('Source has orphan chunk')
    if chunk_cap is not None:
        # Keep every copy of a selected text across doc IDs. A cutoff must not
        # arbitrarily choose one doc ID from a safe fanout group.
        selected_texts = {c['chunk_text'] for c in chunks[:chunk_cap]}
        chunks = [c for c in chunks if c['chunk_text'] in selected_texts]
    if doc_cap is not None:
        protected = {str(c['doc_id']) for c in chunks}
        keep = set(keys[:doc_cap]) | protected
        docs = [d for d in docs if str(d) in keep]
    return dict(entry, relevant_docs=docs, relevant_chunks=chunks)


def export(source, output, expected_ids=range(1, 1201)):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'precision_controls_manifest.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    report = dict(source=str(source.resolve()), source_sha256=digest(source),
                  warning='Unscored controls, not confidence-filtered candidates. Prefix order is not verified score order.',
                  variants={}, queries=[])
    paths = {name: output / f'medxir-base02916-{name}.zip' for name in SPECS}
    for path in paths.values():
        if path.exists() or path.with_suffix('.tmp').exists():
            raise FileExistsError(path)
    totals = {name: dict(docs=0, chunks=0, protected_extra=0) for name in SPECS}
    hashes = {name: hashlib.sha256() for name in SPECS}
    seen = set()
    with ExitStack() as stack:
        writers = {}
        for name, path in paths.items():
            archive = stack.enter_context(zipfile.ZipFile(path.with_suffix('.tmp'), 'w', compression=zipfile.ZIP_BZIP2, compresslevel=9))
            writers[name] = stack.enter_context(archive.open('submission.json', 'w'))
            writers[name].write(b'[')
        archive = stack.enter_context(zipfile.ZipFile(source))
        members = [m for m in archive.infolist() if not m.is_dir()]
        if len(members) != 1 or not members[0].filename.endswith('.json'):
            raise ValueError('Expected one JSON member')
        stream = stack.enter_context(io.TextIOWrapper(archive.open(members[0]), encoding='utf-8-sig'))
        for entry in iter_queries(stream):
            qid = int(entry['id'])
            if qid in seen:
                raise ValueError('Duplicate query')
            row = dict(id=qid, source_docs=len(entry['relevant_docs']), source_chunks=len(entry['relevant_chunks']))
            for name, (dcap, ccap) in SPECS.items():
                result = select(entry, dcap, ccap)
                if dcap is not None and result['relevant_chunks'] != entry['relevant_chunks']:
                    raise AssertionError('Document control changed chunks')
                if ccap is not None and result['relevant_docs'] != entry['relevant_docs']:
                    raise AssertionError('Chunk control changed documents')
                payload = json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode('utf8')
                hashes[name].update(payload)
                if seen:
                    writers[name].write(b',')
                writers[name].write(payload)
                nd, nc = len(result['relevant_docs']), len(result['relevant_chunks'])
                totals[name]['docs'] += nd
                totals[name]['chunks'] += nc
                totals[name]['protected_extra'] += max(0, nd-dcap) if dcap else 0
                row[name] = dict(docs=nd, chunks=nc)
            report['queries'].append(row)
            seen.add(qid)
            if len(seen) % 200 == 0:
                print(f'Exported {len(seen)} queries', flush=True)
        if seen != set(expected_ids):
            raise ValueError('Unexpected query IDs')
        for writer in writers.values():
            writer.write(b']')
    for name, path in paths.items():
        tmp = path.with_suffix('.tmp')
        if tmp.stat().st_size >= 99_000_000:
            raise ValueError('ZIP exceeds 99 MB')
        actual_hash = hashlib.sha256()
        with zipfile.ZipFile(tmp) as archive:
            with io.TextIOWrapper(archive.open('submission.json'), encoding='utf8') as stream:
                for entry in iter_queries(stream):
                    actual_hash.update(json.dumps(entry, ensure_ascii=False, separators=(',', ':')).encode('utf8'))
        if actual_hash.digest() != hashes[name].digest():
            raise ValueError('Readback mismatch')
        tmp.replace(path)
        report['variants'][name] = dict(path=str(path.resolve()), sha256=digest(path), bytes=path.stat().st_size,
            queries=len(seen), docs_per_query=totals[name]['docs']/len(seen),
            chunks_per_query=totals[name]['chunks']/len(seen),
            protected_extra_docs_per_query=totals[name]['protected_extra']/len(seen),
            payload_sha256=hashes[name].hexdigest())
    if digest(source) != report['source_sha256']:
        raise ValueError('Source changed during export')
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(export(args.source, args.output)['variants'], indent=2))
