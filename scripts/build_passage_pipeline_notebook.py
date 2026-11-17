"""Generate CPU preparation / GPU scoring notebooks for source-exact passages."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','bge','qwen'])
    args=parser.parse_args();stage=args.stage
    if stage=='prepare':
        names=['audit_submission_coverage.py','evidence_spans.py','passage_candidates.py','prepare_passage_corpus.py']
        runner='prepare_passage_corpus.py';arguments=[]
    else:
        names=['reranker_inputs.py','score_passage_corpus.py','assemble_passage_submission.py']
        runner='score_passage_corpus.py';arguments=['--model',stage]
    files={name:(ROOT/'scripts'/name).read_text(encoding='utf8') for name in names}
    code='FILES='+repr(files)+'\nARGS='+repr(arguments)+'''\nfrom pathlib import Path
import sys,subprocess
for name,source in FILES.items():Path('/kaggle/working',name).write_text(source,encoding='utf8')
subprocess.run([sys.executable,'-u','/kaggle/working/RUNNER',*ARGS],check=True)
'''.replace('RUNNER',runner)
    compile(code,'passage-'+stage,'exec')
    nb=ROOT/'src/notebooks'/f'kaggle_passage_{stage}.ipynb'
    nb.write_text(json.dumps(dict(cells=[dict(cell_type='code',id='passage-'+stage,metadata={},execution_count=None,
        outputs=[],source=code.splitlines(True))],metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        nbformat=4,nbformat_minor=5),indent=1),encoding='utf8')
    slug=f'medxir-passages-{stage}-20261031'
    config=dict(id='ntklinhfitus/'+slug,title=slug,code_file=nb.relative_to(ROOT).as_posix(),
        kernel_type='notebook',immutable_kernel=True,enable_gpu=stage!='prepare',enable_internet=False,
        kernel_sources=['boinhbo/medxir-package-rerankers'],timeout_seconds=36000,poll_interval_seconds=30)
    if stage=='prepare':
        config.update(dataset_sources=['ntklinhfitus/medxir-chunk-corpus-v2','boinhbo/medxir-docrerank-ens-v2',
            'boinhbo/medxir-retrieval-inputs'],fetch_files=['passage_candidates_manifest.json'],
            required_outputs=['passage_candidates_manifest.json'],metrics_file='passage_candidates_manifest.json',
            json_checks=[dict(path='complete',value=True),dict(path='queries',value=1200)])
    else:
        config['kernel_sources'].append('ntklinhfitus/medxir-passages-prepare-20261031')
        config.update(machine_shape='NvidiaRtxPro6000',expected_hardware_contains='6000',
            competition_sources=['arc-prize-2026-arc-agi-3'],
            fetch_files=[f'passage_scores_{stage}_manifest.json',f'passage_{stage}_submission_manifest.json',f'medxir-passage-{stage}-d200.zip'],
            required_outputs=[f'passage_scores_{stage}_manifest.json',f'passage_{stage}_submission_manifest.json',f'medxir-passage-{stage}-d200.zip'],
            metrics_file=f'passage_scores_{stage}_manifest.json',json_checks=[dict(path='complete',value=True)])
    (ROOT/'configs/kaggle'/f'passage_{stage}_20261031.yaml').write_text(json.dumps(dict(kernel=config),indent=2),encoding='utf8')
    print(nb)


if __name__=='__main__':main()
