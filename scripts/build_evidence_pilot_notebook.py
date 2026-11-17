"""Generate the GPU passage reranking pilot on the shared best ensemble pool."""
from pathlib import Path
import argparse
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slug',default='medxir-evidence-rerank-pilot2-20261031')
    args=parser.parse_args()
    files={n:(ROOT/'scripts'/n).read_text(encoding='utf8') for n in
           ['expand_chunks.py','reranker_inputs.py','evidence_spans.py','pilot_evidence_rerank.py']}
    code='FILES='+repr(files)+'''\nfrom pathlib import Path
import sys,subprocess
for name,source in FILES.items():Path('/kaggle/working',name).write_text(source,encoding='utf8')
subprocess.run([sys.executable,'-u','/kaggle/working/pilot_evidence_rerank.py'],check=True)
'''
    compile(code,'pilot','exec')
    nb=ROOT/'src/notebooks/kaggle_evidence_rerank_pilot.ipynb'
    nb.write_text(json.dumps(dict(cells=[dict(cell_type='code',id='pilot',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))],metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5),indent=1),encoding='utf8')
    slug=args.slug
    config=dict(id='ntklinhfitus/'+slug,title=slug,code_file=nb.relative_to(ROOT).as_posix(),kernel_type='notebook',
                immutable_kernel=True,enable_gpu=True,enable_internet=False,machine_shape='NvidiaRtxPro6000',
                expected_hardware_contains='6000',competition_sources=['arc-prize-2026-arc-agi-3'],
                dataset_sources=['ntklinhfitus/medxir-chunk-corpus-v2','boinhbo/medxir-docrerank-ens-v2','boinhbo/medxir-retrieval-inputs'],
                kernel_sources=['boinhbo/medxir-package-rerankers'],
                timeout_seconds=7200,poll_interval_seconds=30,metrics_file='passage_pilot_report.json',
                fetch_files=['passage_pilot_report.json','passage_review.json'],required_outputs=['passage_pilot_report.json'],
                json_checks=[dict(path='complete',value=True)])
    (ROOT/'configs/kaggle'/f'{slug}_rtx6000.yaml').write_text(json.dumps(dict(kernel=config),indent=2),encoding='utf8')
    print(nb)


if __name__=='__main__':main()
