"""Generate a bounded CPU diagnostic over the best ensemble's candidates."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    files={n:(ROOT/'scripts'/n).read_text(encoding='utf8') for n in ['expand_chunks.py','audit_retrieval_inputs.py']}
    code='FILES='+repr(files)+'''\nfrom pathlib import Path
import sys,subprocess
for name,source in FILES.items():Path('/kaggle/working',name).write_text(source,encoding='utf8')
subprocess.run([sys.executable,'-u','/kaggle/working/audit_retrieval_inputs.py'],check=True)
'''
    compile(code,'audit','exec')
    nb=ROOT/'src/notebooks/kaggle_retrieval_input_audit.ipynb'
    nb.write_text(json.dumps(dict(cells=[dict(cell_type='code',id='audit',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))],metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5),indent=1),encoding='utf8')
    slug='medxir-retrieval-input-audit-20261031'
    config=dict(id='ntklinhfitus/'+slug,title=slug,code_file=nb.relative_to(ROOT).as_posix(),kernel_type='notebook',
                immutable_kernel=True,enable_gpu=False,enable_internet=False,
                dataset_sources=['ntklinhfitus/medxir-chunk-corpus-v2','boinhbo/medxir-docrerank-ens-v2','boinhbo/medxir-retrieval-inputs'],
                kernel_sources=['boinhbo/medxir-package-rerankers'],
                timeout_seconds=3600,poll_interval_seconds=30,metrics_file='retrieval_input_audit.json',
                fetch_files=['retrieval_input_audit.json','retrieval_input_samples.json'],
                required_outputs=['retrieval_input_audit.json'],json_checks=[dict(path='complete',value=True)])
    (ROOT/'configs/kaggle/retrieval_input_audit_cpu.yaml').write_text(json.dumps(dict(kernel=config),indent=2),encoding='utf8')
    print(nb)


if __name__=='__main__':main()
