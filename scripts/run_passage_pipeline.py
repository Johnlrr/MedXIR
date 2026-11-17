"""Local lightweight orchestration. All corpus/model compute stays on Kaggle.

Wait for existing preparation/pilot jobs, check their artifacts, run model jobs,
fetch and audit ZIPs. This script never calls the competition submission API.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

from audit_submission_coverage import audit
from check_passage_pilot import validate

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'runs/passage_pipeline_20261031'
ACCOUNT='ntklinhfitus'
PILOT='ntklinhfitus/medxir-evidence-rerank-pilot2-20261031'
PREPARE='ntklinhfitus/medxir-passages-prepare-20261031'


def cli(*args):
    result=subprocess.run([sys.executable,'-X','utf8','-m','tools.kaggle',*map(str,args)],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8',timeout=900)
    if result.returncode:raise RuntimeError(f'Kaggle command failed: {result.stdout[-1500:]} {result.stderr[-1500:]}')
    return result.stdout


def status(kernel):
    output=cli('kernel','status',kernel,'--account',ACCOUNT)
    decoder=json.JSONDecoder()
    for i,c in enumerate(output):
        if c!='{':continue
        try:value,_=decoder.raw_decode(output[i:])
        except json.JSONDecodeError:continue
        if isinstance(value,dict) and 'status' in value:return value['status']
    raise ValueError('Cannot parse kernel status')


def wait_success(kernels,timeout=36000):
    deadline=time.monotonic()+timeout;last={};heartbeat=0
    pending=set(kernels)
    while pending:
        for kernel in sorted(pending):
            current=status(kernel)
            if last.get(kernel)!=current or time.monotonic()-heartbeat>120:
                print(kernel,current,flush=True)
            last[kernel]=current
            if current=='SUCCESS':pending.remove(kernel)
            elif current not in ('QUEUED','RUNNING'):raise RuntimeError(f'{kernel}: terminal status {current}')
        heartbeat=time.monotonic()
        if pending:
            if time.monotonic()>deadline:raise TimeoutError('Monitoring timed out; remote kernels NOT restarted')
            time.sleep(30)


def fetch(kernel,directory,names):
    directory.mkdir(parents=True,exist_ok=True)
    command=['kernel','fetch',kernel,'--destination',directory,'--account',ACCOUNT]
    for name in names:command.extend(['--file',name])
    cli(*command)
    for name in names:
        if not (directory/name).is_file():raise ValueError('Missing fetched file '+name)


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def check_prepare_against_pilot(candidate_frame,pilot_frame):
    columns=['query_id','doc_id','text_sha256','seed','seed_rank','start','end','chunk_text','passage_tokens']
    qids=set(pilot_frame.query_id)
    left=candidate_frame[candidate_frame.query_id.isin(qids)].copy()
    order=['query_id','text_sha256','start','end']
    a=left.sort_values(order)[columns].reset_index(drop=True)
    b=pilot_frame.sort_values(order)[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_exact=True,check_dtype=False)
    # Fanout arrays are compared separately, canonicalised as integer tuples.
    canonical=lambda f: [tuple(map(int,x)) for x in f.sort_values(order).doc_ids]
    if canonical(left)!=canonical(pilot_frame):raise ValueError('Pilot/preparation fanout mismatch')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    state_path=OUTPUT/'state.json';state=dict(status='WAITING_FOR_INPUTS',submitted={})
    if state_path.exists():state=json.loads(state_path.read_text())
    def save():state_path.write_text(json.dumps(state,indent=2))
    save();wait_success([PILOT,PREPARE])
    pilot=OUTPUT/'pilot2';prepared=OUTPUT/'prepared_sample'
    fetch(PILOT,pilot,['passage_pilot_report.json','passage_candidates.parquet','passage_scores.parquet'])
    report=json.loads((pilot/'passage_pilot_report.json').read_text())
    pilot_frame=pd.read_parquet(pilot/'passage_candidates.parquet')
    check=validate(pilot_frame,pd.read_parquet(pilot/'passage_scores.parquet'),report)
    fetch(PREPARE,prepared,['passage_candidates_manifest.json'])
    manifest=json.loads((prepared/'passage_candidates_manifest.json').read_text())
    if not manifest['complete'] or manifest['queries']!=1200 or set(manifest['spec']['query_ids'])!=set(range(1,1201)):
        raise ValueError('Prepared query set incomplete')
    pilot_qids=set(map(int,pilot_frame.query_id))
    parts=[p for p in manifest['parts'] if pilot_qids.intersection(p['query_ids'])]
    fetch(PREPARE,prepared,[p['path'] for p in parts])
    frames=[]
    for part in parts:
        path=prepared/part['path']
        if digest(path)!=part['sha256']:raise ValueError('Corrupted prepared sample')
        frames.append(pd.read_parquet(path))
    check_prepare_against_pilot(pd.concat(frames,ignore_index=True),pilot_frame)
    del frames,pilot_frame
    state.update(status='INPUTS_VERIFIED',pilot_validation=check,pairs=manifest['pairs'],
                 candidate_manifest_sha256=digest(prepared/'passage_candidates_manifest.json'))
    save();print('Input gates PASS; full pairs',manifest['pairs'],flush=True)
    processes=[]
    for model in ['bge','qwen']:
        kernel=f'{ACCOUNT}/medxir-passages-{model}-20261031'
        if model in state['submitted']:
            print('Existing job, not resubmitting:',kernel,flush=True);continue
        # Persist intent before push: an ambiguous crash cannot silently duplicate a run.
        state['submitted'][model]=dict(kernel=kernel,status='SUBMITTING');save()
        log=(OUTPUT/f'{model}_runner.log').open('a',encoding='utf8')
        command=[sys.executable,'-X','utf8','-m','tools.kaggle','kernel','run','--config',
            f'configs/kaggle/passage_{model}_20261031.yaml','--source','.',
            '--kind','smoke','--run-id',f'passages-{model}-20261031-v1','--account',ACCOUNT]
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        processes.append((process,log));state['submitted'][model]['pid']=process.pid;save()
        print('Started harness',model,process.pid,flush=True)
        # Confirm the push before starting another job. Do not retry failed/ambiguous pushes.
        deadline=time.monotonic()+180
        metadata=ROOT/f'runs/smoke/passages-{model}-20261031-v1/metadata.json'
        while time.monotonic()<deadline:
            if process.poll() is not None and process.returncode:
                raise RuntimeError(f'{model} harness failed; inspect {log.name}')
            if metadata.exists():
                content=json.loads(metadata.read_text())
                if content.get('kernel_version'):
                    state['submitted'][model]['status']='SUBMITTED';save();break
            time.sleep(5)
        else:raise TimeoutError('Push not confirmed; inspect existing job, never automatically resubmit')
    state['status']='SCORING';save()
    wait_success([f'{ACCOUNT}/medxir-passages-{model}-20261031' for model in ['bge','qwen']])
    state['outputs']={}
    for model in ['bge','qwen']:
        target=OUTPUT/model
        names=[f'passage_scores_{model}_manifest.json',f'passage_{model}_submission_manifest.json',f'medxir-passage-{model}-d200.zip']
        fetch(f'{ACCOUNT}/medxir-passages-{model}-20261031',target,names)
        score=json.loads((target/names[0]).read_text());submission=json.loads((target/names[1]).read_text())
        if not score['complete'] or score['pairs']!=manifest['pairs']:
            raise ValueError('Scoring is incomplete')
        if score['spec']['candidate_manifest_sha256']!=state['candidate_manifest_sha256']:
            raise ValueError('Scores use different input candidates')
        zpath=target/names[2]
        if digest(zpath)!=submission['zip_sha256']:raise ValueError('Downloaded ZIP hash mismatch')
        summary=audit(zpath)
        if sorted(summary['qids'])!=list(range(1,1201)) or summary['compression']!=8 or summary['zip_bytes']>=99_000_000:
            raise ValueError('Submission schema/size/query check failed')
        if summary['stats']['duplicate_entries']['max'] or summary['stats']['outside_doc_list']['max']:
            raise ValueError('Submission has duplicate chunks or inconsistent doc list')
        (target/'submission_audit.json').write_text(json.dumps(summary,indent=2))
        state['outputs'][model]=dict(zip=str(zpath),manifest=submission)
        save()
    for process,log in processes:
        log.close()
    state['status']='READY_FOR_USER_REVIEW';save()
    print('READY_FOR_USER_REVIEW. No competition submission was made.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(type(error).__name__,str(error),flush=True)
        raise
