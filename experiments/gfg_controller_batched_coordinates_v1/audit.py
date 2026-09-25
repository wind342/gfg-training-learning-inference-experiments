"""Independently reconcile completed receipts with frozen historical ledgers."""
import json
import hashlib
from pathlib import Path
import statistics
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    receipt=HERE/'verification_v2'
    seal=read(receipt/'RUN_SEAL.json')
    for name,digest in seal['files'].items():
        assert sha(HERE/name)==digest, name
    result=read(receipt/'RESULTS.json')
    assert result['seal_sha256']==sha(receipt/'RUN_SEAL.json')
    cases=0; steps=0
    for history in result['histories']:
        original_path=Path('E:/gfg-evidence/optimizer-capability-benchmark-v1/search')/history['entry']/('controller-'+str(history['config']))/'0/TRAJECTORY.json'
        assert sha(original_path)==history['source_ledger_sha256']
        original=read(original_path)
        assert history==read(receipt/(history['entry']+'.json'))
        for run in history['runs']:
            assert run['exact'] and run['first99_steps']==len(original['steps'])==len(run['steps'])
            assert run['final_state_sha256']==original['final_state_sha256']
            for actual,expected in zip(run['steps'],original['steps']):
                expected=expected['fact']
                assert actual['step']==expected['t']
                assert actual['alpha']==expected['decision']['details']['alpha']
                assert actual['accuracy']==expected['evaluation']['accuracy']
                assert actual['parameters_sha256']==expected['post_parameters_sha256']
                assert actual['update_seconds']>0
                steps+=1
            assert all(s['accuracy']<.99 for s in run['steps'][:-1])
            assert run['steps'][-1]['accuracy']>=.99
            cases+=1
        for metric in ('compute_seconds','checked_wall_seconds'):
            ref=statistics.median(r[metric] for r in history['runs'] if not r['optimized'])
            fast=statistics.median(r[metric] for r in history['runs'] if r['optimized'])
            expected=history['summary'][metric]
            assert ref==expected['reference_median'] and fast==expected['batched_median']
            assert 100*(1-fast/ref)==expected['reduction_percent']
    assert cases==20 and steps==1050
    tag=subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release^{}'],cwd=REPO,text=True).strip()
    assert tag=='36dab5ce347dbbdac157ef23205f556606d18294'
    modified=subprocess.check_output(['git','diff','--name-only',seal['base_commit']],cwd=REPO,text=True).splitlines()
    assert all(p.startswith('experiments/gfg_controller_batched_coordinates_v1/') for p in modified)
    report={'status':'PASS','timed_trajectories':cases,'timed_updates':steps,
            'frozen_tag_commit':tag,'old_tracked_files_unchanged':True,
            'run_seal_verified':True,'historical_ledgers_reconciled':True,
            'files':{str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in HERE.rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.name!='AUDIT.json'}}
    with (HERE/'AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}))


if __name__=='__main__':main()
