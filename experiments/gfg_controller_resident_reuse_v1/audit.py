"""Reconcile every persisted result with the original frozen search ledger."""
from pathlib import Path
import hashlib
import json
import statistics
import subprocess

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=HERE/'verification'
    result=read(root/'RESULTS.json'); seal=read(root/'RUN_SEAL.json')
    assert sha(root/'RUN_SEAL.json')==result['seal_sha256']
    for name,digest in seal['files'].items():assert sha(HERE/name)==digest,name
    assert read(HERE/'EXTRA_CHECKS.json')['status']=='PASS'
    count=steps=0
    for case in result['histories']:
        source=Path('E:/gfg-evidence/optimizer-capability-benchmark-v1/search')/case['entry']/('controller-'+str(case['config']))/'0/TRAJECTORY.json'
        assert sha(source)==case['source_ledger_sha256']
        original=read(source)
        for run in case['runs']:
            assert run==read(root/'receipts'/f"{case['entry']}-{run['repeat']}-{int(run['optimized'])}.json")
            assert run['exact'] and run['first99_steps']==len(original['steps'])==len(run['records'])
            assert run['final_state_sha256']==original['final_state_sha256']
            reused=0
            for actual,expected in zip(run['records'],original['steps']):
                fact=expected['fact']
                assert actual['step']==fact['t']
                assert actual['alpha']==fact['decision']['details']['alpha']
                assert actual['accuracy']==fact['evaluation']['accuracy']
                assert actual['coordinates_sha256']==fact['decision']['coordinates_sha256']
                assert actual['post_parameters_sha256']==fact['post_parameters_sha256']
                if run['optimized'] and (actual['step']-1)%5==0 and actual['alpha']==1.:reused+=1
                steps+=1
            assert reused==run['reused_post_snapshots']
            assert all(r['accuracy']<.99 for r in run['records'][:-1]) and run['records'][-1]['accuracy']>=.99
            assert abs(sum(r['update_seconds'] for r in run['records'])+run['setup_seconds']-run['compute_seconds'])<1e-10
            count+=1
        for metric,s in case['summary'].items():
            ref=statistics.median(r[metric] for r in case['runs'] if not r['optimized'])
            fast=statistics.median(r[metric] for r in case['runs'] if r['optimized'])
            assert ref==s['reference_median'] and fast==s['combined_median']
            assert 100*(1-fast/ref)==s['reduction_percent']
            ratios=[next(r[metric] for r in case['runs'] if r['repeat']==i and not r['optimized'])/next(r[metric] for r in case['runs'] if r['repeat']==i and r['optimized']) for i in range(5)]
            assert ratios==s['paired_speedups']
    assert count==20 and steps==1050
    tag=subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release^{}'],cwd=REPO,text=True).strip()
    assert tag=='36dab5ce347dbbdac157ef23205f556606d18294'
    changed=subprocess.check_output(['git','diff','--name-only',seal['base_commit']],cwd=REPO,text=True).splitlines()
    assert all(p.startswith('experiments/gfg_controller_resident_reuse_v1/') for p in changed)
    report={'status':'PASS','timed_trajectories':count,'timed_updates':steps,
        'frozen_tag_commit':tag,'old_tracked_files_unchanged':True,'all_ten_pairs_faster':
            all(r>1 for c in result['histories'] for r in c['summary']['checked_wall_seconds']['paired_speedups']),
        'files':{str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in HERE.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.name!='AUDIT.json'}}
    with (HERE/'AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}))


if __name__=='__main__':main()
