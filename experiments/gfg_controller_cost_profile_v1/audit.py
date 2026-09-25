"""Read-only result reconciliation; create a new audit receipt without overwriting."""
from pathlib import Path
import hashlib
import json
import statistics
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    seal=read(HERE/'RUN_SEAL.json'); result=read(HERE/'RESULTS.json')
    assert result['seal_sha256']==sha(HERE/'RUN_SEAL.json')
    for name,digest in seal['files'].items(): assert sha(HERE/name)==digest
    stages={}; trajectories=0; updates=0
    for history in result['histories']:
        source=Path('E:/gfg-evidence/optimizer-capability-benchmark-v1/search')/history['entry']/('controller-'+str(history['config']))/'0/TRAJECTORY.json'
        assert sha(source)==history['source_ledger_sha256']
        original=read(source)
        for run in history['runs']:
            assert run==read(HERE/'receipts'/f"{history['entry']}-{run['repeat']}-{int(run['instrumented'])}.json")
            assert run['exact'] and len(run['records'])==len(original['steps'])==run['first99_steps']
            assert run['final_state_sha256']==original['final_state_sha256']
            for actual,expected in zip(run['records'],original['steps']):
                fact=expected['fact']
                assert actual['step']==fact['t'] and actual['alpha']==fact['decision']['details']['alpha']
                assert actual['parameters_sha256']==fact['post_parameters_sha256']
                assert actual['coordinates_sha256']==fact['decision']['coordinates_sha256']
                assert actual['accuracy']==fact['evaluation']['accuracy']
                updates+=1
            assert abs(sum(r['update_seconds'] for r in run['records'])-run['update_seconds'])<1e-12
            if run['instrumented']:
                n=run['first99_steps']; decisions=(n+4)//5
                assert run['stage_counts']['second_directional_derivative']==decisions
                assert run['stage_counts']['first_directional_derivative']==decisions
                assert run['stage_counts']['internal_hash_check']==decisions
                assert run['stage_counts']['external_hash_and_comparison']==n
                assert run['stage_counts']['parameter_d2h_snapshots']==2*n
            trajectories+=1
        prof=[r for r in history['runs'] if r['instrumented']]
        for key,value in history['stage_median_seconds'].items():
            assert value==statistics.median(r['stage_seconds'].get(key,0) for r in prof)
            stages[key]=stages.get(key,0)+value
    assert trajectories==12 and updates==630
    total=sum(stages.values())
    assert abs(total-result['accounted_seconds'])<1e-12
    for row in result['stage_ranking']:
        assert row['seconds_for_two_histories']==stages[row['stage']]
        assert abs(row['percent_accounted']-100*stages[row['stage']]/total)<1e-10
    tag=subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release^{}'],cwd=REPO,text=True).strip()
    assert tag=='36dab5ce347dbbdac157ef23205f556606d18294'
    changed=subprocess.check_output(['git','diff','--name-only',seal['base_commit']],cwd=REPO,text=True).splitlines()
    assert all(p.startswith('experiments/gfg_controller_cost_profile_v1/') for p in changed)
    report={'status':'PASS','timed_trajectories':trajectories,'timed_updates':updates,
        'frozen_tag_commit':tag,'old_tracked_files_unchanged':True,
        'files':{str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in HERE.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.name!='AUDIT.json'}}
    with (HERE/'AUDIT.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,indent=2); stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}))


if __name__=='__main__':main()
