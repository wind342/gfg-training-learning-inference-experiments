"""Independent raw-logit, boundary, decision, ledger and LOO-statistic audit."""
from collections import defaultdict
import math
import statistics
import subprocess
import numpy as np
import torch
from . import run

b=run.b;HERE=run.HERE;OUT=run.OUT


def crossing(record,threshold):
    if record['initial_accuracy']>=threshold:return 0
    for step in record['steps']:
        if step['accuracy']>=threshold:return step['t']
    return None


def main():
    seal=b.read(HERE/'RUN_SEAL.json')
    for name,digest in seal['files'].items():assert b.sha(HERE/name)==digest,name
    assert b.sha(run.resident.HERE/'engine.py')==seal['prior_engine_sha256']
    assert b.sha(b.old.HERE/'RUN_ASSIGNMENT.json')==seal['assignment_sha256']
    entries=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs']
    choices=b.read(HERE/'SELECTION.json');result=b.read(HERE/'RESULTS.json')
    search={};timing={};manifest={};total_updates=0;tuning=defaultdict(lambda:defaultdict(float))
    for path in sorted(OUT.rglob('TRAJECTORY.json')):
        record=b.read(path);manifest[str(path.relative_to(OUT))]=b.sha(path)
        mode,entry,arm,rep=path.relative_to(OUT).parts[:4]
        for name,digest in record['artifacts'].items():
            assert b.sha(path.parent/name)==digest,name
            manifest[str((path.parent/name).relative_to(OUT))]=digest
        with np.load(path.parent/'OUTPUTS.npz',allow_pickle=False) as f:
            logits=f['logits'];targets=f['target']
        assert len(logits)==len(record['steps'])+1
        competitors=logits.copy()
        competitors[:,np.arange(len(targets)),targets]=-np.inf
        gaps=logits[:,np.arange(len(targets)),targets].astype(np.float64)-competitors.max(axis=2).astype(np.float64)
        assert np.isfinite(gaps).all()
        acc=(gaps>0).mean(axis=1)
        assert float(acc[0])==record['initial_accuracy']
        assert b.tensor_sha256(logits[0])==record['initial_logits_sha256']
        coordinate_path=(path.parent if mode=='search' else OUT/'search'/entry/arm/'0')/'COORDINATES.npz'
        coordinates=np.load(coordinate_path,allow_pickle=False) if coordinate_path.exists() else None
        previous=None;alpha=1.;skip=0
        try:
            for i,step in enumerate(record['steps'],1):
                assert step['t']==i and step['accuracy']==float(acc[i])
                assert step['logits_sha256']==b.tensor_sha256(logits[i])
                assert step['lr']==run.RATES[record['config']]*(.1+.9*.5*(1+math.cos(math.pi*(i-1)/199)))
                if record['method']=='controller':
                    if (i-1)%5==0:
                        c={k:coordinates[f'{i}_{k}'] for k in b.KEYS}
                        h=b.identity({k:b.tensor_sha256(torch.from_numpy(c[k])) for k in b.KEYS})
                        assert h==step['coordinates_sha256']
                        alpha,scores=run.independently_choose(c)
                        assert scores==step['details']['scores']
                    elif alpha==1.:skip+=1
                    assert step['details']['alpha']==alpha
                else:assert step['coordinates_sha256'] is None
                fact={k:v for k,v in step.items() if k not in ('sha256','update_seconds','cumulative_compute_seconds','cumulative_checked_wall_seconds')}
                assert fact['previous_sha256']==previous
                previous=b.identity(fact);assert previous==step['sha256']
            assert skip==record['skipped_identity_steps']
        finally:
            if coordinates is not None:coordinates.close()
        if record['initial_accuracy']<.99:
            assert all(a<.99 for a in acc[:-1])
            assert acc[-1]>=.99 or len(record['steps'])==200
        else:assert not record['steps']
        key=(entry,record['method'],record['config'])
        if mode=='search':
            search[key]=record
            tuning[record['method']]['updates']+=len(record['steps'])
            tuning[record['method']]['compute_seconds']+=record['compute_seconds']
            tuning[record['method']]['checked_wall_seconds']+=record['checked_wall_seconds']
        else:timing[(*key,int(rep))]=record
        total_updates+=len(record['steps'])
    assert len(search)==96 and len(timing)==72
    for (entry,method,ci,rep),record in timing.items():
        original=search[(entry,method,ci)]
        assert record['exact_replay'] and record['final_state_sha256']==original['final_state_sha256']
        assert [s['sha256'] for s in record['steps']]==[s['sha256'] for s in original['steps']]
        assert ci==choices[entry]['config']
    for row in entries:
        ranks=[]
        for ci in range(4):
            costs=[]
            for threshold in (.99,.95):
                hits=[crossing(search[(other['entry_id'],m,ci)],threshold) for other in entries if other['entry_id']!=row['entry_id'] for m in run.METHODS]
                vals=[200 if h is None else h for h in hits if h!=0]
                costs.append(statistics.mean(vals))
            ranks.append([*costs,ci])
        assert choices[row['entry_id']]=={'ranks':ranks,'config':min(ranks)[-1]}
    for key,arms in result['thresholds'].items():
        threshold=float(key)
        for method,summary in arms.items():
            rebuilt=[]
            for row in entries:
                entry=row['entry_id'];ci=choices[entry]['config']
                records=[timing[(entry,method,ci,rep)] for rep in range(3)]
                h=crossing(records[0],threshold)
                if h==0:continue
                def value(r,k):return r[k] if h is None else r['steps'][h-1]['cumulative_'+k]
                rebuilt.append({'entry':entry,'hit':h,'restricted_steps':200 if h is None else h,
                    'compute_seconds':statistics.median(value(r,'compute_seconds') for r in records),
                    'checked_wall_seconds':statistics.median(value(r,'checked_wall_seconds') for r in records)})
            assert rebuilt==summary['entries']
            assert summary['reached']==sum(e['hit'] is not None for e in rebuilt)
            for metric in ('restricted_steps','compute_seconds','checked_wall_seconds'):
                assert summary[metric+'_mean']==statistics.mean(e[metric] for e in rebuilt)
    tag=subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release^{}'],cwd=HERE.parents[1],text=True).strip()
    assert tag=='36dab5ce347dbbdac157ef23205f556606d18294'
    report={'status':'PASS','search_records':96,'timing_records':72,'total_updates':total_updates,
        'three_exact_replays_per_selected_arm':True,'raw_logits_and_decisions_recomputed':True,
        'selection_and_statistics_recomputed':True,'frozen_tag_commit':tag,'tuning_costs':dict(tuning),
        'artifact_bytes':sum((OUT/p).stat().st_size for p in manifest),
        'result_sha256':b.sha(HERE/'RESULTS.json'),'files':manifest}
    b.write(HERE/'AUDIT.json',report)
    print({k:v for k,v in report.items() if k!='files'})


if __name__=='__main__':main()
