"""Frozen data selectors and small arm-independent utilities; no V4 policy changes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from experiments.gfg_governor_fewrel_v1.prepare import ROOT as DATA
from experiments.gfg_fewrel_replay_direction_smoke_v1.run import CHECKPOINT, CKSHA
from experiments.gfg_functional_feedback_router_v1.policy import Memory
from experiments.gfg_governor_fewrel_feedback_router_v1.run import extend_memory

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/router-v4-ten-task-validation')
BASE='7ed4fb4c6c0c61f4436ad944c1a88af1650b840f'
DATA_SHA='29450295331dcd56040b9ec60ab2998303988c0e27500e353b7025b977734a8a'
EXCLUDE=(7561,12566,18193,29275,29338,47545,55297)
ARMS=('router','native','replay','random')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def stamp(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)

def unchanged():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)

def sources():
    paths=[REPO/p for p in subprocess.check_output(['git','ls-files','experiments/*.py','experiments/**/*.py'],cwd=REPO,text=True).splitlines()]
    paths+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(set(paths))}

def scope(data,seen):
    y,s=data['y'],data['split']
    probes=np.array([50130 if int(i)==50124 else int(i) for k in range(seen) for i in np.flatnonzero((y==k)&(s==1))[:10]])
    original=np.flatnonzero((y<seen)&(s==2))
    primary=original[~np.isin(original,EXCLUDE)]
    return probes,primary,original

def order(data,task,seed=2021):
    ids=np.flatnonzero((data['split']==0)&(data['y']>=(task-1)*8)&(data['y']<task*8))
    return ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(seed+10000+task-1)).numpy()].reshape(-1,16)

def data_audit(data):
    x,y,s=data['x'],data['y'],data['split'];probes,primary,original=scope(data,80)
    h=[hashlib.sha256(a.tobytes()).hexdigest() for a in x]
    train=set(h[i] for i in np.flatnonzero(s==0));control={h[i] for i in probes};test={h[i] for i in primary}
    assert not(train&control or train&test or control&test)
    assert len(probes)==800 and len(primary)==11193 and len(set(probes))==800
    oldcontrol=[int(i) for k in range(80) for i in np.flatnonzero((y==k)&(s==1))[:10]]
    assert [(int(a),int(b)) for a,b in zip(oldcontrol,probes) if a!=b]==[(50124,50130)]
    assert x[50124].tobytes()==x[49887].tobytes() and y[50124]==y[50130]==71
    eligible=[int(i) for i in np.flatnonzero((y==71)&(s==1)) if i not in oldcontrol and h[i] not in train and h[i] not in {h[j] for j in original} and h[i] not in {h[j] for j in oldcontrol}]
    assert eligible[0]==50130
    bad=[int(i) for i in original if h[i] in train or h[i] in control]
    assert bad==list(EXCLUDE)
    memory=Memory();tasks={}
    for task in range(2,11):
        p,clean,full=scope(data,task*8);extend_memory(memory,data,set(range((task-1)*8)),p,full)
        batches=order(data,task);assert batches.shape==(210,16)
        tasks[task]=dict(probes=p.tolist(),primary_count=len(clean),full_count=len(full),
            batch_sha256=stamp(batches.tolist()),memory={str(k):list(v) for k,v in memory.groups.items()},memory_identity=memory.stamp)
    return dict(status='PASS',replacement={'50124':50130},excluded_test=list(EXCLUDE),
        isolation='exact full encoded input, regardless of relation label; no semantic equivalence claim',tasks=tasks)

def summarize_predictions(indices,predictions,data,task):
    ids=np.asarray(indices);pred=np.asarray(predictions);truth=data['y'][ids];cut=(task-1)*8
    assert len(ids)==len(pred) and np.all((pred>=0)&(pred<task*8))
    per={str(k):float((pred[truth==k]==k).mean()) for k in range(task*8)}
    old=truth<cut;new=~old
    return dict(old=float((pred[old]==truth[old]).mean()) if cut else None,new=float((pred[new]==truth[new]).mean()),
        all=float((pred==truth).mean()),macro_old=float(np.mean([per[str(k)] for k in range(cut)])) if cut else None,
        macro_new=float(np.mean([per[str(k)] for k in range(cut,task*8)])),macro_all=float(np.mean(list(per.values()))),
        per_relation=per,cohorts={str(j+1):float(np.mean([per[str(k)] for k in range(j*8,j*8+8)])) for j in range(task)})

class RandomReplay:
    """Only memory, old group identities, and numeric dose are accepted."""
    def __init__(self,seed):self.rng=np.random.Generator(np.random.PCG64(seed+9000000))
    def choose(self,old,memory,n):
        if n<0 or n>16 or n%2 or n//2>len(old):raise ValueError('invalid exposure dose')
        selected=self.rng.choice(sorted(old),n//2,replace=False).tolist() if n else []
        ids=[int(i) for g in selected for i in self.rng.choice(memory.groups[g],2,replace=False)]
        return selected,ids

def exposure_schedule(rows):
    return [dict(step=r['step'],task=r['task'],old_batch_size=r['info']['feedback_batch_size']) for r in rows]

def trigger(arms):
    r,f,n,z=[arms[a] for a in ('router','replay','native','random')]
    end=lambda a:a['tasks']['10']['primary']
    mean=lambda a,k:float(np.mean([t['primary'][k] for t in a['tasks'].values() if t['task']>1]))
    checks=dict(completed=all(a['status']=='COMPLETED' for a in arms.values()),
        final_old_close=end(r)['macro_old']>=end(f)['macro_old']-.03,
        final_new_close=end(r)['macro_new']>=end(f)['macro_new']-.03,
        mean_old_close=mean(r,'macro_old')>=mean(f,'macro_old')-.03,
        exposure_at_most_65pct=r['old_exposures']<=.65*f['old_exposures'],
        final_random_gap=end(r)['macro_all']>=end(z)['macro_all']+.02,
        mean_random_gap=mean(r,'macro_all')>=mean(z,'macro_all')+.01,
        beats_native_old=end(r)['macro_old']>end(n)['macro_old'],beats_native_all=end(r)['macro_all']>end(n)['macro_all'])
    return dict(criteria=checks,eligible_if_integrity_pass=all(checks.values()),seeds=[2022,2023],statistical_significance_claim=False)
