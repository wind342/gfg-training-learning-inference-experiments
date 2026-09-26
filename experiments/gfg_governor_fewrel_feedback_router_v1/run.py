"""FewRel adapter for bounded feedback routing, with role-isolated data."""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.gfg_fewrel_replay_direction_smoke_v1.run import CHECKPOINT,CKSHA,plan,write
from experiments.gfg_governor_fewrel_v1.prepare import ROOT as DATA,SEED,sha
from experiments.gfg_governor_fewrel_v1.run import evaluate,extend_bank
from experiments.gfg_governor_fewrel_v1.model import RelationModel,BertInstrument
from experiments.gfg_functional_update_governor_v4.runtime import GovernedTraining
from experiments.gfg_functional_feedback_router_v1.policy import Memory,Router
from experiments.gfg_functional_feedback_router_v1.runtime import accumulate,update

ROOT=Path('E:/gfg-evidence/functional-feedback-router-v1')
HERE=Path(__file__).resolve().parent
CORE=HERE.parent/'gfg_functional_feedback_router_v1'

def scope(data,seen):
    y,s=data['y'],data['split']
    probes=np.array([int(i) for k in range(seen) for i in np.flatnonzero((y==k)&(s==1))[:10]])
    test=np.flatnonzero((y<seen)&(s==2))
    return probes,test

def extend_memory(memory,data,old,probes,test):
    x,y,s=data['x'],data['y'],data['split'];forbidden=set(np.concatenate((probes,test)).tolist())
    encoded={x[i].tobytes() for i in forbidden}
    memory.check()
    for g,ids in memory.groups.items():
        if any(x[i].tobytes() in encoded for i in ids):raise ValueError('append-scope memory duplicate collision')
    for g in sorted(old):
        if g not in memory.groups:
            ids=[int(i) for i in np.flatnonzero((y==g)&(s==0)) if x[i].tobytes() not in encoded][:10]
            memory.append(g,ids,completed=old,split=s,labels=y,forbidden=forbidden)

def validate_batches(new,old_ids,data,current,memory,probes,test):
    if any(data['split'][i]!=0 or int(data['y'][i]) not in current for i in new):raise ValueError('invalid current gradient sample')
    memory.check();allowed={i for ids in memory.groups.values() for i in ids}
    if any(i not in allowed or data['split'][i]!=0 or int(data['y'][i]) in current for i in old_ids):raise ValueError('invalid replay gradient sample')
    if (set(new)|set(old_ids)) & (set(probes)|set(test)):raise ValueError('control/test gradient leakage')

def run(arm,stage):
    ROOT.mkdir(parents=True,exist_ok=True);out=ROOT/f'{stage}-{arm}';out.mkdir()
    assert sha(CHECKPOINT)==CKSHA
    data=np.load(DATA/'data.npz')
    assert sha(DATA/'data.npz')=='29450295331dcd56040b9ec60ab2998303988c0e27500e353b7025b977734a8a'
    if stage=='three':
        gate=json.loads((ROOT/'SMOKE_CHECKS.json').read_text())
        assert gate['status']=='PASS' and gate['three_task_gate']
    config=dict(arm=arm,stage=stage,checkpoint_sha256=CKSHA,seed=SEED,
        source_hashes={str(p.relative_to(HERE.parent)):sha(p) for folder in (HERE,CORE) for p in folder.glob('*.py')},
        protocol_sha256=sha(HERE/'PROTOCOL.md'),data_sha256=sha(DATA/'data.npz'),
        budget_updates=60 if stage=='smoke' else 420,cap_seconds=900 if stage=='smoke' else 3600,
        checkpoint='common task1 governor prefix; not independently retrained task1')
    write(out/'CONFIG.json',config);rows=[];metrics=[];start=time.perf_counter()
    try:
        torch.set_num_threads(2);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        x=torch.from_numpy(data['x']);y=torch.from_numpy(data['y'][:,None])
        model=RelationModel().cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-6,betas=(.9,.999),weight_decay=0,fused=True,foreach=False)
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        assert saved['step']==210 and saved['seen']==8
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        history=saved['scheduler'];history.identity=None
        engine=None;px=py=None;memory=Memory();router=Router();fixed_cursor=0
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with (out/'STEPS.jsonl').open('x',encoding='utf8') as f:
            for task in range(1,2 if stage=='smoke' else 3):
                seen=(task+1)*8;old=set(range(task*8));current=set(range(task*8,seen));model.seen=seen
                probes,test=scope(data,seen);extend_memory(memory,data,old,probes,test)
                newx=x[probes].cuda();newy=y[probes].cuda()
                if arm=='router':
                    if engine is None:
                        oldx=newx[:80].clone();oldy=newy[:80].clone();history.bind(oldx,oldy)
                        instrument=BertInstrument(model,chunk=4);instrument.alphas=(.125,.25,.5,1.)
                        engine=GovernedTraining(model,None,mapping=history.mapping,instrument=instrument)
                        engine.scheduler=history;px=oldx;py=oldy
                    extend_bank(engine,px,py,newx,newy)
                px,py=newx,newy
                if task==1:del saved
                def record(offset):
                    ev=evaluate(model,x,y,test);per=ev['per_relation']
                    ev.update(task=task+1,offset=offset,old=float(np.mean([per[str(k)] for k in old])),
                        new=float(np.mean([per[str(k)] for k in current])))
                    metrics.append(ev);print(stage,arm,'task',task+1,'offset',offset,'old',ev['old'],'new',ev['new'],flush=True)
                record(0)
                train=np.flatnonzero(np.isin(data['y'],list(current))&(data['split']==0))
                generator=torch.Generator().manual_seed(SEED+10000+task)
                order=train[torch.randperm(len(train),generator=generator).numpy()]
                count=60 if stage=='smoke' else 210
                for offset in range(1,count+1):
                    t=task*210+offset;ids=order[(offset-1)*16:offset*16].tolist()
                    before_cursor=dict(memory.cursors)
                    if arm=='router':
                        # Preview only on copies to check data roles before any gradient.
                        preview=copy.deepcopy(router).choose(t,engine.mapping.labels,list(engine.scheduler.state),engine.scheduler.level.tolist(),old,copy.deepcopy(memory))
                        oldids=preview['feedback_example_ids'];validate_batches(ids,oldids,data,current,memory,probes,test)
                        info=update(model,opt,engine,router,memory,x,y,ids,px,py,t,old,SEED*100000+t)
                        assert info['feedback_example_ids']==oldids
                    else:
                        torch.cuda.synchronize();tick=time.perf_counter();selected=[];oldids=[]
                        if arm=='replay':
                            groups=sorted(old);selected=[groups[(fixed_cursor+j)%len(groups)] for j in range(min(8,len(groups)))]
                            fixed_cursor=(fixed_cursor+len(selected))%len(groups)
                            for g in selected:oldids.extend(memory.take(g))
                        validate_batches(ids,oldids,data,current,memory,probes,test)
                        losses=accumulate(model,opt,x,y,[ids]+([oldids] if oldids else []),SEED*100000+t);opt.step();torch.cuda.synchronize()
                        info=dict(alpha=1.,instrument_called=False,gate_reason='NOT_MEASURED',seconds=time.perf_counter()-tick,
                            current_loss=losses[0],feedback_loss=losses[1] if oldids else None,
                            router_mode='FIXED_REPLAY' if oldids else 'NATIVE',feedback_example_ids=oldids,
                            feedback_batch_size=len(oldids),selected_feedback_groups=selected,selected_anchor_groups=[],
                            router_activation=False,release_event=False)
                    row=dict(step=t,task=task+1,offset=offset,new_ids=ids,probe_ids=probes.tolist(),memory_identity=memory.stamp,
                        memory_groups={str(g):list(v) for g,v in memory.groups.items()},cursor_before=before_cursor,info=info)
                    rows.append(row);f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                    if (stage=='smoke' and offset%20==0) or (stage=='three' and offset==210):record(offset)
                    if time.perf_counter()-start>config['cap_seconds']:raise TimeoutError('fixed runtime budget')
        result=dict(status='COMPLETED',metrics=metrics,steps=len(rows),wall=time.perf_counter()-start,
            peak_allocated=torch.cuda.max_memory_allocated(),mode_counts=dict(Counter(r['info']['router_mode'] for r in rows)),
            releases=sum(r['info']['release_event'] for r in rows),feedback_steps=sum(bool(r['info']['feedback_batch_size']) for r in rows),
            old_exposures=sum(r['info']['feedback_batch_size'] for r in rows),full_calls=sum(r['info']['instrument_called'] for r in rows),
            alpha_counts=dict(Counter(r['info']['alpha'] for r in rows)),log_sha256=sha(out/'STEPS.jsonl'))
    except Exception:
        result=dict(status='FAILED',traceback=traceback.format_exc(),metrics=metrics,steps=len(rows),wall=time.perf_counter()-start)
    write(out/'RESULT.json',result);print(stage,arm,result['status'],flush=True)
    if result['status']!='COMPLETED':raise RuntimeError(result['traceback'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['native','replay','router']);p.add_argument('--stage',choices=['smoke','three'],default='smoke');a=p.parse_args();run(a.arm,a.stage)
