"""Fixed short replay ablation; reuse the existing predictor and governor."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.gfg_governor_fewrel_v1.model import RelationModel, BertInstrument
from experiments.gfg_governor_fewrel_v1.prepare import ROOT as DATA, SEED, sha
from experiments.gfg_governor_fewrel_v1.run import evaluate, extend_bank
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_functional_update_governor_v4.runtime import GovernedTraining
from experiments.gfg_functional_update_governor_v4.governor import RecoveryGovernor
from experiments.gfg_functional_update_governor_v2.scheduler import native_margins

ROOT=Path('E:/gfg-evidence/fewrel-replay-direction-smoke-v1')
HERE=Path(__file__).resolve().parent
CHECKPOINT=DATA/'3tasks-governor/attempt-1/task1.pt'
CKSHA='bd41bcbd84f280d4de7d721eeee1d7a7bad1ccbb04ceb7d84d0403a6bdbcb676'

def write(path,obj):
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)

def plan(data):
    x,y,s=data['x'],data['y'],data['split']
    probes=np.array([int(i) for k in range(16) for i in np.flatnonzero((y==k)&(s==1))[:10]])
    test=np.flatnonzero((y<16)&(s==2))
    forbidden={x[i].tobytes() for i in np.concatenate((probes,test))}
    memory=[]
    for k in range(8):
        ids=[int(i) for i in np.flatnonzero((y==k)&(s==0)) if x[i].tobytes() not in forbidden][:10]
        assert len(ids)==10
        memory.append(ids)
    train=np.flatnonzero((y>=8)&(y<16)&(s==0))
    g=torch.Generator().manual_seed(SEED+10001)
    order=train[torch.randperm(len(train),generator=g).numpy()][:960].reshape(60,16)
    replay=np.array([[ids[(2*t+j)%10] for ids in memory for j in (0,1)] for t in range(60)])
    assert not set(replay.ravel()) & set(np.concatenate((probes,test)))
    return probes,test,order,replay,memory

def step(model,opt,engine,x,y,new_ids,old_ids,t,px,py):
    torch.cuda.synchronize();start=time.perf_counter()
    opt.zero_grad(set_to_none=True);model.train()
    batches=[new_ids] if old_ids is None else [new_ids,old_ids]
    losses=[]
    for j,ids in enumerate(batches):
        seed=SEED*100000+t+j*10000000
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        loss=model(x[ids].cuda(),y[ids].cuda())[1]/len(batches)
        loss.backward();losses.append(float(loss.detach()))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert np.isfinite(losses).all() and torch.isfinite(norm)
    torch.cuda.synchronize();training=time.perf_counter()-start
    info=dict(alpha=1.,instrument_called=False,training_seconds=training,loss=sum(losses))
    if engine is None:opt.step()
    else:
        s=engine.scheduler;s.bind(px,py);reasons=s.begin(t);events=[]
        if not reasons:
            ids=s.shard(t)
            events=s.observe(ids,native_margins(model,px[ids],py[ids])) if ids else []
            reasons=s.reasons(events)
        info.update(micro_events=events,instrument_reason=reasons)
        if reasons:
            engine.owner.propose(opt)
            m=engine.instrument.measure(engine.owner,px,py)
            d=engine.governor.choose(m);s.measured(t,m.baseline_minima)
            assert engine.governor.summary['scheduler_preview']==s.snapshot(t)
            engine.owner.commit(d.alpha)
            actual=[]
            for b in range(0,len(px),16):actual.extend(native_margins(model,px[b:b+16],py[b:b+16]).tolist())
            info.update(alpha=d.alpha,instrument_called=True,reason=d.reason,
                scores=d.scores,baseline=m.baseline_minima.cpu().tolist(),
                predicted=m.predicted_minima.cpu().tolist(),actual=actual)
        else:opt.step()
        info['history']=s.snapshot(t)
    torch.cuda.synchronize();info['seconds']=time.perf_counter()-start
    return info

def run(arm):
    ROOT.mkdir(parents=True,exist_ok=True);out=ROOT/arm;out.mkdir()
    assert sha(CHECKPOINT)==CKSHA
    data=np.load(DATA/'data.npz');probes,test,order,replay,memory=plan(data)
    assert sha(DATA/'data.npz')=='29450295331dcd56040b9ec60ab2998303988c0e27500e353b7025b977734a8a'
    write(out/'CONFIG.json',dict(arm=arm,checkpoint_sha256=CKSHA,steps=60,seed=SEED,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'PROTOCOL.md'),
        probes=probes.tolist(),test=test.tolist(),new_batches=order.tolist(),old_batches=replay.tolist(),memory=memory))
    rows=[];metrics=[];start=time.perf_counter()
    try:
        torch.set_num_threads(2);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        x=torch.from_numpy(data['x']);y=torch.from_numpy(data['y'][:,None])
        model=RelationModel().cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-6,betas=(.9,.999),weight_decay=0,fused=True,foreach=False)
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        assert saved['step']==210 and saved['seen']==8
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);model.seen=16
        px=x[probes].cuda();py=y[probes].cuda();engine=None
        if arm!='replay_native':
            history=saved['scheduler'];history.identity=None
            oldx=px[:80].clone();oldy=py[:80].clone();history.bind(oldx,oldy)
            engine=GovernedTraining(model,None,mapping=history.mapping,instrument=BertInstrument(model,chunk=4))
            engine.scheduler=history;engine.governor=RecoveryGovernor(history.mapping,history)
            extend_bank(engine,oldx,oldy,px,py)
        del saved
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        def record(offset):
            ev=evaluate(model,x,y,test);per=ev['per_relation']
            ev.update(offset=offset,old=float(np.mean([per[str(k)] for k in range(8)])),
                      new=float(np.mean([per[str(k)] for k in range(8,16)])))
            metrics.append(ev);print(arm,offset,'old',ev['old'],'new',ev['new'],flush=True)
        record(0)
        with (out/'STEPS.jsonl').open('x',encoding='utf8') as f:
            for offset in range(1,61):
                info=step(model,opt,engine,x,y,order[offset-1],None if arm=='new_governor' else replay[offset-1],210+offset,px,py)
                row=dict(offset=offset,info=info);rows.append(row);f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                if offset%20==0:record(offset)
                if time.perf_counter()-start>900:raise TimeoutError('fixed900s budget')
        result=dict(status='COMPLETED',metrics=metrics,wall=time.perf_counter()-start,
            peak_allocated=torch.cuda.max_memory_allocated(),full_calls=sum(r['info']['instrument_called'] for r in rows),
            alpha_counts=dict(Counter(r['info']['alpha'] for r in rows)),log_sha256=sha(out/'STEPS.jsonl'))
    except Exception:
        result=dict(status='FAILED',traceback=traceback.format_exc(),metrics=metrics,steps=len(rows),wall=time.perf_counter()-start)
    write(out/'RESULT.json',result);print(arm,result['status'],flush=True)
    if result['status']!='COMPLETED':raise RuntimeError(result['traceback'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['new_governor','replay_native','replay_governor']);run(p.parse_args().arm)
