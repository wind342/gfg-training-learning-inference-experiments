"""Strict matched-budget comparison; native histories and all identities retained."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from unittest.mock import patch

from experiments.gfg_controller_resident_reuse_v1 import run as resident
from experiments.gfg_controller_resident_reuse_v1.engine import NanoGPTAdapter, ResidentBuffers
from experiments.gfg_nanogpt_functional_online_controller_v1.independent_checker import independently_choose
from .controller import Controller
import numpy as np
import torch

b=resident.b; HERE=Path(__file__).resolve().parent
OUT=Path('E:/gfg-evidence/controller-matched-fused-adamw-v1')
METHODS=('adamw','controller'); RATES=(.0003,.001,.003,.01)
BASE='e5a7dc724d5ac7d99e62dc36264c8fc644edc083'


def optimizer(model,lr):
    return torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=0.,foreach=False,fused=True)


def bridge(rows,seal):
    results=[]
    for case in b.read(resident.HERE/'verification/RESULTS.json')['histories']:
        row=next(r for r in rows if r['entry_id']==case['entry']); ci=case['config']
        original=b.read(b.OUT/'search'/case['entry']/('controller-'+str(ci))/'0/TRAJECTORY.json')
        loaded=b.load_row(row)
        try:
            with patch.object(resident,'Controller',Controller):
                result=resident.execute(row,loaded,ci,original,seal['source_rng_protocol_sha256'],True)
            results.append({'entry':case['entry'],'steps':result['first99_steps'],
                            'exact':True,'final_state_sha256':result['final_state_sha256']})
        finally:loaded[0].close()
    b.write(HERE/'SHORTCUT_BRIDGE.json',results)


def eval_logits(logits,y):
    target=y[:,-1].cpu().numpy()
    other=logits.copy();other[np.arange(len(target)),target]=-np.inf
    gaps=logits[np.arange(len(target)),target].astype(np.float64)-other.max(1).astype(np.float64)
    assert np.isfinite(gaps).all()
    return float((gaps>0).mean())


def run_case(row,loaded,method,ci,rng_hash,mode,repeat=0):
    path=OUT/mode/row['entry_id']/f'{method}-{ci}'/str(repeat)
    if (path/'TRAJECTORY.json').exists():return b.read(path/'TRAJECTORY.json')
    assert not path.exists(),f'partial output preserved: {path}'
    reference=None; old_logits=None
    if mode=='timing':
        source=OUT/'search'/row['entry_id']/f'{method}-{ci}'/'0'
        reference=b.read(source/'TRAJECTORY.json')
        with np.load(source/'OUTPUTS.npz',allow_pickle=False) as f:old_logits=f['logits'].copy()
    rt,start,batches,vx,vy=loaded;model=rt.model
    with torch.no_grad():
        for n,p in model.named_parameters():p.copy_(start[n].to('cuda'))
    model.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.reset_peak_memory_stats()
    b.sync();wall=time.perf_counter();tick=wall
    opt=optimizer(model,RATES[ci])
    audit_buffers=ResidentBuffers(model)
    engine=Controller(model,NanoGPTAdapter(model),b.select,b.phash,b.selection_boundary) if method=='controller' else None
    # Controller uses the same buffers for its computation and the shared auditor.
    if engine is not None:audit_buffers=engine.buffers
    b.sync();construction=time.perf_counter()-tick
    compute=construction
    def evaluate():
        model.eval()
        with torch.no_grad():return model(vx)[0][:,-1,:].cpu().numpy()
    initial=evaluate();initial_accuracy=eval_logits(initial,vy)
    all_logits=[initial];coordinate_records={};steps=[];alpha=1.;chain=None
    assert reference is None or np.array_equal(initial,old_logits[0]),'initial replay'
    for t in range(1,201):
        if (steps[-1]['accuracy'] if steps else initial_accuracy)>=.99:break
        x,y=batches[t-1];seed=b.old.seed_for(row,t-1,rng_hash)
        b.sync();tick=time.perf_counter()
        lr=RATES[ci]*(.1+.9*.5*(1+math.cos(math.pi*(t-1)/199)))
        for group in opt.param_groups:group['lr']=lr
        if engine is not None:
            alpha,details,coords=engine.update(model,opt,t,x,y,vx,vy,seed,alpha)
        else:
            model.train();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            opt.zero_grad(set_to_none=True);loss=model(x,y)[1];loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            details={'loss':float(loss.detach())};coords=None
        b.sync();update_time=time.perf_counter()-tick;compute+=update_time
        coord_hash=None
        if coords is not None:
            coord_hash=b.identity({k:b.tensor_sha256(torch.from_numpy(coords[k])) for k in b.KEYS})
            # Recompute the selection independently; do not omit audit in one arm.
            assert independently_choose(coords)==(alpha,details['scores'])
            if mode=='search':
                for k in b.KEYS:coordinate_records[f'{t}_{k}']=coords[k]
        # Same post-state hash procedure in both arms; retained alpha=1 shortcut
        # saves only update work, not the external audit.
        post_hash=b.phash(audit_buffers.snapshot_model())
        logits=evaluate();accuracy=eval_logits(logits,vy);all_logits.append(logits)
        fact={'t':t,'seed':seed,'lr':lr,'details':details,'coordinates_sha256':coord_hash,
            'post_parameters_sha256':post_hash,'logits_sha256':b.tensor_sha256(logits),
            'accuracy':accuracy,'previous_sha256':chain}
        chain=b.identity(fact)
        record={**fact,'sha256':chain,'update_seconds':update_time,
                'cumulative_compute_seconds':compute,'cumulative_checked_wall_seconds':time.perf_counter()-wall}
        if reference is not None:
            expected=reference['steps'][t-1]
            assert all(record[k]==expected[k] for k in fact),'fact replay'
            assert chain==expected['sha256'] and np.array_equal(logits,old_logits[t]),'logit replay'
        steps.append(record)
    final=b.identity(b.structure({'parameters':b.params(model),'optimizer':opt.state_dict()}))
    if reference is not None:
        assert len(steps)==len(reference['steps']) and final==reference['final_state_sha256'],'final replay'
    b.sync();elapsed=time.perf_counter()-wall
    result={'entry_id':row['entry_id'],'method':method,'config':ci,'initial_accuracy':initial_accuracy,
        'construction_seconds':construction,'compute_seconds':compute,'checked_wall_seconds':elapsed,
        'initial_logits_sha256':b.tensor_sha256(initial),'steps':steps,'final_state_sha256':final,
        'skipped_identity_steps':engine.identity_steps_skipped if engine else 0,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'exact_replay':reference is not None}
    path.mkdir(parents=True)
    with (path/'OUTPUTS.npz').open('xb') as f:np.savez_compressed(f,logits=np.stack(all_logits),target=vy[:,-1].cpu().numpy())
    if coordinate_records:
        with (path/'COORDINATES.npz').open('xb') as f:np.savez_compressed(f,**coordinate_records)
    result['artifacts']={p.name:b.sha(p) for p in path.iterdir() if p.is_file()}
    b.write(path/'TRAJECTORY.json',result)
    print(mode,row['entry_id'],method,ci,repeat,len(steps),round(steps[-1]['accuracy'] if steps else initial_accuracy,4),flush=True)
    return result


def hit(result,threshold):
    if result['initial_accuracy']>=threshold:return 0
    return next((s['t'] for s in result['steps'] if s['accuracy']>=threshold),None)


def selections(rows):
    records={(r['entry_id'],m,ci):b.read(OUT/'search'/r['entry_id']/f'{m}-{ci}'/'0/TRAJECTORY.json') for r in rows for m in METHODS for ci in range(4)}
    choices={}
    for row in rows:
        ranks=[]
        for ci in range(4):
            costs=[]
            for threshold in (.99,.95):
                values=[hit(records[(other['entry_id'],m,ci)],threshold) for other in rows if other['entry_id']!=row['entry_id'] for m in METHODS]
                values=[200 if v is None else v for v in values if v!=0]
                costs.append(statistics.mean(values))
            ranks.append([*costs,ci])
        choices[row['entry_id']]={'config':min(ranks)[-1],'ranks':ranks}
    return choices


def summarize(rows,choices):
    result={};per_history=[]
    for threshold in (.95,.99):
        result[str(threshold)]={}
        for method in METHODS:
            entries=[]
            for row in rows:
                ci=choices[row['entry_id']]['config']
                records=[b.read(OUT/'timing'/row['entry_id']/f'{method}-{ci}'/str(rep)/'TRAJECTORY.json') for rep in range(3)]
                h=hit(records[0],threshold)
                if h==0:continue
                def cost(record,key):
                    if h is None:return record[key]
                    return record['steps'][h-1]['cumulative_'+key]
                entries.append({'entry':row['entry_id'],'hit':h,'restricted_steps':200 if h is None else h,
                    'compute_seconds':statistics.median(cost(r,'compute_seconds') for r in records),
                    'checked_wall_seconds':statistics.median(cost(r,'checked_wall_seconds') for r in records)})
            result[str(threshold)][method]={'eligible':len(entries),'reached':sum(e['hit'] is not None for e in entries),
                'restricted_steps_mean':statistics.mean(e['restricted_steps'] for e in entries),
                'compute_seconds_mean':statistics.mean(e['compute_seconds'] for e in entries),
                'checked_wall_seconds_mean':statistics.mean(e['checked_wall_seconds'] for e in entries),'entries':entries}
    for row in rows:
        per_history.append({'entry':row['entry_id'],'lr':RATES[choices[row['entry_id']]['config']],
            'all_lr_first99':{str(lr):{m:hit(b.read(OUT/'search'/row['entry_id']/f'{m}-{ci}'/'0/TRAJECTORY.json'),.99) for m in METHODS} for ci,lr in enumerate(RATES)}})
    return {'status':'COMPLETE_PENDING_INDEPENDENT_AUDIT','thresholds':result,'per_history':per_history}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','summarize']);args=ap.parse_args()
    b.old.setup()
    oldrepo=b.REPO;b.REPO=resident.previous.DEPENDENCIES.parent
    oldseal=b.sealcheck();b.REPO=oldrepo
    rows=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs']
    if args.action=='summarize':
        b.write(HERE/'RESULTS.json',summarize(rows,b.read(HERE/'SELECTION.json')));return
    if (HERE/'RUN_SEAL.json').exists():
        for n,d in b.read(HERE/'RUN_SEAL.json')['files'].items():assert b.sha(HERE/n)==d,n
    else:
        assert not OUT.exists()
        b.write(HERE/'RUN_SEAL.json',{'base_commit':BASE,'files':{p.name:b.sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md')},
            'utc':b.old.utc(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'assignment_sha256':b.sha(b.old.HERE/'RUN_ASSIGNMENT.json'),
            'prior_engine_sha256':b.sha(resident.HERE/'engine.py')})
    if not (HERE/'SHORTCUT_BRIDGE.json').exists():bridge(rows,oldseal)
    for i,row in enumerate(rows):
        loaded=b.load_row(row)
        try:
            for ci in range(4):
                order=METHODS if (i+ci)%2==0 else METHODS[::-1]
                for method in order:run_case(row,loaded,method,ci,oldseal['source_rng_protocol_sha256'],'search')
        finally:loaded[0].close();del loaded;gc.collect()
    choices=selections(rows)
    if (HERE/'SELECTION.json').exists():assert b.read(HERE/'SELECTION.json')==choices
    else:b.write(HERE/'SELECTION.json',choices)
    for i,row in enumerate(rows):
        loaded=b.load_row(row)
        try:
            for rep in range(3):
                for method in (METHODS if (i+rep)%2==0 else METHODS[::-1]):
                    run_case(row,loaded,method,choices[row['entry_id']]['config'],oldseal['source_rng_protocol_sha256'],'timing',rep)
        finally:loaded[0].close();del loaded;gc.collect()
    b.write(HERE/'RESULTS.json',summarize(rows,choices))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
