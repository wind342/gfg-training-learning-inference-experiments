"""Frozen bounded gate and parent orchestration; no original evidence writes."""
from pathlib import Path
from unittest.mock import patch
import copy
import gc
import json
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from experiments.gfg_controller_large_model_runtime_v1 import phase1 as old
from experiments.gfg_controller_gpu_boundary_v1.runtime import GPUBlockwiseController
from .distributed_entry import EmulatedController

b=old.b;HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
OUT=Path('E:/gfg-evidence/controller-distributed-runtime-v1');BASE='e49cb97db7467fbda8fa9d06c8c4a2a945707d43'


def gate(row,world,lr,rng):
    loaded=b.load_row(row);rt,start,batches,vx,vy=loaded;model=rt.model
    old.reset(model,start);oracle=copy.deepcopy(model)
    opt=old.prior.native.optimizer(oracle,lr);ref=GPUBlockwiseController(oracle,audit=True,hash_parameters=b.phash)
    capture={};raw={};receipts=[];alpha=1.
    def hook(stage,engine):
        if stage=='proposal':
            own=engine.owner
            capture['candidate']={n:own.gather(n,'candidate').view_as(p).cpu() for n,p in own.full.items()}
            capture['delta']={n:own.gather(n,'delta').view_as(p).cpu() for n,p in own.full.items()}
            capture['moments']=all(torch.equal(own.optimizer_state(n,k).cpu(),opt.state[p][k].cpu()) for n,p in oracle.named_parameters() for k in ('step','exp_avg','exp_avg_sq'))
            capture['states']={f'{i}_{k}':own.optimizer_state(n,k).cpu().numpy() for i,(n,p) in enumerate(oracle.named_parameters()) for k in ('step','exp_avg','exp_avg_sq')}
            capture['reference_states']={f'{i}_{k}':opt.state[p][k].cpu().numpy().copy() for i,(n,p) in enumerate(oracle.named_parameters()) for k in ('step','exp_avg','exp_avg_sq')}
    engine=EmulatedController(model,lambda ps:torch.optim.AdamW(ps,lr=lr,betas=(.9,.98),weight_decay=0,foreach=False,fused=True),world,audit_hook=hook)
    try:
        for t in range(1,7):
            before=b.params(oracle);actual_before=b.params(model);seed=b.old.seed_for(row,t-1,rng);native=opt.step;proposal={}
            def observed(*a,**kw):
                result=native(*a,**kw);proposal.update(b.params(oracle));return result
            with patch.object(opt,'step',observed):alpha,rd,_=ref.update(oracle,opt,t,*batches[t-1],vx,vy,seed,alpha)
            details=engine.step(t,*batches[t-1],vx,vy,seed,lr)
            expected_delta={n:proposal[n]-before[n] for n in proposal}
            current=b.params(model);reference_current=b.params(oracle)
            for i,n in enumerate(proposal):
                for key,values in [('before',actual_before),('candidate',capture['candidate']),('reference_candidate',proposal),('delta',capture['delta']),('reference_delta',expected_delta),('parameters',current),('reference_parameters',reference_current)]:
                    raw[f'{t}_{key}_{i}']=values[n].numpy()
            for key in capture['states']:
                raw[f'{t}_state_{key}']=capture['states'][key]
                raw[f'{t}_reference_state_{key}']=capture['reference_states'][key]
            checks={'candidate':b.phash(capture['candidate'])==b.phash(proposal),'delta':b.phash(capture['delta'])==b.phash(expected_delta),
                'moments':capture['moments'],'alpha':details['alpha']==alpha,'scores':details.get('scores')==rd.get('scores'),
                'parameters':b.phash(b.params(model))==b.phash(b.params(oracle))}
            z,_=old.evaluate(model,vx,vy);oz,_=old.evaluate(oracle,vx,vy);checks['logits']=np.array_equal(z,oz)
            raw[f'{t}_logits']=z;raw[f'{t}_reference_logits']=oz
            if engine.last is not None:
                for j in range(3):
                    x=engine.last['hidden'][j].cpu().numpy();y=ref.last_debug['hidden'][j].cpu().numpy()
                    raw[f'{t}_h{j}']=x;raw[f'{t}_reference_h{j}']=y;checks[f'h{j}']=np.array_equal(x,y)
                raw[f'{t}_local_minima']=torch.stack(engine.last['local_minima']).cpu().numpy()
                raw[f'{t}_minima']=engine.last['minima'].cpu().numpy();raw[f'{t}_reference_minima']=ref.last_debug['minima'].cpu().numpy()
                raw[f'{t}_pre_logits']=engine.last['pre_logits'].cpu().numpy()
                checks['minima']=np.array_equal(raw[f'{t}_minima'],raw[f'{t}_reference_minima'])
            receipts.append({'t':t,'checks':{k:bool(v) for k,v in checks.items()},'alpha':details['alpha'],'reference_alpha':alpha,
                'scores':details.get('scores'),'reference_scores':rd.get('scores'),
                'candidate_sha256':b.phash(capture['candidate']),'reference_candidate_sha256':b.phash(proposal),
                'delta_sha256':b.phash(capture['delta']),'reference_delta_sha256':b.phash(expected_delta),
                'post_sha256':b.phash(b.params(model)),'reference_post_sha256':b.phash(b.params(oracle))})
        name=f'w{world}-lr{lr}';path=OUT/'gate'/name;path.mkdir(parents=True)
        with (path/'RAW.npz').open('xb') as f:np.savez_compressed(f,**raw)
        result={'world_size':world,'mode':'EMULATED','lr':lr,'receipts':receipts,
            'exact':all(all(r['checks'].values()) for r in receipts),'rank_controller_bytes':[2*r.bytes() for r in engine.owner.ranks],
            'rank_parameter_bytes':[r.bytes() for r in engine.owner.ranks],'parameter_count':sum(p.numel() for p in model.parameters()),
            'events':engine.owner.events,'peak_block_materialization':engine.owner.peak_materialized,
            'communication':engine.last['communication'] if engine.last else None,'raw_sha256':b.sha(path/'RAW.npz')}
        result['parameter_shapes']={n:list(p.shape) for n,p in model.named_parameters()}
        result['parameter_names']=[n for n,p in model.named_parameters()]
        result['targets']=vy[:,-1].cpu().tolist()
        result['memory']={'physical_peak_allocated':torch.cuda.max_memory_allocated(),'physical_peak_reserved':torch.cuda.max_memory_reserved(),
            'scope':'single physical GPU including audit oracle; NOT runtime-only or per-rank peak',
            'rank_gradient_bytes':[r.bytes() for r in engine.owner.ranks],
            'rank_optimizer_state_bytes':[sum(v.numel()*v.element_size() for state in o.state.values() for v in state.values() if isinstance(v,torch.Tensor)) for o in engine.owner.optimizers],
            'rank_candidate_bytes':[r.bytes() for r in engine.owner.ranks],'rank_delta_bytes':[r.bytes() for r in engine.owner.ranks]}
        # Last step6 is a decision; communication is available.
        b.write(path/'RESULT.json',result);print('GATE',name,result['exact'],flush=True)
        return {'name':name,'exact':result['exact'],'world_size':world}
    finally:rt.close()


def main():
    b.old.setup();assert old.git('rev-parse','HEAD')==BASE and not OUT.exists()
    hardware={'torch':torch.__version__,'devices':torch.cuda.device_count(),'gpu':torch.cuda.get_device_name(0),
        'vram':torch.cuda.get_device_properties(0).total_memory,'nccl':torch.distributed.is_nccl_available(),'gloo':torch.distributed.is_gloo_available()}
    seal={'base':BASE,'hardware':hardware,'sources':{p.name:b.sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md')},
        'old_files':{n:b.sha(REPO/n) for n in old.git('ls-files').splitlines()},'utc':b.old.utc(),
        'frozen_tag':old.git('rev-parse',old.prior.TAG),'frozen_commit':old.git('rev-parse',old.prior.TAG+'^{}')}
    b.write(HERE/'DISTRIBUTED_SEAL.json',seal);OUT.mkdir()
    row=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs'][0];rng=b.read(old.HERE/'PHASE1_SEAL.json')['source_rng_protocol_sha256']
    results=[]
    for world in (1,2,3):
        for lr in (.001,.003,.01):
            torch.cuda.reset_peak_memory_stats()
            results.append(gate(row,world,lr,rng));gc.collect();torch.cuda.empty_cache()
    b.write(HERE/'GATE_RESULTS.json',{'cases':results,'all_exact':all(r['exact'] for r in results)})


if __name__=='__main__':
    try:main()
    except Exception:
        with (HERE/f'FAILURE_{time.time_ns()}.txt').open('x',encoding='utf-8') as f:f.write(traceback.format_exc())
        raise
