"""Portable real FSDP gate against an independent full-batch e49 oracle.

torchrun --standalone --nproc-per-node=2 -m \
experiments.gfg_controller_distributed_runtime_v2.distributed_gate --lr .003

Run .001/.003/.01 separately. This is audit code, not a runtime dependency.
Requires an existing evidence destination and historical witness on the host.
No multi-GPU claim is made merely because this entry point exists.
"""
import argparse
import copy
from datetime import timedelta
import os
from unittest.mock import patch
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from .evidence import OUT, setup, write, sha
from .identity import CanonicalParameterRegistry
from .adapters import NanoGPTAdapter
from .controller import FunctionalController
from .gates import cpu, snapshot, expanded, old, b, GPUBlockwiseController


def main():
    p=argparse.ArgumentParser();p.add_argument('--lr',type=float,required=True);a=p.parse_args()
    setup();world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK'])
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    backend='nccl' if dist.is_nccl_available() else 'gloo'
    if world>1 and backend!='nccl':raise RuntimeError('Multi-CUDA FSDP requires NCCL')
    dist.init_process_group(backend,timeout=timedelta(seconds=120))
    root=OUT/'distributed-oracle-gate-v2'/f'world{world}-lr{a.lr}'/f'rank{rank}'
    root.mkdir(parents=True,exist_ok=False)
    try:
        row=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs'][0]
        rng=b.read(old.HERE/'PHASE1_SEAL.json')['source_rng_protocol_sha256']
        rt,start,batches,vx,vy=b.load_row(row);model=rt.model;old.reset(model,start)
        oracle=copy.deepcopy(model);registry=CanonicalParameterRegistry(model)
        adapter=NanoGPTAdapter(model);identity=registry.receipt(model)
        units=adapter.wrap_fsdp(model,init_device_mesh('cuda',(world,)))
        opt=old.prior.native.optimizer(model,a.lr)
        reference_opt=old.prior.native.optimizer(oracle,a.lr)
        reference=GPUBlockwiseController(oracle,audit=True,hash_parameters=b.phash)
        engine=FunctionalController(model,adapter,registry=registry,units=units,
                                    group=dist.group.WORLD,audit=True)
        captured={};native_propose=engine.owner.propose
        def observe_proposal(o):
            native_propose(o)
            captured['candidate']=expanded(engine.owner.proposal.candidate,engine.owner.parameters())
            captured['delta']=expanded(engine.owner.proposal.delta,engine.owner.parameters())
        engine.owner.propose=observe_proposal
        raw={};records=[];alpha=1.
        for t in range(1,7):
            before=snapshot(oracle);ref_candidate={};original_step=reference_opt.step
            def observe_reference(*args,**kwargs):
                value=original_step(*args,**kwargs);ref_candidate.update(snapshot(oracle));return value
            native_clip=torch.nn.utils.clip_grad_norm_;norms={}
            def observer(side):
                def call(parameters,*args,**kwargs):
                    parameters=list(parameters)
                    for i,q in enumerate(parameters):raw[f'{t}/{side}/raw_gradient/{i}']=cpu(q.grad).numpy()
                    value=native_clip(parameters,*args,**kwargs);norms[side]=cpu(value)
                    return value
                return call
            x,y=batches[t-1];assert len(x)%world==0
            lo,hi=len(x)*rank//world,len(x)*(rank+1)//world
            seed=b.old.seed_for(row,t-1,rng)
            with patch.object(torch.nn.utils,'clip_grad_norm_',observer('actual')):
                actual=engine.update(opt,t,x[lo:hi],y[lo:hi],vx,vy,seed,a.lr)
            with patch.object(reference_opt,'step',observe_reference), \
                 patch.object(torch.nn.utils,'clip_grad_norm_',observer('reference')):
                alpha,ref_info,_=reference.update(oracle,reference_opt,t,x,y,vx,vy,seed,alpha)
            pairs={'norm':(norms['actual'],norms['reference'])}
            for i,(name,q) in enumerate(registry.resolve(model).items()):
                rq=oracle.get_parameter(name)
                pairs[f'candidate/{i}']=(captured['candidate'][name],ref_candidate[name])
                pairs[f'delta/{i}']=(captured['delta'][name],ref_candidate[name]-before[name])
                pairs[f'parameter/{i}']=(cpu(q),cpu(rq))
                pairs[f'gradient/{i}']=(cpu(q.grad),cpu(rq.grad))
                for kind in ('step','exp_avg','exp_avg_sq'):
                    pairs[f'moment/{kind}/{i}']=(cpu(opt.state[q][kind]),cpu(reference_opt.state[rq][kind]))
            with torch.no_grad():pairs['logits']=(cpu(model(vx)[0]),cpu(oracle(vx)[0]))
            if engine.last:
                for j in range(3):pairs[f'h{j}']=(cpu(engine.last['hidden'][j]),cpu(reference.last_debug['hidden'][j]))
                pairs['minima']=(cpu(engine.last['minima']),cpu(reference.last_debug['minima']))
            checks={}
            for name,(left,right) in pairs.items():
                raw[f'{t}/actual/{name}']=left.numpy();raw[f'{t}/reference/{name}']=right.numpy()
                checks[name]={'byte_equal':torch.equal(left,right),
                    'max_abs':float((left.double()-right.double()).abs().max())}
            records.append({'step':t,'checks':checks,'alpha_equal':actual['alpha']==alpha,
                'scores_equal':actual.get('scores')==ref_info.get('scores'),
                'actual':actual,'reference_alpha':alpha,'reference_scores':ref_info.get('scores')})
            assert registry.receipt(model)==identity
        with (root/'RAW.npz').open('xb') as f:np.savez_compressed(f,**raw)
        exact=all(all(c['byte_equal'] for c in r['checks'].values()) and r['alpha_equal'] and r['scores_equal'] for r in records)
        write(root/'RESULT.json',{'world_size':world,'rank':rank,'lr':a.lr,'records':records,
            'classification':'BYTE_EXACT' if exact else 'MISMATCH_REQUIRES_REVIEW',
            'tied_identity':identity,'raw_sha256':sha(root/'RAW.npz'),
            'batch_semantics':'One fixed global batch split equally over FSDP ranks; oracle uses full global batch',
            'tolerance_upgrade':False})
        rt.close()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
