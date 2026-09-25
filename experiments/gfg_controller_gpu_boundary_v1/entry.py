"""Standalone training loop: no scientific oracle, CPU model hash, or old ledger.

A local factory supplies model, optimizer, batches, seeds, learning_rates,
probe_inputs, probe_targets and optionally separate evaluation inputs/targets.
No model/data is downloaded. Backend is explicit, not selected after outcomes.
"""
import argparse
import importlib
import json
from pathlib import Path
import time
import torch
from .runtime import GPUBlockwiseController
from experiments.gfg_controller_large_model_runtime_v1.blockwise_runtime import BlockwiseController


@torch.no_grad()
def accuracy(model,x,y):
    model.eval();z=model(x)[0][:,-1,:];target=y[:,-1]
    truth=z.gather(1,target[:,None]).squeeze(1)
    z.scatter_(1,target[:,None],-torch.inf)
    return int((truth>z.max(1).values).sum())/target.numel()


def train(model,optimizer,batches,seeds,learning_rates,probe_inputs,probe_targets,*,
          evaluation_inputs=None,evaluation_targets=None,backend='gpu',target_tile=256,competitor_tile=128,
          stop_accuracy=.99,evaluate_every=1):
    if backend not in ('gpu','cpu'):raise ValueError('backend must be explicit gpu or cpu')
    if len(batches)!=len(seeds) or len(batches)!=len(learning_rates):raise ValueError('schedule length mismatch')
    if evaluate_every<1:raise ValueError('evaluate_every must be positive')
    # Model/optimizer creation and data placement are owned by the caller.
    engine=GPUBlockwiseController(model,target_tile=target_tile,competitor_tile=competitor_tile) if backend=='gpu' else BlockwiseController(model,audit=False)
    ex=probe_inputs if evaluation_inputs is None else evaluation_inputs
    ey=probe_targets if evaluation_targets is None else evaluation_targets
    torch.cuda.synchronize();start=time.perf_counter();alpha=1.;records=[]
    initial=accuracy(model,ex,ey);acc=initial
    torch.cuda.reset_peak_memory_stats()
    for t,((x,y),seed,lr) in enumerate(zip(batches,seeds,learning_rates),1):
        if acc>=stop_accuracy:break
        for group in optimizer.param_groups:group['lr']=lr
        alpha,details,_=engine.update(model,optimizer,t,x,y,probe_inputs,probe_targets,seed,alpha)
        evaluated=t%evaluate_every==0 or t==len(batches)
        if evaluated:acc=accuracy(model,ex,ey)
        records.append({'t':t,'alpha':alpha,'accuracy':acc if evaluated else None,'details':details})
    torch.cuda.synchronize()
    return {'backend':backend,'initial_accuracy':initial,'final_accuracy':acc,'steps':records,
        'checked_wall_seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated(),
        'peak_reserved':torch.cuda.max_memory_reserved(),'control_buffer_bytes':engine.buffers.allocated_bytes,
        'boundary_decision_d2h_bytes':engine.decision_transfer_bytes if backend=='gpu' else None,
        'probe_count':len(probe_inputs),'evaluation_count':len(ex),'evaluate_every':evaluate_every,
        'has_per_step_model_hash':False,'cadence':5,'target_tile':target_tile,'competitor_tile':competitor_tile}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factory',required=True,help='trusted local module:function returning train() arguments')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--backend',choices=('gpu','cpu'),required=True)
    p.add_argument('--target-tile',type=int,default=256);p.add_argument('--competitor-tile',type=int,default=128)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    module,name=args.factory.split(':');config=getattr(importlib.import_module(module),name)()
    result=train(**config,backend=args.backend,target_tile=args.target_tile,competitor_tile=args.competitor_tile)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
