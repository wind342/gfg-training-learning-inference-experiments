"""Supplementary instrumented timing, never used for capability/preset selection.

Seal this file before invocation. Synchronizing instrumentation is deliberately
separate from scale_job's uninstrumented end-to-end time-to-capability runs.
"""
import argparse
from collections import defaultdict
import math
import time
from types import SimpleNamespace
from unittest.mock import patch
import torch
from . import controller as runtime
from .evidence import OUT, setup, write, sha
from .scale_job import create, update, banks, evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scale',choices=('70','160'),required=True)
    a=parser.parse_args();setup();width={'70':704,'160':1024}[a.scale]
    args=SimpleNamespace(scale=a.scale,seed=701,lr=.0001*math.sqrt(64/width)*.125,
                         arm='dynamic',cadence=5)
    x,y,control,evaluation=banks();x=x.cuda();y=y.cuda()
    vx,vy=x[control],y[control];ex,ey=x[evaluation],y[evaluation]
    gen=torch.Generator().manual_seed(99702)
    batches=[torch.randint(529,(128,),generator=gen).cuda() for _ in range(6)]
    objects=create(args);model,registry,opt,engine=objects
    # Same model warmup; this is NOT a seed-to-capability comparison.
    for t in (1,2):update(args,objects,t,x[batches[t-1]],y[batches[t-1]],vx,vy)
    measurements=[];times=defaultdict(float);phase=['update']

    def timed(name,fn):
        def wrapper(*pos,**kw):
            torch.cuda.synchronize();start=time.perf_counter()
            result=fn(*pos,**kw)
            torch.cuda.synchronize();times[phase[0]+'/'+name]+=time.perf_counter()-start
            return result
        return wrapper

    original_forward=model.forward;original_backward=torch.Tensor.backward
    with patch.object(model,'forward',timed('forward',original_forward)), \
         patch.object(torch.Tensor,'backward',timed('backward',original_backward)), \
         patch.object(runtime,'clip',timed('global_clip',runtime.clip)), \
         patch.object(opt,'step',timed('optimizer_step',opt.step)), \
         patch.object(engine.owner,'propose',timed('proposal_including_optimizer',engine.owner.propose)), \
         patch.object(engine.adapter,'jets',timed('directional_including_materialization',engine.adapter.jets)), \
         patch.object(runtime,'calculate',timed('boundary_including_selector',runtime.calculate)), \
         patch.object(engine.owner,'commit',timed('commit',engine.owner.commit)):
        for t in range(3,7):
            times.clear();phase[0]='update';torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();start=time.perf_counter()
            info=update(args,objects,t,x[batches[t-1]],y[batches[t-1]],vx,vy)
            torch.cuda.synchronize();update_wall=time.perf_counter()-start
            phase[0]='evaluation';start=time.perf_counter();_,accuracy=evaluate(model,ex,ey)
            torch.cuda.synchronize();evaluation_wall=time.perf_counter()-start
            measurements.append({'step':t,'decision':(t-1)%5==0,'seconds':dict(times),
                'instrumented_update_wall':update_wall,'evaluation_wall':evaluation_wall,
                'accuracy':accuracy,'alpha':info['alpha'],
                'peak_allocated':torch.cuda.max_memory_allocated(),
                'peak_reserved':torch.cuda.max_memory_reserved()})
    size=lambda v:v.numel()*v.element_size()
    memory={'parameters':sum(size(p) for p in model.parameters()),
            'gradients':sum(size(p.grad) for p in model.parameters() if p.grad is not None),
            'optimizer':sum(size(v) for st in opt.state.values() for v in st.values() if isinstance(v,torch.Tensor)),
            'candidate':sum(size(v) for v in engine.owner.proposal.candidate.values()),
            'delta':sum(size(v) for v in engine.owner.proposal.delta.values())}
    write(OUT/'profiling'/f'{a.scale}.json',{
        'scale':a.scale,'seed':701,'source_sha256':sha(__file__),
        'kind':'synchronizing_diagnostic_not_production_benchmark',
        'world_size':1,'memory_bytes':memory,'rows':measurements,
        'communication_seconds':'NOT_APPLICABLE_SINGLE_GPU',
        'directional_workspace_isolated':'NOT_MEASURED',
        'note':'Nested timers overlap; optimizer is inside proposal. Do not sum all rows.'})


if __name__=='__main__':main()
