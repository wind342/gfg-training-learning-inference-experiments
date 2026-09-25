"""External audit fixture for the portable launcher, NOT a runtime dependency."""
import os
import torch
from .evidence import OUT,write,setup
from .adapters import NanoGPTAdapter
from .identity import CanonicalParameterRegistry
from experiments.gfg_controller_distributed_runtime_v1.checks import old,b


def factory(rank,world_size,device):
    setup();row=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs'][0];rng=b.read(old.HERE/'PHASE1_SEAL.json')['source_rng_protocol_sha256']
    rt,start,batches,vx,vy=b.load_row(row);model=rt.model;old.reset(model,start)
    adapter=NanoGPTAdapter(model);registry=CanonicalParameterRegistry(model);records=[]
    lr=float(os.environ.get('GFG_V2_LAUNCH_LR','.003'))
    def callback(t,info,engine):
        records.append({'step':t,'info':info})
        if t==6:write(OUT/'launcher'/f'world{world_size}-lr{lr}-rank{rank}.json',{'records':records,'tied_identity':registry.receipt(model),
            'world':world_size,'rank':rank,'packed':engine.owner.packer.events,'status':'COMPLETED','fixture':'identical-global-batch partitioned over ranks'})
    sliced=[]
    for x,y in batches[:6]:
        assert len(x)%world_size==0
        lo=len(x)*rank//world_size;hi=len(x)*(rank+1)//world_size;sliced.append((x[lo:hi],y[lo:hi]))
    return {'model':model,'adapter':adapter,'registry':registry,'optimizer_factory':lambda ps:torch.optim.AdamW(ps,lr=lr,betas=(.9,.98),weight_decay=0,fused=True,foreach=False),
        'batches':sliced,'seeds':[b.old.seed_for(row,t,rng) for t in range(6)],'learning_rates':[lr]*6,'probes':vx,'targets':vy,'callback':callback}
