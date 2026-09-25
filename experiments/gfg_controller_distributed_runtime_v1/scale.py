"""Isolated single-GPU controlled-scale checks using the certified e49 runtime."""
import argparse
import gc
import hashlib
import importlib.util
import math
import sys
import time
import traceback
import numpy as np
import torch
from .checks import b,OUT,HERE
from experiments.gfg_controller_gpu_boundary_v1.runtime import GPUBlockwiseController
from experiments.gfg_controller_large_model_runtime_v1.runtime import ProposalStorage

CONFIGS={'70':(12,704,11),'160':(13,1024,16)}


def banks():
    ids=sorted(range(529),key=lambda i:hashlib.sha256(f'distributed-scale-bank-v1:{i}'.encode()).digest())
    x=torch.tensor([[i//23,i%23,23] for i in range(529)],dtype=torch.long)
    y=torch.tensor([[-1,-1,(i//23+i%23)%23] for i in range(529)],dtype=torch.long)
    return x,y,ids[:212],ids[212:]


def model_for(scale,seed):
    spec=importlib.util.spec_from_file_location('sealed_scale_nanogpt',str(b.old.TRAINER/'model.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    layers,width,heads=CONFIGS[scale];torch.manual_seed(seed)
    return module.GPT(module.GPTConfig(n_layer=layers,n_head=heads,n_embd=width,vocab_size=24,block_size=3,bias=False,dropout=0)).cuda()


@torch.no_grad()
def evaluate(model,x,y):
    model.eval();z=model(x)[0][:,-1,:];truth=z.gather(1,y[:,-1,None]).squeeze(1)
    other=z.clone();other.scatter_(1,y[:,-1,None],-torch.inf)
    gap=truth.double()-other.max(1).values.double()
    return z.cpu().numpy(),int((gap>0).sum())/len(x)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scale',choices=CONFIGS,required=True);p.add_argument('--arm',choices=('native','fixed','dynamic'),required=True)
    p.add_argument('--phase',required=True);p.add_argument('--seed',type=int,default=101);p.add_argument('--lr',type=float,required=True);p.add_argument('--steps',type=int,required=True)
    a=p.parse_args();b.old.setup();path=OUT/'scale'/a.scale/a.phase/f'{a.arm}-seed{a.seed}-lr{a.lr}'
    assert not path.exists();path.mkdir(parents=True);stage='model';started=time.perf_counter();records=[];raw={}
    try:
        model=model_for(a.scale,a.seed);count=sum(p.numel() for p in model.parameters())
        x,y,control,evaluation=banks();x=x.cuda();y=y.cuda();vx,vy=x[control],y[control];ex,ey=x[evaluation],y[evaluation]
        gen=torch.Generator().manual_seed(a.seed+99001);batches=[torch.randint(529,(128,),generator=gen).cuda() for _ in range(a.steps)]
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.98),weight_decay=0,fused=True,foreach=False)
        engine=GPUBlockwiseController(model) if a.arm=='dynamic' else ProposalStorage(model) if a.arm=='fixed' else None
        raw['targets']=ey[:,-1].cpu().numpy();z,acc=evaluate(model,ex,ey);raw['0_logits']=z
        torch.cuda.synchronize();wall=time.perf_counter();torch.cuda.reset_peak_memory_stats();alpha=1.
        for t,ids in enumerate(batches,1):
            if a.phase!='smoke' and acc>=.99:break
            stage=f'update-{t}';torch.cuda.synchronize();tick=time.perf_counter()
            seed=a.seed*1000+t
            if a.arm=='dynamic':alpha,details,_=engine.update(model,opt,t,x[ids],y[ids],vx,vy,seed,alpha)
            else:
                model.train();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);opt.zero_grad(set_to_none=True)
                loss=model(x[ids],y[ids])[1];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if a.arm=='native':opt.step();alpha=1.
                else:engine.form(opt);engine.commit(.25);alpha=.25
                details={'loss':float(loss.detach()),'alpha':alpha}
            torch.cuda.synchronize();update_seconds=time.perf_counter()-tick
            z,acc=evaluate(model,ex,ey);raw[f'{t}_logits']=z
            records.append({'t':t,'accuracy':acc,'alpha':alpha,'details':details,'update_seconds':update_seconds,'wall':time.perf_counter()-wall})
            print(a.scale,a.phase,a.arm,a.lr,t,acc,flush=True) if t%20==0 or t==a.steps else None
            if time.perf_counter()-wall>180:break
        result={'status':'RESOURCE_LIMITED' if time.perf_counter()-wall>180 else 'COMPLETED','scale':a.scale,'phase':a.phase,'arm':a.arm,'seed':a.seed,'lr':a.lr,
            'parameters':count,'records':records,'initial_accuracy':evaluate_logits(raw['0_logits'],raw['targets']),
            'final_accuracy':acc,'wall':time.perf_counter()-wall,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
            'control_buffer_bytes':0 if engine is None else engine.allocated_bytes if a.arm=='fixed' else engine.buffers.allocated_bytes,
            'control_ids':control,'evaluation_ids':evaluation,'training_support':'all529, not held-out-example generalization'}
        with (path/'RAW.npz').open('xb') as f:np.savez_compressed(f,**raw)
        result['raw_sha256']=b.sha(path/'RAW.npz');b.write(path/'RESULT.json',result)
    except Exception as e:
        if raw:
            with (path/'PARTIAL_RAW.npz').open('xb') as f:np.savez_compressed(f,**raw)
        b.write(path/'RESULT.json',{'status':'OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else 'FAILED',
            'scale':a.scale,'phase':a.phase,'arm':a.arm,'seed':a.seed,'lr':a.lr,'stage':stage,'records':records,
            'exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'elapsed':time.perf_counter()-started,
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
            'partial_raw_sha256':b.sha(path/'PARTIAL_RAW.npz') if raw else None})


def evaluate_logits(z,y):
    comp=z.copy();comp[np.arange(len(y)),y]=-np.inf
    return float((z[np.arange(len(y)),y].astype(np.float64)-comp.max(1).astype(np.float64)>0).mean())


if __name__=='__main__':main()
