"""Isolated warmed capability runs; no checkpoint or raw-array commits."""
import argparse
import gc
import random
import time
import traceback
import numpy as np
import torch
from .evidence import OUT,write,sha,setup
from .identity import CanonicalParameterRegistry
from .ownership import Proposal
from .adapters import NanoGPTAdapter
from .controller import FunctionalController
from experiments.gfg_controller_distributed_runtime_v1.scale import model_for,banks,evaluate,evaluate_logits


def create(a):
    model=model_for(a.scale,a.seed);registry=CanonicalParameterRegistry(model)
    opt=torch.optim.AdamW(registry.resolve(model).values(),lr=a.lr,betas=(.9,.98),weight_decay=0,fused=True,foreach=False)
    controller=FunctionalController(model,NanoGPTAdapter(model),registry=registry,cadence=a.cadence) if a.arm=='dynamic' else Proposal(registry.resolve(model)) if a.arm=='fixed' else None
    return model,registry,opt,controller


def update(a,objects,t,x,y,vx,vy):
    model,registry,opt,engine=objects;seed=a.seed*1000+t
    if a.arm=='dynamic':return engine.update(opt,t,x,y,vx,vy,seed,a.lr)
    model.train();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);opt.zero_grad(set_to_none=True)
    loss=model(x,y)[1];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if a.arm=='fixed':engine.form(registry.resolve(model),opt);engine.commit(registry.resolve(model),.25)
    else:opt.step()
    return {'alpha':.25 if a.arm=='fixed' else 1.,'loss':float(loss.detach())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--scale',choices=('70','160'),required=True);p.add_argument('--arm',choices=('native','fixed','dynamic'),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--phase',required=True)
    p.add_argument('--cadence',type=int,default=5);p.add_argument('--probes',type=int,default=212);p.add_argument('--subset-seed',type=int,default=11)
    a=p.parse_args();setup();name=f'{a.arm}-s{a.seed}-lr{a.lr}-k{a.cadence}-n{a.probes}-u{a.subset_seed}'
    path=OUT/'scale'/a.scale/a.phase/name;path.mkdir(parents=True,exist_ok=False);raw={};rows=[];stage='setup'
    try:
        x,y,control,ev=banks();selected=list(control)
        if a.probes<212:random.Random(a.subset_seed).shuffle(selected);selected=selected[:a.probes]
        x=x.cuda();y=y.cuda();vx,vy=x[selected],y[selected];ex,ey=x[ev],y[ev]
        gen=torch.Generator().manual_seed(a.seed+99001);indices=[torch.randint(529,(128,),generator=gen).cuda() for _ in range(200)]
        stage='warmup';warm=create(a)
        for t in (1,2):update(a,warm,t,x[indices[t-1]],y[indices[t-1]],vx,vy)
        del warm;gc.collect();torch.cuda.empty_cache()
        stage='initialization';objects=create(a);model=objects[0];parameters=sum(p.numel() for p in model.parameters())
        raw['targets']=ey[:,-1].cpu().numpy();z,acc=evaluate(model,ex,ey);raw['0_logits']=z
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for t,ids in enumerate(indices,1):
            if acc>=.99:break
            stage=f'update-{t}';tick=time.perf_counter();info=update(a,objects,t,x[ids],y[ids],vx,vy)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            z,acc=evaluate(model,ex,ey);raw[f'{t}_logits']=z
            rows.append({'t':t,'accuracy':acc,'update_seconds':seconds,'wall':time.perf_counter()-start,'info':info})
            if t%50==0:print(a.scale,a.phase,a.arm,a.lr,t,acc,flush=True)
            if time.perf_counter()-start>300:break
        status='COMPLETED' if time.perf_counter()-start<=300 else 'RESOURCE_LIMITED'
        result={'status':status,**vars(a),'parameters':parameters,'control_ids':selected,'evaluation_ids':ev,'rows':rows,'final_accuracy':acc,
            'wall':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
            'training_evaluation_overlap':True,'warmup_updates_excluded':2,'threshold':.99,'cap':200}
    except Exception as e:
        result={'status':'OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else 'FAILED',**vars(a),'stage':stage,'rows':rows,'message':str(e),'traceback':traceback.format_exc()}
    if raw:
        with (path/'RAW.npz').open('xb') as f:np.savez_compressed(f,**raw)
        result['raw_sha256']=sha(path/'RAW.npz')
    write(path/'RESULT.json',result)


if __name__=='__main__':main()
