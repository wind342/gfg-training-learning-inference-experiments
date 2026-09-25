"""Sealed numerical/trajectory/standalone-entry and timing validation."""
from pathlib import Path
import gc
import math
import subprocess
import time
import traceback
import numpy as np
import torch
from experiments.gfg_controller_large_model_runtime_v1 import phase1 as old
from experiments.gfg_controller_large_model_runtime_v1.runtime import streamed_boundary,KEYS,ALPHAS
from .runtime import gpu_boundary,GPUBlockwiseController
from .entry import train

b=old.b;HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
BASE='992ab35645b60d60f5eb4ab4590b7faaa79ff2df'
OUT=Path('E:/gfg-evidence/controller-gpu-boundary-v1')
TILES=((32,8),(64,64),(256,128))


def comparison(a,z):
    a=np.asarray(a);z=np.asarray(z)
    return {'exact':bool(np.array_equal(a,z)),'close':bool(np.allclose(a,z,rtol=1e-6,atol=1e-8)),
        'different':int(np.count_nonzero(a!=z)),'max_abs':float(np.max(np.abs(a-z)))}


def compare_boundary(ref,gpu):
    aa,ss,cc,mm,competitors=ref;ga,gs,gc,gm=gpu
    aligned={k:np.take_along_axis(gc[k].cpu().numpy(),competitors,1) for k in KEYS}
    margins=np.stack([mm[a] for a in ALPHAS]);actual=gm.cpu().numpy()
    return {'coordinates':{k:comparison(cc[k],aligned[k]) for k in KEYS},
        'minima':comparison(margins,actual),'scores_exact':ss==gs,'alpha_exact':aa==ga,
        'reference_alpha':aa,'gpu_alpha':ga},aligned,actual


def synthetics():
    rng=np.random.default_rng(20260925);records=[]
    for dtype in (np.float32,np.float64):
        for n,v,h in ((1,2,3),(7,5,17),(53,11,32),(212,24,64),(7,4097,65)):
            values=[rng.normal(size=s).astype(dtype) for s in ((n,v),(v,h),(v,h),(n,h),(n,h),(n,h))]
            targets=rng.integers(v,size=n);ref=streamed_boundary(values[0],targets,*values[1:],retain_coordinates=True)
            tensors=[torch.from_numpy(x).cuda() for x in values];yy=torch.from_numpy(targets).cuda()
            for tt,ct in TILES:
                output=gpu_boundary(tensors[0],yy,*tensors[1:],target_tile=tt,competitor_tile=ct,audit=True)
                result,aligned,actual=compare_boundary(ref,output)
                name=f'synthetic-{len(records)}.npz'
                with (OUT/name).open('xb') as f:
                    np.savez_compressed(f,**{'gpu_'+k:aligned[k] for k in KEYS},**{'cpu_'+k:ref[2][k] for k in KEYS},
                        gpu_minima=actual,cpu_minima=np.stack([ref[3][a] for a in ALPHAS]))
                result['artifact']=name;result['artifact_sha256']=b.sha(OUT/name)
                records.append({'shape':[n,v,h],'dtype':str(dtype),'tile':[tt,ct],**result})
    z=torch.zeros((4,3),device='cuda');w=torch.zeros((3,5),device='cuda');h=torch.zeros((4,5),device='cuda')
    a,s,_,_=gpu_boundary(z,torch.tensor([0,1,2,0],device='cuda'),w,w,h,h,h)
    assert a==1. and all(v['predicted_correct_count']==0 and v['median_predicted_minimum_margin']==0 for v in s.values())
    return {'cases':records,'zero_tie':'PASS','all_alpha_exact':all(x['alpha_exact'] for x in records),
        'all_coordinates_exact':all(all(c['exact'] for c in x['coordinates'].values()) for x in records)}


class CheckedController(GPUBlockwiseController):
    def __init__(self,model):
        super().__init__(model,audit=True,hash_parameters=b.phash)
        self.check=None;self.arrays={};self.counter=0

    def response(self,model,inputs,targets):
        a,s,c,m,_=super().response(model,inputs,targets);d=self.last_debug
        values=[d['pre_logits'],model.transformer.wte.weight.detach(),self.buffers.delta['transformer.wte.weight'],*d['hidden']]
        values=[z.detach().cpu().numpy() for z in values]
        ref=streamed_boundary(values[0],targets[:,-1].cpu().numpy(),*values[1:],retain_coordinates=True)
        self.check,aligned,minima=compare_boundary(ref,(a,s,c,m));self.counter+=1
        for k in KEYS:
            self.arrays[f'{self.counter}_gpu_{k}']=aligned[k];self.arrays[f'{self.counter}_cpu_{k}']=ref[2][k]
        self.arrays[f'{self.counter}_gpu_minima']=minima
        self.arrays[f'{self.counter}_cpu_minima']=np.stack([ref[3][a] for a in ALPHAS])
        return a,s,c,m,None


def run_checked(row,loaded,ci,rng):
    rt,start,batches,vx,vy=loaded;model=rt.model;old.reset(model,start)
    opt=old.prior.native.optimizer(model,old.prior.RATES[ci]);controller=CheckedController(model)
    ref=old.prior.record('dynamic','search',row['entry_id'],ci)
    initial,acc=old.evaluate(model,vx,vy);outputs=[initial];steps=[];alpha=1.;chain=None
    for i in range(1,201):
        if acc>=.99:break
        seed=b.old.seed_for(row,i-1,rng);lr=old.prior.RATES[ci]*(.1+.9*.5*(1+math.cos(math.pi*(i-1)/199)))
        opt.param_groups[0]['lr']=lr;controller.check=None
        alpha,details,_=controller.update(model,opt,i,*batches[i-1],vx,vy,seed,alpha)
        logits,acc=old.evaluate(model,vx,vy);outputs.append(logits);post=b.phash(b.params(model))
        expected=ref['steps'][i-1] if i<=len(ref['steps']) else None
        fact={'t':i,'seed':seed,'lr':lr,'details':details,'accuracy':acc,'numerical_check':controller.check,
            'post_parameters_sha256':post,'logits_sha256':b.tensor_sha256(logits),
            'old_post_exact':expected is not None and post==expected['post_parameters_sha256'],
            'old_logits_exact':expected is not None and b.tensor_sha256(logits)==expected['logits_sha256'],
            'previous_sha256':chain}
        chain=b.identity(fact);steps.append({**fact,'sha256':chain})
    final=b.identity(b.structure({'parameters':b.params(model),'optimizer':opt.state_dict()}))
    path=OUT/'checked'/row['entry_id']/str(ci);path.mkdir(parents=True)
    with (path/'RAW.npz').open('xb') as f:np.savez_compressed(f,logits=np.stack(outputs),targets=vy[:,-1].cpu().numpy(),**controller.arrays)
    result={'entry':row['entry_id'],'config':ci,'steps':steps,'final_state_sha256':final,
        'initial_logits_sha256':b.tensor_sha256(initial),'raw_sha256':b.sha(path/'RAW.npz'),
        'trajectory_exact':len(steps)==len(ref['steps']) and all(s['old_post_exact'] and s['old_logits_exact'] for s in steps) and final==ref['final_state_sha256']}
    b.write(path/'TRAJECTORY.json',result)
    print('CHECKED',row['entry_id'],ci,len(steps),result['trajectory_exact'],flush=True)
    return {'entry':row['entry_id'],'config':ci,'updates':len(steps),'trajectory_exact':result['trajectory_exact']}


def config(row,loaded,ci,rng):
    rt,start,batches,vx,vy=loaded;model=rt.model;old.reset(model,start)
    return {'model':model,'optimizer':old.prior.native.optimizer(model,old.prior.RATES[ci]),'batches':batches,
        'seeds':[b.old.seed_for(row,i,rng) for i in range(200)],
        'learning_rates':[old.prior.RATES[ci]*(.1+.9*.5*(1+math.cos(math.pi*i/199))) for i in range(200)],
        'probe_inputs':vx,'probe_targets':vy}


def run_entry(row,loaded,ci,rng,backend='gpu',tile=(256,128)):
    cfg=config(row,loaded,ci,rng);result=train(**cfg,backend=backend,target_tile=tile[0],competitor_tile=tile[1])
    result['entry']=row['entry_id'];result['config']=ci
    # External validation only: not called by entry.train or inside its timer.
    result['final_state_sha256']=b.identity(b.structure({'parameters':b.params(cfg['model']),'optimizer':cfg['optimizer'].state_dict()}))
    ref=old.prior.record('dynamic','search',row['entry_id'],ci)
    logits,acc=old.evaluate(cfg['model'],cfg['probe_inputs'],cfg['probe_targets'])
    result['final_logits_sha256']=b.tensor_sha256(logits)
    result['trajectory_exact']=len(result['steps'])==len(ref['steps']) and result['final_state_sha256']==ref['final_state_sha256'] and all(
        s['alpha']==z['details']['alpha'] and s['accuracy']==z['accuracy'] for s,z in zip(result['steps'],ref['steps']))
    return result


def main():
    b.old.setup();assert old.git('rev-parse','HEAD')==BASE and not OUT.exists()
    files=old.git('ls-files').splitlines()
    b.write(HERE/'SEAL.json',{'base':BASE,'utc':b.old.utc(),'sources':{p.name:b.sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md')},
        'old_files':{p:b.sha(REPO/p) for p in files},'frozen_tag':old.git('rev-parse',old.prior.TAG),
        'frozen_commit':old.git('rev-parse',old.prior.TAG+'^{}'),'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(0),'tiles':TILES})
    OUT.mkdir();b.write(HERE/'SYNTHETIC.json',synthetics())
    seal=b.read(old.HERE/'PHASE1_SEAL.json');rng=seal['source_rng_protocol_sha256']
    rows=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs'];checked=[];standalone=[]
    for row in rows:
        loaded=b.load_row(row)
        try:
            for ci in range(3):
                checked.append(run_checked(row,loaded,ci,rng))
                result=run_entry(row,loaded,ci,rng)
                b.write(OUT/'entry'/row['entry_id']/f'{ci}.json',result)
                standalone.append({'entry':row['entry_id'],'config':ci,'updates':len(result['steps']),'trajectory_exact':result['trajectory_exact']})
                print('ENTRY',row['entry_id'],ci,len(result['steps']),result['trajectory_exact'],flush=True)
        finally:loaded[0].close();del loaded;gc.collect();torch.cuda.empty_cache()
    timing=[]
    # Round-robin fresh states; first entire run per configuration is excluded warmup.
    for rep in range(4):
        for backend,tile in [('cpu',(32,8)),*[("gpu",t) for t in TILES]]:
            loaded=b.load_row(rows[0])
            try:
                gc.collect();torch.cuda.empty_cache()
                result=run_entry(rows[0],loaded,1,rng,backend,tile);result['rep']=rep;result['warmup']=rep==0
                b.write(OUT/'timing'/f'{backend}-{tile[0]}-{tile[1]}-{rep}.json',result)
                timing.append({'backend':backend,'tile':tile,'rep':rep,'wall':result['checked_wall_seconds'],'exact':result['trajectory_exact']})
            finally:loaded[0].close();del loaded;gc.collect();torch.cuda.empty_cache()
    b.write(HERE/'RESULTS.json',{'checked':checked,'entry':standalone,'timing':timing,'scale_training':'NOT_STARTED'})


if __name__=='__main__':
    try:main()
    except Exception:
        with (HERE/f'FAILURE_{time.time_ns()}.txt').open('x',encoding='utf-8') as f:f.write(traceback.format_exc())
        raise
