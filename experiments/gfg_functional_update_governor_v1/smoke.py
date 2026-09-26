"""Two fixed 70M project smoke runs only; reuse prior setup and reference results."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from experiments.gfg_controller_distributed_runtime_v1.scale import model_for,banks,evaluate
from experiments.gfg_controller_distributed_runtime_v2.evidence import setup
from experiments.gfg_controller_distributed_runtime_v2.adapters import NanoGPTAdapter
from .runtime import GovernedTraining

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUT=Path('E:/gfg-evidence/functional-update-governor-v1')
RATES={'safe':3.7688918072220455e-6,'near':6.030226891555273e-5}
CONFIG={'alphas':[.125,.25,.5,1.,2.],'preservation_penalty':1.,'release_steps':0,
        'max_instrument_interval':20,'history_window':20,'history_warmup':5,'mad_k':3.,
        'mad_relative_floor':.001,'mad_absolute_floor':1e-12,'backend':'exact'}


def write(path,obj):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def create(lr):
    model=model_for('70',201)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=0,fused=True,foreach=False)
    engine=GovernedTraining(model,NanoGPTAdapter(model))
    return model,opt,engine


def run(case):
    setup();lr=RATES[case];root=OUT/case
    root.mkdir(parents=True,exist_ok=True)
    number=1
    while (root/f'attempt-{number}').exists():number+=1
    path=root/f'attempt-{number}';path.mkdir()
    source_hashes={p.name:sha(p) for p in HERE.glob('*.py')}
    write(path/'config.json',{'case':case,'seed':201,'lr':lr,'cap':200,'threshold':.99,
        'config':CONFIG,'source_hashes':source_hashes,'created_time_ns':time.time_ns(),
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
    rows=[];raw={};stage='setup'
    try:
        x,y,control,ev=banks();x=x.cuda();y=y.cuda()
        vx,vy=x[control],y[control];ex,ey=x[ev],y[ev]
        gen=torch.Generator().manual_seed(201+99001)
        indices=[torch.randint(529,(128,),generator=gen).cuda() for _ in range(200)]
        stage='warmup';model,opt,engine=create(lr)
        for t in (1,2):engine.update(opt,t,x[indices[t-1]],y[indices[t-1]],vx,vy,201000+t,lr)
        del model,opt,engine;gc.collect();torch.cuda.empty_cache()
        stage='initialize';model,opt,engine=create(lr)
        parameter_count=sum(p.numel() for p in model.parameters())
        raw['targets']=ey[:,-1].cpu().numpy();z,accuracy=evaluate(model,ex,ey);raw['0_logits']=z
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for t,ids in enumerate(indices,1):
            if accuracy>=.99:break
            stage=f'update-{t}';tick=time.perf_counter()
            info=engine.update(opt,t,x[ids],y[ids],vx,vy,201000+t,lr)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            z,accuracy=evaluate(model,ex,ey);raw[f'{t}_logits']=z
            rows.append({'t':t,'accuracy':accuracy,'update_seconds':seconds,'wall':time.perf_counter()-started,'info':info})
            if t%25==0 or accuracy>=.99:print(case,t,round(accuracy,6),'instruments',engine.instrument_calls,flush=True)
            if time.perf_counter()-started>300:break
        wall=time.perf_counter()-started
        result={'status':'COMPLETED' if wall<=300 else 'RESOURCE_LIMITED','case':case,'seed':201,'lr':lr,
            'parameters':parameter_count,'rows':rows,'final_accuracy':accuracy,'wall':wall,
            'reach99':accuracy>=.99,'steps_to99':next((r['t'] for r in rows if r['accuracy']>=.99),None),
            'instrument_calls':engine.instrument_calls,'instrument_fraction':engine.instrument_calls/len(rows),
            'alpha_histogram':dict(Counter(r['info']['alpha'] for r in rows)),
            'decision_alpha_histogram':dict(Counter(r['info']['alpha'] for r in rows if r['info']['instrument_called'])),
            'trigger_histogram':dict(Counter(reason for r in rows for reason in r['info']['instrument_reason'] if reason!='NONE')),
            'execution_paths':dict(Counter(r['info']['execution_path'] for r in rows)),
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
            'config':CONFIG,'source_hashes':source_hashes,'control_ids':control,'evaluation_ids':ev,
            'training_evaluation_overlap':True,'warmup_updates_excluded':2,
            'time_scope':'all updates and every-step evaluation; initialization and warmup excluded'}
    except Exception:
        result={'status':'FAILED','case':case,'stage':stage,'rows':rows,'traceback':traceback.format_exc()}
    if raw:
        with (path/'RAW.npz').open('xb') as f:np.savez_compressed(f,**raw)
        result['raw_sha256']=sha(path/'RAW.npz')
    write(path/'RESULT.json',result)
    print('RESULT',path,result['status'],result.get('steps_to99'),flush=True)
    if result['status']=='FAILED':raise RuntimeError(result['traceback'])


def summarize():
    dest=HERE/'results';dest.mkdir(exist_ok=True)
    comparisons={}
    for case in RATES:
        attempts=sorted((OUT/case).glob('attempt-*/RESULT.json'))
        if len(attempts)!=1:raise RuntimeError('multiple attempts: explicitly audit before summarizing')
        p=attempts[0];d=json.loads(p.read_text())
        assert d['status']=='COMPLETED'
        assert d['raw_sha256']==sha(p.parent/'RAW.npz')
        with np.load(p.parent/'RAW.npz') as raw:
            targets=raw['targets']
            for row in d['rows']:
                z=raw[f"{row['t']}_logits"];other=z.copy();other[np.arange(len(z)),targets]=-np.inf
                acc=float((z[np.arange(len(z)),targets].astype(np.float64)-other.max(1).astype(np.float64)>0).mean())
                assert acc==row['accuracy']
        calls=sum(r['info']['instrument_called'] for r in d['rows'])
        assert calls==d['instrument_calls']
        assert all(r['info']['alpha']==1. for r in d['rows'] if not r['info']['instrument_called'])
        current_steps=len(d['rows']);scheduled_old=(current_steps+4)//5
        history={}
        oldroot=Path('E:/gfg-evidence/controller-distributed-runtime-v2/attempt-2/scale/70')
        for label,folder,arm in [('native','formal-'+case,'native'),('tuned_native','formal-'+case+'-tuned','native'),('v2_dynamic','formal-'+case,'dynamic')]:
            matches=list((oldroot/folder).glob(arm+'-s201-*/RESULT.json'));assert len(matches)==1
            h=json.loads(matches[0].read_text());hit=next((r for r in h['rows'] if r['accuracy']>=.99),None)
            assert h['control_ids']==d['control_ids'] and h['evaluation_ids']==d['evaluation_ids']
            history[label]={'steps_to99':hit['t'] if hit else None,'wall_to99':hit['wall'] if hit else None,
                'source':str(matches[0]),'source_sha256':sha(matches[0]),'lr':h['lr']}
        comparisons[case]={'new_steps_to99':d['steps_to99'],'new_wall_to99':d['rows'][-1]['wall'],
            'instrument_calls':calls,'instrument_fraction':d['instrument_fraction'],
            'v2_cadence_calls_over_same_new_horizon':scheduled_old,'call_reduction_same_horizon':1-calls/scheduled_old,
            'v2_actual_calls_to_own_99':(history['v2_dynamic']['steps_to99']+4)//5,
            'decision_alpha_histogram':d['decision_alpha_histogram'],'execution_paths':d['execution_paths'],
            'peak_allocated':d['peak_allocated'],'historical_references':history,
            'time_comparison':'historical, same hardware and timing scope; not contemporaneously rerun'}
        write(dest/(case+'-RESULT.json'),d)
    write(dest/'COMPARISON.json',comparisons)
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('safe','near','summary'));a=p.parse_args()
    if a.action=='summary':summarize()
    else:run(a.action)
