"""Fixed small sequential benchmark: same batches, optimizer and evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_functional_update_governor_v3.scheduler import GroupSentinel
from experiments.gfg_functional_update_governor_v4.governor import RecoveryGovernor
from experiments.gfg_functional_update_governor_v4.runtime import GovernedTraining
from .model import RelationModel,BertInstrument
from .prepare import ROOT,SEED,sha

HERE=Path(__file__).resolve().parent
CONFIG=dict(seed=SEED,batch_size=16,epochs_per_task=1,lr=5e-6,betas=[.9,.999],weight_decay=0,
    clip=1.,max_length=256,probes_per_relation=10,instrument_chunk=4,threshold=.9,
    max_interval=50,micro_every=1,alphas=[.125,.25,.5,1.,2.],time_limit_per_arm_seconds=3600,
    evaluation='all seen classes, no task identity oracle; full disjoint test split',
    scope='one-epoch sequential-finetuning ablation on CRL split; not reproduction of CRL method/score')


def json_write(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,separators=(',',':'),allow_nan=False)


def extend_bank(engine,old_x,old_y,new_x,new_y):
    """Only explicit append at a task boundary; old identities/data stay exact."""
    old=engine.scheduler
    if old_x is not None:
        old.bind(old_x,old_y)
        if not torch.equal(old_x,new_x[:len(old_x)]) or not torch.equal(old_y,new_y[:len(old_y)]):
            raise ValueError('old probe identities/data changed at task transition')
    mapping=GroupMap(new_y[:,-1].cpu().numpy())
    if tuple(mapping.labels[:len(old.mapping.labels)])!=old.mapping.labels:
        raise ValueError('old group labels changed')
    new=GroupSentinel(mapping)
    for k,label in enumerate(old.mapping.labels):
        if mapping.counts[k]!=old.mapping.counts[k]:raise ValueError('old group size changed')
        new.state[k]=old.state[k]
        for field in ('level','anchor','capabilities','q10','last_observed'):
            getattr(new,field)[k]=getattr(old,field)[k]
    new.last_step=old.last_step
    # FIRST_STEP performs a full measurement of the new competitor/probe scope.
    new.last_instrument=None;new.cursor=0;new.bind(new_x,new_y)
    engine.mapping=mapping;engine.scheduler=new
    engine.governor=RecoveryGovernor(mapping,new)


@torch.no_grad()
def evaluate(model,x,y,indices):
    model.eval();predictions=[]
    for begin in range(0,len(indices),16):
        ids=indices[begin:begin+16];z=model(x[ids].cuda())[0][:,-1]
        predictions.extend(z.argmax(1).cpu().tolist())
    truth=y[indices,-1].numpy();pred=np.array(predictions)
    return dict(accuracy=float((truth==pred).mean()),
        per_relation={str(k):float((truth[truth==k]==pred[truth==k]).mean()) for k in np.unique(truth)},
        indices=indices.tolist(),predictions=predictions)


def run(arm,tasks,resume_three=False):
    root=ROOT/f'{tasks}tasks-{arm}';root.mkdir(exist_ok=True)
    attempt=root/'attempt-1';attempt.mkdir() # no silent reruns/best-selection
    prefix=ROOT/f'3tasks-{arm}'/'attempt-1' if resume_three else None
    if prefix is not None and tasks!=10:raise ValueError('only extend three tasks to ten')
    config=dict(CONFIG,arm=arm,tasks=tasks,prefix=str(prefix) if prefix else None,
        source_hashes={p.name:sha(p) for p in HERE.glob('*.py')},
        data_manifest_sha256=sha(ROOT/'DATA_MANIFEST.json'),data_sha256=sha(ROOT/'data.npz'),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    json_write(attempt/'CONFIG.json',config)
    rows=[];metrics=[];start=time.perf_counter();stage='load'
    try:
        torch.set_num_threads(2);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=np.load(ROOT/'data.npz');x=torch.from_numpy(data['x']);y=torch.from_numpy(data['y'][:,None]);split=data['split'];labels=data['y']
        model=RelationModel().cuda();opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],
            betas=tuple(CONFIG['betas']),weight_decay=0,fused=True,foreach=False)
        engine=None;probe_x=probe_y=None;step=0;old_probe_ids=[];checkpoint_ids=[];start_task=0;prefix_wall=0.
        if prefix is not None:
            pd=json.loads((prefix/'RESULT.json').read_text());pc=json.loads((prefix/'CONFIG.json').read_text())
            assert pd['status']=='COMPLETED' and pc['data_sha256']==config['data_sha256']
            for key in CONFIG:
                assert pc[key]==CONFIG[key],key
            checkpoint=Path(pd['checkpoints'][-1]['path']);assert sha(checkpoint)==pd['checkpoints'][-1]['sha256']
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);model.seen=saved['seen']
            assert model.seen==24 and saved['step']==630
            step=saved['step'];start_task=3;rows=list(pd['rows']);metrics=list(pd['metrics'])
            checkpoint_ids=list(pd['checkpoints']);prefix_wall=pd['wall'];old_probe_ids=metrics[-1]['probe_ids']
            probe_x=x[old_probe_ids].cuda();probe_y=y[old_probe_ids].cuda()
            if arm=='governor':
                mapping=GroupMap(probe_y[:,-1].cpu().numpy());history=saved['scheduler']
                assert history.mapping.identity==mapping.identity and history.last_step==step
                engine=GovernedTraining(model,None,mapping=mapping,instrument=BertInstrument(model,chunk=CONFIG['instrument_chunk']))
                history.identity=None;history.bind(probe_x,probe_y)
                engine.scheduler=history;engine.governor=RecoveryGovernor(mapping,history)
                engine.instrument_calls=pd['full_calls']
            del saved
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with (attempt/'STEPS.jsonl').open('x',encoding='utf8') as stream:
            for task in range(start_task,tasks):
                stage=f'task{task+1}';seen=(task+1)*8;model.seen=seen;task_start=time.perf_counter()
                probe_ids=[int(i) for k in range(seen) for i in np.flatnonzero((labels==k)&(split==1))[:10]]
                assert probe_ids[:len(old_probe_ids)]==old_probe_ids and all(split[i]==1 for i in probe_ids)
                new_x=x[probe_ids].cuda();new_y=y[probe_ids].cuda()
                if arm=='governor':
                    if engine is None:
                        engine=GovernedTraining(model,None,mapping=GroupMap(new_y[:,-1].cpu().numpy()),
                            instrument=BertInstrument(model,chunk=CONFIG['instrument_chunk']))
                    else:extend_bank(engine,probe_x,probe_y,new_x,new_y)
                probe_x,probe_y=new_x,new_y;old_probe_ids=probe_ids
                train=np.flatnonzero((labels>=task*8)&(labels<seen)&(split==0))
                test=np.flatnonzero((labels<seen)&(split==2))
                if set(probe_ids)&set(test.tolist()):raise ValueError('test leakage')
                generator=torch.Generator().manual_seed(SEED+10000+task)
                before=evaluate(model,x,y,test)
                epoch_metrics=[]
                for epoch in range(CONFIG['epochs_per_task']):
                    order=train[torch.randperm(len(train),generator=generator).numpy()]
                    for begin in range(0,len(order),CONFIG['batch_size']):
                        ids=order[begin:begin+CONFIG['batch_size']];step+=1
                        bx=x[ids].cuda();by=y[ids].cuda();torch.cuda.synchronize();tick=time.perf_counter()
                        if arm=='governor':
                            info=engine.update(opt,step,bx,by,probe_x,probe_y,SEED*100000+step,CONFIG['lr'])
                            if info['instrument_called']:
                                # Observe only the selected real endpoint, never use it to reselect.
                                audit_start=time.perf_counter();actual=[];model.eval()
                                with torch.no_grad():
                                    for a in range(0,len(probe_x),16):
                                        z=model(probe_x[a:a+16])[0][:,-1];truth=probe_y[a:a+16,-1,None]
                                        margin=z.gather(1,truth).squeeze(1).double()-z.scatter(1,truth,-torch.inf).max(1).values.double()
                                        actual.extend(margin.cpu().tolist())
                                info['actual_endpoint']=engine.mapping.values(actual)
                                info['actual_endpoint_margins']=actual
                                info['endpoint_audit_seconds']=time.perf_counter()-audit_start
                        else:
                            model.train();torch.manual_seed(SEED*100000+step);torch.cuda.manual_seed_all(SEED*100000+step)
                            opt.zero_grad(set_to_none=True);loss=model(bx,by)[1];loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['clip']);opt.step()
                            info=dict(alpha=1.,instrument_called=False,loss=float(loss.detach()))
                        torch.cuda.synchronize()
                        row=dict(step=step,task=task+1,epoch=epoch+1,seconds=time.perf_counter()-tick,info=info)
                        stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');stream.flush()
                        rows.append({k:v for k,v in row.items() if k!='info'}|dict(alpha=info['alpha'],full=info['instrument_called'],
                            reason=info.get('intervention_reason'),micro=info.get('micro_probe_called',False)))
                        if step%50==0:print(arm,'task',task+1,'step',step,'wall',round(time.perf_counter()-start,1),'full',sum(r['full'] for r in rows),flush=True)
                        if time.perf_counter()-start>CONFIG['time_limit_per_arm_seconds']:raise TimeoutError('fixed arm runtime limit')
                    ev=evaluate(model,x,y,test);epoch_metrics.append(ev)
                per=ev['per_relation'];current=float(np.mean([per[str(k)] for k in range(task*8,seen)]))
                old=float(np.mean([per[str(k)] for k in range(task*8)])) if task else None
                forgetting=[];bwt=[]
                for k in range(task*8):
                    history=[m['per_relation'][str(k)] for m in metrics if str(k) in m['per_relation']]
                    forgetting.append(max(history)-per[str(k)]);bwt.append(per[str(k)]-history[0])
                metric=dict(task=task+1,**ev,current_accuracy=current,old_accuracy=old,
                    average_forgetting=float(np.mean(forgetting)) if forgetting else None,
                    backward_transfer=float(np.mean(bwt)) if bwt else None,
                    wall=prefix_wall+time.perf_counter()-start,task_wall=time.perf_counter()-task_start,
                    before_current_scope=before,epochs=epoch_metrics,probe_ids=probe_ids,
                    control_state=engine.scheduler.snapshot(step) if engine else None,
                    time_to_target_accuracy='NOT_ESTIMATED: test evaluated at fixed task/epoch boundaries, not used for stopping')
                metrics.append(metric);print('TASK_RESULT',arm,task+1,'new',current,'old',old,flush=True)
                checkpoint=attempt/f'task{task+1}.pt'
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'seen':seen,
                    'scheduler':engine.scheduler if engine else None},checkpoint)
                checkpoint_ids.append({'path':str(checkpoint),'sha256':sha(checkpoint)})
        result=dict(status='COMPLETED',metrics=metrics,rows=rows,checkpoints=checkpoint_ids,
            steps=len(rows),full_calls=sum(r['full'] for r in rows),micro_calls=sum(r['micro'] for r in rows),
            alpha_histogram=dict(Counter(r['alpha'] for r in rows)),peak_allocated=torch.cuda.max_memory_allocated(),
            wall=prefix_wall+time.perf_counter()-start,continuation_wall=time.perf_counter()-start,
            prefix=str(prefix) if prefix else None,prefix_result_sha256=sha(prefix/'RESULT.json') if prefix else None)
    except Exception:
        result=dict(status='FAILED',stage=stage,traceback=traceback.format_exc(),metrics=metrics,rows=rows,wall=time.perf_counter()-start)
    json_write(attempt/'RESULT.json',result)
    print('RESULT',str(attempt),result['status'],flush=True)
    if result['status']!='COMPLETED':raise RuntimeError(result['traceback'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=('native','governor'));parser.add_argument('--tasks',type=int,choices=(3,10),default=3)
    parser.add_argument('--resume-three',action='store_true')
    args=parser.parse_args();run(args.arm,args.tasks,args.resume_three)
