"""Independent ten-task harness; imports the frozen V4 mechanism unchanged."""
import argparse
import copy
import json
import time
import traceback
from collections import Counter
import numpy as np
import torch
from .common import (HERE,REPO,ROOT,BASE,DATA,CHECKPOINT,CKSHA,DATA_SHA,ARMS,
    sha,stamp,write,unchanged,sources,scope,order,summarize_predictions,RandomReplay,exposure_schedule)
from experiments.gfg_governor_fewrel_v1 import run as fewrel
from experiments.gfg_governor_fewrel_v1.model import RelationModel,BertInstrument
from experiments.gfg_governor_fewrel_feedback_router_v1.run import Memory,extend_memory,validate_batches
from experiments.gfg_router_v4_partial_capability.policy import Router,PartialSentinel,complete_old
from experiments.gfg_router_v3_three_task_continuation.run import preserving_extend
from experiments.gfg_functional_feedback_router_v1 import runtime
from experiments.gfg_functional_feedback_router_v3.dose import accumulate
from experiments.gfg_functional_update_governor_v4.runtime import GovernedTraining

def admission():
    unchanged();p=json.loads((ROOT/'PREFLIGHT.json').read_text())
    assert p['status']=='PASS' and p['source_hashes']==sources()
    assert sha(CHECKPOINT)==CKSHA and sha(DATA/'data.npz')==DATA_SHA
    return p

def extend(engine,px,py,nx,ny):
    completed=set(engine.scheduler.mapping.labels)
    audit=preserving_extend(fewrel.extend_bank,engine,px,py,nx,ny)
    audit['converted']=complete_old(engine.scheduler,completed)
    audit['states_after']=list(engine.scheduler.state)
    return audit

def run(arm,seed=2021):
    assert arm in ARMS and seed==2021,'Additional independent-seed prefixes require separate admitted execution'
    p=admission();out=ROOT/f'seed-{seed}'/arm;out.mkdir(parents=True)
    data=np.load(DATA/'data.npz');random=RandomReplay(seed);schedule=None
    if arm=='random':
        schedule=json.loads((ROOT/f'seed-{seed}'/'EXPOSURE_SCHEDULE.json').read_text())
        assert len(schedule['rows'])==1890
        assert all(set(r)=={'step','task','old_batch_size'} for r in schedule['rows'])
    config=dict(base=BASE,arm=arm,seed=seed,source_hashes=p['source_hashes'],preflight_sha256=sha(ROOT/'PREFLIGHT.json'),
        protocol_sha256=sha(HERE/'PROTOCOL.md'),checkpoint_sha256=CKSHA,data_sha256=DATA_SHA,
        tasks=10,epochs_per_task=1,current_batch_size=16,lr=5e-6,cap_seconds=21600,
        shared_prefix='original seed2021 task1 checkpoint; inherited model/optimizer/sentinel',
        exposure_schedule_sha256=sha(ROOT/f'seed-{seed}'/'EXPOSURE_SCHEDULE.json') if schedule else None,
        optimizer='AdamW fused=True foreach=False, betas(.9,.999), weight_decay0, clip1',
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    write(out/'CONFIG.json',config)
    start=time.perf_counter();metrics=[];transitions=[];rows=[];stage='load';result=None
    try:
        torch.set_num_threads(2);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        x=torch.from_numpy(data['x']);y=torch.from_numpy(data['y'][:,None])
        model=RelationModel().cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-6,betas=(.9,.999),weight_decay=0,fused=True,foreach=False)
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        assert saved['step']==210 and saved['seen']==8
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);model.seen=8
        history=saved['scheduler'];history.identity=None
        assert all(group['lr']==5e-6 for group in opt.param_groups)
        runtime.accumulate=accumulate;fewrel.GroupSentinel=PartialSentinel
        engine=None;px=py=None;memory=Memory();router=Router();fixed_cursor=0;evaluation_seconds=0.
        torch.cuda.synchronize();load_seconds=time.perf_counter()-start;train_start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        def record(task):
            nonlocal evaluation_seconds
            tick=time.perf_counter();_,primary,full=scope(data,task*8)
            ev=fewrel.evaluate(model,x,y,full);pred=np.array(ev['predictions']);keep=np.isin(full,primary)
            metric=dict(task=task,step=task*210,indices=full.tolist(),predictions=pred.tolist(),
                primary=summarize_predictions(primary,pred[keep],data,task),
                original=summarize_predictions(full,pred,data,task),
                control_state=engine.scheduler.snapshot(task*210) if engine else None)
            metrics.append(metric);evaluation_seconds+=time.perf_counter()-tick
            print('TASK_END',arm,task,json.dumps(metric['primary']),flush=True)
        record(1)
        with (out/'STEPS.jsonl').open('x',encoding='utf8') as f:
            for task in range(2,11):
                stage=f'task{task}';seen=task*8;old=set(range(seen-8));current=set(range(seen-8,seen));model.seen=seen
                probes,primary,full=scope(data,seen);extend_memory(memory,data,old,probes,full)
                assert {str(k):list(v) for k,v in memory.groups.items()}==p['data_audit']['tasks'][str(task)]['memory']
                nx=x[probes].cuda();ny=y[probes].cuda()
                if arm=='router':
                    if engine is None:
                        ox=nx[:80].clone();oy=ny[:80].clone();history.bind(ox,oy)
                        instrument=BertInstrument(model,chunk=4);instrument.alphas=(.125,.25,.5,1.)
                        engine=GovernedTraining(model,None,mapping=history.mapping,instrument=instrument)
                        engine.scheduler=history;px=ox;py=oy
                    transitions.append(extend(engine,px,py,nx,ny))
                px,py=nx,ny
                if task==2:del saved
                batches=order(data,task,seed)
                assert stamp(batches.tolist())==p['data_audit']['tasks'][str(task)]['batch_sha256']
                for offset,ids0 in enumerate(batches,1):
                    t=(task-1)*210+offset;ids=ids0.tolist();cursor_before=dict(memory.cursors)
                    torch.cuda.synchronize();tick=time.perf_counter()
                    if arm=='router':
                        receiving=dict(labels=list(engine.mapping.labels),**engine.scheduler.snapshot(t-1))
                        preview=copy.deepcopy(router).choose(t,engine.mapping.labels,list(engine.scheduler.state),engine.scheduler.level.tolist(),old,copy.deepcopy(memory))
                        validate_batches(ids,preview['feedback_example_ids'],data,current,memory,probes,primary)
                        info=runtime.update(model,opt,engine,router,memory,x,y,ids,px,py,t,old,seed*100000+t)
                        assert preview['feedback_example_ids']==info['feedback_example_ids']
                        info['allocation_receiving_history']=receiving
                    else:
                        selected=[];oldids=[]
                        if arm=='replay':
                            groups=sorted(old);selected=[groups[(fixed_cursor+j)%len(groups)] for j in range(8)]
                            fixed_cursor=(fixed_cursor+8)%len(groups)
                            for g in selected:oldids.extend(memory.take(g))
                        elif arm=='random':
                            dose=schedule['rows'][t-211];assert dose['step']==t and dose['task']==task
                            selected,oldids=random.choose(old,memory,dose['old_batch_size'])
                        validate_batches(ids,oldids,data,current,memory,probes,primary)
                        losses=accumulate(model,opt,x,y,[ids]+([oldids] if oldids else []),seed*100000+t);opt.step()
                        info=dict(alpha=1.,instrument_called=False,gate_reason='NOT_MEASURED',current_loss=losses[0],
                            feedback_loss=losses[1] if oldids else None,feedback_example_ids=oldids,feedback_batch_size=len(oldids),
                            selected_feedback_groups=selected,router_mode=arm.upper(),release_event=False,
                            selected_anchor_groups=[],selected_partial_groups=[])
                    n=info['feedback_batch_size'];info.update(old_loss_weight=n/(16+n),current_loss_weight=16/(16+n),
                        weighted_training_loss=(16*info['current_loss']+n*(info['feedback_loss'] or 0.))/(16+n))
                    torch.cuda.synchronize();info['update_seconds_inclusive']=time.perf_counter()-tick
                    row=dict(step=t,task=task,offset=offset,new_ids=ids,probe_ids=probes.tolist(),memory_identity=memory.stamp,
                        memory_groups={str(k):list(v) for k,v in memory.groups.items()},cursor_before=cursor_before,info=info)
                    rows.append(row);f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');f.flush()
                    if offset%50==0:print('PROGRESS',arm,task,offset,'old_exposure',sum(r['info']['feedback_batch_size'] for r in rows),'wall',round(time.perf_counter()-train_start,1),flush=True)
                    if time.perf_counter()-train_start>21600:raise TimeoutError('frozen6h arm budget exceeded')
                record(task)
        torch.cuda.synchronize();continuation_wall=time.perf_counter()-train_start
        stage='final_checkpoint';save_start=time.perf_counter()
        final=out/'final.pt'
        with final.open('xb') as f:
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),seen=model.seen,step=2100,
                scheduler=engine.scheduler if engine else None,router=router if engine else None,memory=memory),f)
        final_id=dict(sha256=sha(final),bytes=final.stat().st_size,path=str(final))
        result=dict(status='COMPLETED',arm=arm,steps=len(rows),metrics=metrics,transitions=transitions,
            old_exposures=sum(r['info']['feedback_batch_size'] for r in rows),
            wall=continuation_wall,load_seconds=load_seconds,evaluation_seconds=evaluation_seconds,
            checkpoint_seconds=time.perf_counter()-save_start,wall_total=time.perf_counter()-start,
            update_seconds=sum(r['info']['update_seconds_inclusive'] for r in rows),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            final_checkpoint=final_id,log_sha256=sha(out/'STEPS.jsonl'),config_sha256=sha(out/'CONFIG.json'))
    except Exception:
        result=dict(status='FAILED',arm=arm,stage=stage,traceback=traceback.format_exc(),steps=len(rows),metrics=metrics,
            wall_total=time.perf_counter()-start,log_sha256=sha(out/'STEPS.jsonl') if (out/'STEPS.jsonl').exists() else None)
    write(out/'RESULT.json',result);print('ARM_RESULT',arm,result['status'],flush=True)
    if result['status']!='COMPLETED':raise RuntimeError(result['traceback'])
    if arm=='router':
        write(ROOT/f'seed-{seed}'/'EXPOSURE_SCHEDULE.json',dict(rows=exposure_schedule(rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=ARMS);a=p.parse_args();run(a.arm)
