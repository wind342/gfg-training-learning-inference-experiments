"""Non-overlapping synchronized wall-time attribution with exact replay checks."""
from collections import defaultdict
from pathlib import Path
import statistics
import time
from unittest.mock import patch
import json

from experiments.gfg_controller_batched_coordinates_v1 import run as prior
from experiments.gfg_controller_batched_coordinates_v1.kernel import boundary_coordinates
from experiments.gfg_controller_batched_coordinates_v1.nanogpt_adapter import coordinates as original_coordinates
from experiments.gfg_nanogpt_joint_rotation_branch_v1.analysis import HiddenStateModel
import numpy as np
import torch
from torch.func import functional_call, jvp
from torch.nn.attention import sdpa_kernel, SDPBackend

b=prior.b
HERE=Path(__file__).resolve().parent
BASE='ada27cd1dbd1f30cab6769cf93c967d2bdfcdf5c'


class Timer:
    def __init__(self):
        self.seconds=defaultdict(float)
        self.counts=defaultdict(int)

    def call(self, name, fn, *args, **kwargs):
        b.sync(); tick=time.perf_counter()
        result=fn(*args, **kwargs)
        b.sync()
        self.seconds[name]+=time.perf_counter()-tick
        self.counts[name]+=1
        return result


def coordinate_profile(model, parameters, delta, inputs, targets, timer):
    names=tuple(n for n,_ in model.named_parameters())
    assert set(names)==set(parameters)==set(delta)
    params,tangent=timer.call('coordinate_h2d',lambda:(
        {n:parameters[n].to(inputs.device) for n in names},
        {'gpt.'+n:delta[n].to(inputs.device) for n in names}))
    prior_mode=model.training; model.eval()
    try:
        with torch.no_grad():
            logits,_=timer.call('target_forward',functional_call,model,params,(inputs,targets),tie_weights=True,strict=True)
            recorded=timer.call('readout_d2h_cast',lambda:logits[:,-1,:].cpu().numpy().astype(np.float64))
            wrapper=HiddenStateModel(model)
            prefixed={'gpt.'+n:v for n,v in params.items()}
            def forward(values):
                return functional_call(wrapper,values,(inputs,),tie_weights=True)
            def first(values):
                return jvp(forward,(values,),(tangent,))[1]
            with sdpa_kernel(SDPBackend.MATH):
                h,hp=timer.call('first_directional_derivative',jvp,forward,(prefixed,),(tangent,))
                _,hpp=timer.call('second_directional_derivative',jvp,first,(prefixed,),(tangent,))
            hs=timer.call('readout_d2h_cast',lambda:[v.detach().cpu().numpy().astype(np.float64) for v in (h,hp,hpp)])
    finally: model.train(prior_mode)
    y=timer.call('readout_d2h_cast',lambda:targets[:,-1].cpu().numpy())
    return timer.call('boundary_aggregation',boundary_coordinates,recorded,y,
        parameters['transformer.wte.weight'].numpy(),delta['transformer.wte.weight'].numpy(),*hs)


def instrumented_update(model,opt,lr,t,x,y,vx,vy,seed,alpha,timer):
    def setup():
        model.train(); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    timer.call('update_setup',setup)
    before=timer.call('parameter_d2h_snapshots',b.params,model)
    timer.call('zero_grad',opt.zero_grad,set_to_none=True)
    loss=timer.call('training_forward',model,x,y)[1]
    timer.call('training_backward',loss.backward)
    timer.call('gradient_clip',torch.nn.utils.clip_grad_norm_,model.parameters(),1.)
    timer.call('adamw_update',opt.step)
    details={'loss':timer.call('loss_readout',lambda:float(loss.detach()))}
    proposal=timer.call('parameter_d2h_snapshots',b.params,model)
    delta=timer.call('delta_cpu',lambda:{n:proposal[n]-before[n] for n in before})
    coords=None
    if (t-1)%5==0:
        with b.selection_boundary():
            coords=coordinate_profile(model,before,delta,vx,vy,timer)
            alpha,scores=timer.call('alpha_selection',b.select,coords)
        snapshot=timer.call('internal_audit_d2h',b.params,model)
        def check():
            assert b.phash(snapshot)==b.phash(proposal),'SELECTOR_MUTATION'
        timer.call('internal_hash_check',check)
        details['scores']=scores
    details['alpha']=alpha
    if alpha!=1.:
        def apply():
            with torch.no_grad():
                for n,p in model.named_parameters():
                    p.copy_((before[n]+alpha*delta[n]).to(p.device))
        timer.call('scaled_application_cpu_h2d',apply)
    return alpha,details,coords


def execute(row,loaded,config,reference,rng_hash,instrumented):
    rt,start,batches,vx,vy=loaded; model=rt.model
    with torch.no_grad():
        for n,p in model.named_parameters():p.copy_(start[n].to('cuda'))
    model.zero_grad(set_to_none=True)
    timer=Timer(); b.sync(); wall=time.perf_counter()
    opt=timer.call('optimizer_construction',b.make_optimizer,model,'controller',b.RATES['controller'][config])
    initial=timer.call('evaluation',b.evaluate,model,vx,vy)
    assert initial==reference['initial']
    alpha=1.; records=[]; update_total=0.
    with patch.object(b,'coordinates',original_coordinates):
        for t in range(1,61):
            seed=b.old.seed_for(row,t-1,rng_hash); x,y=batches[t-1]
            b.sync(); tick=time.perf_counter()
            if instrumented:
                alpha,details,coords=instrumented_update(model,opt,b.RATES['controller'][config],t,x,y,vx,vy,seed,alpha,timer)
            else:
                alpha,details,coords=b.update(model,opt,'controller',b.RATES['controller'][config],t,x,y,vx,vy,seed,alpha)
            b.sync(); elapsed=time.perf_counter()-tick; update_total+=elapsed
            expected=reference['steps'][t-1]['fact']
            snapshot=timer.call('external_audit_d2h',b.params,model)
            def check():
                assert details==expected['decision']['details'],('decision',t)
                coord_hash=None
                if coords is not None:
                    coord_hash=b.identity({k:b.tensor_sha256(torch.from_numpy(coords[k])) for k in b.KEYS})
                    assert coord_hash==expected['decision']['coordinates_sha256'],('coordinates',t)
                parameter_hash=b.phash(snapshot)
                assert parameter_hash==expected['post_parameters_sha256'],('parameters',t)
                return parameter_hash,coord_hash
            parameter_hash,coord_hash=timer.call('external_hash_and_comparison',check)
            ev=timer.call('evaluation',b.evaluate,model,vx,vy)
            assert ev==expected['evaluation'],('evaluation',t)
            records.append({'step':t,'alpha':alpha,'accuracy':ev['accuracy'],
                'coordinates_sha256':coord_hash,'parameters_sha256':parameter_hash,'update_seconds':elapsed})
            if ev['accuracy']>=.99:break
    def final_check():
        digest=b.identity(b.structure({'parameters':b.params(model),'optimizer':opt.state_dict()}))
        assert digest==reference['final_state_sha256'],'final state'
        return digest
    final=timer.call('final_state_audit',final_check)
    assert len(records)==len(reference['steps'])
    b.sync(); elapsed=time.perf_counter()-wall
    return {'instrumented':instrumented,'exact':True,'first99_steps':len(records),
        'update_seconds':update_total,'checked_wall_seconds':elapsed,
        'stage_seconds':dict(timer.seconds),'stage_counts':dict(timer.counts),
        'final_state_sha256':final,'records':records}


def main():
    b.old.setup()
    saved_repo=b.REPO; b.REPO=prior.DEPENDENCIES.parent
    previous=b.sealcheck(); b.REPO=saved_repo
    for name,digest in b.read(prior.HERE/'verification_v2/RUN_SEAL.json')['files'].items():
        assert b.sha(prior.HERE/name)==digest,name
    sources={p.name:b.sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md')}
    b.write(HERE/'RUN_SEAL.json',{'base_commit':BASE,'utc':b.old.utc(),'files':sources,
        'prior_result_sha256':b.sha(prior.HERE/'verification_v2/RESULTS.json'),
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__})
    rows={r['entry_id']:r for r in b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs']}
    histories=[]
    for case in b.read(prior.HERE/'verification_v2/RESULTS.json')['histories']:
        entry=case['entry']; ci=case['config']; row=rows[entry]
        source=b.OUT/'search'/entry/('controller-'+str(ci))/'0/TRAJECTORY.json'
        assert b.sha(source)==case['source_ledger_sha256']
        reference=b.read(source); loaded=b.load_row(row)
        try:
            for instrumented in (False,True):
                execute(row,loaded,ci,reference,previous['source_rng_protocol_sha256'],instrumented)
            runs=[]
            for repeat in range(3):
                for instrumented in ((False,True) if repeat%2==0 else (True,False)):
                    run=execute(row,loaded,ci,reference,previous['source_rng_protocol_sha256'],instrumented)
                    run['repeat']=repeat; runs.append(run)
                    b.write(HERE/'receipts'/f'{entry}-{repeat}-{int(instrumented)}.json',run)
                    print(entry,repeat,'profile' if instrumented else 'reference',round(run['update_seconds'],4),flush=True)
            prof=[r for r in runs if r['instrumented']]
            names=sorted(set().union(*(r['stage_seconds'] for r in prof)))
            medians={k:statistics.median(r['stage_seconds'].get(k,0.) for r in prof) for k in names}
            history={'entry':entry,'config':ci,'source_ledger_sha256':b.sha(source),
                'stage_median_seconds':medians,'runs':runs,
                'reference_update_median':statistics.median(r['update_seconds'] for r in runs if not r['instrumented']),
                'profile_update_median':statistics.median(r['update_seconds'] for r in prof)}
            histories.append(history)
        finally:loaded[0].close()
    totals=defaultdict(float)
    for history in histories:
        for k,v in history['stage_median_seconds'].items():totals[k]+=v
    accounted=sum(totals.values())
    ranking=[{'stage':k,'seconds_for_two_histories':v,'percent_accounted':100*v/accounted}
             for k,v in sorted(totals.items(),key=lambda kv:-kv[1])]
    assert all(b.sha(HERE/n)==d for n,d in sources.items())
    b.write(HERE/'RESULTS.json',{'status':'PASS','seal_sha256':b.sha(HERE/'RUN_SEAL.json'),
        'histories':histories,'stage_ranking':ranking,'accounted_seconds':accounted})
    print(json.dumps(ranking,indent=2),flush=True)


if __name__=='__main__':main()
