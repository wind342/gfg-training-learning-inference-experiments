"""Isolated one-update allocation diagnostic, not a time-to-capability test."""
import gc
import statistics
import time
import torch
from . import phase1 as run
from .blockwise_runtime import BlockwiseController

b=run.b;HERE=run.HERE


def case(row,mode,rep,seed):
    loaded=b.load_row(row);rt,start,batches,vx,vy=loaded;model=rt.model
    run.reset(model,start);model.zero_grad(set_to_none=True)
    opt=run.prior.native.optimizer(model,.003)
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    if mode=='old':engine=run.prior.native.Controller(model,run.NanoGPTAdapter(model),b.select,b.phash,b.selection_boundary)
    elif mode=='global':engine=run.RuntimeController(model)
    elif mode=='blockwise':engine=BlockwiseController(model)
    else:engine=None
    after_constructor=torch.cuda.memory_allocated();torch.cuda.synchronize();tick=time.perf_counter()
    x,y=batches[0]
    if engine is not None:alpha,_,_=engine.update(model,opt,1,x,y,vx,vy,seed,1.)
    else:
        model.train();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        opt.zero_grad(set_to_none=True);loss=model(x,y)[1];loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();alpha=1.
    torch.cuda.synchronize();elapsed=time.perf_counter()-tick
    record={'mode':mode,'rep':rep,'baseline_allocated':baseline,'after_constructor_allocated':after_constructor,
        'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
        'update_seconds_diagnostic':elapsed,'alpha':alpha,'parameter_count':sum(p.numel() for p in model.parameters())}
    policy='fixed1' if mode=='native' else 'dynamic';ref=run.prior.record(policy,'search',row['entry_id'],1)['steps'][0]
    post=b.phash(b.params(model));assert post==ref['post_parameters_sha256']
    record['post_parameters_sha256']=post
    rt.close();return record


def main():
    b.old.setup();assert b.read(HERE/'blockwise_validation/AUDIT.json')['status']=='PASS'
    rows=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs'];row=rows[0]
    seal=b.read(HERE/'PHASE1_SEAL.json');seed=b.old.seed_for(row,0,seal['source_rng_protocol_sha256'])
    b.write(HERE/'MEMORY_SEAL.json',{'source_sha256':b.sha(HERE/'memory.py'),
        'blockwise_audit_sha256':b.sha(HERE/'blockwise_validation/AUDIT.json'),
        'entry':row['entry_id'],'lr':.003,'step':1,'repetitions':3,'seed':seed,
        'modes':['native','old','global','blockwise'],'purpose':'one-refresh allocation diagnostic; not steady-state or large-model peak'})
    records=[]
    for rep in range(3):
        for mode in ('native','old','global','blockwise'):
            records.append(case(row,mode,rep,seed));gc.collect();torch.cuda.empty_cache()
    result={'records':records,'median_by_mode':{m:{k:statistics.median(r[k] for r in records if r['mode']==m)
        for k in ('peak_allocated','peak_reserved','update_seconds_diagnostic')} for m in ('native','old','global','blockwise')},
        'buffer_formula':{'old_bytes_per_parameter':20,'new_bytes_per_parameter':8,'reduction_fraction':.6},
        'lower_bound_estimates_not_measurements':{str(p):{'old_fp32_state_plus_buffers_bytes':36*p,
            'new_fp32_state_plus_buffers_bytes':24*p,'control_buffers_bytes':8*p} for p in (70000000,160000000)},
        'exclusions':'Large model activations, JVP workspace, allocator reserve and framework overhead are not included in estimates.',
        'memory_seal_sha256':b.sha(HERE/'MEMORY_SEAL.json')}
    assert all(r['peak_allocated']>=r['after_constructor_allocated']>=r['baseline_allocated'] for r in records)
    b.write(HERE/'PHASE1_MEMORY.json',result);print(result['median_by_mode'])


if __name__=='__main__':main()
