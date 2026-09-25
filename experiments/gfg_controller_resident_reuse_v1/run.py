"""Exact replay and paired timing of the combined implementation."""
import argparse
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from experiments.gfg_controller_batched_coordinates_v1 import run as previous
from experiments.gfg_controller_batched_coordinates_v1.nanogpt_adapter import coordinates as reference_coordinates
from .engine import ResidentBuffers, shared_derivatives, NanoGPTAdapter, Controller
import torch
import numpy as np
from torch.func import jvp

b=previous.b
HERE=Path(__file__).resolve().parent


def unit_tests():
    results=[]
    for device in ('cpu','cuda'):
        for width in (3,17,64):
            torch.manual_seed(91+width)
            model=torch.nn.Sequential(torch.nn.Linear(width,7),torch.nn.Linear(7,2)).to(device)
            buffers=ResidentBuffers(model)
            before=b.params(model); buffers.capture('before')
            with torch.no_grad():
                for p in model.parameters():p.add_(torch.randn_like(p)*.001)
            proposal=b.params(model); buffers.capture('proposal'); buffers.form_delta()
            delta={n:proposal[n]-before[n] for n in before}
            assert all(torch.equal(delta[n],buffers.named['delta'][n].cpu()) for n in delta)
            for alpha in (.25,.5,.75,1.):
                with torch.no_grad():
                    for n,p in model.named_parameters():p.copy_(proposal[n].to(device))
                buffers.apply(alpha)
                expected=proposal if alpha==1. else {n:before[n]+alpha*delta[n] for n in before}
                assert b.phash(b.params(model))==b.phash(expected)
                snap=buffers.snapshot_model()
                assert b.phash(snap)==b.phash(expected)
                buffers.capture('audit')
                assert b.phash(snap)==b.phash(expected),'snapshot aliased resident storage'
            p={'x':torch.randn(width,device=device)}; d={'x':torch.randn(width,device=device)}
            def forward(v):return torch.sin(v['x'])*v['x']**2
            h,hp=jvp(forward,(p,),(d,))
            def first(v):return jvp(forward,(v,),(d,))[1]
            _,hpp=jvp(first,(p,),(d,))
            actual=shared_derivatives(forward,p,d)
            assert all(torch.equal(a,c) for a,c in zip((h,hp,hpp),actual))
            results.append({'device':device,'width':width,'buffers_and_derivatives_exact':True})
    return results


def execute(row,loaded,config,reference,rng_hash,optimized):
    rt,start,batches,vx,vy=loaded; model=rt.model
    with torch.no_grad():
        for n,p in model.named_parameters():p.copy_(start[n].to('cuda'))
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    b.sync(); wall=time.perf_counter(); tick=wall
    opt=b.make_optimizer(model,'controller',b.RATES['controller'][config])
    engine=Controller(model,NanoGPTAdapter(model),b.select,b.phash,b.selection_boundary) if optimized else None
    b.sync(); setup=time.perf_counter()-tick
    compute=setup; initial=b.evaluate(model,vx,vy)
    assert initial==reference['initial']
    records=[]; alpha=1.; reuse_count=0
    with patch.object(b,'coordinates',reference_coordinates):
        for t in range(1,61):
            seed=b.old.seed_for(row,t-1,rng_hash); x,y=batches[t-1]
            b.sync(); tick=time.perf_counter()
            if optimized:
                alpha,details,coords=engine.update(model,opt,t,x,y,vx,vy,seed,alpha)
            else:
                alpha,details,coords=b.update(model,opt,'controller',b.RATES['controller'][config],t,x,y,vx,vy,seed,alpha)
            b.sync(); elapsed=time.perf_counter()-tick; compute+=elapsed
            expected=reference['steps'][t-1]['fact']
            assert details==expected['decision']['details'],('DECISION',row['entry_id'],t)
            coord_hash=None
            if coords is not None:
                coord_hash=b.identity({k:b.tensor_sha256(torch.from_numpy(coords[k])) for k in b.KEYS})
                assert coord_hash==expected['decision']['coordinates_sha256'],('COORDINATES',row['entry_id'],t)
            post=engine.post_snapshot() if optimized else b.params(model)
            if optimized and engine.reused_post_snapshot is not None:reuse_count+=1
            digest=b.phash(post)
            assert digest==expected['post_parameters_sha256'],('PARAMETERS',row['entry_id'],t)
            ev=b.evaluate(model,vx,vy)
            assert ev==expected['evaluation'],('EVALUATION',row['entry_id'],t)
            records.append({'step':t,'alpha':alpha,'coordinates_sha256':coord_hash,
                'post_parameters_sha256':digest,'accuracy':ev['accuracy'],'update_seconds':elapsed})
            if ev['accuracy']>=.99:break
    assert len(records)==len(reference['steps'])
    final=b.identity(b.structure({'parameters':b.params(model),'optimizer':opt.state_dict()}))
    assert final==reference['final_state_sha256'],'FINAL_STATE'
    b.sync(); full=time.perf_counter()-wall
    return {'optimized':optimized,'exact':True,'first99_steps':len(records),
        'compute_seconds':compute,'setup_seconds':setup,'checked_wall_seconds':full,
        'reused_post_snapshots':reuse_count,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'records':records,'final_state_sha256':final}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--unit-only',action='store_true')
    ap.add_argument('--output',type=Path,default=HERE/'verification'); args=ap.parse_args()
    b.old.setup()
    if args.unit_only:print(json.dumps(unit_tests()));return
    old_repo=b.REPO; b.REPO=previous.DEPENDENCIES.parent
    prior_seal=b.sealcheck(); b.REPO=old_repo
    for name,digest in b.read(previous.HERE/'verification_v2/RUN_SEAL.json')['files'].items():
        assert b.sha(previous.HERE/name)==digest,name
    sources={p.name:b.sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md')}
    out=args.output
    b.write(out/'RUN_SEAL.json',{'base_commit':'689140d90fe7508392e9fcd65b7c9da44352c72a',
        'files':sources,'utc':b.old.utc(),'gpu':torch.cuda.get_device_name(0),
        'torch':torch.__version__,'prior_results_sha256':b.sha(previous.HERE/'verification_v2/RESULTS.json')})
    try:
        b.write(out/'UNIT_TESTS.json',unit_tests())
        rows={r['entry_id']:r for r in b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs']}
        histories=[]
        for case in b.read(previous.HERE/'verification_v2/RESULTS.json')['histories']:
            entry=case['entry'];ci=case['config']; row=rows[entry]
            source=b.OUT/'search'/entry/('controller-'+str(ci))/'0/TRAJECTORY.json'
            assert b.sha(source)==case['source_ledger_sha256']
            reference=b.read(source);loaded=b.load_row(row)
            try:
                for fast in (False,True):
                    warm=execute(row,loaded,ci,reference,prior_seal['source_rng_protocol_sha256'],fast)
                    b.write(out/'warmup'/f'{entry}-{int(fast)}.json',warm)
                    print('warmup',entry,fast,'exact',flush=True)
                runs=[]
                for rep in range(5):
                    for fast in ((False,True) if rep%2==0 else (True,False)):
                        result=execute(row,loaded,ci,reference,prior_seal['source_rng_protocol_sha256'],fast)
                        result['repeat']=rep;runs.append(result)
                        b.write(out/'receipts'/f'{entry}-{rep}-{int(fast)}.json',result)
                        print(entry,rep,'combined' if fast else 'reference',round(result['compute_seconds'],5),round(result['checked_wall_seconds'],5),flush=True)
                summaries={}
                for metric in ('compute_seconds','checked_wall_seconds'):
                    ref=statistics.median(r[metric] for r in runs if not r['optimized'])
                    fast=statistics.median(r[metric] for r in runs if r['optimized'])
                    summaries[metric]={'reference_median':ref,'combined_median':fast,
                        'reduction_percent':100*(1-fast/ref),'paired_speedups':[
                        next(r[metric] for r in runs if r['repeat']==rep and not r['optimized'])/
                        next(r[metric] for r in runs if r['repeat']==rep and r['optimized']) for rep in range(5)]}
                histories.append({'entry':entry,'config':ci,'source_ledger_sha256':b.sha(source),
                    'summary':summaries,'runs':runs})
            finally:loaded[0].close()
        assert all(b.sha(HERE/n)==digest for n,digest in sources.items())
        b.write(out/'RESULTS.json',{'status':'PASS','seal_sha256':b.sha(out/'RUN_SEAL.json'),'histories':histories})
        print('PASS',json.dumps([h['summary'] for h in histories]),flush=True)
    except Exception as exc:
        b.write(out/'FAILURE.json',{'error':repr(exc),'utc':b.old.utc()})
        raise


if __name__=='__main__':main()
