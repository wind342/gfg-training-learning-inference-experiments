"""Read-only reuse of historical evidence; write only this experiment's receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEPENDENCIES = Path('D:/codex/gfg-optimizer-capability-benchmark-v1/runtime')
for name in ('Sophia', 'sls', 'optimizers'):
    sys.path.insert(0, str(DEPENDENCIES/name))
sys.path.insert(0, str(REPO/'src'))
import numpy as np
import torch
from experiments.gfg_optimizer_capability_benchmark_v1 import benchmark as b
from experiments.gfg_controller_batched_coordinates_v1.kernel import boundary_coordinates, KEYS
from experiments.gfg_controller_batched_coordinates_v1.nanogpt_adapter import coordinates


def loop(logits, targets, w, dw, h, hp, hpp):
    logits, w, dw, h, hp, hpp = [np.asarray(v,dtype=np.float64) for v in (logits,w,dw,h,hp,hpp)]
    n, c = logits.shape
    out = {k: np.empty((n, c-1)) for k in KEYS}
    comp = []
    for i, y in enumerate(targets):
        order = np.argsort(logits[i])[::-1]
        cs = order[order != y]; comp.append(cs)
        wb, db = w[y][None, :] - w[cs], dw[y][None, :] - dw[cs]
        out['g0'][i] = logits[i, y] - logits[i, cs]
        out['first_gap'][i] = (db @ h[i] + wb @ hp[i]).astype(np.float32).astype(np.float64)
        out['joint_rotation'][i] = (db @ hp[i]).astype(np.float32).astype(np.float64)
        out['hidden_curvature'][i] = (.5 * (wb @ hpp[i])).astype(np.float32).astype(np.float64)
    return {**out, 'competitors': np.array(comp)}


def tests():
    rng = np.random.default_rng(20260925)
    receipts = []
    for n, c, d in [(1,2,3), (7,5,17), (53,11,32), (212,24,64), (257,37,19)]:
        values = [rng.normal(size=s) for s in [(n,c),(c,d),(c,d),(n,d),(n,d),(n,d)]]
        values[0][0] = 0. # ties preserve exact competitor ordering
        args = (values[0], rng.integers(c,size=n), *values[1:])
        reference = loop(*args)
        for chunk in (1, 16, 256):
            actual = boundary_coordinates(*args, chunk_size=chunk)
            assert all(np.array_equal(reference[k],actual[k]) for k in (*KEYS,'competitors'))
            assert b.select(reference) == b.select(actual)
        receipts.append({'targets':n,'classes':c,'width':d,'exact':True})
    bad = list(args); bad[1] = np.full(n, c)
    for v in (bad, [np.full_like(args[0],np.nan),*args[1:]]):
        try: boundary_coordinates(*v)
        except ValueError: pass
        else: raise AssertionError('invalid input accepted')
    return receipts


def micro(args):
    assert all(np.array_equal(loop(*args)[k], boundary_coordinates(*args)[k]) for k in KEYS)
    times = {'loop':[], 'batch':[]}
    for r in range(9):
        for name in (('loop','batch') if r%2==0 else ('batch','loop')):
            fn = loop if name=='loop' else boundary_coordinates
            start = time.perf_counter()
            for _ in range(100): fn(*args)
            times[name].append((time.perf_counter()-start)/100)
    return {'seconds':times,'speedup':statistics.median(times['loop'])/statistics.median(times['batch'])}


def trajectory(row, loaded, config, reference, rng_hash, optimized):
    rt, start, batches, vx, vy = loaded
    model = rt.model
    with torch.no_grad():
        for n,p in model.named_parameters(): p.copy_(start[n].to('cuda'))
    model.zero_grad(set_to_none=True)
    b.sync(); wall = time.perf_counter(); tick = wall
    opt = b.make_optimizer(model,'controller',b.RATES['controller'][config])
    b.sync(); compute = time.perf_counter()-tick
    assert b.evaluate(model,vx,vy) == reference['initial']
    alpha = 1.; records = []; sample = None
    with patch.object(b, 'coordinates', coordinates if optimized else ORIGINAL):
        for t in range(1,61):
            seed = b.old.seed_for(row,t-1,rng_hash)
            x,y = batches[t-1]
            b.sync(); tick = time.perf_counter()
            alpha, details, coords = b.update(model,opt,'controller',b.RATES['controller'][config],t,x,y,vx,vy,seed,alpha)
            b.sync(); seconds = time.perf_counter()-tick; compute += seconds
            expected = reference['steps'][t-1]['fact']
            assert details == expected['decision']['details'], ('decision',t)
            if coords is not None:
                digest = b.identity({k:b.tensor_sha256(torch.from_numpy(coords[k])) for k in KEYS})
                assert digest == expected['decision']['coordinates_sha256'], ('coordinates',t)
            state_hash = b.phash(b.params(model))
            assert state_hash == expected['post_parameters_sha256'], ('parameters',t)
            evaluation = b.evaluate(model,vx,vy)
            assert evaluation == expected['evaluation'], ('evaluation',t)
            records.append({'step':t,'alpha':alpha,'accuracy':evaluation['accuracy'],
                            'parameters_sha256':state_hash,'update_seconds':seconds})
            if evaluation['accuracy'] >= .99: break
    assert len(records)==len(reference['steps']), 'crossing mismatch'
    final = b.identity(b.structure({'parameters':b.params(model),'optimizer':opt.state_dict()}))
    assert final == reference['final_state_sha256'], 'final state mismatch'
    b.sync(); elapsed = time.perf_counter()-wall
    return {'optimized':optimized,'compute_seconds':compute,'checked_wall_seconds':elapsed,
            'first99_steps':len(records),'final_state_sha256':final,'steps':records,'exact':True}


ORIGINAL = b.coordinates


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--receipt-dir',type=Path,default=HERE/'verification'); args=parser.parse_args()
    checks=tests()
    if args.self_test: print(json.dumps(checks)); return
    b.old.setup()
    # Verify previous sealed sources, dependency identities and historical inputs.
    saved_repo=b.REPO
    b.REPO=DEPENDENCIES.parent
    prior_seal=b.sealcheck()
    b.REPO=saved_repo
    sources={p.name:b.sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md')}
    seal={'files':sources,'base_commit':'77333c8aa890aa1b406ff62086fb555d16eba16b',
          'prior_freeze_sha256':b.sha(b.HERE/'FREEZE_SEAL.json'),'utc':b.old.utc(),
          'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'numpy':np.__version__,
          'shape_tests':checks}
    receipt=args.receipt_dir
    b.write(receipt/'RUN_SEAL.json',seal)
    selections=b.read(b.HERE/'LOO_SELECTION.json')['entries']
    rows={r['entry_id']:r for r in b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs']}
    output=[]
    eligible=[]
    for entry in sorted(selections):
        ci=selections[entry]['controller']['config']
        ledger=b.read(b.OUT/'search'/entry/('controller-'+str(ci))/'0/TRAJECTORY.json')
        if ledger['initial']['accuracy'] < .99: eligible.append(entry)
    for entry in eligible[:2]:
        row=rows[entry]; config=selections[entry]['controller']['config']
        source=b.OUT/'search'/entry/('controller-'+str(config))/'0/TRAJECTORY.json'
        reference=b.read(source); loaded=b.load_row(row)
        assert reference['initial']['accuracy'] < .99
        try:
            # Collect an actual pre-response coordinate input for isolated timing.
            rt,start,batches,vx,vy=loaded
            captured=[]
            def capture(*a,**kw): captured.append(tuple(np.array(v,copy=True) for v in a)); return boundary_coordinates(*a,**kw)
            with patch('experiments.gfg_controller_batched_coordinates_v1.nanogpt_adapter.boundary_coordinates',capture):
                warm=trajectory(row,loaded,config,reference,prior_seal['source_rng_protocol_sha256'],True)
            trajectory(row,loaded,config,reference,prior_seal['source_rng_protocol_sha256'],False)
            isolated=micro(captured[0])
            runs=[]
            for rep in range(5):
                for fast in ((False,True) if rep%2==0 else (True,False)):
                    result=trajectory(row,loaded,config,reference,prior_seal['source_rng_protocol_sha256'],fast)
                    result['repeat']=rep; runs.append(result)
                    print(entry,rep,'batch' if fast else 'loop',round(result['compute_seconds'],5),flush=True)
            summary={}
            for key in ('compute_seconds','checked_wall_seconds'):
                baseline=statistics.median(r[key] for r in runs if not r['optimized'])
                fast=statistics.median(r[key] for r in runs if r['optimized'])
                summary[key]={'reference_median':baseline,'batched_median':fast,
                              'reduction_percent':100*(1-fast/baseline),
                              'paired_speedups':[next(r[key] for r in runs if r['repeat']==i and not r['optimized'])/next(r[key] for r in runs if r['repeat']==i and r['optimized']) for i in range(5)]}
            output.append({'entry':entry,'source_ledger_sha256':b.sha(source),'config':config,
                           'summary':summary,'micro':isolated,'runs':runs})
            b.write(receipt/(entry+'.json'),output[-1])
        finally: loaded[0].close()
    assert all(b.sha(HERE/p)==h for p,h in sources.items()), 'sources changed during run'
    b.write(receipt/'RESULTS.json',{'status':'PASS','histories':output,'seal_sha256':b.sha(receipt/'RUN_SEAL.json')})
    print('PASS',json.dumps([o['summary'] for o in output]),flush=True)


if __name__=='__main__': main()
