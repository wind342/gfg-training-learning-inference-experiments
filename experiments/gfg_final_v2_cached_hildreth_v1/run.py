"""Create-only validation and separate deterministic standalone replay."""
from .core import *
from experiments.gfg_final_architecture_v2_safe_screening_v3.run import load
import argparse
import traceback


def guard():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)
    assert subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release^{}'],cwd=REPO,text=True).strip()=='36dab5ce347dbbdac157ef23205f556606d18294'


def history():
    with gzip.open(OLD_ROOT/'STEPS.jsonl.gz','rt',encoding='utf8') as f:return [json.loads(x) for x in f]


def inputs():
    oldpre=read(OLD_ROOT/'PREFLIGHT.json')
    for q,v in oldpre['inputs'].items():assert sha(q)==v,q
    return oldpre['inputs']


def prediction_check(head,d):
    test=torch.load(INPUT/'TEST_FEATURES.pt',weights_only=False,map_location='cpu')
    with torch.no_grad():pred=fixed_affine(p.snapshot(head),test['features'].cuda(),80).argmax(1).cpu().tolist()
    oldev=read(OLD_ROOT/'EVALUATION.json');assert pred==oldev['predictions']
    return dict(predictions=pred,metrics=output_metrics(d,pred,10),old_predictions_identical=True)


def validate(data,dest,kind='cached'):
    dest.mkdir(exist_ok=False)
    d,F,initial,h,y,ids,n,K=data;historical=history()
    head,shadow,opt=make_state(initial);g=Geometry(h,y,n,72,80,Meter())
    endpoint=FullEndpoint(g);refmeter=Meter();reference=instrument_reference(refmeter)
    if kind=='transition':
        from .transition import TransitionEndpoint,Observer
        endpoint=TransitionEndpoint(g,K)
        callstream=gzip.open(dest/'CALLS.jsonl.gz','xt',encoding='utf8')
        observer=Observer(endpoint,K,ids[:n],callstream)
    else:observer=None
    rows=[];reserves_checked=0
    with gzip.open(dest/'STEPS.jsonl.gz','xt',encoding='utf8') as stream:
        for step,current in enumerate(order(d,10),1):
            _,loss=ce_grad(head,F[current],torch.from_numpy(d['y'][current]).cuda(),80)
            norm=nn.utils.clip_grad_norm_(head.parameters(),1.);assert np.isfinite(loss) and torch.isfinite(norm)
            before=p.snapshot(head);propose(head,shadow,opt);native=p.snapshot(shadow);mh=moments_hash(opt)
            ref,_,report=reference(before,native,h,y,n,K,ids[:n],seen=80)
            def audit(reserve,new):
                nonlocal reserves_checked
                for i,c in new:
                    t=int(g.labels[i]);bt,bc=reserve.pair_values[(i,c)]
                    assert bt==refmeter.reference_budget[i,t] and bc==refmeter.reference_budget[i,c],(step,i,t,c,bt,bc,'reserve arithmetic')
                    reserves_checked+=1
            if observer:observer.step=step
            evaluate=observer.evaluate if observer else endpoint.evaluate
            executed,r=projection(before,native,g,K,ids[:n],evaluate,audit)
            compare(r,report);assert equal(executed,ref),(step,'executed head')
            head.load_state_dict(executed);actual=evaluate(p.snapshot(head),r['selected_controls'],'commit')
            assert np.all(actual['C']>=K)
            assert head_hash(head)==historical[step-1]['head_sha256'],(step,'old head')
            assert moments_hash(opt)==mh==historical[step-1]['moments_sha256']
            assert old.opt_counter(opt)==1890+step
            row=dict(step=step,head_sha256=head_hash(head),moments_sha256=mh,report=r,checks='PASS')
            stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush();rows.append(row)
            if step==1 or step%30==0:print('VALIDATE',kind,step,'sweeps',r['sweep_count'],flush=True)
    assert g.builds==1
    ev=prediction_check(head,d);write(dest/'EVALUATION.json',ev)
    save(dest/'CHECKPOINT.pt',dict(head=p.cpu_state(p.snapshot(head)),optimizer=opt.state_dict()))
    result=dict(status='PASS',steps=210,head_matches=210,moments_matches=210,deficits_selections_active_pairs_sweeps_identical=True,
        primal_and_dual_byte_identical=True,pair_reserves_byte_identical=True,checked_pairs=reserves_checked,
        final_predictions_identical=True,floor_violations=0,geometry_builds=g.builds,sweeps=sum(r['report']['sweep_count'] for r in rows))
    if observer:
        callstream.close()
        result.update(false_safe=0,false_wrong=0,observer_calls=observer.calls,certified_positive=observer.positive,
                      certified_negative=observer.negative,anchor_updates=endpoint.anchor_updates,
                      executed_anchor_full_refresh_rows=endpoint.refresh_rows)
    write(dest/'CHECKS.json',result);guard();print(json.dumps(result),flush=True)
    return result


def replay(data,dest,kind,profile=False):
    output=dest/(('PROFILE_' if profile else 'TIMED_')+kind+'.json')
    if output.exists():raise FileExistsError(output)
    d,F,initial,h,y,ids,n,K=data;historical=history()
    head,shadow,opt=make_state(initial);meter=Meter(profile)
    sync();t=time.perf_counter()
    if kind=='full':g=None;endpoint=None;reference=instrument_reference(meter)
    else:
        g=Geometry(h,y,n,72,80,meter);endpoint=FullEndpoint(g)
        if kind=='transition':
            from .transition import TransitionEndpoint
            endpoint=TransitionEndpoint(g,K)
    sync();setup=time.perf_counter()-t
    common=[];control=[];hashes=[];sweeps=0;budget_entries=budget_tiles=0
    for step,current in enumerate(order(d,10),1):
        sync();t=time.perf_counter()
        ce_grad(head,F[current],torch.from_numpy(d['y'][current]).cuda(),80)
        nn.utils.clip_grad_norm_(head.parameters(),1.)
        before=p.snapshot(head);propose(head,shadow,opt);native=p.snapshot(shadow)
        sync();common.append(time.perf_counter()-t)
        mh=moments_hash(opt)
        sync();t=time.perf_counter()
        if kind=='full':
            executed,_,r=reference(before,native,h,y,n,K,ids[:n],seen=80)
            head.load_state_dict(executed)
            m=margins(meter.affine(p.snapshot(head),h,80),y)
            C=counts(m[:n].cpu().numpy(),y[:n].cpu().numpy(),72)
        else:
            executed,r=projection(before,native,g,K,ids[:n],endpoint.evaluate)
            head.load_state_dict(executed)
            C=endpoint.evaluate(p.snapshot(head),r['selected_controls'],'commit')['C']
        sync();control.append(time.perf_counter()-t)
        assert np.all(C>=K)
        assert head_hash(head)==historical[step-1]['head_sha256'],(kind,step,'head')
        assert moments_hash(opt)==mh==historical[step-1]['moments_sha256']
        # Check exact original primal and dual identities independently of timing.
        oldr=historical[step-1]['report']
        for key in ('final_active','final_duals','selected_controls','sweep_count','skip'):
            assert r[key]==oldr[key],(kind,step,key)
        for a,b in zip(r['sweeps'],oldr['sweeps']):
            assert all(a[k]==b[k] for k in ('sweep','safe','constraints','primal_sha256','dual_sha256'))
        sweeps+=r['sweep_count'];budget_entries+=r.get('padded_budget_entries',n*80 if r['projection_triggered'] else 0)
        budget_tiles+=r.get('budget_tiles',int(r['projection_triggered']))
        hashes.append(dict(step=step,head=head_hash(head),moments=mh))
        if step%60==0:print('PROFILE' if profile else 'TIMED',kind,step,flush=True)
    ev=prediction_check(head,d)
    result=dict(kind=kind,profiled=profile,steps=210,setup_wall=setup,common_update_wall=sum(common),control_wall=sum(control),
        total_wall=setup+sum(common)+sum(control),components=dict(meter.seconds),exact_old_rows=endpoint.old_rows if endpoint else n*meter.calls,
        total_affine_rows=meter.rows,padded_affine_rows=meter.padded,gemm_calls=meter.gemms,endpoint_calls=meter.calls,
        sweeps=sweeps,padded_budget_entries=budget_entries,budget_gemm_calls=budget_tiles,
        per_step_control=control,verification=hashes,final_predictions_identical=ev['old_predictions_identical'])
    if kind=='transition':result.update(executed_anchor_full_refresh_rows=endpoint.refresh_rows,anchor_updates=endpoint.anchor_updates)
    write(output,result)
    print(kind,'profile' if profile else 'standalone',result['total_wall'],dict(meter.seconds),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['validate','benchmark','profile','transition','transition-benchmark','transition-profile'])
    args=parser.parse_args();guard();p.setup()
    if args.mode=='validate':
        ROOT.mkdir(exist_ok=False)
        write(ROOT/'PREFLIGHT.json',dict(base=BASE,inputs=inputs(),runtime=runtime(),
            sources={q.name:sha(q) for q in HERE.glob('*.py')},protocol_sha256=sha(HERE/'PROTOCOL.md')))
    data=load()
    if args.mode=='validate':validate(data,ROOT/'cached_validation')
    elif args.mode=='transition':
        assert read(ROOT/'cached_validation/CHECKS.json')['status']=='PASS'
        if (ROOT/'transition_validation').exists():raise FileExistsError(ROOT/'transition_validation')
        from .tests import preflight
        write(ROOT/'TRANSITION_PREFLIGHT.json',preflight())
        validate(data,ROOT/'transition_validation','transition')
    elif args.mode.startswith('transition-'):
        assert read(ROOT/'transition_validation/CHECKS.json')['status']=='PASS'
        replay(data,ROOT,'transition',args.mode=='transition-profile')
    else:
        assert read(ROOT/'cached_validation/CHECKS.json')['status']=='PASS'
        for kind in ('full','cached'):replay(data,ROOT,kind,args.mode=='profile')
    guard()


if __name__=='__main__':
    try:main()
    except Exception:
        if ROOT.exists():write(ROOT/f'FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
