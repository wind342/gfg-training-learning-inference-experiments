"""A only: independent trajectories, original cached Hildreth actuator."""
from .core import *
import argparse
import traceback


def evaluate_final(head,d,arm):
    test=torch.load(INPUT/'TEST_FEATURES.pt',weights_only=False,map_location='cpu')
    h=test['features'].cuda();state=p.snapshot(head);kind,tile=ARMS[arm]
    pred=EndpointKernel(h,kind,tile)(state).argmax(1).cpu().tolist()
    ref=fixed_affine(state,h,80).argmax(1).cpu().tolist()
    hist=read(Path('E:/gfg-evidence/final-v2-cached-hildreth-v1/cached_validation/EVALUATION.json'))['predictions']
    return dict(predictions=pred,fixed256_predictions=ref,metrics=output_metrics(d,pred,10),
                fixed256_metrics=output_metrics(d,ref,10),same_head_prediction_differences=sum(a!=b for a,b in zip(pred,ref)),
                historical_prediction_differences=sum(a!=b for a,b in zip(pred,hist)))


def run(data,arm,mode):
    d,F,initial,h,y,ids,n,K=data;kind,tile=ARMS[arm]
    dest=ROOT/arm/mode;dest.mkdir(parents=True,exist_ok=False)
    if mode=='validate':write(dest/'NUMERICAL_PREFLIGHT.json',numerical_preflight(data,arm))
    else:assert read(ROOT/arm/'validate/CHECKS.json')['status']=='PASS'
    historical=history();head,shadow,opt=make_state(initial)
    sync();t=time.perf_counter();meter=EndpointMeter(h,kind,tile,mode=='profile')
    g=Geometry(h,y,n,72,80,meter);endpoint=FullEndpoint(g);sync();setup=time.perf_counter()-t
    common=[];control=[];rows=[];floor=0
    stream=gzip.open(dest/'STEPS.jsonl.gz','xt',encoding='utf8')
    callstream=gzip.open(dest/'CALLS.jsonl.gz','xt',encoding='utf8') if mode=='validate' else None
    obs=Observer(endpoint,K,ids[:n],callstream) if callstream else None
    with stream:
        for step,current in enumerate(order(d,10),1):
            sync();t=time.perf_counter()
            _,loss=ce_grad(head,F[current],torch.from_numpy(d['y'][current]).cuda(),80)
            norm=nn.utils.clip_grad_norm_(head.parameters(),1.);assert np.isfinite(loss) and torch.isfinite(norm)
            before=p.snapshot(head);propose(head,shadow,opt);native=p.snapshot(shadow)
            sync();common.append(time.perf_counter()-t);mh=moments_hash(opt)
            if obs:obs.step=step
            ev=obs.evaluate if obs else endpoint.evaluate
            sync();t=time.perf_counter()
            executed,r=projection(before,native,g,K,ids[:n],ev)
            head.load_state_dict(executed);actual=ev(p.snapshot(head),r['selected_controls'],'commit')
            sync();control.append(time.perf_counter()-t)
            violation=int((actual['C']<K).sum());floor+=violation;assert violation==0
            assert moments_hash(opt)==mh and old.opt_counter(opt)==1890+step
            oldrow=historical[step-1]
            row=dict(step=step,head_sha256=head_hash(head),moments_sha256=mh,report=r,floor_violations=violation,
                historical_head_identical=head_hash(head)==oldrow['head_sha256'],historical_moments_identical=mh==oldrow['moments_sha256'],
                historical_control_identity=same_report(r,oldrow['report']))
            rows.append(row);stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush()
            if step==1 or step%60==0:print('A',mode,arm,step,'sweeps',r['sweep_count'],flush=True)
    if callstream:callstream.close()
    if mode=='validate':save(dest/'CHECKPOINT.pt',dict(head=p.cpu_state(p.snapshot(head)),optimizer=opt.state_dict()))
    else:
        with gzip.open(ROOT/arm/'validate/STEPS.jsonl.gz','rt') as f:validated=[json.loads(s) for s in f]
        for a,b in zip(rows,validated):
            assert all(a[k]==b[k] for k in ('step','head_sha256','moments_sha256','report','floor_violations')),'independent replay differs from validated trajectory'
    ev=evaluate_final(head,d,arm);write(dest/'EVALUATION.json',ev)
    report=dict(status='PASS',arm=arm,mode=mode,tile=tile,steps=210,setup_wall=setup,
        control_wall=sum(control),total_wall=setup+sum(common)+sum(control),common_wall=sum(common),
        wall_includes_observer=mode=='validate',profile_components=dict(meter.seconds),
        gemm_api_calls=meter.api_calls,tile_matrix_products=meter.gemms,padded_rows=meter.padded,
        endpoint_calls=meter.calls,exact_old_rows=endpoint.old_rows,sweeps=sum(r['report']['sweep_count'] for r in rows),
        skips=sum(r['report']['skip'] for r in rows),floor_violations=floor,
        head_matches=sum(r['historical_head_identical'] for r in rows),moments_matches=sum(r['historical_moments_identical'] for r in rows),
        control_matches={k:sum(r['historical_control_identity'][k] for r in rows) for k in rows[0]['historical_control_identity']},
        metrics=ev['metrics'],fixed256_metrics=ev['fixed256_metrics'],historical_prediction_differences=ev['historical_prediction_differences'])
    if obs:report.update(sign_differences=obs.signs,old_sign_differences=obs.old_signs,executed_old_sign_differences=obs.executed_signs,
        max_logit_difference=obs.max_error,argmax_differences=obs.pred_diffs,count_difference_calls=obs.count_diff_calls,
        choice_difference_calls=obs.choice_diff_calls,fixed256_executed_floor_violations=obs.reference_floor)
    write(dest/'CHECKS.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','fixed256_metrics')}),flush=True);guard()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['validate','timed','profile']);ap.add_argument('--arm',choices=list(ARMS))
    a=ap.parse_args();guard();p.setup()
    if not ROOT.exists():
        assert a.mode=='validate';ROOT.mkdir(exist_ok=False)
        write(ROOT/'PREFLIGHT.json',dict(base=BASE,inputs=inputs(),runtime=runtime(),arms=ARMS,
            sources={q.name:sha(q) for q in HERE.glob('*.py')},protocol_sha256=sha(HERE/'PROTOCOL.md')))
    data=load()
    for arm in [a.arm] if a.arm else ARMS:run(data,arm,a.mode)
    inputs();guard()


if __name__=='__main__':
    try:main()
    except Exception:
        if ROOT.exists():write(ROOT/f'FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
