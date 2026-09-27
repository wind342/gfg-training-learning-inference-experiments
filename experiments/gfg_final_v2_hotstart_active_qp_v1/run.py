"""B only: original fixed256 endpoint, independent warm-start actuator."""
from .core import *
from experiments.gfg_final_v2_cached_hildreth_v1.run import inputs
from experiments.gfg_final_architecture_v2_safe_screening_v3.run import load
import argparse
import traceback


def guard():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',
        ':(exclude)experiments/gfg_final_v2_endpoint_kernel_v1',
        ':(exclude)experiments/gfg_final_v2_hotstart_active_qp_v1'],cwd=REPO,check=True,capture_output=True)


class ColdAudit:
    def __init__(self,stream):self.stream=stream;self.step=0;self.rows=[]
    def __call__(self,G,b,warm,x,info,outer):
        t=time.perf_counter();cold,ci=active_qp(G,b,np.zeros_like(warm));coldwall=time.perf_counter()-t
        error=float(np.max(np.abs(x-cold),initial=0.))
        obj=lambda v:float(.5*v@G@v-b@v)
        same=info['status']==ci['status']=='PASS' and abs(obj(x)-obj(cold))<1e-8
        if info['status']=='PASS':assert same and error<1e-6,(self.step,outer,'warm/cold QP disagreement')
        row=dict(step=self.step,outer=outer,constraints=len(b),positive_warm=int((warm>0).sum()),
            warm_iterations=info['iterations'],cold_iterations=ci['iterations'],warm_status=info['status'],cold_status=ci['status'],
            cold_wall=coldwall,max_dual_difference=error,objective_difference=abs(obj(x)-obj(cold)),solutions_agree=same,
            initial_warm_residual=info['initial'],initial_cold_residual=ci['initial'])
        self.rows.append(row);self.stream.write(json.dumps(row,separators=(',',':'))+'\n');self.stream.flush()


def final_eval(head,d):
    test=torch.load(INPUT/'TEST_FEATURES.pt',weights_only=False,map_location='cpu')
    with torch.no_grad():pred=fixed_affine(p.snapshot(head),test['features'].cuda(),80).argmax(1).cpu().tolist()
    baseline=read(Path('E:/gfg-evidence/final-v2-cached-hildreth-v1/cached_validation/EVALUATION.json'))['predictions']
    return dict(predictions=pred,metrics=output_metrics(d,pred,10),baseline_prediction_differences=sum(a!=b for a,b in zip(pred,baseline)))


def run(data,mode):
    dest=ROOT/mode;dest.mkdir(exist_ok=False)
    d,F,initial,h,y,ids,n,K=data;head,shadow,opt=make_state(initial)
    with gzip.open('E:/gfg-evidence/final-v2-cached-hildreth-v1/cached_validation/STEPS.jsonl.gz','rt') as f:baseline=[json.loads(s) for s in f]
    sync();t=time.perf_counter();meter=Meter(mode=='profile');g=Geometry(h,y,n,72,80,meter)
    endpoint=FullEndpoint(g);actuator=Actuator(g);sync();setup=time.perf_counter()-t
    auditstream=gzip.open(dest/'COLD_AUDIT.jsonl.gz','xt',encoding='utf8') if mode=='validate' else None
    audit=ColdAudit(auditstream) if auditstream else None;actuator.audit=audit
    rows=[];common=[];control=[]
    with gzip.open(dest/'STEPS.jsonl.gz','xt',encoding='utf8') as stream:
        for step,current in enumerate(order(d,10),1):
            sync();t=time.perf_counter()
            _,loss=ce_grad(head,F[current],torch.from_numpy(d['y'][current]).cuda(),80)
            norm=nn.utils.clip_grad_norm_(head.parameters(),1.);assert np.isfinite(loss) and torch.isfinite(norm)
            before=p.snapshot(head);propose(head,shadow,opt);native=p.snapshot(shadow);sync();common.append(time.perf_counter()-t)
            mh=moments_hash(opt)
            if audit:audit.step=step
            sync();t=time.perf_counter();executed,r=actuator.project(before,native,K,ids[:n],endpoint.evaluate)
            head.load_state_dict(executed);actual=endpoint.evaluate(p.snapshot(head),r['selected_controls'],'commit')
            sync();control.append(time.perf_counter()-t)
            assert np.all(actual['C']>=K) and moments_hash(opt)==mh and old.opt_counter(opt)==1890+step
            row=dict(step=step,head_sha256=head_hash(head),moments_sha256=mh,report=r,floor_violations=0,
                historical_head_identical=head_hash(head)==baseline[step-1]['head_sha256'],historical_moments_identical=mh==baseline[step-1]['moments_sha256'])
            rows.append(row);stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush()
            if step==1 or step%30==0:print('B',mode,step,'solves',r['outer_solves'],'pivots',r['solver_iterations'],'skip',r['skip'],flush=True)
    if auditstream:auditstream.close()
    if mode=='validate':save(dest/'CHECKPOINT.pt',dict(head=p.cpu_state(p.snapshot(head)),optimizer=opt.state_dict()))
    else:
        with gzip.open(ROOT/'validate/STEPS.jsonl.gz','rt') as f:validated=[json.loads(s) for s in f]
        assert rows==validated,'warm actuator replay differs from validation'
    ev=final_eval(head,d);write(dest/'EVALUATION.json',ev)
    summary=dict(status='PASS',mode=mode,steps=210,floor_violations=0,setup_wall=setup,
        control_wall=sum(control),total_wall=setup+sum(common)+sum(control),common_wall=sum(common),profile_components=dict(meter.seconds),
        control_includes_cold_audit=mode=='validate',endpoint_calls=meter.calls,exact_old_rows=endpoint.old_rows,
        gemm_calls=meter.gemms,padded_rows=meter.padded,solver_iterations=sum(x['report']['solver_iterations'] for x in rows),
        outer_solves=sum(x['report']['outer_solves'] for x in rows),linear_solves=sum(x['report']['linear_solves'] for x in rows),
        skips=sum(x['report']['skip'] for x in rows),mean_active_constraints=float(np.mean([len(x['report']['final_active']) for x in rows])),
        final_active_constraints=len(rows[-1]['report']['final_active']),
        mean_inherited_pairs=float(np.mean([x['report']['inherited_pairs'] for x in rows][1:])),
        mean_filtered_previous=float(np.mean([x['report']['filtered_previous_pairs'] for x in rows][1:])),
        gram_cache=dict(unique_constraints=len(actuator.cache.pairs),computed_nonzero_upper_entries=actuator.cache.computed,
                        reused_entries=actuator.cache.reused,zero_entries=actuator.cache.zero_entries,requests=actuator.cache.requests),
        head_matches=sum(x['historical_head_identical'] for x in rows),moments_matches=sum(x['historical_moments_identical'] for x in rows),
        metrics=ev['metrics'],baseline_prediction_differences=ev['baseline_prediction_differences'])
    if audit:
        summary['warm_cold']=dict(qps=len(audit.rows),warm_iterations=sum(x['warm_iterations'] for x in audit.rows),
            cold_iterations=sum(x['cold_iterations'] for x in audit.rows),agree=sum(x['solutions_agree'] for x in audit.rows),
            warm_fewer=sum(x['warm_iterations']<x['cold_iterations'] for x in audit.rows),
            warm_more=sum(x['warm_iterations']>x['cold_iterations'] for x in audit.rows))
    write(dest/'CHECKS.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='metrics'}),flush=True);guard()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['validate','timed','profile']);a=ap.parse_args()
    guard();p.setup();assert read(ROOT/'CONTINUITY.json')['decision']=='PROCEED'
    assert read(ROOT/'SOLVER_PREFLIGHT.json')['status']=='PASS'
    if a.mode=='validate':
        write(ROOT/'PREFLIGHT.json',dict(base=BASE,inputs=inputs(),runtime=runtime(),
            sources={q.name:sha(q) for q in HERE.glob('*.py')},protocol_sha256=sha(HERE/'PROTOCOL.md')))
    else:assert read(ROOT/'validate/CHECKS.json')['status']=='PASS'
    run(load(),a.mode);inputs();guard()


if __name__=='__main__':
    try:main()
    except Exception:
        if ROOT.exists():write(ROOT/f'FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
