"""Independent replay + standalone same-trajectory timing, no observer feedback."""
from ..run import load, guard
from ..core import *
from contextlib import nullcontext
import cProfile
import pstats


def records(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(line) for line in f]


class NoTiming:
    def region(self, name):return nullcontext()


def replay(kind, data, rows, profiling=False):
    d,F,initial,h,y,ids,n,K=data
    head,shadow,opt=make_state(initial)
    s=None
    sync();start=time.perf_counter()
    if kind=='v3':
        s=Screen(h,y,n,72,80,K);s.timings=NoTiming()
    elif kind=='v2':
        from experiments.gfg_final_architecture_v2_safe_screening_v2.core import Screen as V2Screen
        s=V2Screen(h,y,n,72,80);s.begin(p.snapshot(head),1)
    sync();setup=time.perf_counter()-start
    common_times=[];control_times=[];evidence=[]
    profiler=cProfile.Profile() if profiling else None
    for step,current in enumerate(order(d,10),1):
        # Profile first 30 updates only, separate from any speed measurement.
        if profiling and step>30:break
        sync();start=time.perf_counter()
        _,loss=ce_grad(head,F[current],torch.from_numpy(d['y'][current]).cuda(),80)
        nn.utils.clip_grad_norm_(head.parameters(),1.)
        before=p.snapshot(head);propose(head,shadow,opt);native=p.snapshot(shadow)
        sync();common_times.append(time.perf_counter()-start)
        mh=moments_hash(opt)
        sync();start=time.perf_counter()
        if profiler:profiler.enable()
        if kind=='full':
            executed,_,report=reference_projection(before,native,h,y,n,K,ids[:n],seen=80)
            head.load_state_dict(executed)
            final=margins(fixed_affine(executed,h,80),y)
            C=counts(final[:n].cpu().numpy(),y[:n].cpu().numpy(),72)
        else:
            if kind=='v2' and step>1:s.begin(before,step)
            executed,report=screened_projection(before,native,s,K,ids[:n],lambda state,sel,label:s.evaluate(state,sel))
            head.load_state_dict(executed)
            actual=s.evaluate(p.snapshot(head),set(report['selected_controls']))
            C=actual['C']
            if kind=='v2':s.finish()
        if profiler:profiler.disable()
        sync();control_times.append(time.perf_counter()-start)
        assert np.all(C>=K)
        assert head_hash(head)==rows[step-1]['head_sha256'],(kind,step,'head')
        assert moments_hash(opt)==mh==rows[step-1]['moments_sha256']
        assert old.opt_counter(opt)==1890+step
        if kind=='full':compare_reports(rows[step-1]['report'],report,K)
        else:
            for key in ('skip','sweep_count','final_active','final_duals','selected_controls','roundoff_budget_sha256'):
                assert report[key]==rows[step-1]['report'][key]
        evidence.append(dict(step=step,head=head_hash(head),moments=mh,head_moments_decisions_match=True))
        if not profiling and step%60==0:print('STANDALONE',kind,step,flush=True)
    if profiling:
        stats=pstats.Stats(profiler)
        values=[]
        for (file,line,name),(cc,nc,tt,ct,callers) in stats.stats.items():
            values.append(dict(file=file,line=line,name=name,primitive_calls=cc,total_calls=nc,self_seconds=tt,cumulative_seconds=ct))
        return dict(profiled_steps=len(evidence),kind=kind,profiling_overhead_not_used_for_speed=True,
                    top_self=sorted(values,key=lambda v:-v['self_seconds'])[:35],
                    top_cumulative=sorted(values,key=lambda v:-v['cumulative_seconds'])[:35])
    final=torch.load(ROOT/'CHECKPOINT.pt',weights_only=False,map_location='cuda')
    assert equal(p.snapshot(head),final['head']) and equal(opt.state_dict(),final['optimizer'])
    result=dict(kind=kind,steps=210,setup_wall=setup,common_update_wall=sum(common_times),
                control_wall=sum(control_times),algorithm_wall=setup+sum(common_times)+sum(control_times),
                per_step_control_wall=control_times,verification=evidence)
    if kind=='v3':result.update(exact_old_rows=s.total_exact,padded_rows=s.total_padded,gemm_calls=s.total_gemms,
                               additional_refresh_forward_rows=0,cache_refresh_rows=s.total_refresh)
    return result


def main():
    p.setup();guard()
    assert read(ROOT/'RESULTS.json')['gate']=='PASS'
    pre=read(ROOT/'PREFLIGHT.json')
    assert all(sha(REPO/q)==v for q,v in pre['sources'].items())
    assert all(sha(q)==v for q,v in pre['inputs'].items())
    rows=records(ROOT/'STEPS.jsonl.gz');calls=records(ROOT/'CALLS.jsonl.gz')
    assert len(rows)==210 and all(not c['false_safe'] and c['deficits_identical'] and c['choice_identical'] and c['lower_count_valid'] and c['max_margin_difference']==0. for c in calls)
    data=load()
    # No controller observers, diagnostic hashes outside timing, or CUDA events.
    # Same explicit commit/recheck contract. Fixed order disclosed, no best-of.
    outcomes={}
    for kind in ('full','v3','v2'):
        outcomes[kind]=replay(kind,data,rows)
        write(ROOT/f'STANDALONE_{kind}.json',outcomes[kind])
        print(kind,outcomes[kind]['algorithm_wall'],flush=True)
    bench={k:{a:b for a,b in v.items() if a not in ('verification','per_step_control_wall')} for k,v in outcomes.items()}
    write(ROOT/'BENCHMARK.json',dict(order=['full','v3','v2'],single_measurement_not_best_of=True,methods=bench,
        v3_speedup_vs_full=bench['full']['algorithm_wall']/bench['v3']['algorithm_wall'],
        v3_speedup_vs_v2=bench['v2']['algorithm_wall']/bench['v3']['algorithm_wall'],
        note='Separate deterministic replays, matched state trajectory, all hashes checked outside timing. No profiling events or full observer in v3 runtime. One measurement per path; no confidence interval.'))
    profile=replay('v3',data,rows,profiling=True)
    write(ROOT/'PROFILE.json',profile)
    final=torch.load(ROOT/'CHECKPOINT.pt',weights_only=False,map_location='cuda')
    test=torch.load(INPUT/'TEST_FEATURES.pt',weights_only=False,map_location='cpu')
    ev=read(ROOT/'EVALUATION.json');d=data[0]
    with torch.no_grad():pred=fixed_affine(final['head'],test['features'].cuda(),80).argmax(1).cpu().tolist()
    assert pred==ev['predictions'] and output_metrics(d,pred,10)==ev['metrics']
    assert all(sha(REPO/q)==v for q,v in pre['sources'].items()) and all(sha(q)==v for q,v in pre['inputs'].items());guard()
    checks=dict(status='PASS',candidate_observer_calls=len(calls),false_safe=0,floor_violations=0,
                independently_replayed_full_reference_steps=210,standalone_v2_steps=210,standalone_v3_steps=210,
                all_head_and_optimizer_moments_identical=True,deficits_and_control_decisions_identical=True,
                final_test_recomputed=True,old_v2_test_predictions_identical=True,
                cached_count_semantics='safe relations lower bounds only; full counts independently observed',
                inputs_and_frozen_sources_unchanged=True,old_experiments_unchanged=True)
    write(ROOT/'CHECKS.json',checks)
    print(json.dumps(dict(checks=checks,benchmark=bench),indent=2),flush=True)


if __name__=='__main__':main()
