"""Run is the outer unit. No target-level pseudo-replication or refitting."""
from collections import Counter
from .common import *

def summarize(root=OUTPUT):
    rows=read(HERE / 'RUN_ASSIGNMENT.json')['runs']
    results=[]
    curves={b:[] for b in BRANCHES}
    total_alphas=Counter()
    for row in rows:
        runs={}
        for b in BRANCHES:
            br=root / row['entry_id'] / b
            require(read(br / 'COMPLETE.json')['horizon']==HORIZON,'INCOMPLETE_BRANCH')
            evals=[read(br / 'evaluations' / f'{t:04d}.json') for t in range(0,HORIZON+1,CADENCE)]
            acc=np.array([e['accuracy'] for e in evals])
            correct=np.array([e['correct'] for e in evals],dtype=bool)
            times=np.arange(0,HORIZON+1,CADENCE)
            auc=float(np.trapezoid(acc,times)/HORIZON)
            previous=correct[:-1]
            following=correct[1:]
            # A return after a prior observed correct state is a recovery;
            # any wrong->correct interval is also reported, without inventing
            # a group-level formation threshold or claiming an exact crossing time.
            ever_correct=np.maximum.accumulate(correct,axis=0)
            recoveries=int(((~previous)&following&ever_correct[:-1]).sum())
            first_acquisitions=int(((~ever_correct[:-1])&following).sum())
            selections=[read(br / 'selections' / f'{t:04d}.json')['selected_alpha'] for t in range(1,HORIZON+1)]
            counts=Counter(map(str,selections))
            if b=='controller': total_alphas.update(counts)
            steps=[read(br / 'steps' / f'{t:04d}.json') for t in range(1,HORIZON+1)]
            runs[b]={'normalized_auc':auc,'final_accuracy':float(acc[-1]),
                'initial_accuracy':float(acc[0]),'declines':int((previous&(~following)).sum()),
                'wrong_to_correct':int(((~previous)&following).sum()),'recoveries_after_prior_correct':recoveries,
                'first_observed_acquisitions':first_acquisitions,
                'final_mean_margin':evals[-1]['mean_margin'],'final_median_margin':evals[-1]['median_margin'],
                'alpha_counts':dict(sorted(counts.items())),
                'alpha_changes':sum(a!=c for a,c in zip(selections,selections[1:])),
                'mean_training_loss':float(np.mean([s['proposal']['loss'] for s in steps])),
                'mean_proposed_norm':float(np.mean([s['proposal']['proposed_norm'] for s in steps])),
                'mean_applied_norm':float(np.mean([s['application']['applied_norm'] for s in steps])),
                'timing_seconds':read(br / 'COMPLETE.json')['timing_seconds']}
            curves[b].append({'accuracy':acc.tolist(),'mean_margin':[e['mean_margin'] for e in evals]})
        comparisons={b:{'auc_delta':runs['controller']['normalized_auc']-runs[b]['normalized_auc'],
                         'final_delta':runs['controller']['final_accuracy']-runs[b]['final_accuracy']} for b in ('native','fixed')}
        results.append({'entry_id':row['entry_id'],'source_step':row['source_step'],'branches':runs,'comparisons':comparisons})
    comparisons={}
    for b in ('native','fixed'):
        a=np.array([r['comparisons'][b]['auc_delta'] for r in results])
        f=np.array([r['comparisons'][b]['final_delta'] for r in results])
        comparisons[b]={'mean_auc_delta':float(a.mean()),'auc_wins':int((a>0).sum()),'auc_ties':int((a==0).sum()),'auc_losses':int((a<0).sum()),
            'mean_final_delta':float(f.mean()),'final_wins':int((f>0).sum()),'final_ties':int((f==0).sum()),'final_losses':int((f<0).sum()),
            'primary_pass':bool(a.mean()>=0.01 and (a>0).sum()>=9)}
    supported=all(v['primary_pass'] for v in comparisons.values())
    totals={b:{k:float(sum(r['branches'][b][k] for r in results)) for k in ('declines','recoveries_after_prior_correct','wrong_to_correct','first_observed_acquisitions')}
            for b in BRANCHES}
    timings={b:{k:sum(r['branches'][b]['timing_seconds'][k] for r in results)
                   for k in results[0]['branches'][b]['timing_seconds']} for b in BRANCHES}
    return {'schema':SCHEMA+'-results','protocol_sha256':sha(HERE / 'PROTOCOL_FREEZE.md'),
        'verdict':'ONLINE_FUNCTIONAL_CONTROLLER_SUPPORTED' if supported else 'ONLINE_FUNCTIONAL_CONTROLLER_NOT_SUPPORTED',
        'run_count':12,'horizon':HORIZON,'cadence':CADENCE,'outer_unit':'run',
        'comparisons':comparisons,'per_run':results,'controller_alpha_counts':dict(sorted(total_alphas.items())),
        'pooled_descriptive_curves':{b:{k:np.mean([v[k] for v in curves[b]],axis=0).tolist() for k in ('accuracy','mean_margin')} for b in BRANCHES},
        'evaluation_times':list(range(0,HORIZON+1,CADENCE)), 'transition_totals_descriptive':totals,'timings_seconds':timings,
        'controller_training_plus_selection_cost_ratio_vs_native':(timings['controller']['training']+timings['controller']['selection'])/timings['native']['training'],
        'controller_wall_cost_ratio_vs_native':timings['controller']['wall_total']/timings['native']['wall_total'],
        'boundary':'Same control/evaluation targets; no held-out-generalization, generic optimizer, or wall-budget-efficiency claim.'}

def main():
    verify_seal()
    result=summarize()
    write_new(HERE / 'RESULTS.json',result)
    lines=['# Online functional amplitude controller','',result['verdict'],'',
        'Frozen primary: run-mean normalized AUC improvement >= 1 pp and >= 9/12 winning runs against BOTH controls.','',
        '| Comparator | Mean AUC delta (pp) | AUC W/T/L | Mean final delta (pp) | Final W/T/L |','|---|---:|---|---:|---|']
    for name,c in result['comparisons'].items():
        lines.append(f"| {name} | {100*c['mean_auc_delta']:.6f} | {c['auc_wins']}/{c['auc_ties']}/{c['auc_losses']} | {100*c['mean_final_delta']:.6f} | {c['final_wins']}/{c['final_ties']}/{c['final_losses']} |")
    lines+=['','## Per-run comparisons','','| Run | Source step | AUC vs native (pp) | AUC vs fixed (pp) | Final vs native (pp) | Final vs fixed (pp) |','|---|---:|---:|---:|---:|---:|']
    for r in result['per_run']:
        n,f=r['comparisons']['native'],r['comparisons']['fixed']
        lines.append(f"| {r['entry_id']} | {r['source_step']} | {100*n['auc_delta']:.6f} | {100*f['auc_delta']:.6f} | {100*n['final_delta']:.6f} | {100*f['final_delta']:.6f} |")
    lines+=['','Alpha counts: '+str(result['controller_alpha_counts']),
        '',f"Measured training+selection cost ratio vs native: {result['controller_training_plus_selection_cost_ratio_vs_native']:.3f}x.",
        f"Measured wall-time ratio including evidence: {result['controller_wall_cost_ratio_vs_native']:.3f}x.",
        '', 'Transitions are descriptive counts at the 10-step evaluation grid, not unobserved exact crossing times.',
        'No new capability-formation threshold was introduced. These selected checkpoints need not begin before formation.',
        result['boundary'],'']
    with (HERE / 'RESULTS.md').open('x',encoding='utf-8',newline='\n') as f:
        f.write('\n'.join(lines))
    print(result['verdict'],result['comparisons'],flush=True)

if __name__=='__main__': main()
