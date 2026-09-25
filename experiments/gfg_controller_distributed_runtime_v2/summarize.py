"""Post-outcome descriptive aggregation; never selects rates, seeds or policies."""
from collections import Counter, defaultdict
import statistics
from .evidence import OUT, read, write


def mean(xs):return statistics.mean(xs) if xs else None
def median(xs):return statistics.median(xs) if xs else None


def first_hit(r):
    return next((v for v in r.get('rows',[]) if v['accuracy']>=.99),None)


def aggregate(rows):
    hits=[first_hit(r) for r in rows]
    updates=[(r,v) for r in rows for v in r.get('rows',[])]
    alpha=Counter(str(v['info']['alpha']) for r,v in updates)
    decision=[v['update_seconds'] for r,v in updates if r['arm']=='dynamic' and (v['t']-1)%r['cadence']==0]
    ordinary=[v['update_seconds'] for r,v in updates if r['arm']!='dynamic' or (v['t']-1)%r['cadence']!=0]
    return {'runs':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),
        'reach99':sum(h is not None for h in hits),
        'steps_each':[h['t'] if h else None for h in hits],
        'restricted_steps_mean':mean([h['t'] if h else 201 for h in hits]),
        'tokens_to99_each':[h['t']*128*3 if h else None for h in hits],
        'wall_to99_each':[h['wall'] if h else None for h in hits],
        'wall_to99_mean_successful_only':mean([h['wall'] for h in hits if h]),
        'observed_wall_mean':mean([r.get('wall') for r in rows if r.get('wall') is not None]),
        'median_update_seconds_decision':median(decision),
        'median_update_seconds_nondecision':median(ordinary),
        'alpha_histogram':dict(alpha),
        'intervention_rate':mean([v['info']['alpha']<1 for r,v in updates]),
        'peak_allocated':max([r.get('peak_allocated',0) for r in rows],default=0),
        'peak_reserved':max([r.get('peak_reserved',0) for r in rows],default=0)}


def main():
    jobs=[read(p) for p in sorted((OUT/'scale').rglob('RESULT.json'))]
    groups=defaultdict(list)
    for r in jobs:
        if 'cadence' not in r:continue
        groups[(r['scale'],r['phase'],r['arm'],r['lr'],r['cadence'],r['probes'],r['subset_seed'])].append(r)
    rows=[]
    for k,rs in groups.items():
        rs.sort(key=lambda r:r['seed'])
        rows.append(dict(zip(('scale','phase','arm','lr','cadence','probes','subset_seed'),k),
                         seeds=[r['seed'] for r in rs],**aggregate(rs)))
    pairs=[]
    for scale in ('70','160'):
        for phase in ('formal-safe','formal-near','formal-severe'):
            arms={a:{r['seed']:r for r in jobs if r.get('scale')==scale and r.get('phase')==phase and r.get('arm')==a} for a in ('native','fixed','dynamic')}
            for against in ('native','fixed'):
                outcomes=[]
                for seed in sorted(set(arms['dynamic'])&set(arms[against])):
                    d,b=first_hit(arms['dynamic'][seed]),first_hit(arms[against][seed])
                    ds,bs=(d['t'] if d else 201),(b['t'] if b else 201)
                    outcomes.append({'seed':seed,'step_verdict':'win' if ds<bs else 'loss' if ds>bs else 'tie',
                        'dynamic_steps':d['t'] if d else None,'baseline_steps':b['t'] if b else None,
                        'wall_difference_when_both_reach':d['wall']-b['wall'] if d and b else None})
                if outcomes:pairs.append({'scale':scale,'phase':phase,'dynamic_against':against,
                    'step_wins_losses_ties':dict(Counter(x['step_verdict'] for x in outcomes)), 'pairs':outcomes})
    write(OUT/'CAPABILITY_SUMMARY.json',{'groups':rows,'paired_comparisons':pairs,
        'restricted_failure_value':201,'threshold':.99,'tokens_per_update':384,
        'wall_note':'Successful-only wall means must not be compared without reach counts. RESOURCE_LIMITED is censored; not a completed-200 failure.',
        'timing_note':'Decision/nondecision medians are descriptive, not a causal isolated overhead estimate.',
        'generalization_note':'317 controller-disjoint evaluation queries are included in the training distribution.'})


if __name__=='__main__':main()
