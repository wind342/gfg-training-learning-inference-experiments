"""Recompute all three arms and target coverage without new native execution."""
from collections import Counter
from fractions import Fraction
import json
import numpy as np
from experiments.gfg_functional_feedback_router_v3 import check as v3check
from .gate import HERE,ROOT,BASE,audit_smoke,normalize

def longest(flags):
    best=current=0
    for v in flags:
        current=current+1 if v else 0;best=max(best,current)
    return best

def coverage(rows,router):
    old=range((rows[0]['task']-1)*8);stats={};weights=Counter();batches=Counter();risk_counts=Counter()
    for row in rows:
        i=row['info'];n=len(i['feedback_example_ids']);nc=len(row['new_ids'])
        weights[str(Fraction(n,nc+n))]+=1;batches[n]+=1
        if router:
            h=i['allocation_receiving_history'];selected=set(i['selected_feedback_groups']);anchors=set(i['selected_anchor_groups'])
            degraded={g for g,s in zip(h['labels'],h['states']) if g in old and s=='DEGRADED'}
            risk={g for g,s,l in zip(h['labels'],h['states'],h['levels']) if g in old and s=='PROTECTED' and l==1}
            stable={g for g,s,l in zip(h['labels'],h['states'],h['levels']) if g in old and s=='PROTECTED' and l==0}
            assert degraded==set(i['active_degraded_groups']) and risk==set(i['active_at_risk_groups'])
            assert anchors<=stable and len(anchors)<=1 and selected-anchors<=degraded|risk
            assert i['old_loss_weight']==n/(nc+n) and i['current_loss_weight']==nc/(nc+n)
            assert i['weighted_training_loss']==(nc*i['current_loss']+n*(i['feedback_loss'] or 0.))/(nc+n)
            risk_counts[len(degraded|risk)]+=1
    for g in old:
        selected=[g in r['info']['selected_feedback_groups'] for r in rows]
        risk=[g in r['info'].get('active_degraded_groups',[])+r['info'].get('active_at_risk_groups',[]) for r in rows]
        stat=dict(replay_occurrences=2*sum(selected),selected_steps=sum(selected),
            max_consecutive_no_feedback=longest([not v for v in selected]))
        if router:
            forming=[r['info']['allocation_receiving_history']['states'][g]=='FORMING' for r in rows]
            stat.update(risk_steps=sum(risk),risk_unserved_steps=sum(a and not b for a,b in zip(risk,selected)),
                max_consecutive_risk_unserved=longest([a and not b for a,b in zip(risk,selected)]),
                forming_steps=sum(forming),max_consecutive_forming_unserved=longest([a and not b for a,b in zip(forming,selected)]),
                anchor_steps=sum(g in r['info']['selected_anchor_groups'] for r in rows))
        stats[g]=stat
    info=[r['info'] for r in rows]
    return dict(old_exposures=sum(i['feedback_batch_size'] for i in info),
        feedback_active_fraction=sum(bool(i['feedback_batch_size']) for i in info)/len(rows),
        modes=dict(Counter(i['router_mode'] for i in info)),releases=sum(i['release_event'] for i in info),
        activations=sum(i['router_activation'] for i in info),per_relation=stats,
        old_batch_histogram=dict(batches),old_loss_weight_histogram=dict(weights),
        observed_risk_count_histogram=dict(risk_counts),
        full_predictor_calls=sum(i['instrument_called'] for i in info),
        emergency_shrink_calls=sum(i['gate_reason']=='EMERGENCY_SHRINK' for i in info),
        alpha_below_one=sum(i['alpha']<1 for i in info),gate_counts=dict(Counter(i['gate_reason'] for i in info)),
        alpha_counts=dict(Counter(i['alpha'] for i in info)),
        summed_update_seconds=sum(i['seconds'] for i in info))

def main():
    admission=audit_smoke();base=v3check.base;base.ROOT=ROOT;base.HERE=HERE;base.CORE=HERE;base.Router=v3check.Router
    assert json.loads((ROOT/'SMOKE_CHECKS.json').read_text())==admission
    arms={};initial=None;orders=None
    for arm in ('native','replay','router'):
        summary,p=base.verify(arm,'three')
        c=json.loads((ROOT/f'three-{arm}/CONFIG.json').read_text())
        assert c['continuation_base']==BASE and c['admission']==admission and c['seed']==2021 and c['budget_updates']==420
        for field in ('imported_sources','inherited_sources','router_v2_sources'):
            for rel,h in c[field].items():assert base.sha(HERE.parent/rel)==h,rel
        result=json.loads((ROOT/f'three-{arm}/RESULT.json').read_text())
        rows=list(map(json.loads,(ROOT/f'three-{arm}/STEPS.jsonl').read_text().splitlines()))
        if initial is None:initial=p;orders=[r['new_ids'] for r in rows]
        assert p==initial and [r['new_ids'] for r in rows]==orders
        assert all(len(r['new_ids'])==16 for r in rows)
        summary['tasks']={}
        for task in (2,3):
            selected=[r for r in rows if r['task']==task]
            assert len(selected)==210 and len(set(sum([r['new_ids'] for r in selected],[])))==3360
            v=coverage(selected,arm=='router')
            endpoints=[m for m in result['metrics'] if m['task']==task]
            assert [m['offset'] for m in endpoints]==[0,210]
            v['start_accuracy']={k:endpoints[0][k] for k in ('old','new','accuracy','per_relation')}
            v['end_accuracy']={k:endpoints[1][k] for k in ('old','new','accuracy','per_relation')}
            per=endpoints[1]['per_relation']
            v['end_cohort_accuracy']={f'task{j+1}':float(np.mean([per[str(k)] for k in range(j*8,(j+1)*8)])) for j in range(task)}
            summary['tasks'][task]=v
        if arm=='router':
            assert len(result['task_boundary_audit'])==2
            boundary=result['task_boundary_audit'][1]
            previous=rows[209]['info']['history']
            for dst,src in [('state','states'),('level','levels'),('anchor','anchor_q10'),('capabilities','capabilities'),('q10','q10')]:
                assert boundary['preserved'][dst]==previous[src]
            assert boundary['old_labels']==list(range(16)) and boundary['new_labels']==list(range(24))
            assert boundary['last_step']==420 and rows[210]['step']==421
            # Match the first 60 scientific outcomes to the existing smoke, excluding timing.
            smoke=list(map(json.loads,(v3check.ROOT/'smoke-router/STEPS.jsonl').read_text().splitlines()))
            for new,old in zip(rows[:60],smoke):
                for key in ('new_ids','probe_ids','memory_identity','memory_groups','cursor_before'):assert new[key]==old[key]
                for key,val in old['info'].items():
                    if key!='seconds':assert new['info'][key]==val,(new['step'],key)
        summary['evaluation_seconds']=result['evaluation_seconds']
        arms[arm]=summary
    output=dict(status='PASS',base_commit=BASE,admission=admission,arms=arms,
        checker_boundary='hashes, role isolation, routing and cursors replayed from recorded histories; metrics recomputed from predictions; not an independent native rerun',
        scope='shared historical task1 checkpoint, 210 task2 and 210 task3 updates per arm, no policy tuning',
        risk_definition='pre-update observed router state; not privileged test truth; FORMING old groups may be ineligible')
    base.write(ROOT/'CHECKS.json',output);(HERE/'results').mkdir(exist_ok=True);base.write(HERE/'results/CHECKS.json',output)
    print(json.dumps({a:dict(wall=v['wall'],peak=v['peak_allocated'],tasks={t:{k:s[k] for k in ('end_accuracy','old_exposures','feedback_active_fraction','modes','old_loss_weight_histogram')} for t,s in v['tasks'].items()}) for a,v in arms.items()},indent=2))

if __name__=='__main__':main()
