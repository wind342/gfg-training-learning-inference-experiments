"""Read-only record verification for the fixed PARTIAL ablation."""
from collections import Counter
from fractions import Fraction
import json
import numpy as np
from experiments.gfg_functional_feedback_router_v3 import check as v3check
from experiments.gfg_router_v3_three_task_continuation.check import longest
from .run import HERE,ROOT,BASE,admit,prior_gate
from .policy import Router

class AuditRouter(Router):
    def __init__(self):super().__init__();self.completed=set()
    def choose(self,step,labels,states,levels,old_groups,memory):
        states=list(states)
        if set(old_groups)!=self.completed:
            states=['PARTIAL' if g in old_groups and s=='FORMING' else s for g,s in zip(labels,states)]
            self.completed=set(old_groups)
        return super().choose(step,labels,states,levels,old_groups,memory)

def stats(rows,router):
    old=range((rows[0]['task']-1)*8);info=[r['info'] for r in rows];groups={}
    for g in old:
        served=[g in i['selected_feedback_groups'] for i in info]
        risk=[g in i.get('active_degraded_groups',[])+i.get('active_at_risk_groups',[]) for i in info]
        partial=[g in i.get('active_partial_groups',[]) for i in info]
        groups[g]=dict(replay_occurrences=2*sum(served),partial_replay_occurrences=2*sum(g in i.get('selected_partial_groups',[]) for i in info),
            anchor_replay_occurrences=2*sum(g in i['selected_anchor_groups'] for i in info),
            risk_steps=sum(risk),partial_steps=sum(partial),
            max_consecutive_no_feedback=longest([not s for s in served]),
            max_consecutive_partial_unserved=longest([p and not s for p,s in zip(partial,served)]),
            risk_unserved_steps=sum(p and not s for p,s in zip(risk,served)))
    for row in rows:
        i=row['info'];n=len(i['feedback_example_ids']);nc=len(row['new_ids'])
        if router:
            h=i['allocation_receiving_history'];state=dict(zip(h['labels'],h['states']));lev=dict(zip(h['labels'],h['levels']))
            assert all(state[g]!='FORMING' for g in old)
            partial={g for g in old if state[g]=='PARTIAL'};risk={g for g in old if state[g]=='DEGRADED' or (state[g]=='PROTECTED' and lev[g]==1)}
            selected=set(i['selected_feedback_groups']);sp=set(i['selected_partial_groups']);sa=set(i['selected_anchor_groups'])
            assert partial==set(i['active_partial_groups']) and sp<=partial
            assert len(sp)+len(sa)<=1 and selected-sp-sa<=risk
            assert all(state[g]=='PROTECTED' and lev[g]==0 for g in sa)
            assert not sp or row['step']%2==0
            assert i['old_loss_weight']==n/(nc+n) and i['current_loss_weight']==nc/(nc+n)
            assert i['weighted_training_loss']==(nc*i['current_loss']+n*(i['feedback_loss'] or 0.))/(nc+n)
    return dict(old_exposures=sum(i['feedback_batch_size'] for i in info),active_fraction=sum(bool(i['feedback_batch_size']) for i in info)/len(info),
        modes=dict(Counter(i['router_mode'] for i in info)),releases=sum(i['release_event'] for i in info),
        per_relation=groups,batch_histogram=dict(Counter(i['feedback_batch_size'] for i in info)),
        old_weight_histogram=dict(Counter(str(Fraction(i['feedback_batch_size'],16+i['feedback_batch_size'])) for i in info)),
        full_calls=sum(i['instrument_called'] for i in info),emergency_shrink=sum(i['gate_reason']=='EMERGENCY_SHRINK' for i in info),
        gates=dict(Counter(i['gate_reason'] for i in info)),alphas=dict(Counter(i['alpha'] for i in info)),
        update_seconds=sum(i['seconds'] for i in info))

def lifecycle(rows,result,initial):
    previous=initial;transitions=result['partial_boundary_transitions'];assert len(transitions)==2
    for row in rows:
        i=row['info'];n=row['task']*8;old=n-8
        states=list(previous['states']);levels=list(previous['levels'])
        if row['offset']==1:
            converted=[g for g,s in enumerate(states) if s=='FORMING']
            states=['PARTIAL' if s=='FORMING' else s for s in states]+['FORMING']*8;levels+=[0]*8
            t=transitions[row['task']-2]
            assert t['converted']==converted and t['states']==states and t['completed']==list(range(old))
        h=i['allocation_receiving_history']
        assert h['states']==states and h['levels']==levels and h['labels']==list(range(n))
        previous=i['history']

def main():
    admission=admit();base=v3check.base;base.ROOT=ROOT;base.HERE=HERE;base.CORE=HERE;base.Router=AuditRouter
    prior=json.loads((prior_gate.ROOT/'CHECKS.json').read_text());arms={};orders=None;initial_prediction=None
    initial=json.loads((base.DATA/'3tasks-governor/attempt-1/RESULT.json').read_text())['metrics'][0]['control_state']
    for arm in ('native','replay','router'):
        summary,p=base.verify(arm,'three')
        c=json.loads((ROOT/f'three-{arm}/CONFIG.json').read_text());assert c['v4_base']==BASE and c['admission']==admission
        for field in ('imported_sources','inherited_sources','router_v2_sources'):
            for rel,h in c[field].items():assert base.sha(HERE.parent/rel)==h,rel
        assert base.sha(HERE/'policy.py')==c['policy_sha256']
        rows=list(map(json.loads,(ROOT/f'three-{arm}/STEPS.jsonl').read_text().splitlines()))
        result=json.loads((ROOT/f'three-{arm}/RESULT.json').read_text())
        if orders is None:orders=[r['new_ids'] for r in rows];initial_prediction=p
        assert orders==[r['new_ids'] for r in rows] and p==initial_prediction
        if arm=='router':lifecycle(rows,result,initial)
        tasks={}
        for task in (2,3):
            rr=[r for r in rows if r['task']==task];assert len(rr)==210 and len(set(sum([r['new_ids'] for r in rr],[])))==3360
            t=stats(rr,arm=='router');end=[m for m in result['metrics'] if m['task']==task and m['offset']==210][0]
            start=[m for m in result['metrics'] if m['task']==task and m['offset']==0][0]
            t.update(end={k:end[k] for k in ('old','new','accuracy','per_relation')},start={k:start[k] for k in ('old','new','accuracy','per_relation')},
                cohorts={f'task{j+1}':float(np.mean([end['per_relation'][str(k)] for k in range(j*8,j*8+8)])) for j in range(task)})
            tasks[task]=t
        summary['tasks']=tasks;summary['partial_boundary_transitions']=result['partial_boundary_transitions']
        if arm!='router':
            old=json.loads((prior_gate.ROOT/f'three-{arm}/RESULT.json').read_text())
            summary['baseline_repetition_prediction_identity']=[a['predictions']==b['predictions'] for a,b in zip(result['metrics'],old['metrics'])]
        arms[arm]=summary
    output=dict(status='PASS',base_commit=BASE,admission=admission,arms=arms,historical_v3=prior['arms']['router'],
        checker_boundary='source/data/checkpoint/record hashes; recorded-history routing and lifecycle replay; actual prediction metrics and dose verification; no independent native rerun',
        scope='single fixed three-arm three-task experiment; no parameter sweep or strategy changes after running',
        inherited_duplicate='original control/test duplicate at test7561 retained; sensitivity in metrics; old memory excludes encoded control/test duplicates')
    base.write(ROOT/'CHECKS.json',output);(HERE/'results').mkdir(exist_ok=True);base.write(HERE/'results/CHECKS.json',output)
    print(json.dumps({a:{t:{k:s[k] for k in ('end','old_exposures','active_fraction','modes','batch_histogram')} for t,s in v['tasks'].items()} for a,v in arms.items()},indent=2))

if __name__=='__main__':main()
