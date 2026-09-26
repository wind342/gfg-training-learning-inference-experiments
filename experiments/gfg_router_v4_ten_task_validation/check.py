"""Independent record/identity checker; never trains or changes recorded outcomes."""
from collections import Counter
from fractions import Fraction
import json
import numpy as np
import torch
from .common import (HERE,ROOT,BASE,DATA,ARMS,sha,stamp,write,unchanged,scope,order,
    summarize_predictions,RandomReplay,trigger,CHECKPOINT,CKSHA,sources)
from experiments.gfg_governor_fewrel_feedback_router_v1.run import Memory,extend_memory,validate_batches
from experiments.gfg_router_v4_partial_capability.policy import Router
from experiments.gfg_functional_feedback_router_v1.policy import emergency_gate
from experiments.gfg_functional_update_governor_v1.instrument import Measurement
from experiments.gfg_functional_update_governor_v3.groups import GroupMap

def longest(values):
    high=run=0
    for value in values:run=run+1 if value else 0;high=max(high,run)
    return high

def statistics(rows,router):
    info=[r['info'] for r in rows];groups=sorted({g for r in rows for g in range((r['task']-1)*8)})
    per={}
    for g in groups:
        relevant=[r for r in rows if g<(r['task']-1)*8]
        selected=[g in r['info']['selected_feedback_groups'] for r in relevant]
        risk=[g in r['info'].get('active_degraded_groups',[])+r['info'].get('active_at_risk_groups',[]) for r in relevant]
        per[str(g)]=dict(replay_exposures=sum(sum(int(v)==g for v in r['info']['selected_feedback_groups'])*2 for r in relevant),
            selected_steps=sum(selected),eligible_steps=len(relevant),max_no_feedback_steps=longest([not v for v in selected]),
            risk_steps=sum(risk) if router else None,risk_unserved_steps=sum(a and not b for a,b in zip(risk,selected)) if router else None,
            max_risk_unserved_streak=longest([a and not b for a,b in zip(risk,selected)]) if router else None)
    counts=[len(i.get('active_degraded_groups',[]))+len(i.get('active_at_risk_groups',[])) for i in info]
    hits=total=changes=changehits=0;confusion=Counter();transitions=Counter()
    for i in info:
        if i['instrument_called']:
            p=np.array(i['predicted'])[(.125,.25,.5,1.).index(i['alpha'])]>0;a=np.array(i['actual'])>0;b=np.array(i['baseline'])>0
            total+=len(a);hits+=int((p==a).sum());mask=a!=b;changes+=int(mask.sum());changehits+=int(((p==a)&mask).sum())
            for actual,pred in zip(a,p):confusion[f'actual_{int(actual)}_pred_{int(pred)}']+=1
            for before,actual,pred in zip(b,a,p):transitions[f'{int(before)}->{int(actual)}_pred_{int(pred)}']+=1
    return dict(updates=len(rows),old_exposures=sum(i['feedback_batch_size'] for i in info),
        feedback_active_fraction=sum(bool(i['feedback_batch_size']) for i in info)/len(info),
        feedback_batch_histogram=dict(Counter(str(i['feedback_batch_size']) for i in info)),
        old_loss_weight_histogram=dict(Counter(str(Fraction(i['feedback_batch_size'],16+i['feedback_batch_size'])) for i in info)),
        per_relation=per,modes=dict(Counter(i['router_mode'] for i in info)),release_events=sum(i['release_event'] for i in info),
        active_risk_histogram=dict(Counter(str(n) for n in counts)) if router else None,
        active_risk_over_cap_steps=sum(n>8 for n in counts) if router else None,
        coalition_saturated_steps=sum(len(i['selected_feedback_groups'])==8 for i in info),
        risk_unserved_group_steps=sum(v['risk_unserved_steps'] for v in per.values()) if router else None,
        full_predictor_calls=sum(i['instrument_called'] for i in info),boundary_hits=hits,boundary_total=total,
        boundary_agreement=hits/total if total else None,actual_boundary_transitions=changes,transition_hits=changehits,
        transition_agreement=changehits/changes if changes else None,boundary_confusion=dict(confusion),transition_confusion=dict(transitions),
        emergency_shrink=sum(i['gate_reason']=='EMERGENCY_SHRINK' for i in info),
        actual_shrink=sum(i['alpha']<1 for i in info),alpha_histogram=dict(Counter(str(i['alpha']) for i in info)),
        gate_histogram=dict(Counter(i['gate_reason'] for i in info)),update_seconds=sum(i['update_seconds_inclusive'] for i in info))

def verify(arm,pre,data,schedule):
    root=ROOT/'seed-2021'/arm;c=json.loads((root/'CONFIG.json').read_text());r=json.loads((root/'RESULT.json').read_text())
    assert r['status']=='COMPLETED' and r['steps']==1890 and c['arm']==arm and c['seed']==2021
    assert c['base']==BASE and c['source_hashes']==pre['source_hashes']==sources()
    assert c['preflight_sha256']==sha(ROOT/'PREFLIGHT.json') and c['checkpoint_sha256']==sha(CHECKPOINT)==CKSHA
    assert c['protocol_sha256']==sha(HERE/'PROTOCOL.md') and c['data_sha256']==sha(DATA/'data.npz')
    assert r['log_sha256']==sha(root/'STEPS.jsonl') and r['config_sha256']==sha(root/'CONFIG.json')
    assert sha(root/'final.pt')==r['final_checkpoint']['sha256']
    if arm=='random':assert c['exposure_schedule_sha256']==sha(ROOT/'seed-2021'/'EXPOSURE_SCHEDULE.json')
    rows=[json.loads(line) for line in (root/'STEPS.jsonl').read_text().splitlines()];assert len(rows)==1890
    memory=Memory();router=Router();random=RandomReplay(2021);fixed_cursor=0;previous=None;taskrows={}
    initial=json.loads((DATA/'3tasks-governor/attempt-1/RESULT.json').read_text())['metrics'][0]['control_state']
    last=210
    for task in range(2,11):
        probes,primary,full=scope(data,task*8);old=set(range((task-1)*8));current=set(range((task-1)*8,task*8))
        extend_memory(memory,data,old,probes,full);batches=order(data,task);rr=[z for z in rows if z['task']==task];assert len(rr)==210
        taskrows[task]=rr
        for offset,row in enumerate(rr,1):
            i=row['info'];t=row['step'];assert t==last+1 and row['offset']==offset;last=t
            assert row['new_ids']==batches[offset-1].tolist() and row['probe_ids']==probes.tolist()
            assert row['memory_groups']=={str(g):list(v) for g,v in memory.groups.items()} and row['memory_identity']==memory.stamp
            assert row['cursor_before']=={str(g):v for g,v in memory.cursors.items()}
            n=i['feedback_batch_size'];ids=i['feedback_example_ids']
            assert n==len(ids)==2*len(i['selected_feedback_groups']) and 0<=n<=16
            validate_batches(row['new_ids'],ids,data,current,memory,probes,primary)
            assert i['old_loss_weight']==n/(16+n) and i['current_loss_weight']==16/(16+n)
            assert i['weighted_training_loss']==(16*i['current_loss']+n*(i['feedback_loss'] or 0.))/(16+n)
            assert i['update_seconds_inclusive']>=0
            if arm=='router':
                h=i['allocation_receiving_history'];assert h['labels']==list(range(task*8))
                prev=initial if previous is None else previous
                for key in ('states','capabilities','q10','anchor_q10','levels'):
                    expected=list(prev[key])
                    if offset==1:
                        if key=='states':expected=['PARTIAL' if s=='FORMING' else s for s in expected]+['FORMING']*8
                        else:expected += [0]*8
                    assert h[key]==expected,(task,offset,key)
                decision=router.choose(t,h['labels'],h['states'],h['levels'],old,memory)
                for key,value in decision.items():assert json.loads(json.dumps(value))==i[key],(t,key)
                assert not any(s=='FORMING' for s in h['states'][:(task-1)*8])
                if i['instrument_called']:
                    m=Measurement((.125,.25,.5,1.),torch.tensor(i['baseline'],dtype=torch.float64),torch.tensor(i['predicted'],dtype=torch.float64),{})
                    mapping=GroupMap(np.repeat(np.arange(task*8),10));gate=emergency_gate(m,mapping)
                    assert gate==i['gate'] and gate['alpha']==i['alpha'] and gate['reason']==i['gate_reason']
                    actual=mapping.values(i['actual']);assert actual['capabilities']==i['history']['capabilities'] and actual['q10']==i['history']['q10']
                else:assert i['alpha']==1 and i['gate_reason']=='NOT_MEASURED'
                previous=i['history']
            elif arm=='replay':
                groups=sorted(old);sel=[groups[(fixed_cursor+j)%len(groups)] for j in range(8)];fixed_cursor=(fixed_cursor+8)%len(groups)
                assert sel==i['selected_feedback_groups'] and ids==[v for g in sel for v in memory.take(g)] and n==16
            elif arm=='random':
                dose=schedule[t-211];assert dose['step']==t and dose['task']==task and dose['old_batch_size']==n
                selected,expected=random.choose(old,memory,n);assert selected==i['selected_feedback_groups'] and expected==ids
            else:assert ids==[] and n==0
            if arm!='router':assert i['alpha']==1 and not i['instrument_called'] and i['gate_reason']=='NOT_MEASURED'
    tasks={};assert len(r['metrics'])==10
    for task,m in enumerate(r['metrics'],1):
        probes,primary,full=scope(data,task*8);assert m['task']==task and m['step']==task*210 and m['indices']==full.tolist()
        pred=np.asarray(m['predictions']);cleanpred=pred[np.isin(full,primary)]
        primary_stats=summarize_predictions(primary,cleanpred,data,task);full_stats=summarize_predictions(full,pred,data,task)
        assert primary_stats==m['primary'] and full_stats==m['original']
        item=dict(task=task,primary=primary_stats,original=full_stats)
        if task>1:
            item.update(statistics(taskrows[task],arm=='router'))
            if arm=='router':
                assert m['control_state']==taskrows[task][-1]['info']['history']
                item['state_counts_all']=dict(Counter(m['control_state']['states']))
                item['state_counts_old']=dict(Counter(m['control_state']['states'][:(task-1)*8]))
            else:item.update(state_counts_all=None,state_counts_old=None)
        tasks[str(task)]=item
    stats=statistics(rows,arm=='router');assert r['old_exposures']==stats['old_exposures']
    assert r['update_seconds']==stats['update_seconds']
    return dict(status=r['status'],tasks=tasks,**stats,wall=r['wall'],load_seconds=r['load_seconds'],evaluation_seconds=r['evaluation_seconds'],
        wall_total=r['wall_total'],checkpoint_seconds=r['checkpoint_seconds'],peak_allocated=r['peak_allocated'],peak_reserved=r['peak_reserved'],
        hashes={f:sha(root/f) for f in ('CONFIG.json','RESULT.json','STEPS.jsonl','final.pt')}),rows

def main():
    unchanged();pre=json.loads((ROOT/'PREFLIGHT.json').read_text());assert pre['status']=='PASS'
    data=np.load(DATA/'data.npz');schedule=json.loads((ROOT/'seed-2021'/'EXPOSURE_SCHEDULE.json').read_text())['rows']
    assert len(schedule)==1890 and all(set(x)=={'step','task','old_batch_size'} for x in schedule)
    arms={};orders=None;prefix=None
    for arm in ARMS:
        arms[arm],rows=verify(arm,pre,data,schedule)
        if arm=='router':assert schedule==[dict(step=z['step'],task=z['task'],old_batch_size=z['info']['feedback_batch_size']) for z in rows]
        batch_order=[r['new_ids'] for r in rows]
        if orders is None:orders=batch_order
        assert orders==batch_order
        first=json.loads((ROOT/'seed-2021'/arm/'RESULT.json').read_text())['metrics'][0]['predictions']
        if prefix is None:prefix=first
        assert first==prefix
    assert arms['random']['old_exposures']==arms['router']['old_exposures']
    assert arms['replay']['old_exposures']==1890*16
    comparison={}
    for task in range(2,11):
        r,f,z=[arms[a]['tasks'][str(task)] for a in ('router','replay','random')]
        assert r['old_exposures']==z['old_exposures'] and r['feedback_batch_histogram']==z['feedback_batch_histogram']
        comparison[str(task)]=dict(exposure_ratio=r['old_exposures']/f['old_exposures'],
            router_minus_fixed_pp={k:100*(r['primary'][k]-f['primary'][k]) for k in ('macro_old','macro_new','macro_all')},
            router_minus_random_pp={k:100*(r['primary'][k]-z['primary'][k]) for k in ('macro_old','macro_new','macro_all')})
    result=dict(status='PASS',base=BASE,seed=2021,arms=arms,comparison=comparison,
        total_exposure_ratio=arms['router']['old_exposures']/arms['replay']['old_exposures'],
        repeat_trigger=trigger(arms),checker_boundary='record-derived metrics, deterministic routing/random sampling/dose/gate and hash verification; not independent native inference rerun',
        isolation=pre['data_audit'],primary_metric='macro relation values for comparisons, micro values also retained',
        confound='Router emergency gate differs from random; whole-mechanism not isolated-routing comparison')
    write(ROOT/'CHECKS.json',result);write(HERE/'results/CHECKS.json',result)
    print(json.dumps(dict(status=result['status'],ratio=result['total_exposure_ratio'],trigger=result['repeat_trigger']),indent=2))

if __name__=='__main__':main()
