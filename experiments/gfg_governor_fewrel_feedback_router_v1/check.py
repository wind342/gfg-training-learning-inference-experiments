"""Recompute metrics/routing/gates and verify role and identity invariants."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from experiments.gfg_functional_feedback_router_v1.policy import Memory,Router,emergency_gate
from experiments.gfg_functional_update_governor_v1.instrument import Measurement
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_functional_update_governor_v3.scheduler import GroupSentinel
from experiments.gfg_fewrel_replay_direction_smoke_v1.run import ROOT as PREVIOUS
from .run import ROOT,DATA,HERE,CORE,CHECKPOINT,CKSHA,SEED,scope,extend_memory,validate_batches,sha,write,plan

def verify(arm,stage):
    root=ROOT/f'{stage}-{arm}';r=json.loads((root/'RESULT.json').read_text());c=json.loads((root/'CONFIG.json').read_text())
    assert r['status']=='COMPLETED' and r['steps']==(60 if stage=='smoke' else 420)
    assert sha(CHECKPOINT)==CKSHA==c['checkpoint_sha256'] and c['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    assert c['data_sha256']==sha(DATA/'data.npz')
    for rel,h in c['source_hashes'].items():assert sha(HERE.parent/rel)==h,rel
    assert r['log_sha256']==sha(root/'STEPS.jsonl')
    rows=list(map(json.loads,(root/'STEPS.jsonl').read_text().splitlines()));assert len(rows)==r['steps']
    data=np.load(DATA/'data.npz');memory=Memory();router=Router();counts=Counter();modes=Counter();gate_counts=Counter()
    pergroup=Counter();active=old_exposures=releases=activations=hits=total=changes=change_hits=full=0
    s=None;prev_task=0;fixed_cursor=0;last=None
    initial=json.loads((DATA/'3tasks-governor/attempt-1/RESULT.json').read_text())['metrics'][0]['control_state']
    fieldmap={'states':'state','capabilities':'capabilities','q10':'q10','anchor_q10':'anchor','levels':'level'}
    for row in rows:
        task=row['task'];offset=row['offset'];t=row['step'];i=row['info']
        assert t==(last+1 if last is not None else 211);last=t
        seen=task*8;old=set(range(seen-8));current=set(range(seen-8,seen))
        if task!=prev_task:
            probes,test=scope(data,seen);extend_memory(memory,data,old,probes,test)
            if s is None:
                s=GroupSentinel(GroupMap(np.repeat(np.arange(seen),10)))
                for src,dst in fieldmap.items():
                    value=initial[src]
                    getattr(s,dst)[:8]=value
            else:
                new=GroupSentinel(GroupMap(np.repeat(np.arange(seen),10)));n=len(s.state)
                for dst in fieldmap.values():getattr(new,dst)[:n]=getattr(s,dst)
                s=new
            train=np.flatnonzero(np.isin(data['y'],list(current))&(data['split']==0))
            g=torch.Generator().manual_seed(SEED+10000+task-1)
            order=train[torch.randperm(len(train),generator=g).numpy()]
            prev_task=task
        assert row['new_ids']==order[(offset-1)*16:offset*16].tolist()
        assert row['probe_ids']==probes.tolist() and row['memory_identity']==memory.stamp
        assert row['memory_groups']=={str(g):list(v) for g,v in memory.groups.items()}
        assert {int(k):v for k,v in row['cursor_before'].items()}==memory.cursors
        oldids=i['feedback_example_ids']
        if arm=='router':
            expected=router.choose(t,s.mapping.labels,list(s.state),list(s.level),old,memory)
            for k,v in expected.items():assert i[k]==v,(t,k)
            if i['instrument_called']:
                m=Measurement((.125,.25,.5,1.),torch.tensor(i['baseline'],dtype=torch.float64),torch.tensor(i['predicted'],dtype=torch.float64),{})
                gate=emergency_gate(m,s.mapping);assert gate==i['gate'] and gate['alpha']==i['alpha'] and gate['reason']==i['gate_reason']
                p=np.array(i['predicted'])[[.125,.25,.5,1.].index(i['alpha'])]>0;a=np.array(i['actual'])>0;b=np.array(i['baseline'])>0
                hits+=int((p==a).sum());total+=len(a);mask=a!=b;changes+=int(mask.sum());change_hits+=int(((p==a)&mask).sum())
                actual=s.mapping.values(i['actual'])
                assert actual['capabilities']==i['history']['capabilities'] and actual['q10']==i['history']['q10']
            else:assert i['alpha']==1. and i['gate_reason']=='NOT_MEASURED'
            for src,dst in fieldmap.items():getattr(s,dst)[:]=i['history'][src]
        elif arm=='replay':
            gs=sorted(old);selected=[gs[(fixed_cursor+j)%len(gs)] for j in range(min(8,len(gs)))];fixed_cursor=(fixed_cursor+len(selected))%len(gs)
            expected=[v for g in selected for v in memory.take(g)]
            assert expected==oldids and i['selected_feedback_groups']==selected
        else:assert not oldids and i['alpha']==1.
        validate_batches(row['new_ids'],oldids,data,current,memory,probes,test)
        assert len(oldids)==i['feedback_batch_size']<=16
        assert set(i['selected_feedback_groups'])<=old
        assert len(oldids)==2*len(i['selected_feedback_groups'])
        for g in i['selected_feedback_groups']:pergroup[g]+=2
        active+=bool(oldids);old_exposures+=len(oldids);releases+=i['release_event'];activations+=i['router_activation']
        counts[i['alpha']]+=1;modes[i['router_mode']]+=1;gate_counts[i['gate_reason']]+=1;full+=i['instrument_called']
    assert active==r['feedback_steps'] and old_exposures==r['old_exposures'] and releases==r['releases'] and full==r['full_calls']
    assert dict(modes)==r['mode_counts'] and {str(k):v for k,v in counts.items()}==r['alpha_counts']
    metrics=[];initial_prediction=None
    for m in r['metrics']:
        probes,test=scope(data,m['task']*8);assert m['indices']==test.tolist()
        truth=data['y'][test];p=np.array(m['predictions']);assert m['accuracy']==float((p==truth).mean())
        per={str(k):float((p[truth==k]==k).mean()) for k in range(m['task']*8)};assert per==m['per_relation']
        cut=(m['task']-1)*8
        assert m['old']==float(np.mean([per[str(k)] for k in range(cut)]))
        assert m['new']==float(np.mean([per[str(k)] for k in range(cut,m['task']*8)]))
        keep=test!=7561
        clean={k:float((p[(truth==k)&keep]==k).mean()) for k in range(m['task']*8)}
        metrics.append({k:m[k] for k in ('task','offset','old','new','accuracy')}|dict(duplicate_sensitivity_new=float(np.mean([clean[k] for k in range(cut,m['task']*8)]))))
        if initial_prediction is None:initial_prediction=m['predictions']
    summary=dict(metrics=metrics,wall=r['wall'],peak_allocated=r['peak_allocated'],old_exposures=old_exposures,
        feedback_steps=active,feedback_fraction=active/len(rows),mode_counts=dict(modes),releases=releases,activations=activations,
        per_relation_exposures=dict(pergroup),full_calls=full,gate_counts=dict(gate_counts),alpha_counts=dict(counts),
        prediction_hits=hits,prediction_total=total,actual_transitions=changes,predicted_transitions=change_hits,
        hashes={n:sha(root/n) for n in ('CONFIG.json','RESULT.json','STEPS.jsonl')})
    if stage=='three':
        prefix=json.loads((DATA/'3tasks-governor/attempt-1/RESULT.json').read_text())['metrics'][0]
        h=[prefix['per_relation']];endpoints=[]
        for m in r['metrics']:
            if m['offset']!=210:continue
            current=m['per_relation'];fg=[];bwt=[]
            for k in range((m['task']-1)*8):
                history=[v[str(k)] for v in h if str(k) in v];fg.append(max(history)-current[str(k)]);bwt.append(current[str(k)]-history[0])
            endpoints.append(dict(task=m['task'],average_forgetting=float(np.mean(fg)),backward_transfer=float(np.mean(bwt))))
            h.append(current)
        summary['forgetting']=endpoints
        summary['shared_task1']={k:prefix[k] for k in ('current_accuracy','accuracy','task_wall')}
    return summary,initial_prediction

def main(stage):
    arms={};initial=None
    for arm in ('native','router') if stage=='smoke' else ('native','replay','router'):
        summary,p=verify(arm,stage);arms[arm]=summary
        if initial is None:initial=p
        assert p==initial
    if stage=='smoke':
        cfg=json.loads((PREVIOUS/'replay_native/CONFIG.json').read_text());ref=json.loads((PREVIOUS/'replay_native/RESULT.json').read_text())
        assert sha(PREVIOUS/'replay_native/RESULT.json')=='5b7a3ded69ed51f0af357acfc8b600353010d0dd61800d9197e534de86d463a8'
        p,t,order,rep,mem=plan(np.load(DATA/'data.npz'))
        assert cfg['checkpoint_sha256']==CKSHA and cfg['new_batches']==order.tolist() and cfg['old_batches']==rep.tolist() and cfg['memory']==mem
        assert initial==ref['metrics'][0]['predictions']
        arms['replay_reference']=dict(metrics=[{k:m[k] for k in ('offset','old','new','accuracy')} for m in ref['metrics']],
            wall=ref['wall'],peak_allocated=ref['peak_allocated'],old_exposures=960,feedback_steps=60,feedback_fraction=1.,
            result_sha256=sha(PREVIOUS/'replay_native/RESULT.json'))
    result=dict(status='PASS',stage=stage,arms=arms,
        check_boundary='recomputed routing/cursors from recorded histories, gates from predictions, test metrics from recorded predictions; not an independent native rerun',
        unchanged_duplicate='inherited probe/test index7561; secondary sensitivity shown; no training memory duplicate',
        scope='same task1 checkpoint; no test-driven gradient/control; short engineering comparison')
    if stage=='smoke':
        a=arms['router'];result['three_task_gate']=bool(a['activations']>0 and a['releases']>0 and a['metrics'][-1]['new']>=.7)
        result['gate_rule']='integrity PASS, activation>0, release>0, final new accuracy>=70%; no efficacy claim'
    name='SMOKE_CHECKS.json' if stage=='smoke' else 'THREE_CHECKS.json'
    write(ROOT/name,result);public=HERE/'results';public.mkdir(exist_ok=True);write(public/name,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','three'],default='smoke');main(p.parse_args().stage)
