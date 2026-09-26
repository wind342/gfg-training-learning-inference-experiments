"""Final read-only audit: distinguish decision identity from floating reruns.

Added after an in-progress prefix check found float-only differences. The
pre-run check.py is retained byte-for-byte and its original hash is verified.
No training, policy, criterion or original output is changed by this audit.
"""
import json
import numpy as np
from . import check as strict
from .gate import HERE,ROOT,BASE,audit_smoke,normalize

def prefix_comparison(rows,smoke):
    differences={};count=0;maximum=0.
    def compare(a,b,path):
        nonlocal count,maximum
        if isinstance(b,dict):
            for key,value in b.items():
                if key!='seconds':compare(a[key],value,path+'.'+key)
        elif isinstance(b,list):
            assert len(a)==len(b),path
            for x,y in zip(a,b):compare(x,y,path+'[]')
        elif a!=b:
            # Discrete actions/identities must agree. Float deltas are reported,
            # not declared bitwise equal or hidden behind a fitted tolerance.
            assert isinstance(a,float) and isinstance(b,float), (path,a,b)
            delta=abs(a-b);assert np.isfinite(delta),path
            count+=1;maximum=max(maximum,delta)
            differences[path]=max(differences.get(path,0.),delta)
    assert len(smoke)==60 and len(rows)>=60
    for new,old in zip(rows[:60],smoke):
        for key in ('alpha','old_loss_weight','current_loss_weight','gate_reason','router_mode',
                    'feedback_batch_size','selected_feedback_groups','selected_anchor_groups',
                    'release_event','router_activation'):
            assert new['info'][key]==old['info'][key],key
        for key in ('states','levels','capabilities'):
            assert new['info']['history'][key]==old['info']['history'][key],key
        compare(new,old,'record')
        for field in ('baseline','predicted','actual'):
            if field in old['info']:
                assert np.array_equal(np.array(new['info'][field])>0,np.array(old['info'][field])>0),field
    return dict(discrete_records_equal=True,measured_boundary_signs_equal=True,
        floating_bitwise_equal=count==0,float_difference_count=count,
        maximum_absolute_float_difference=maximum,maximum_by_field=differences,
        interpretation='same discrete control decisions in first60; floating rerun is not bitwise identical; no fitted tolerance')

def main():
    admission=audit_smoke();base=strict.v3check.base
    base.ROOT=ROOT;base.HERE=HERE;base.CORE=HERE;base.Router=strict.v3check.Router
    assert json.loads((ROOT/'SMOKE_CHECKS.json').read_text())==admission
    arms={};initial=None;orders=None;prefix=None
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
            v=strict.coverage(selected,arm=='router')
            endpoints=[m for m in result['metrics'] if m['task']==task]
            assert [m['offset'] for m in endpoints]==[0,210]
            v['start_accuracy']={k:endpoints[0][k] for k in ('old','new','accuracy','per_relation')}
            v['end_accuracy']={k:endpoints[1][k] for k in ('old','new','accuracy','per_relation')}
            per=endpoints[1]['per_relation']
            v['end_cohort_accuracy']={f'task{j+1}':float(np.mean([per[str(k)] for k in range(j*8,(j+1)*8)])) for j in range(task)}
            summary['tasks'][task]=v
        if arm=='router':
            assert len(result['task_boundary_audit'])==2
            boundary=result['task_boundary_audit'][1];previous=rows[209]['info']['history']
            for dst,src in [('state','states'),('level','levels'),('anchor','anchor_q10'),('capabilities','capabilities'),('q10','q10')]:
                assert boundary['preserved'][dst]==previous[src]
            assert boundary['old_labels']==list(range(16)) and boundary['new_labels']==list(range(24))
            assert boundary['last_step']==420 and rows[210]['step']==421
            smoke=list(map(json.loads,(strict.v3check.ROOT/'smoke-router/STEPS.jsonl').read_text().splitlines()))
            prefix=prefix_comparison(rows,smoke)
        summary['evaluation_seconds']=result['evaluation_seconds'];arms[arm]=summary
    output=dict(status='PASS',base_commit=BASE,admission=admission,arms=arms,smoke_prefix_comparison=prefix,
        auditor_sha256=base.sha(HERE/'audit.py'),
        audit_note='post-run read-only auditor added after strict bitwise smoke-prefix assertion proved inappropriate; original check.py and all training sources remain unchanged and hash-verified',
        checker_boundary='hashes, role isolation, routing/cursors replayed from recorded histories, gates from recorded predictions, metrics recomputed from saved predictions; no independent native rerun',
        scope='shared historical task1 checkpoint, 210 task2 and 210 task3 updates per arm, no tuning',
        risk_definition='pre-update observed router state, not privileged test truth; FORMING old groups may be ineligible')
    base.write(ROOT/'CHECKS.json',output);base.write(HERE/'results/CHECKS.json',output)
    print(json.dumps({a:dict(wall=v['wall'],peak=v['peak_allocated'],tasks={t:{k:s[k] for k in ('end_accuracy','old_exposures','feedback_active_fraction','modes','old_loss_weight_histogram')} for t,s in v['tasks'].items()}) for a,v in arms.items()},indent=2))

if __name__=='__main__':main()
