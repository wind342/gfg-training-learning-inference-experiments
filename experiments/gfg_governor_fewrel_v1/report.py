"""Read-only recomputation of test metrics and recorded capability decisions."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.gfg_functional_update_governor_v1.tests import measured
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_functional_update_governor_v3.scheduler import GroupSentinel
from experiments.gfg_functional_update_governor_v4.governor import RecoveryGovernor
from .prepare import ROOT,sha


def summarize(tasks):
    data=np.load(ROOT/'data.npz');truth=data['y'];inputs=data['x'];result={}
    for arm in ('native','governor'):
        root=ROOT/f'{tasks}tasks-{arm}'/'attempt-1'
        config=json.loads((root/'CONFIG.json').read_text());d=json.loads((root/'RESULT.json').read_text())
        assert config['data_sha256']==sha(ROOT/'data.npz') and d['status']=='COMPLETED'
        for name in ('model.py','prepare.py'):
            assert config['source_hashes'][name]==sha(Path(__file__).parent/name)
        checked=[];s=None;previous_task=0;calls=0;timings=Counter();modes=Counter();alphas=Counter()
        prediction_hits=prediction_total=0;changes=[];last_step=0;observed_micro=0;transitions=Counter();conflicts=0;opportunities=Counter()
        paths=[root/'STEPS.jsonl'];peak=d['peak_allocated']
        if d.get('prefix'):
            prefix=Path(d['prefix']);assert sha(prefix/'RESULT.json')==d['prefix_result_sha256']
            prefix_config=json.loads((prefix/'CONFIG.json').read_text())
            for name in ('model.py','prepare.py'):
                assert prefix_config['source_hashes'][name]==config['source_hashes'][name]
            peak=max(peak,json.loads((prefix/'RESULT.json').read_text())['peak_allocated'])
            paths.insert(0,prefix/'STEPS.jsonl')
        for log_path in paths:
          with log_path.open() as f:
            for line in f:
                row=json.loads(line);i=row['info'];t=row['step'];task=row['task']
                assert 1<=task<=tasks and t==last_step+1
                last_step=t;observed_micro+=int(i.get('micro_probe_called',False))
                alphas[str(i['alpha'])]+=1
                if arm=='native':assert i['alpha']==1.;continue
                if task!=previous_task:
                    old=s;s=GroupSentinel(GroupMap([k for k in range(task*8) for _ in range(10)]))
                    if old is not None:
                        n=len(old.state);s.state[:n]=old.state
                        for field in ('level','anchor','capabilities','q10','last_observed'):getattr(s,field)[:n]=getattr(old,field)
                        s.last_step=old.last_step
                    g=RecoveryGovernor(s.mapping,s);previous_task=task
                reasons=s.begin(t)
                if not reasons:
                    ids=s.shard(t);assert ids==i['micro_indices']
                    events=s.observe(ids,i['micro_margins']) if ids else []
                    assert events==i['micro_events'];reasons=s.reasons(events)
                assert (reasons or ['NONE'])==i['instrument_reason']
                if reasons:
                    calls+=1;m=measured(i['baseline_minima'],i['hierarchy']['predicted_minima']);decision=g.choose(m)
                    assert decision.alpha==i['alpha'] and decision.scores==i['scores']
                    assert g.summary==i['hierarchy']
                    assert s.measured(t,m.baseline_minima)==i['full_refresh_events']
                    actual=np.array(i['actual_endpoint_margins']);assert s.mapping.values(actual)==i['actual_endpoint']
                    predicted=m.predicted_minima[m.alphas.index(i['alpha'])].numpy()
                    prediction_hits+=int(((actual>0)==(predicted>0)).sum());prediction_total+=len(actual)
                    baseline=np.array(i['baseline_minima'])
                    transitions['correct_to_wrong']+=int(((baseline>0)&(actual<=0)).sum())
                    transitions['wrong_to_correct']+=int(((baseline<=0)&(actual>0)).sum())
                    conflicts+=int(i['scores']['1.0']['protected_violation_count']>0)
                    scores=list(i['scores'].values())
                    opportunities['calls_with_overall_count_difference']+=int(len({q['correct_count'] for q in scores})>1)
                    opportunities['calls_with_recovery_debt']+=int(bool(i['hierarchy']['degraded_groups']))
                    if i['hierarchy']['degraded_groups']:
                        opportunities['debt_calls_with_different_debt_capability']+=int(len({q['degraded_macro_capability'] for q in scores})>1)
                        opportunities['debt_calls_with_any_predicted_recovery']+=int(any(q['degraded_recovery_count']>0 for q in scores))
                    modes[i['intervention_reason']]+=1
                    changes.append(dict(step=t,task=task,alpha=i['alpha'],degraded=i['hierarchy']['degraded_groups'],
                        protected=i['hierarchy']['protected_groups'],
                        predicted_gain=i['scores'][str(i['alpha'])]['delta_capability'],
                        actual_gain=i['actual_endpoint']['capability']-i['hierarchy']['current']['capability']))
                assert s.snapshot(t)==i['group_observation']
                timings.update(i['timing']);timings['endpoint_audit']+=i.get('endpoint_audit_seconds',0.)
        assert last_step==tasks*210==d['steps']
        assert observed_micro==d['micro_calls'] and calls==d['full_calls']
        assert len(d['metrics'])==tasks
        for metric in d['metrics']:
            ids=np.array(metric['indices']);pred=np.array(metric['predictions']);seen=metric['task']*8
            assert np.all(data['split'][ids]==2) and np.all(truth[ids]<seen)
            assert np.array_equal(ids,np.flatnonzero((data['split']==2)&(truth<seen)))
            assert not(set(ids.tolist())&set(metric['probe_ids']))
            assert all(data['split'][i]==1 and truth[i]<seen for i in metric['probe_ids'])
            acc=float((truth[ids]==pred).mean());assert acc==metric['accuracy']
            per={str(k):float((pred[truth[ids]==k]==k).mean()) for k in range(seen)}
            assert per==metric['per_relation']
            assert metric['current_accuracy']==float(np.mean([per[str(k)] for k in range(seen-8,seen)]))
            if seen>8:
                assert metric['old_accuracy']==float(np.mean([per[str(k)] for k in range(seen-8)]))
                old_metrics=d['metrics'][:metric['task']-1]
                forgetting=[];bwt=[]
                for k in range(seen-8):
                    history=[m['per_relation'][str(k)] for m in old_metrics if str(k) in m['per_relation']]
                    forgetting.append(max(history)-per[str(k)]);bwt.append(per[str(k)]-history[0])
                assert float(np.mean(forgetting))==metric['average_forgetting']
                assert float(np.mean(bwt))==metric['backward_transfer']
            entry={k:metric[k] for k in ('task','accuracy','current_accuracy','old_accuracy','average_forgetting','backward_transfer','wall','task_wall','per_relation')}
            entry['control_state']=metric['control_state']
            probe_content={(int(truth[j]),inputs[j].tobytes()) for j in metric['probe_ids']}
            keep=np.array([(int(truth[j]),inputs[j].tobytes()) not in probe_content for j in ids])
            dedup_per={str(k):float((pred[keep&(truth[ids]==k)]==k).mean()) for k in range(seen)}
            entry['secondary_excluding_probe_content_duplicates']=dict(
                label='post-hoc data-integrity sensitivity, not replacement primary metric',
                excluded_test_indices=ids[~keep].tolist(),test_count=int(keep.sum()),
                accuracy=float((truth[ids][keep]==pred[keep]).mean()),
                current_macro_accuracy=float(np.mean([dedup_per[str(k)] for k in range(seen-8,seen)])),
                old_macro_accuracy=float(np.mean([dedup_per[str(k)] for k in range(seen-8)])) if seen>8 else None)
            if seen>8:
                before=metric['before_current_scope'];before_ids=np.array(before['indices']);before_pred=np.array(before['predictions'])
                assert np.array_equal(before_ids,ids)
                old_mask=truth[before_ids]<seen-8
                old_before=float((truth[before_ids][old_mask]==before_pred[old_mask]).mean())
                previous=d['metrics'][metric['task']-2]['accuracy']
                entry.update(old_accuracy_before_current_training=old_before,
                    old_scope_expansion_delta=old_before-previous,
                    old_training_delta=metric['old_accuracy']-old_before)
            checked.append(entry)
        result[arm]=dict(status='PASS',metrics=checked,full_calls=calls,alpha_histogram=dict(alphas),
            micro_calls=d['micro_calls'],timing_totals=dict(timings),reason_histogram=dict(modes),
            selected_endpoint_boundary_accuracy=prediction_hits/prediction_total if prediction_total else None,
            endpoint_count=prediction_total,decisions=changes,wall=d['wall'],peak_allocated=peak,
            selected_endpoint_transitions=dict(transitions),native_protected_conflict_calls=conflicts,
            candidate_opportunities=dict(opportunities),
            log_identities=[dict(path=str(p),sha256=sha(p)) for p in paths],
            steps_log_sha256=sha(root/'STEPS.jsonl'),result_sha256=sha(root/'RESULT.json'),
            checkpoints=d['checkpoints'],configuration=config)
    n=result['native'];g=result['governor']
    result['comparison']=dict(wall_ratio=g['wall']/n['wall'],
        old_accuracy_difference=g['metrics'][-1]['old_accuracy']-n['metrics'][-1]['old_accuracy'],
        new_accuracy_difference=g['metrics'][-1]['current_accuracy']-n['metrics'][-1]['current_accuracy'],
        overall_accuracy_difference=g['metrics'][-1]['accuracy']-n['metrics'][-1]['accuracy'],
        note='controller retains old validation probes; native has no old-data feedback; no replay baseline claim')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=int,default=3);args=p.parse_args()
    report=summarize(args.tasks)
    path=ROOT/f'{args.tasks}tasks-CHECKS.json'
    with path.open('x') as f:json.dump(report,f,separators=(',',':'),allow_nan=False)
    print(json.dumps({k:({x:v[x] for x in ['metrics','full_calls','wall','peak_allocated','selected_endpoint_boundary_accuracy']} if k!='comparison' else v) for k,v in report.items()},indent=2))
