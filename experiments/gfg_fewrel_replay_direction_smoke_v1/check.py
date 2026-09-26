"""Recompute metrics, split/batch identities and the original-path regression."""
import json
from pathlib import Path
import numpy as np
from .run import ROOT,DATA,HERE,CHECKPOINT,CKSHA,plan,sha,write

def main():
    data=np.load(DATA/'data.npz');probes,test,order,replay,memory=plan(data)
    assert sha(CHECKPOINT)==CKSHA
    historical={r['step']:r['info'] for r in map(json.loads,(DATA/'3tasks-governor/attempt-1/STEPS.jsonl').read_text().splitlines()) if 211<=r['step']<=270}
    reports={};initial=None
    for arm in ('new_governor','replay_native','replay_governor'):
        root=ROOT/arm
        if arm=='replay_governor':
            failure=json.loads((root/'RESULT.json').read_text())
            assert failure['status']=='FAILED' and failure['steps']==0
            root=ROOT/'resource_retry'/arm
        c=json.loads((root/'CONFIG.json').read_text());r=json.loads((root/'RESULT.json').read_text())
        if arm=='replay_governor':
            assert c['resource_retry']['wrapper_sha256']==sha(HERE/'retry_resource.py')
            assert c['resource_retry']['failed_result_sha256']==sha(ROOT/arm/'RESULT.json')
        rows=list(map(json.loads,(root/'STEPS.jsonl').read_text().splitlines()))
        assert r['status']=='COMPLETED' and len(rows)==60
        assert c['checkpoint_sha256']==CKSHA and c['source_sha256']==sha(HERE/'run.py') and c['protocol_sha256']==sha(HERE/'PROTOCOL.md')
        assert c['new_batches']==order.tolist() and c['old_batches']==replay.tolist() and c['memory']==memory
        assert c['test']==test.tolist() and c['probes']==probes.tolist()
        assert r['log_sha256']==sha(root/'STEPS.jsonl')
        if initial is None:initial=r['metrics'][0]['predictions']
        assert r['metrics'][0]['predictions']==initial
        assert [m['offset'] for m in r['metrics']]==[0,20,40,60]
        sensitivity=[]
        for m in r['metrics']:
            assert m['indices']==test.tolist()
            pred=np.array(m['predictions']);truth=data['y'][test]
            assert m['accuracy']==float((pred==truth).mean())
            for k in range(16):assert m['per_relation'][str(k)]==float((pred[truth==k]==k).mean())
            for key,ks in [('old',range(8)),('new',range(8,16))]:assert m[key]==float(np.mean([m['per_relation'][str(k)] for k in ks]))
            keep=test!=7561
            per={k:float((pred[(truth==k)&keep]==k).mean()) for k in range(16)}
            sensitivity.append(dict(offset=m['offset'],old=float(np.mean([per[k] for k in range(8)])),
                new=float(np.mean([per[k] for k in range(8,16)]))))
        hits=total=changed=changed_hits=0;max_regression_error=0.
        for offset,row in enumerate(rows,1):
            assert row['offset']==offset;i=row['info']
            if arm=='new_governor':
                old=historical[210+offset]
                assert i['alpha']==old['alpha'] and i['instrument_called']==old['instrument_called']
                if i['instrument_called']:
                    err=float(np.max(np.abs(np.array(i['actual'])-old['actual_endpoint_margins'])))
                    max_regression_error=max(max_regression_error,err)
            if i['instrument_called']:
                pred=np.array(i['predicted'])[[.125,.25,.5,1.,2.].index(i['alpha'])]>0
                actual=np.array(i['actual'])>0;before=np.array(i['baseline'])>0
                hits+=int((pred==actual).sum());total+=len(actual)
                mask=actual!=before;changed+=int(mask.sum());changed_hits+=int(((pred==actual)&mask).sum())
        if arm=='new_governor':assert max_regression_error<1e-5,max_regression_error
        reports[arm]=dict(metrics=[{k:m[k] for k in ('offset','old','new','accuracy')} for m in r['metrics']],
            resource_retry=c.get('resource_retry'),
            posthoc_exclude_inherited_duplicate_7561=sensitivity,
            wall=r['wall'],peak_allocated=r['peak_allocated'],full_calls=r['full_calls'],alpha_counts=r['alpha_counts'],
            prediction_hits=hits,prediction_total=total,actual_transitions=changed,predicted_actual_transitions=changed_hits,
            original_path_max_endpoint_error=max_regression_error if arm=='new_governor' else None,
            raw_hashes={n:sha(root/n) for n in ('CONFIG.json','RESULT.json','STEPS.jsonl')})
    result=dict(status='PASS',arms=reports,
        limitations=['one seed,60 updates; not full continual learning or recovery',
        'replay adds old training exposure; no equal-total-samples comparison',
        'existing inherited probe/test duplicate disclosed in parent FewRel results remains unchanged',
        'checker recomputes logs; no independent full rerun or final checkpoint saved'])
    write(ROOT/'CHECKS.json',result)
    public=HERE/'results';public.mkdir(exist_ok=True)
    write(public/'CHECKS.json',result)
    failed=ROOT/'replay_governor/RESULT.json';failure=json.loads(failed.read_text())
    write(public/'RESOURCE_FAILURE.json',dict(status=failure['status'],completed_steps=failure['steps'],
        reason='MemoryError: workspace budget cannot fit the minimum exact tile',
        raw_result_sha256=sha(failed),wall=failure['wall'],
        treatment='preserved unchanged; separate cache-release-only retry, same scientific settings'))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
