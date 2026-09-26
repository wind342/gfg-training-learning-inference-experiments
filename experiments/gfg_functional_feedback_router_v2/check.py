"""Reuse V1 integrity checker with V2 allocation; preserve all historical results."""
import json
from experiments.gfg_governor_fewrel_feedback_router_v1 import check as old
from .run import HERE,ROOT
from .policy import Router

def main():
    original_root=old.ROOT
    old.ROOT=ROOT;old.HERE=HERE;old.CORE=HERE;old.Router=Router
    summary,initial=old.verify('router','smoke')
    cfg=json.loads((ROOT/'smoke-router/CONFIG.json').read_text())
    for rel,h in cfg['inherited_sources'].items():assert old.sha(HERE.parent/rel)==h,rel
    previous=json.loads((original_root/'SMOKE_CHECKS.json').read_text())
    assert previous['status']=='PASS'
    for arm in ('native','router'):
        assert old.sha(original_root/f'smoke-{arm}/RESULT.json')==previous['arms'][arm]['hashes']['RESULT.json']
    original=json.loads((original_root/'smoke-router/RESULT.json').read_text())
    assert initial==original['metrics'][0]['predictions']
    rows=list(map(json.loads,(ROOT/'smoke-router/STEPS.jsonl').read_text().splitlines()))
    from collections import Counter
    sizes=Counter();single=[]
    for r in rows:
        i=r['info'];n=len(i['active_degraded_groups'])+len(i['active_at_risk_groups'])
        assert len(i['selected_anchor_groups'])<=1 and i['feedback_batch_size']<=2*min(8,n+1)
        if n==1:
            assert i['feedback_batch_size']<=4;single.append(r['offset'])
        sizes[i['feedback_batch_size']]+=1
    summary['batch_size_histogram']=dict(sizes)
    summary['single_risk_steps_no_overfill']=single
    summary['old_exposure_saving_vs_fixed']=1-summary['old_exposures']/960
    result=dict(status='PASS',router_v2=summary,historical=previous['arms'],
        historical_checks_sha256=old.sha(original_root/'SMOKE_CHECKS.json'),
        ten_step_structural_gate='PASS: all first10 recorded coalitions satisfy sparse-anchor bounds',
        scope='fixed60-step fix validation only; no three-task run or post-result tuning',
        checker_boundary=previous['check_boundary'],inherited_duplicate=previous['unchanged_duplicate'])
    old.write(ROOT/'CHECKS.json',result);public=HERE/'results';public.mkdir(exist_ok=True);old.write(public/'CHECKS.json',result)
    print(json.dumps({k:v for k,v in summary.items() if k!='hashes'},indent=2))

if __name__=='__main__':main()
