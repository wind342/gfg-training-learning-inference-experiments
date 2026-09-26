"""Validate sample-count dose and unchanged routing/gate against actual records."""
from collections import Counter
from fractions import Fraction
import json
from experiments.gfg_governor_fewrel_feedback_router_v1 import check as base
from experiments.gfg_functional_feedback_router_v2.policy import Router
from experiments.gfg_functional_feedback_router_v2.run import ROOT as V2ROOT
from .run import HERE,ROOT

def main():
    base.ROOT=ROOT;base.HERE=HERE;base.CORE=HERE;base.Router=Router
    summary,initial=base.verify('router','smoke')
    c=json.loads((ROOT/'smoke-router/CONFIG.json').read_text())
    for field in ('inherited_sources','router_v2_sources'):
        for rel,h in c[field].items():assert base.sha(HERE.parent/rel)==h,rel
    assert c['v3_base_commit']=='cf6c0f30d0e35173771ab2fc52fa2474b1d940ee'
    rows=list(map(json.loads,(ROOT/'smoke-router/STEPS.jsonl').read_text().splitlines()))
    batches=Counter();weights=Counter();old_coefficient_sum=0.
    for r in rows:
        i=r['info'];nc=len(r['new_ids']);no=len(i['feedback_example_ids']);total=nc+no
        assert nc==i['current_batch_size']==16
        assert i['current_loss_weight']==nc/total and i['old_loss_weight']==no/total
        assert i['weighted_training_loss']==(nc*i['current_loss']+no*(i['feedback_loss'] or 0.))/total
        assert len(i['selected_anchor_groups'])<=1
        batches[no]+=1;weights[str(Fraction(no,total))]+=1;old_coefficient_sum+=no/total
    prior=json.loads((V2ROOT/'CHECKS.json').read_text());assert prior['status']=='PASS'
    oldresult=json.loads((V2ROOT/'smoke-router/RESULT.json').read_text())
    assert base.sha(V2ROOT/'smoke-router/RESULT.json')==prior['router_v2']['hashes']['RESULT.json']
    assert initial==oldresult['metrics'][0]['predictions']
    summary.update(feedback_batch_histogram=dict(batches),old_loss_weight_histogram=dict(weights),
        cumulative_old_loss_coefficient=old_coefficient_sum,cumulative_current_loss_coefficient=60-old_coefficient_sum)
    v2rows=list(map(json.loads,(V2ROOT/'smoke-router/STEPS.jsonl').read_text().splitlines()))
    oldweights=Counter('1/2' if r['info']['feedback_batch_size'] else '0' for r in v2rows)
    v2=prior['router_v2']|dict(old_loss_weight_histogram=dict(oldweights))
    result=dict(status='PASS',router_v3=summary,router_v2=v2,
        new_only_native=prior['historical']['native'],fixed_replay_native=prior['historical']['replay_reference'],
        historical_checks_sha256=base.sha(V2ROOT/'CHECKS.json'),
        scope='only sample-count dose changed; actual closed-loop coalitions may differ',
        check_boundary=prior['checker_boundary'],inherited_duplicate=prior['inherited_duplicate'])
    base.write(ROOT/'CHECKS.json',result);public=HERE/'results';public.mkdir(exist_ok=True);base.write(public/'CHECKS.json',result)
    print(json.dumps({k:v for k,v in summary.items() if k!='hashes'},indent=2))

if __name__=='__main__':main()
