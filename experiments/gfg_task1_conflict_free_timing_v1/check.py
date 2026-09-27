"""Read-only verification of inputs, paired identities, timings and test readout."""
import json
from pathlib import Path
import numpy as np
import torch
from .run import ROOT,HERE,REPO,ARMS,BASE,setup,FrozenRelationModel,freeze,features
from experiments.gfg_head_only_exact_monitor_smoke_v1.core import equal,tensor_hash,head_hash,moments_hash
from experiments.gfg_router_v4_ten_task_validation.common import sha,write,DATA,DATA_SHA,order,scope

def main():
    pre=json.loads((ROOT/'PREFLIGHT.json').read_text())
    r=json.loads((ROOT/'RESULTS.json').read_text())
    sealed=json.loads((ROOT/'TRAINING_SEALED.json').read_text())
    rows=[json.loads(s) for s in (ROOT/'STEPS.jsonl').read_text().splitlines()]
    assert len(rows)==210 and sha(ROOT/'STEPS.jsonl')==sealed['ledger_sha256']
    assert sha(ROOT/'INITIAL.pt')==pre['initial_sha256'] and sha(ROOT/'FINAL.pt')==r['final_sha256']
    assert sha(DATA/'data.npz')==DATA_SHA
    assert all(sha(REPO/p)==v for p,v in pre['sources'].items())
    assert all(sha(p)==v for p,v in pre['pretrained_files'].items())
    d=np.load(DATA/'data.npz');batches=order(d,1)
    for i,row in enumerate(rows):
        assert row['step']==i+1 and row['ids']==batches[i].tolist()
        a,b=[row['arms'][k] for k in ARMS]
        for k in ('loss','head_sha256','moments_sha256'):assert a[k]==b[k]
        assert not b['projection'] and not b['skip']
    final=torch.load(ROOT/'FINAL.pt',map_location='cpu',weights_only=False)
    assert equal(final[ARMS[0]],final[ARMS[1]])
    for arm in ARMS:
        for k in ('seconds','bert_seconds','head_update_seconds','controller_seconds'):
            assert abs(sum(row['arms'][arm][k] for row in rows)-r['summary'][arm]['timing'][k])<1e-10
        assert all(int(state['step'].item())==210 for state in final[arm]['optimizer']['state'].values())
        for item in r['curve']:assert item[ARMS[0]]==item[ARMS[1]]
        ids=np.asarray(r['test_ids']);pred=np.asarray(r['test'][arm]['predictions'])
        exact_macro=float(np.mean([np.mean(pred[d['y'][ids]==g]==g) for g in range(8)]))
        assert abs(exact_macro-r['summary'][arm]['test_accuracy'])<1e-7
    # Independently regenerate pretrained frozen features and actual test logits.
    setup();m=FrozenRelationModel().cuda();freeze(m);m.eval()
    assert tensor_hash(m.bert.state_dict())==pre['backbone_sha256']
    assert head_hash(m.head)==pre['initial_head_sha256']
    m.head.load_state_dict(final[ARMS[0]]['head'])
    assert head_hash(m.head)==rows[-1]['arms'][ARMS[0]]['head_sha256']
    ids=scope(d,8)[1];assert ids.tolist()==r['test_ids']
    with torch.no_grad():
        h=features(m,torch.from_numpy(d['x']),ids).cuda()
        pred=m.head(h)[:,:8].argmax(1).cpu().tolist()
    assert pred==r['test'][ARMS[0]]['predictions']==r['test'][ARMS[1]]['predictions']
    check=dict(status='PASS',paired_steps=210,head_and_optimizer_identity=True,
        input_and_source_hashes=True,batch_order_identity=True,timings_recomputed=True,
        independently_recomputed_test_predictions=len(pred),test_correct=sum(a==int(b) for a,b in zip(pred,d['y'][ids])),
        final_optimizer_step=210,projection_steps=0,skip_steps=0)
    write(ROOT/'INDEPENDENT_CHECK.json',check)
    curve=[dict(step=v['step'],validation_accuracy=v[ARMS[0]]['macro']) for v in r['curve']]
    out=HERE/'results'
    write(out/'RESULTS.json',dict(base=BASE,
        summary=r['summary'],curve=curve,peak_cuda_bytes=r['peak_cuda_bytes'],cache_seconds=r['cache_seconds'],
        controller_over_native_wall_ratio=r['controller_over_native_wall_ratio'],
        exact_test_correct=check['test_correct'],test_count=len(pred)))
    write(out/'CHECKS.json',check)
    write(out/'TIMING_LEDGER.json',rows)
    write(out/'ARTIFACT_MANIFEST.json',dict(local_root=str(ROOT),
        artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in ROOT.iterdir() if p.is_file()}))
    print(json.dumps(check,indent=2))

if __name__=='__main__':main()
