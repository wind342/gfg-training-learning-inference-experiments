"""One read-only evaluation phase, permitted only after ALL new training ends."""
import time
from .core import *

def main():
    pre=guard();setup();barrier=json.loads((ROOT/'TRAINING_COMPLETE.json').read_text())
    for arm in ARMS:
        r=json.loads((ROOT/arm/'TRAIN_RESULT.json').read_text());assert r['status']=='COMPLETED'
        assert sha(ROOT/arm/'TRAIN_RESULT.json')==barrier['result_hashes'][arm]
        assert r['finished_unix']<=barrier['created_unix'] and r['audit_sha256']==sha(ROOT/arm/'AUDIT.pt')
    start=time.time();names=[BASELINE]+list(ARMS);heads={};controls={}
    for name in names:
        p=BASE_ROOT/'head_only_exact_monitor' if name==BASELINE else ROOT/name
        a=torch.load(p/'AUDIT.pt',map_location='cpu',weights_only=False);heads[name]={k:v.cuda() for k,v in a['heads'][-1].items()}
        rr=[json.loads(s) for s in (p/'STEPS.jsonl').read_text().splitlines()]
        controls[name]=dict(final=rr[-1]['observation']['candidate_stats'],
            old_correct_to_wrong=sum(int(np.sum((np.array(r['observation']['baseline'])[:80]>0)&(np.array(r['observation']['candidate'])[:80]<=0))) for r in rr),
            newly_unsafe_steps=sum(bool(r['observation']['newly_threatened_by_this_update']) for r in rr),
            newly_unsafe_group_occurrences=sum(len(r['observation']['newly_threatened_by_this_update']) for r in rr))
        del a
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);model=FrozenRelationModel().cuda();model.load_state_dict(saved['model']);freeze(model);model.seen=16;del saved
    assert tensor_hash(model.bert.state_dict())==pre['backbone_initial_sha256']
    d=np.load(DATA/'data.npz');x=torch.from_numpy(d['x']);_,primary,full=scope(d,16)
    ids=np.flatnonzero((d['split']==0)&(d['y']>=8)&(d['y']<16));out={a:dict(controls=controls[a]) for a in names}
    with torch.no_grad():
        for label,index in (('training',ids),('test',full)):
            pred={a:[] for a in names};restricted={a:[] for a in names}
            for b in range(0,len(index),16):
                h=model.features(x[index[b:b+16]].cuda())
                for a,hd in heads.items():
                    z=nn.functional.linear(h,hd['weight'],hd['bias'])[:,:16];pred[a]+=z.argmax(1).cpu().tolist()
                    if label=='training':restricted[a]+=(z[:,8:].argmax(1)+8).cpu().tolist()
            for a in names:
                pp=np.array(pred[a]);truth=d['y'][index]
                if label=='training':out[a]['training']=dict(indices=index.tolist(),predictions=pred[a],new_only_predictions=restricted[a],
                    accuracy_16class=float((pp==truth).mean()),accuracy_new_only=float((np.array(restricted[a])==truth).mean()),n=len(index))
                else:out[a]['test']=dict(indices=full.tolist(),predictions=pred[a],
                    primary=summarize_predictions(primary,pp[np.isin(full,primary)],d,2),original=summarize_predictions(full,pp,d,2))
            print('EVALUATED',label,flush=True)
    baseline=json.loads((BASE_ROOT/'head_only_exact_monitor'/'RESULT.json').read_text())
    assert out[BASELINE]['test']['predictions']==baseline['metrics']['predictions']
    assert out[BASELINE]['test']['primary']==baseline['metrics']['primary']
    assert tensor_hash(model.bert.state_dict())==pre['backbone_initial_sha256']
    write(ROOT/'EVALUATION.json',dict(status='PASS',started_unix=start,training_barrier_sha256=sha(ROOT/'TRAINING_COMPLETE.json'),
        source_hashes=sources(),baseline_retrained=False,baseline_test_matches=True,arms=out))
    for a in names:print(a,out[a]['training']['accuracy_16class'],out[a]['training']['accuracy_new_only'],out[a]['test']['primary']['macro_new'],flush=True)

if __name__=='__main__':main()
