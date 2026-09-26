"""Three fixed trainings, then one shared test evaluation; no test selection."""
import traceback
from .core import *

def build_cache(model,d):
    old=torch.load(PARENT_ROOT/'head_only_exact_cut'/'AUDIT.pt',map_location='cpu',weights_only=False)
    original=scope(d,16)[0];H=old['control_features'];del old
    assert H.shape==(160,1536)
    byid={int(i):H[j].clone() for j,i in enumerate(original)}
    extra=[int(i) for i in bank_ids(d,140) if int(i) not in byid];assert len(extra)==1040
    x=torch.from_numpy(d['x'])
    with torch.no_grad():
        for b in range(0,len(extra),16):
            ids=extra[b:b+16];h=model.features(x[ids].cuda()).cpu()
            for i,v in zip(ids,h):byid[i]=v.clone()
    ids=bank_ids(d,140)
    cache=dict(indices=ids.tolist(),features=torch.stack([byid[int(i)] for i in ids]))
    with (ROOT/'CACHE.pt').open('xb') as f:torch.save(cache,f)
    return byid

def train(arm,bank,pre):
    setup();n,kind=ARMS[arm];out=ROOT/arm;out.mkdir(exist_ok=False);rows=[];start=time.perf_counter()
    write(out/'CONFIG.json',dict(base=BASE,arm=arm,protection_per_relation=n,cut=kind,epsilon=EPS if kind=='tight' else None,
        seed=2021,lr=LR,steps=210,batch_size=16,replay=0,source_hashes=sources(),checkpoint_sha256=CKSHA,data_sha256=DATA_SHA))
    try:
        d=np.load(DATA/'data.npz');x=torch.from_numpy(d['x']);y=torch.from_numpy(d['y']);ids=bank_ids(d,n);old_count=n*8
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        model=FrozenRelationModel().cuda();model.load_state_dict(saved['model']);freeze(model);model.seen=16
        shadow,opt=make_candidate(model,saved['optimizer']);del saved
        assert head_hash(model.head)==pre['head_initial_sha256'] and moments_hash(opt)==pre['head_optimizer_initial_sha256']
        set_lr(opt,LR);initialopt=copy.deepcopy(opt.state_dict());backbone=tensor_hash(model.bert.state_dict())
        assert backbone==pre['backbone_initial_sha256'];versions=[p._version for p in model.bert.parameters()]
        h=torch.stack([bank[int(i)] for i in ids]).cuda();labels=y[ids].cuda();cachehash=tensor_hash({'H':h});hv=h._version
        history=[cpu_state(snapshot(model.head))];natives=[];cuts=[]
        sync(h);trainstart=time.perf_counter()
        with (out/'STEPS.jsonl').open('x',encoding='utf8') as f:
            for step,batch in enumerate(order(d,2),1):
                tick=time.perf_counter();current=batch.tolist();seed=2021*100000+210+step
                model.train();assert all(not m.training for m in model.bert.modules())
                model.head.zero_grad(set_to_none=True);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with torch.no_grad():features=model.features(x[current].cuda())
                loss=nn.functional.cross_entropy(model.head(features)[:,:16],y[current].cuda());loss.backward()
                gn=nn.utils.clip_grad_norm_(model.head.parameters(),1.);assert torch.isfinite(gn) and torch.isfinite(loss)
                propose(model.head,shadow,opt);before=snapshot(model.head);native=snapshot(shadow)
                rh=rng();mh=moments_hash(opt);hh=head_hash(model.head)
                executed,candidate,report=cut(before,native,h,labels,old_count,kind)
                assert equal(rh,rng()) and mh==moments_hash(opt) and hh==head_hash(model.head)
                assert versions==[p._version for p in model.bert.parameters()] and all(p.grad is None and not p.requires_grad for p in model.bert.parameters())
                assert hv==h._version
                with torch.no_grad():
                    for k,p in model.head.named_parameters():p.copy_(executed[k])
                actual=margins(logits(model.head,h),labels).cpu().tolist()
                assert actual==(report['baseline'] if report['skip'] else report['cut'])
                assert not any(a>0 and b<=0 for a,b in zip(report['baseline'][:old_count],actual[:old_count]))
                report['executed']=actual
                history.append(cpu_state(executed));natives.append(cpu_state(native));cuts.append(cpu_state(candidate))
                sync(h);row=dict(step=step,current_ids=current,seed=seed,lr=LR,old_replay=0,report=report,
                    cut_geometry=geometry(before,native,candidate),executed_geometry=geometry(before,native,executed),
                    head_before_sha256=hh,head_after_sha256=head_hash(model.head),optimizer_sha256=mh,
                    native_sha256=tensor_hash(native),cut_sha256=tensor_hash(candidate),
                    moments_unmodified_by_cut=True,bert_frozen_eval=True,seconds=time.perf_counter()-tick)
                rows.append(row);f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');f.flush()
                if step in (1,50,100,150,210):print(arm,step,report['action'],'old',sum(v>0 for v in actual[:old_count]),'skips',sum(r['report']['skip'] for r in rows),flush=True)
                if time.perf_counter()-trainstart>7200:raise TimeoutError('fixed 2h cap')
        sync(h);wall=time.perf_counter()-trainstart
        assert backbone==tensor_hash(model.bert.state_dict()) and cachehash==tensor_hash({'H':h})
        with (out/'AUDIT.pt').open('xb') as f:torch.save(dict(heads=history,natives=natives,cuts=cuts,control_features=h.cpu(),
            control_ids=ids.tolist(),initial_optimizer=initialopt,final_optimizer=opt.state_dict()),f)
        result=dict(status='COMPLETED',arm=arm,steps=210,training_wall=wall,no_test_inference=True,finished_unix=time.time(),
            summary=summary(rows,old_count),head_sha256=head_hash(model.head),optimizer_sha256=moments_hash(opt),
            backbone_sha256=backbone,cache_sha256=cachehash,audit_sha256=sha(out/'AUDIT.pt'),log_sha256=sha(out/'STEPS.jsonl'),
            config_sha256=sha(out/'CONFIG.json'))
    except Exception:result=dict(status='FAILED',steps=len(rows),traceback=traceback.format_exc())
    write(out/'TRAIN_RESULT.json',result)
    if result['status']!='COMPLETED':raise RuntimeError(result['traceback'])
    print('FINISHED',arm,result['summary'],'wall',wall,flush=True)
    return result

def evaluate(pre,results):
    guard();barrier=json.loads((ROOT/'TRAINING_COMPLETE.json').read_text())
    for arm,r in results.items():
        assert r['status']=='COMPLETED' and r['no_test_inference']
        assert sha(ROOT/arm/'TRAIN_RESULT.json')==barrier['train_result_hashes'][arm]
        assert sha(ROOT/arm/'AUDIT.pt')==r['audit_sha256']
    started=time.time();assert started>=max(r['finished_unix'] for r in results.values())
    setup();saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=FrozenRelationModel().cuda();model.load_state_dict(saved['model']);freeze(model);model.eval();del saved
    heads={}
    for arm in ARMS:
        a=torch.load(ROOT/arm/'AUDIT.pt',map_location='cpu',weights_only=False)
        heads[arm]={k:v.cuda() for k,v in a['heads'][-1].items()};del a
    d=np.load(DATA/'data.npz');x=torch.from_numpy(d['x']);_,ids,_=scope(d,16);preds={k:[] for k in ARMS}
    with torch.no_grad():
        for b in range(0,len(ids),16):
            h=model.features(x[ids[b:b+16]].cuda())
            for arm,state in heads.items():preds[arm]+=affine(state,h).argmax(1).cpu().tolist()
    assert tensor_hash(model.bert.state_dict())==pre['backbone_initial_sha256']
    ev=dict(status='PASS',started_unix=started,barrier_sha256=sha(ROOT/'TRAINING_COMPLETE.json'),indices=ids.tolist(),
        arms={arm:dict(predictions=preds[arm],test=summarize_predictions(ids,preds[arm],d,2)) for arm in ARMS})
    write(ROOT/'EVALUATION.json',ev)
    print('TEST',json.dumps({k:v['test'] for k,v in ev['arms'].items()}),flush=True)

def main():
    pre=guard();setup();d=np.load(DATA/'data.npz')
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    m=FrozenRelationModel().cuda();m.load_state_dict(saved['model']);freeze(m);m.eval();del saved
    assert tensor_hash(m.bert.state_dict())==pre['backbone_initial_sha256']
    bank=build_cache(m,d);del m;torch.cuda.empty_cache()
    results={}
    for arm in ARMS:results[arm]=train(arm,bank,pre);torch.cuda.empty_cache()
    write(ROOT/'TRAINING_COMPLETE.json',dict(status='COMPLETED',finished_unix=time.time(),
        train_result_hashes={arm:sha(ROOT/arm/'TRAIN_RESULT.json') for arm in ARMS},cache_sha256=sha(ROOT/'CACHE.pt')))
    evaluate(pre,results)

if __name__=='__main__':main()
