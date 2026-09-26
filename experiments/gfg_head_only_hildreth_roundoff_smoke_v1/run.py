"""One fixed trajectory, followed by read-only train/test evaluation."""
import traceback
from .core import *

def train(pre):
    setup();out=ROOT/ARM;out.mkdir(exist_ok=False);rows=[]
    write(out/'CONFIG.json',dict(base=BASE,arm=ARM,seed=2021,lr=LR,updates=210,batch_size=16,old_controls=1120,
        replay=0,max_sweeps=MAX_SWEEPS,epsilon=EPS,roundoff_rule='gamma_(augmented_dimension+2) times receiving/native absolute-product envelope; pair sums',
        numerical_check_sha256=sha(ROOT/'NUMERICAL_CHECK.json'),source_hashes=sources(),checkpoint_sha256=CKSHA,data_sha256=DATA_SHA))
    try:
        d=np.load(DATA/'data.npz');x=torch.from_numpy(d['x']);y=torch.from_numpy(d['y']);ids=bank_ids(d,140)
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        m=FrozenRelationModel().cuda();m.load_state_dict(saved['model']);freeze(m);m.seen=16
        shadow,opt=make_candidate(m,saved['optimizer']);del saved
        assert head_hash(m.head)==pre['head_initial_sha256'] and moments_hash(opt)==pre['head_optimizer_initial_sha256']
        set_lr(opt,LR);initialopt=copy.deepcopy(opt.state_dict());bh=tensor_hash(m.bert.state_dict())
        assert bh==pre['backbone_initial_sha256'];versions=[p._version for p in m.bert.parameters()]
        cache=torch.load(COVER_ROOT/'CACHE.pt',map_location='cpu',weights_only=False);assert cache['indices']==ids.tolist()
        h=cache['features'].cuda();labels=y[ids].cuda();hh=tensor_hash({'H':h});hv=h._version
        history=[cpu_state(snapshot(m.head))];natives=[];projected=[]
        sync(h);started=time.perf_counter()
        with (out/'STEPS.jsonl').open('x',encoding='utf8') as f:
            for step,batch in enumerate(order(d,2),1):
                tick=time.perf_counter();seed=2021*100000+210+step;current=batch.tolist()
                m.train();assert all(not z.training for z in m.bert.modules()) and opt.param_groups[0]['lr']==LR
                m.head.zero_grad(set_to_none=True);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with torch.no_grad():features=m.features(x[current].cuda())
                loss=nn.functional.cross_entropy(m.head(features)[:,:16],y[current].cuda());loss.backward()
                gn=nn.utils.clip_grad_norm_(m.head.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(gn)
                propose(m.head,shadow,opt);before=snapshot(m.head);native=snapshot(shadow)
                rh=rng();mh=moments_hash(opt);headbefore=head_hash(m.head)
                executed,candidate,report=hildreth(before,native,h,labels)
                assert equal(rh,rng()) and mh==moments_hash(opt) and headbefore==head_hash(m.head)
                assert hv==h._version and versions==[p._version for p in m.bert.parameters()]
                assert all(p.grad is None and not p.requires_grad for p in m.bert.parameters())
                with torch.no_grad():
                    for k,p in m.head.named_parameters():p.copy_(executed[k])
                actual=margins(logits(m.head,h),labels).cpu().tolist()
                assert actual==(report['baseline'] if report['skip'] else report['projected'])
                assert not any(a>0 and b<=0 for a,b in zip(report['baseline'][:1120],actual[:1120]))
                if report['native_violating'] and not report['skip']:
                    assert all(actual[i]>=EPS for i in report['eligible'])
                report['executed']=actual
                history.append(cpu_state(executed));natives.append(cpu_state(native));projected.append(cpu_state(candidate))
                sync(h);row=dict(step=step,current_ids=current,seed=seed,lr=LR,old_replay=0,report=report,
                    projected_geometry=geometry(before,native,candidate),executed_geometry=geometry(before,native,executed),
                    head_before_sha256=headbefore,head_after_sha256=head_hash(m.head),native_sha256=tensor_hash(native),
                    projected_sha256=tensor_hash(candidate),optimizer_sha256=mh,moments_unmodified_by_projection=True,
                    bert_frozen_eval=True,seconds=time.perf_counter()-tick)
                rows.append(row);f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');f.flush()
                if step==1 or step%10==0:print('STEP',step,report['action'],'sweeps',report['sweep_count'],'active',len(report['final_active']),'skips',sum(r['report']['skip'] for r in rows),flush=True)
        sync(h);wall=time.perf_counter()-started
        assert bh==tensor_hash(m.bert.state_dict()) and hh==tensor_hash({'H':h})
        with (out/'AUDIT.pt').open('xb') as f:torch.save(dict(heads=history,natives=natives,projected=projected,
            control_ids=ids.tolist(),initial_optimizer=initialopt,final_optimizer=opt.state_dict()),f)
        result=dict(status='COMPLETED',arm=ARM,steps=210,no_test_inference=True,finished_unix=time.time(),training_wall=wall,
            summary=summarize(rows),head_sha256=head_hash(m.head),optimizer_sha256=moments_hash(opt),backbone_sha256=bh,cache_sha256=hh,
            audit_sha256=sha(out/'AUDIT.pt'),log_sha256=sha(out/'STEPS.jsonl'),config_sha256=sha(out/'CONFIG.json'))
    except Exception:result=dict(status='FAILED',steps=len(rows),traceback=traceback.format_exc())
    write(out/'TRAIN_RESULT.json',result)
    if result['status']!='COMPLETED':raise RuntimeError(result['traceback'])
    return result

def evaluate(pre,r):
    guard();assert r['status']=='COMPLETED' and r['no_test_inference']
    barrier=json.loads((ROOT/'TRAINING_COMPLETE.json').read_text())
    assert barrier['train_result_sha256']==sha(ROOT/ARM/'TRAIN_RESULT.json') and sha(ROOT/ARM/'AUDIT.pt')==r['audit_sha256']
    started=time.time();assert started>=r['finished_unix'];setup()
    a=torch.load(ROOT/ARM/'AUDIT.pt',map_location='cpu',weights_only=False)
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    m=FrozenRelationModel().cuda();m.load_state_dict(saved['model']);m.head.load_state_dict(a['heads'][-1]);freeze(m);m.eval();m.seen=16;del saved,a
    d=np.load(DATA/'data.npz');x=torch.from_numpy(d['x']);trainids=np.flatnonzero((d['split']==0)&(d['y']>=8)&(d['y']<16))
    ids=scope(d,16)[1];results={}
    with torch.no_grad():
        for name,selected in [('training',trainids),('test',ids)]:
            pred=[]
            for b in range(0,len(selected),16):pred+=m(x[selected[b:b+16]].cuda())[0][:,-1].argmax(1).cpu().tolist()
            results[name]=dict(indices=selected.tolist(),predictions=pred)
            if name=='training':results[name]['accuracy16']=float((np.array(pred)==d['y'][selected]).mean())
            else:results[name]['macro']=summarize_predictions(selected,pred,d,2)
    assert head_hash(m.head)==r['head_sha256'] and tensor_hash(m.bert.state_dict())==pre['backbone_initial_sha256']
    write(ROOT/'EVALUATION.json',dict(status='PASS',started_unix=started,barrier_sha256=sha(ROOT/'TRAINING_COMPLETE.json'),result=results))
    print('EVALUATION',results['training']['accuracy16'],results['test']['macro'],flush=True)

def main():
    pre=guard();r=train(pre)
    write(ROOT/'TRAINING_COMPLETE.json',dict(status='COMPLETED',train_result_sha256=sha(ROOT/ARM/'TRAIN_RESULT.json'),finished_unix=time.time()))
    evaluate(pre,r)

if __name__=='__main__':main()
