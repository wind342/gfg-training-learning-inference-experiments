"""Two continuously maintained heads; frozen shared backbone, fixed protocol."""
import gc
import traceback
from .core import *

@torch.no_grad()
def cache_controls(m,d):
    original=torch.load(COVER_ROOT/'CACHE.pt',map_location='cpu',weights_only=False)
    ids=clean_validation(d);positions={int(v):j for j,v in enumerate(ids)}
    features=torch.empty(len(ids),1536);existing={int(i):h for i,h in zip(original['indices'],original['features'])}
    missing=[]
    for i in ids:
        if int(i) in existing:features[positions[int(i)]]=existing[int(i)]
        else:missing.append(int(i))
    x=torch.from_numpy(d['x']);started=time.perf_counter();m.eval()
    for begin in range(0,len(missing),16):
        batch=missing[begin:begin+16];values=m.features(x[batch].cuda()).cpu()
        for i,h in zip(batch,values):features[positions[i]]=h
        if begin%2048==0:print('CACHE',begin,'/',len(missing),flush=True)
    assert torch.isfinite(features).all()
    checkids,_=task_bank(d,2)
    assert checkids.tolist()==original['indices']
    assert torch.equal(features[[positions[int(i)] for i in checkids]],original['features'])
    save_tensor(ROOT/'CONTROL_CACHE.pt',dict(indices=ids.tolist(),features=features))
    write(ROOT/'CACHE_MANIFEST.json',dict(sha256=sha(ROOT/'CONTROL_CACHE.pt'),parent_sha256=sha(COVER_ROOT/'CACHE.pt'),
        rows=len(ids),reused_rows=len(existing),new_rows=len(missing),seconds=time.perf_counter()-started,
        backbone_sha256=tensor_hash(m.bert.state_dict()),features_sha256=tensor_hash({'H':features})))
    return features,positions

def train_task(m,arm,state,task,d,h,labels,ids,n,pre,parent):
    folder=ROOT/arm/f'task{task}';folder.mkdir(parents=True,exist_ok=False)
    head,shadow,opt=state;m.head=head;m.seen=task*8;m.train()
    assert opt_counter(opt)==210*(task-1);head_start=head_hash(head);mom_start=moments_hash(opt)
    initial_opt=copy.deepcopy(opt.state_dict());versions=[p._version for p in m.bert.parameters()]
    boundary=transition(head,h,labels,n,task) if arm==ARMS[1] else None
    write(folder/'CONFIG.json',dict(base=BASE,arm=arm,task=task,lr=LR,seed=2021,updates=210,batch_size=16,
        source_hashes=pre['source_hashes'],bank_ids=ids.tolist() if arm==ARMS[1] else [],old_count=n if arm==ARMS[1] else 0,
        head_start=head_start,moments_start=mom_start,optimizer_step_start=opt_counter(opt),
        checkpoint_sha256=CKSHA,control_cache_sha256=sha(ROOT/'CONTROL_CACHE.pt'),transition=boundary))
    heads=[cpu_state(snapshot(head))];natives=[];projected=[];grads=[];rows=[]
    x=torch.from_numpy(d['x']);truth=torch.from_numpy(d['y']);hh=tensor_hash({'H':h}) if arm==ARMS[1] else None
    sync(h);torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with (folder/'STEPS.jsonl').open('x',encoding='utf8') as f:
        for step,batch in enumerate(order(d,task),1):
            tick=time.perf_counter();global_step=210*(task-1)+step;seed=2021*100000+global_step
            m.train();assert all(not z.training for z in m.bert.modules());head.zero_grad(set_to_none=True)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            with torch.no_grad():features=m.features(x[batch].cuda())
            loss=nn.functional.cross_entropy(head(features)[:,:task*8],truth[batch].cuda());loss.backward()
            gradnorm=nn.utils.clip_grad_norm_(head.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(gradnorm)
            grads.append({name:p.grad.detach().cpu().clone() for name,p in head.named_parameters()})
            propose(head,shadow,opt);before=snapshot(head);native=snapshot(shadow)
            assert opt_counter(opt)==global_step and opt.param_groups[0]['lr']==LR
            mh=moments_hash(opt);rr=rng();report=None
            if arm==ARMS[1]:
                executed,candidate,report=hildreth(before,native,h,labels,n,task*8)
                assert equal(rr,rng()) and moments_hash(opt)==mh
            else:executed=candidate=native
            with torch.no_grad():
                for name,p in head.named_parameters():p.copy_(executed[name])
            assert all(p.grad is None and not p.requires_grad for p in m.bert.parameters())
            assert versions==[p._version for p in m.bert.parameters()]
            row=dict(task=task,step=step,global_step=global_step,seed=seed,current_ids=batch.tolist(),
                loss=float(loss),gradient_norm_before_clip=float(gradnorm),head_before_sha256=tensor_hash(before),
                native_sha256=tensor_hash(native),head_after_sha256=head_hash(head),optimizer_sha256=mh,
                optimizer_step=opt_counter(opt),moments_unmodified_by_projection=True,bert_frozen_eval=True)
            if report is not None:
                actual=margins(logits(head,h,task*8),labels).cpu().tolist()
                assert actual==(report['baseline'] if report['skip'] else report['projected'])
                assert not any(a>0 and b<=0 for a,b in zip(report['baseline'][:n],actual[:n]))
                if report['native_violating'] and not report['skip']:assert all(actual[i]>=EPS for i in report['eligible'])
                report['executed']=actual
                row.update(report=report,projected_sha256=tensor_hash(candidate),
                    projected_geometry=geometry(before,native,candidate),executed_geometry=geometry(before,native,executed))
                projected.append(cpu_state(candidate))
                if task==2:
                    assert equal(cpu_state(before),parent['heads'][step-1])
                    assert equal(cpu_state(native),parent['natives'][step-1])
                    assert equal(cpu_state(candidate),parent['projected'][step-1])
                    assert equal(cpu_state(executed),parent['heads'][step])
            heads.append(cpu_state(executed));natives.append(cpu_state(native));sync(h)
            row['seconds']=time.perf_counter()-tick;rows.append(row)
            f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');f.flush()
            if step==1 or step%30==0:
                print('TRAIN',arm,'task',task,'step',step,'skips',sum(r.get('report',{}).get('skip',False) for r in rows),
                    'sweeps',report['sweep_count'] if report else 0,flush=True)
    sync(h);wall=time.perf_counter()-started;peak=torch.cuda.max_memory_allocated()
    assert tensor_hash(m.bert.state_dict())==pre['backbone_initial_sha256']
    if hh:assert tensor_hash({'H':h})==hh
    save_tensor(folder/'AUDIT.pt',dict(heads=heads,natives=natives,projected=projected,clipped_gradients=grads,
        initial_optimizer=initial_opt,final_optimizer=copy.deepcopy(opt.state_dict())))
    r=dict(status='COMPLETED',arm=arm,task=task,steps=210,training_wall=wall,peak_cuda_bytes=peak,
        head_start=head_start,moments_start=mom_start,head_end=head_hash(head),moments_end=moments_hash(opt),
        optimizer_step=opt_counter(opt),backbone_sha256=tensor_hash(m.bert.state_dict()),no_test_in_training=True,
        transition=boundary,summary=summary(rows,n) if arm==ARMS[1] else None,
        audit_sha256=sha(folder/'AUDIT.pt'),log_sha256=sha(folder/'STEPS.jsonl'),config_sha256=sha(folder/'CONFIG.json'),
        finished_unix=time.time())
    write(folder/'TRAIN_RESULT.json',r);return r

@torch.no_grad()
def evaluate(m,states,task,d,pre):
    folder=ROOT/f'evaluation_task{task}';folder.mkdir(exist_ok=False)
    barriers={a:sha(ROOT/a/f'task{task}'/'TRAIN_RESULT.json') for a in ARMS}
    write(folder/'BARRIER.json',dict(training_results=barriers,created_unix=time.time()))
    head_ids={a:head_hash(s[0]) for a,s in states.items()};moment_ids={a:moments_hash(s[2]) for a,s in states.items()}
    rr=rng();was=m.training;active=m.head;m.eval();m.seen=task*8
    ids=scope(d,task*8)[1];x=torch.from_numpy(d['x']);pred={a:[] for a in ARMS};start=time.perf_counter()
    for begin in range(0,len(ids),16):
        batch=ids[begin:begin+16];features=m.features(x[batch].cuda())
        for arm,(head,_,_) in states.items():pred[arm]+=head(features)[:,:task*8].argmax(1).cpu().tolist()
    m.head=active;m.train(was);torch.set_rng_state(rr[0]);torch.cuda.set_rng_state_all(rr[1])
    assert head_ids=={a:head_hash(s[0]) for a,s in states.items()}
    assert moment_ids=={a:moments_hash(s[2]) for a,s in states.items()} and equal(rr,rng())
    assert tensor_hash(m.bert.state_dict())==pre['backbone_initial_sha256']
    metrics={}
    for arm in ARMS:
        score=summarize_predictions(ids,pred[arm],d,task)
        score['previous_task_macro']=float(np.mean([score['per_relation'][str(g)] for g in range(8*(task-2),8*(task-1))]))
        metrics[arm]=dict(indices=ids.tolist(),predictions=pred[arm],metrics=score,head_sha256=head_ids[arm])
    write(folder/'EVALUATION.json',dict(status='PASS',task=task,barrier_sha256=sha(folder/'BARRIER.json'),
        finished_unix=time.time(),seconds=time.perf_counter()-start,arms=metrics,readonly=True))
    print('EVALUATED',task,{a:{k:metrics[a]['metrics']['macro_'+k] for k in ('old','new','all')} for a in ARMS},flush=True)

def main():
    pre=guard();assert not (ROOT/'EXECUTION.json').exists();write(ROOT/'EXECUTION.json',dict(status='STARTED',base=BASE,started_unix=time.time()))
    try:
        setup();d=np.load(DATA/'data.npz');saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        m=FrozenRelationModel().cuda();m.load_state_dict(saved['model']);freeze(m)
        assert tensor_hash(m.bert.state_dict())==pre['backbone_initial_sha256']
        states={};original=m.head
        for arm in ARMS:
            m.head=copy.deepcopy(original);shadow,opt=make_candidate(m,saved['optimizer']);set_lr(opt,LR)
            states[arm]=(m.head,shadow,opt)
        del saved,original
        cache,positions=cache_controls(m,d)
        parent=torch.load(PARENT_ROOT/'140_controls_hildreth_roundoff'/'AUDIT.pt',map_location='cpu',weights_only=False)
        for task in range(2,11):
            guard(False);ids,n=task_bank(d,task)
            h=cache[[positions[int(i)] for i in ids]].cuda();labels=torch.from_numpy(d['y'][ids]).cuda()
            for arm in ARMS:train_task(m,arm,states[arm],task,d,h,labels,ids,n,pre,parent)
            evaluate(m,states,task,d,pre)
            if task==2:del parent;parent=None
            del h,labels;gc.collect()
        guard();write(ROOT/'COMPLETE.json',dict(status='COMPLETED',tasks=list(range(2,11)),arms=list(ARMS),finished_unix=time.time()))
    except Exception:
        write(ROOT/'FAILURE.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise

if __name__=='__main__':main()
