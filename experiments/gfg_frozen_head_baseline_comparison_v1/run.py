"""Fixed comparison with compact checkpoints; test is evaluator-only."""
from .core import *
from .statistics import Cumulative
from .tests import tests
import traceback

def train_task(arm,state,rng,F,d,task,output=None):
    method,epochs=METHODS[arm];head,shadow,opt=state
    folder=Path(output) if output else ROOT/arm/f'task{task}';folder.mkdir(parents=True,exist_ok=False)
    h,y,ids,n=bank(d,F,task);labels=d['y'][ids[:n]];seen=task*8
    bm,K=monitor(head,h,y,n,seen)
    before_head=head_hash(head);before_moments=moments_hash(opt)
    write(folder/'CONFIG.json',dict(task=task,method=method,epochs=epochs,K=K.tolist(),bank_ids=ids[:n].tolist(),
        head_start=before_head,moments_start=before_moments,optimizer_step=old.opt_counter(opt),rng_start=rng.bit_generator.state))
    rows=[];total_old_reads=0;total_old_backward=0;algorithm=0.;observer=0.;audit=0.;sync();wallstart=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with gzip.open(folder/'STEPS.jsonl.gz','wt',encoding='utf8') as log:
        for epoch in range(1,epochs+1):
            for local,current in enumerate(order(d,task),1):
                sync();start=time.perf_counter();selected=[];certificate=None;projection=None
                if method=='er':
                    selected=rng.choice(ids[:n],16,replace=False).tolist()
                    inputs=np.concatenate((current,selected));g,loss=ce_grad(head,F[inputs],torch.from_numpy(d['y'][inputs]).cuda(),seen)
                    old_reads=16;old_backward=16
                else:
                    refs=[]
                    if method in ('agem','gem'):
                        pools=[ids[:n]] if method=='agem' else [ids[:n][labels//8==j] for j in range(task-1)]
                        for pool in pools:
                            sample=rng.choice(pool,16,replace=False).tolist();selected+=sample
                            rg,_=ce_grad(head,F[sample],torch.from_numpy(d['y'][sample]).cuda(),seen);refs.append(rg)
                    g,loss=ce_grad(head,F[current],torch.from_numpy(d['y'][current]).cuda(),seen)
                    if refs:g,certificate=project_gradient(g,refs,method);assign_grad(head,g)
                    old_reads=len(selected);old_backward=len(selected)
                norm=nn.utils.clip_grad_norm_(head.parameters(),1.)
                assert torch.isfinite(norm) and np.isfinite(loss)
                before=p.snapshot(head) if method=='v2' else None
                propose(head,shadow,opt)
                if method=='v2':
                    native=p.snapshot(shadow)
                    executed,_,projection=floor_projection(before,native,h,y,n,K,ids[:n],seen=seen)
                    with torch.no_grad():
                        for name,q in head.named_parameters():q.copy_(executed[name])
                        actual=margins(head(h)[:,:seen],y).cpu().tolist()
                    assert actual==projection['executed']
                    # baseline+native+one candidate scan/sweep+actual commit recheck.
                    old_reads=n*(3+projection['sweep_count'])
                else:commit(head,shadow)
                sync();algo=time.perf_counter()-start;algorithm+=algo
                start=time.perf_counter();am,C=monitor(head,h,y,n,seen);sync();obs=time.perf_counter()-start;observer+=obs
                D=np.maximum(K-C,0);lost=int(((bm>0)&(am<=0)).sum());gained=int(((bm<=0)&(am>0)).sum())
                if method=='v2':assert not D.any()
                start=time.perf_counter()
                row=dict(task=task,epoch=epoch,local=local,current_ids=current.tolist(),old_sample_ids=selected,
                    loss=loss,gradient_norm_before_clip=float(norm),counts=C.tolist(),deficits=D.tolist(),
                    correct_to_wrong=lost,wrong_to_correct=gained,old_feature_reads=old_reads,old_backward_examples=old_backward,
                    algorithm_seconds=algo,observer_seconds=obs,head_sha256=head_hash(head),moments_sha256=moments_hash(opt),
                    optimizer_step=old.opt_counter(opt),gradient_projection=certificate)
                if projection:
                    row['functional_projection']={k:projection[k] for k in ('projection_triggered','skip','sweep_count','final_active',
                        'final_duals','selected_controls','native_counts','executed_counts','floor_violations','roundoff_budget_sha256','sweeps')}
                    row['geometry']=p.geometry(before,native,executed)
                log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                audit+=time.perf_counter()-start;rows.append(row);bm=am;total_old_reads+=old_reads;total_old_backward+=old_backward
                if local%210==0:print('TRAIN',arm,'task',task,'epoch',epoch,'floor-deficit',int(D.sum()),flush=True)
    sync();wall=time.perf_counter()-wallstart
    save(folder/'CHECKPOINT.pt',dict(head=p.cpu_state(p.snapshot(head)),optimizer=opt.state_dict(),rng=rng.bit_generator.state))
    result=dict(status='COMPLETED',arm=arm,task=task,steps=len(rows),K=K.tolist(),end_counts=C.tolist(),
        algorithm_wall=algorithm,observer_wall=observer,audit_wall=audit,loop_wall=wall,
        floor_violating_steps=sum(any(v>0 for v in r['deficits']) for r in rows),
        relation_floor_violations=sum(sum(v>0 for v in r['deficits']) for r in rows),
        max_total_deficit=max(sum(r['deficits']) for r in rows),
        correct_to_wrong=sum(r['correct_to_wrong'] for r in rows),wrong_to_correct=sum(r['wrong_to_correct'] for r in rows),
        old_feature_reads=total_old_reads,old_backward_examples=total_old_backward,observer_old_reads=n*len(rows),
        projection_steps=sum(r.get('functional_projection',{}).get('projection_triggered',False) for r in rows),
        skips=sum(r.get('functional_projection',{}).get('skip',False) for r in rows),
        gradient_projection_steps=sum(bool(r['gradient_projection'] and r['gradient_projection']['triggered']) for r in rows),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),head_end=head_hash(head),moments_end=moments_hash(opt),
        optimizer_step=old.opt_counter(opt),checkpoint_sha256=sha(folder/'CHECKPOINT.pt'),log_sha256=sha(folder/'STEPS.jsonl.gz'))
    write(folder/'TRAIN_RESULT.json',result)
    return result

@torch.no_grad()
def evaluate(states,stats,task,d):
    # Load held-out features ONLY in this post-training evaluator.
    test=torch.load(ROOT/'TEST_FEATURES.pt',map_location='cpu',weights_only=False)
    ids=scope(d,task*8)[1];assert test['indices'][:len(ids)]==ids.tolist()
    F=test['features'][:len(ids)].cuda();arms={};sync();start=time.perf_counter()
    for arm,(head,_,opt) in states.items():
        identity=head_hash(head);moment=moments_hash(opt)
        pred=head(F)[:,:task*8].argmax(1).cpu().tolist()
        assert head_hash(head)==identity and moments_hash(opt)==moment
        arms[arm]=dict(predictions=pred,metrics=output_metrics(d,pred,task),head_sha256=identity)
    for arm,model in stats.items():
        pred=[]
        for begin in range(0,len(F),256):pred+=model.logits(F[begin:begin+256])[:,:task*8].argmax(1).cpu().tolist()
        arms[arm]=dict(predictions=pred,metrics=output_metrics(d,pred,task))
    sync();write(ROOT/f'EVALUATION_TASK{task}.json',dict(task=task,indices=ids.tolist(),arms=arms,wall=time.perf_counter()-start,readonly=True))
    print('EVAL task',task,{a:round(v['metrics']['macro_all']*100,3) for a,v in arms.items()},flush=True)

def main():
    p.setup();guards();cache=json.loads((ROOT/'CACHE.json').read_text());assert cache['status']=='PASS'
    assert all(sha(ROOT/name)==h for name,h in cache['artifacts'].items())
    check=tests();p.setup()
    pre=dict(status='PASS',base=BASE,sources=sources(),tests=check,cache_sha256=sha(ROOT/'CACHE.json'),runtime=runtime(),created_unix=time.time())
    write(ROOT/'PREFLIGHT.json',pre)
    d=np.load(DATA/'data.npz');features=torch.load(ROOT/'FEATURES.pt',map_location='cpu',weights_only=False)
    F=features['features'].cuda();initial=torch.load(ROOT/'INITIAL.pt',weights_only=False,map_location='cpu')
    states={arm:make_state(initial) for arm in METHODS};randoms={arm:np.random.default_rng(2021+9000000) for arm in METHODS}
    stats={kind:Cumulative(kind) for kind in ('ridge','ranpac_rp')}
    for arm,model in stats.items():
        result=model.train_task(F,d,1);model.save_task(1);write(ROOT/arm/'task1'/'TRAIN_RESULT.json',result)
    evaluate(states,stats,1,d)
    for task in range(2,11):
        guards(pre)
        for arm in METHODS:train_task(arm,states[arm],randoms[arm],F,d,task)
        for arm,model in stats.items():
            folder=ROOT/arm/f'task{task}';folder.mkdir(parents=True,exist_ok=False)
            ids,n=old.task_bank(d,task);h=F[ids[:n]];y=torch.from_numpy(d['y'][ids[:n]]).cuda()
            sync();obs_start=time.perf_counter()
            with torch.no_grad():entry=margins(model.logits(h)[:,:task*8],y).cpu().numpy()
            K=counts(entry,d['y'][ids[:n]],8*(task-1));sync();obs=time.perf_counter()-obs_start
            result=model.train_task(F,d,task);sync();obs_start=time.perf_counter()
            with torch.no_grad():end=margins(model.logits(h)[:,:task*8],y).cpu().numpy()
            C=counts(end,d['y'][ids[:n]],8*(task-1));result.update(K=K.tolist(),end_counts=C.tolist(),
                deficits=np.maximum(K-C,0).tolist(),diagnostic_old_reads=2*n,observer_wall=obs+time.perf_counter()-obs_start)
            model.save_task(task);write(folder/'TRAIN_RESULT.json',result)
        evaluate(states,stats,task,d)
    for model in stats.values():model.save_final()
    guards(pre);write(ROOT/'COMPLETED.json',dict(status='COMPLETED',finished_unix=time.time(),sources=pre['sources']))

if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/f'FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
