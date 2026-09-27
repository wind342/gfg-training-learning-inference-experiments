"""Two unchanged-controller continuations; expansion observer cannot steer them."""
from .common import *
from .observer import audit,full_logits,summarize
import traceback


def start_state(budget,checkpoint=None):
    source=checkpoint or PARENT/f'budget{budget}/CHECKPOINT.pt'
    state=torch.load(source,weights_only=True)
    head=model();head.load_state_dict(state['model']);shadow=copy.deepcopy(head)
    opt=optimizer(shadow);opt.load_state_dict(state['optimizer'])
    return head,shadow,opt


def make_bank(X,y,control,task,meter):
    yc=y.cpu().numpy();ids=np.flatnonzero(control&(yc<7*(task-1)))
    assert len(ids)==70*(task-1) and np.bincount(yc[ids]).tolist()==[10]*(7*(task-1))
    bank=Bank(X[ids],y[ids],ids,task*7,meter)
    predictor=Predictor(bank,read(PARENT/'NUMERICS.json')['predictor_dense_flops_per256'])
    current=np.flatnonzero((yc//7==task-1)&~control)
    return bank,predictor,ids,current


def one_update(head,shadow,opt,bank,predictor,X,y,batch,K,previous_z):
    sync();t=time.perf_counter();before=params(head)
    loss=native_step(head,shadow,opt,X[batch],y[batch],bank.seen);native=params(shadow)
    sync();common=time.perf_counter()-t;moments=opt_hash(opt)
    sync();t=time.perf_counter()
    predicted=predictor.predict(before,native);actual=bank.exact(native)
    executed,decision=project(before,native,bank,K,prediction=predicted,native_endpoint=actual)
    commit(head,executed);z=bank.exact(params(head));C=bank.counts(z)
    assert np.all(C>=K);sync();wall=time.perf_counter()-t
    assert opt_hash(opt)==moments
    prediction=measure(bank,previous_z,predicted,actual,K);prediction['components_sha256']=predictor.hashes()
    was=bank.margins(previous_z)>0;now=bank.margins(z)>0
    row=dict(batch_ids=batch.tolist(),current_loss=loss,before_sha256=phash(before),native_sha256=phash(native),
        executed_sha256=phash(executed),head_sha256=thash(head.state_dict()),moments_sha256=moments,optimizer_steps=opt_steps(opt),
        endpoint_sha256=ahash(z),counts=C.tolist(),predictor=prediction,decision=decision,
        correct_to_wrong=int((was&~now).sum()),wrong_to_correct=int((~was&now).sum()),floor_violations=0,
        old_ce_backward_examples=0,common_wall=common,control_wall=wall)
    return row,z


def preflight(X,y,control):
    # Hand-computed three rows: strict steal, exact tie, already wrong.
    z=np.zeros((3,77),dtype=np.float32)
    z[0,:3]=[3.,1.,4.];z[1,:3]=[3.,1.,3.];z[2,:3]=[1.,3.,4.]
    report=summarize(z,np.array([0,0,0]),np.array([10,11,12]),2,3)
    assert report['correct_to_wrong']==2 and report['strict_competition_flips']==1 and report['new_class_ties']==1
    assert report['argmax_correct_to_wrong']==1 and report['wrong_to_correct']==0
    cases=[]
    for budget in (210,630):
        h,s,o=start_state(budget);assert opt_steps(o)==[2*budget]*4
        last=records(PARENT/f'budget{budget}/STEPS.jsonl.gz')[-1]
        assert thash(h.state_dict())==last['head_sha256'] and opt_hash(o)==last['moments_sha256']
        meter=Meter();bank,predictor,ids,current=make_bank(X,y,control,3,meter)
        before=state_identity(h,s,o)
        zz=full_logits(params(h),X[ids]);repeated=full_logits(params(h),X[ids]);assert np.array_equal(zz,repeated)
        summarize(zz,y[ids].cpu().numpy(),ids,14,21)
        assert state_identity(h,s,o)==before
        entry=bank.exact(params(h));assert entry.tobytes()==zz[:,:21].copy().tobytes()
        K=bank.counts(entry);batch=schedule(current,3,budget)[0]
        with_observer,_=one_update(h,s,o,bank,predictor,X,y,batch,K,entry)
        h2,s2,o2=start_state(budget);bank2,pr2,_,_=make_bank(X,y,control,3,Meter())
        entry2=bank2.exact(params(h2));without_observer,_=one_update(h2,s2,o2,bank2,pr2,X,y,batch,bank2.counts(entry2),entry2)
        ignore={'common_wall','control_wall'}
        assert {k:v for k,v in with_observer.items() if k not in ignore}=={k:v for k,v in without_observer.items() if k not in ignore}
        cases.append(dict(budget=budget,task2_checkpoint_matches=True,observer_state_unchanged=True,
            observer_production_logits_byte_identical=True,first_update_observer_on_off_identical=True))
    return dict(status='PASS',strict_tie_and_wrong_cases=True,wrong_to_correct_invariant=True,cases=cases)


def freeze(access):
    assert read(PARENT/'FINAL.json')['status']=='PASS' and read(PARENT/'AUDIT.json')['status']=='PASS'
    hashes={str(p):sha(p) for p in inputs()};parent=read(PARENT/'FREEZE.json');cache=read(OLD/'CACHE.json')
    assert hashes[str(OLD/'TRAIN_FEATURES.pt')]==cache['feature_file_sha256'] and cache['frozen_bert_unchanged']
    for name,digest in parent['sources'].items():assert sha(PARENT_SOURCE/name)==digest,name
    for budget in (210,630):
        checks=read(PARENT/f'budget{budget}/CHECKS.json');assert checks['status']=='PASS'
        assert hashes[str(PARENT/f'budget{budget}/CHECKPOINT.pt')]==checks['checkpoint_sha256']
    for name,record in cache['model_files'].items():
        path=Path('E:/gfg-models/bert-base-uncased-v4')/name
        hashes[str(path)]=sha(path);assert hashes[str(path)]==record['sha256']
    write(ROOT/'FREEZE.json',dict(base=BASE,sources=sources(),inputs=hashes,budgets=[210,630],tasks=list(range(3,12)),
        source_reuse='Unmodified imported task2 production Bank/Predictor/project/native_step/schedule/model/optimizer.',
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),threads=torch.get_num_threads()),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),test_files_opened=0,access=access.report()))


def run_arm(budget,X,y,control):
    head,shadow,opt=start_state(budget);yc=y.cpu().numpy();summaries=[]
    source=PARENT/f'budget{budget}/CHECKPOINT.pt'
    for task in range(3,12):
        assert shutil.disk_usage(ROOT.parent).free>3*1024**3,'Disk safety stop; preserve completed checkpoints'
        assert opt_steps(opt)==[(task-1)*budget]*4
        folder=ROOT/f'budget{budget}/task{task}';folder.mkdir(parents=True,exist_ok=False)
        ids=np.flatnonzero(control&(yc<7*(task-1)))
        expansion=audit(head,shadow,opt,X,y,ids,task,budget,folder)
        print('EXPANSION',budget,f'T{task-1}->T{task}',expansion['correct_before'],'->',expansion['correct_after'],
              'loss',expansion['expansion_loss'],'relations_to_zero',expansion['relations_to_zero'],flush=True)
        meter=Meter();bank,predictor,actual_ids,current=make_bank(X,y,control,task,meter)
        assert np.array_equal(ids,actual_ids)
        z=bank.exact(params(head));K=bank.counts(z)
        assert K.tolist()==expansion['C_after'] # check only; K comes exclusively from production Bank
        batches=schedule(current,task,budget)
        write(folder/'CONFIG.json',dict(task=task,budget=budget,seen=7*task,old_controls=len(ids),bank_ids=ids.tolist(),
            K=K.tolist(),K_semantics='production post-expansion exact counts',schedule_sha256=ahash(batches),
            source_checkpoint=str(source),source_checkpoint_sha256=sha(source),head_entry=thash(head.state_dict()),
            moments_entry=opt_hash(opt),optimizer_steps=opt_steps(opt)))
        common=wall=0.;rollbacks=inner=outer=unsafe=floors=cw=wc=0;preds=[];events=Counter()
        torch.cuda.reset_peak_memory_stats()
        with gzip.open(folder/'STEPS.jsonl.gz','xt',encoding='utf8') as out:
            for step,batch in enumerate(batches,1):
                assert not control[batch].any() and np.all(yc[batch]//7==task-1)
                row,z=one_update(head,shadow,opt,bank,predictor,X,y,batch,K,z)
                assert opt_steps(opt)==[(task-1)*budget+step]*4
                row.update(step=step,task=task,budget=budget)
                out.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                common+=row['common_wall'];wall+=row['control_wall'];preds.append(row['predictor'])
                d=row['decision'];rollbacks+=int(d['rollback']);unsafe+=int(d['triggered'])
                inner+=d['inner_sweeps'];outer+=d['outer_iterations'];events.update(q['decision'] for q in d['rounds'])
                cw+=row['correct_to_wrong'];wc+=row['wrong_to_correct']
                if step%60==0:
                    out.flush();print('TRAIN',budget,task,step,'rollback',rollbacks,'wall',round(common+wall,2),flush=True)
        save(folder/'CHECKPOINT.pt',dict(model=head.state_dict(),optimizer=opt.state_dict(),task=task))
        summary=dict(status='COMPLETED',budget=budget,task=task,updates=budget,K=K.tolist(),final_counts=bank.counts(z).tolist(),
            unsafe_native=unsafe,rollbacks=rollbacks,inner_sweeps=inner,outer_iterations=outer,decisions=dict(events),
            floor_violations=floors,correct_to_wrong=cw,wrong_to_correct=wc,predictor=aggregate(preds),
            head_final=thash(head.state_dict()),moments_final=opt_hash(opt),optimizer_steps=opt_steps(opt),
            common_wall=common,control_wall=wall,algorithm_wall=common+wall,region_seconds=dict(meter.seconds),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),old_ce_backward_examples=0,test_files_opened=0,
            checkpoint_sha256=sha(folder/'CHECKPOINT.pt'),expansion_sha256=sha(folder/'EXPANSION.json'))
        write(folder/'CHECKS.json',summary);summaries.append(summary);source=folder/'CHECKPOINT.pt'
        print('TASK COMPLETE',budget,task,'floor0','rollbacks',rollbacks,flush=True)
    return summaries


def main():
    setup();guard();ROOT.mkdir(exist_ok=False);access=Access();freeze(access)
    X,y,control=load();assert y.min()==0 and y.max()==76
    write(ROOT/'PREFLIGHT.json',preflight(X,y,control));setup()
    reports={str(b):run_arm(b,X,y,control) for b in (210,630)}
    verify_freeze()
    write(ROOT/'TRAINING_COMPLETE.json',dict(status='COMPLETED',arms=reports,updates=7560,access=access.report(),test_files_opened=0))
    print('COMPLETE: both arms through task11; independent audit pending',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        if ROOT.exists():write(ROOT/f'FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
