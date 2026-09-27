"""Only task2 controller runs, both fixed budgets. No continuation/evaluation entrypoint."""
from .cutter import *
from .preflight import install,verify_inputs,identity
from collections import Counter

def history(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]

def distribution(v):
    v=[float(x) for x in v if x is not None]
    return dict(n=len(v),median=float(np.median(v)),mean=float(np.mean(v)),min=min(v),max=max(v)) if v else None

def run(budget,X,y,control):
    folder=ROOT/f'budget{budget}';folder.mkdir(exist_ok=False);yc=y.cpu().numpy();current=np.flatnonzero((yc//7==1)&~control)
    old=np.flatnonzero((yc<7)&control);oldtrain=np.flatnonzero((yc<7)&~control);arrived=np.flatnonzero((yc<14)&~control)
    assert len(old)==70 and np.bincount(yc[old],minlength=7).tolist()==[10]*7
    source=OLD/f'formal/native_{budget}/task1/CHECKPOINT.pt';state=torch.load(source,weights_only=True)
    head=model();head.load_state_dict(state['model']);shadow=copy.deepcopy(head);opt=optimizer(shadow);opt.load_state_dict(state['optimizer'])
    assert opt_steps(opt)==[budget]*4;meter=Meter();sync();start=time.perf_counter()
    bank=Bank(X[old],y[old],old,14,meter);z=bank.exact(params(head));K=bank.counts(z);previous=bank.margins(z)>0
    sync();entry_wall=time.perf_counter()-start;entry=metrics(head,X[current],y[current],14);batches=schedule(current,2,budget)
    common=control_wall=observer=0.;rows=[];backward_examples=0;torch.cuda.reset_peak_memory_stats()
    with gzip.open(folder/'STEPS.jsonl.gz','xt',encoding='utf8') as output:
        for step,batch in enumerate(batches,1):
            assert not control[batch].any() and np.all(yc[batch]//7==1) and not np.intersect1d(batch,old).size
            sync();start=time.perf_counter();before=params(head);loss=native_step(head,shadow,opt,X[batch],y[batch],14);native=params(shadow)
            sync();common+=time.perf_counter()-start;backward_examples+=len(batch)
            moment=opt_hash(opt);assert opt_steps(opt)==[budget+step]*4
            beforehash=phash(before);nativehash=phash(native);sync();start=time.perf_counter()
            executed,decision=project(before,native,bank,K);commit(head,executed);z=bank.exact(params(head));counts=bank.counts(z)
            assert np.all(counts>=K);sync();control_wall+=time.perf_counter()-start
            sync();start=time.perf_counter();assert opt_hash(opt)==moment
            dn=tuple(b-a for a,b in zip(before,native));de=tuple(b-a for a,b in zip(before,executed))
            normn=float(torch.sqrt(sum((v.double()**2).sum() for v in dn)));norme=float(torch.sqrt(sum((v.double()**2).sum() for v in de)))
            dot=float(sum((a.double()*b.double()).sum() for a,b in zip(dn,de)));now=bank.margins(z)>0
            row=dict(step=step,batch_ids=batch.tolist(),current_loss=loss,before_sha256=beforehash,native_sha256=nativehash,
                executed_sha256=phash(params(head)),head_sha256=thash(head.state_dict()),moments_sha256=moment,optimizer_steps=opt_steps(opt),
                endpoint_sha256=ahash(z),counts=counts.tolist(),decision=decision,native_norm=normn,executed_norm=norme,
                norm_ratio=norme/max(normn,1e-300),cosine=dot/(normn*norme) if normn*norme else None,
                correct_to_wrong=int((previous&~now).sum()),wrong_to_correct=int((~previous&now).sum()),
                floor_violations=0,old_ce_backward_examples=0)
            previous=now;rows.append(row);output.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
            sync();observer+=time.perf_counter()-start
            if step%60==0:print('smoke',budget,step,'rollback',sum(r['decision']['rollback'] for r in rows),flush=True)
    end=metrics(head,X[current],y[current],14);train=dict(current=end,old=metrics(head,X[oldtrain],y[oldtrain],14),all=metrics(head,X[arrived],y[arrived],14))
    rounds=[q for r in rows for q in r['decision']['rounds']];events=Counter(q['decision'] for q in rounds);trigger=[r for r in rows if r['decision']['triggered']]
    rollbacks=sum(r['decision']['rollback'] for r in rows);inner=sum(q['solver']['status']=='INNER_LIMIT' for q in rounds)
    safe=events['INNER_LIMIT_ACTUAL_SAFE_ACCEPT'];cont=events['INNER_LIMIT_ACTUAL_UNSAFE_CONTINUE']
    cap_outer=sum(q['solver']['status']=='INNER_LIMIT' and q['decision']=='OUTER_LIMIT_ROLLBACK' for q in rounds)
    assert inner==safe+cont+cap_outer
    repair=(len(trigger)-rollbacks)/len(trigger) if trigger else None;ratio=distribution([r['norm_ratio'] for r in trigger])
    gates=dict(floor_zero=True,unsafe_exposure=len(trigger)>=20,rollback_rate=rollbacks/budget<=.1,
        successful_repair=repair is not None and repair>=.9,new_train_macro=end['macro']>=.5,current_ce=end['ce']<=.9*entry['ce'],
        movement=ratio is not None and ratio['median']>=.1,old_ce_backward_zero=True,state_moment_continuity=True)
    status='PASS' if all(gates.values()) else 'INCONCLUSIVE' if not gates['unsafe_exposure'] else 'FAIL'
    save(folder/'CHECKPOINT.pt',dict(model=head.state_dict(),optimizer=opt.state_dict(),task=2))
    current_cost=read(OLD/'NUMERICS.json')['ce_forward_backward_dense_flops']['16']*budget
    report=dict(status=status,budget=budget,task=2,checkpoint_source=str(source),checkpoint_source_sha256=sha(source),K=K.tolist(),schedule_sha256=ahash(batches),
        unsafe_native_steps=len(trigger),native_safe_steps=budget-len(trigger),ordinary_pass_actual_safe=events['PASS_ACTUAL_SAFE_ACCEPT'],
        inner_limit=inner,inner_limit_actual_safe_accept=safe,inner_limit_actual_unsafe_continue=cont,inner_limit_outer_rollback=cap_outer,
        cap_safe_acceptance_rate=safe/inner if inner else None,final_rollback=rollbacks,successful_repair_rate=repair,
        floor_violations=0,decisions=dict(events),outer_iterations=len(rounds),inner_sweeps=sum(q['solver']['sweeps'] for q in rounds),
        jacobian_calls=meter.calls['jacobian_batches'],jacobian_pairs=meter.calls['jacobian_pairs'],
        endpoint_calls=dict(entry=1,native=budget,projected=len(rounds),commit=budget,total=bank.calls),actual_endpoint_rechecks=len(rounds)+budget,
        norm_ratio_triggered=ratio,cosine_nonzero_triggered=distribution([r['cosine'] for r in trigger]),
        correct_to_wrong=sum(r['correct_to_wrong'] for r in rows),wrong_to_correct=sum(r['wrong_to_correct'] for r in rows),
        entry_current_train=entry,train_metrics=train,gate=gates,independent_audit_status='PENDING',
        current_backward_examples=backward_examples,old_ce_backward_examples=0,old_ce_backward_flops=0,
        common_wall=common,entry_wall=entry_wall,control_wall=control_wall,total_algorithm_wall=common+entry_wall+control_wall,observer_wall=observer,
        region_seconds=dict(meter.seconds),endpoint_flops=meter.flops['endpoint'],cutter_flops=sum(v for k,v in meter.flops.items() if k!='endpoint'),
        current_forward_backward_dense_flops=current_cost,flop_ledger=dict(meter.flops),
        flop_coverage='Dense multiply-add lower bounds plus frozen analytic/coordinate estimates; not full hardware FLOPs.',
        control_bank_reads=bank.rows,peak_cuda_bytes=torch.cuda.max_memory_allocated(),test_files_opened=0,
        checkpoint_sha256=sha(folder/'CHECKPOINT.pt'))
    assert bank.calls==1+budget+len(rounds)+budget
    write(folder/'CHECKS.json',report);print('completed',budget,status,'caps',inner,'safe',safe,'continue',cont,'new train',end['macro'],flush=True)
    return report

def main():
    access=install();setup();guard();verify_inputs();identity();assert read(ROOT/'NUMERICS.json')['status']=='PASS'
    for name,digest in read(ROOT/'FREEZE.json')['sources'].items():assert sha(HERE/name)==digest,name
    write(ROOT/'RUN_START.json',dict(task=2,budgets=[210,630],source_hashes={p.name:sha(p) for p in HERE.glob('*.py')}))
    X,y,c=load();reports=[run(b,X,y,c) for b in (210,630)];verify_inputs();guard()
    write(ROOT/'SMOKE.json',dict(status='PASS' if all(r['status']=='PASS' for r in reports) else 'FAIL',reports=reports,
        stop_after_smoke=True,no_formal_entrypoint=True,access=access.report(),audit_pending=True))

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(ROOT/f'FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
