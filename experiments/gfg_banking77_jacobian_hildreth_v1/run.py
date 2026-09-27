"""Prospectively gated task2 runs, with immutable step evidence and replay."""
from .cutter import *
from collections import Counter

def history(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]

def run(kind,budget,phase='smoke',task=2):
    setup();guard();folder=ROOT/phase/f'{kind}_{budget}'/f'task{task}';folder.mkdir(parents=True,exist_ok=False)
    X,y,controls=load();yc=y.cpu().numpy();seen=task*7
    source=OLD/'formal'/f'native_{budget}'/'task1/CHECKPOINT.pt' if task==2 else ROOT/'formal'/f'jacobian_{budget}'/f'task{task-1}/CHECKPOINT.pt'
    if task==3:source=ROOT/'smoke'/f'jacobian_{budget}'/'task2/CHECKPOINT.pt'
    state=torch.load(source,weights_only=True);head=model();head.load_state_dict(state['model']);shadow=copy.deepcopy(head)
    opt=optimizer(shadow);opt.load_state_dict(state['optimizer']);assert opt_steps(opt)==[budget*(task-1)]*4
    current=np.flatnonzero((yc//7==task-1)&~controls);oldids=np.flatnonzero((yc<seen-7)&controls)
    oldtrain=np.flatnonzero((yc<seen-7)&~controls);arrived=np.flatnonzero((yc<seen)&~controls)
    batches=schedule(current,task,budget);meter=Meter();sync();clock=time.perf_counter()
    bank=Bank(X[oldids],y[oldids],oldids,seen,meter);entry=bank.exact(params(head));K=bank.counts(entry)
    previous=bank.counts(entry[:,:seen-7]);entry_wall=time.perf_counter()-clock
    entry_metric=metrics(head,X[current],y[current],seen);previous_correct=bank.margins(entry)>0
    reference=history(OLD/'formal'/f'native_{budget}'/f'task{task}/STEPS.jsonl.gz') if kind=='native' else None
    replay=history(ROOT/'smoke'/f'{kind}_{budget}'/f'task{task}/STEPS.jsonl.gz') if phase=='replay' else None
    rows=[];common_wall=control_wall=observer_wall=0.;failures=Counter();torch.cuda.reset_peak_memory_stats()
    with gzip.open(folder/'STEPS.jsonl.gz','xt',encoding='utf8') as stream:
        for step,batch in enumerate(batches,1):
            sync();start=time.perf_counter();before=params(head)
            loss=native_step(head,shadow,opt,X[batch],y[batch],seen);native=params(shadow)
            sync();common_wall+=time.perf_counter()-start
            mh=opt_hash(opt);assert opt_steps(opt)==[budget*(task-1)+step]*4
            sync();start=time.perf_counter()
            if kind=='jacobian':
                assert np.all(np.bincount(bank.labels[previous_correct],minlength=seen-7)>=K)
                executed,decision=project(before,native,bank,K);commit(head,executed)
                z=bank.exact(params(head));C=bank.counts(z);violation=int(np.maximum(K-C,0).sum())
                assert violation==0,(task,step,C,K)
                sync();control_wall+=time.perf_counter()-start
            else:
                commit(head,native);executed=native;decision={};violation=None;C=None;z=None
                sync();common_wall+=time.perf_counter()-start
            sync();start=time.perf_counter();assert opt_hash(opt)==mh
            dn=tuple(b-a for a,b in zip(before,native));de=tuple(b-a for a,b in zip(before,executed))
            normn=float(torch.sqrt(sum((v.double()**2).sum() for v in dn)))
            norme=float(torch.sqrt(sum((v.double()**2).sum() for v in de)))
            dot=float(sum((a.double()*b.double()).sum() for a,b in zip(dn,de)))
            changed={}
            if z is not None:
                now=bank.margins(z)>0
                changed=dict(correct_to_wrong=int((previous_correct&~now).sum()),wrong_to_correct=int((~previous_correct&now).sum()))
                previous_correct=now
            row=dict(step=step,loss=loss,head_sha256=thash(head.state_dict()),moments_sha256=mh,
                counts=C.tolist() if C is not None else None,floor_violations=violation,decision=decision,
                native_norm=normn,executed_norm=norme,norm_ratio=norme/max(normn,1e-300),
                cosine=dot/(normn*norme) if normn*norme>0 else None,**changed)
            if reference:
                assert row['head_sha256']==reference[step-1]['head_sha256'],('native_head',budget,step)
                assert mh==reference[step-1]['moments_sha256'],('native_moments',budget,step)
            if replay:assert row==replay[step-1],('replay_mismatch',budget,step)
            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');rows.append(row)
            if decision.get('failure_reason'):failures[decision['failure_reason']]+=1
            sync();observer_wall+=time.perf_counter()-start
            if step%30==0:print(phase,kind,budget,'task',task,'step',step,'failures',dict(failures),flush=True)
    final=metrics(head,X[current],y[current],seen)
    trainmetrics=dict(current=final,old=metrics(head,X[oldtrain],y[oldtrain],seen),all=metrics(head,X[arrived],y[arrived],seen))
    triggered=[r for r in rows if r['decision'].get('triggered')];rolled=[r for r in rows if r['decision'].get('rollback')]
    ratio=float(np.median([r['norm_ratio'] for r in triggered])) if triggered else None
    repair=(len(triggered)-len(rolled))/len(triggered) if triggered else None
    gates=dict(floor_zero=all(r['floor_violations']==0 for r in rows),exposure=len(triggered)>=20,
        repair_rate=repair is not None and repair>=.9,rollback_rate=len(rolled)/budget<=.1,
        current_macro=final['macro']>=.5,current_ce=final['ce']<=.9*entry_metric['ce'],
        movement=ratio is not None and ratio>=.1)
    verdict=('PASS' if all(gates.values()) else 'INCONCLUSIVE' if not gates['exposure'] else 'FAIL') if kind=='jacobian' else 'PASS'
    save(folder/'CHECKPOINT.pt',dict(model=head.state_dict(),optimizer=opt.state_dict(),task=task))
    trainflops=budget*read(OLD/'NUMERICS.json')['ce_forward_backward_dense_flops']['16']
    report=dict(status=verdict,kind=kind,budget=budget,task=task,source_checkpoint=str(source),source_checkpoint_sha256=sha(source),
        K=K.tolist(),expansion_losses=(previous-K).tolist(),schedule_sha256=ahash(batches),
        current_entry=entry_metric,train_metrics=trainmetrics,smoke_gates=gates if kind=='jacobian' else None,
        unsafe_native=len(triggered),safe_native=budget-len(triggered) if kind=='jacobian' else None,
        repaired=len(triggered)-len(rolled),rollback=len(rolled),repair_rate=repair,norm_ratio_median_triggered=ratio,
        failure_counts=dict(failures),floor_violations=sum(r['floor_violations'] or 0 for r in rows) if kind=='jacobian' else None,
        old_ce_backward_examples=0,old_ce_backward_flops=0,test_access=False,
        common_wall=common_wall,control_wall=control_wall,entry_wall=entry_wall,
        algorithm_wall=common_wall+control_wall+(entry_wall if kind=='jacobian' else 0),observer_wall=observer_wall,
        regions_seconds=dict(meter.seconds),flop_ledger=dict(meter.flops),current_ce_dense_flops=trainflops,
        flop_coverage='Dense multiply-add lower bounds and named analytic/coordinate estimates; excludes optimizer, some elementwise/reductions/GELU costs. Not exact total hardware FLOPs.',
        endpoint_calls=bank.calls,endpoint_rows=bank.rows,padded_endpoint_rows=bank.calls*bank.padded,
        control_bank_reads=bank.rows if kind=='jacobian' else 0,derivative_calls=dict(meter.calls),
        outer_iterations=sum(r['decision'].get('outer_iterations',0) for r in rows),
        inner_sweeps=sum(r['decision'].get('inner_sweeps',0) for r in rows),
        correct_to_wrong=sum(r.get('correct_to_wrong',0) for r in rows),wrong_to_correct=sum(r.get('wrong_to_correct',0) for r in rows),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),model_bytes=sum(v.numel()*v.element_size() for v in head.parameters()),
        bank_gpu_bytes=sum(v.numel()*v.element_size() for v in (bank.x,bank.hidden,bank.out,bank.y)),
        bank_cpu_bytes=bank.H.nbytes+bank.absH.nbytes+bank.labels.nbytes+bank.ids.nbytes+bank.inputGram.nbytes,
        optimizer_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor)),
        checkpoint_sha256=sha(folder/'CHECKPOINT.pt'),native_reference_identity=bool(reference),replay_identity=bool(replay))
    write(folder/'CHECKS.json',report);print(phase,kind,budget,verdict,'new train',final['macro'],'repair',repair,'wall',report['algorithm_wall'],flush=True)
    return report

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    guard();assert read(ROOT/'NUMERICS.json')['status']=='PASS'
    if not args.replay:
        write(ROOT/'SMOKE_SOURCE_FREEZE.json',dict(sources={p.name:sha(p) for p in HERE.glob('*.py')},protocol_sha256=sha(HERE/'PROTOCOL.md')))
    else:
        for name,digest in read(ROOT/'SMOKE_SOURCE_FREEZE.json')['sources'].items():assert sha(HERE/name)==digest
    reports=[]
    for budget in (210,630):
        for kind in (('jacobian',) if args.replay else ('native','jacobian')):reports.append(run(kind,budget,'replay' if args.replay else 'smoke'))
    verdict='PASS' if all(r['status']=='PASS' for r in reports) else 'STOP_AFTER_SMOKE'
    write(ROOT/('REPLAY.json' if args.replay else 'SMOKE.json'),dict(status=verdict,reports=reports,test_access=False))

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(ROOT/f'FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
