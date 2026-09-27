"""One task10 arm, full observer and paired fixed-arithmetic controller."""
from .core import *
import traceback


def guard():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)
    assert subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release^{}'],cwd=REPO,text=True).strip() == '36dab5ce347dbbdac157ef23205f556606d18294'


class Observer:
    def __init__(self, screen, K, ids, stream=None):
        self.s, self.K, self.ids, self.stream = screen, K, ids, stream
        self.wall, self.step = 0., 0
        self.records = []

    @torch.no_grad()
    def evaluate(self, state, selected, label):
        r = self.s.evaluate(state, selected)
        sync(); start=time.perf_counter()
        s=self.s
        z=fixed_affine(state,s.h,s.seen);m=margins(z,s.y)
        correct=m>0
        C=counts(m[:s.n].cpu().numpy(),s.labels,s.groups)
        false_safe=torch.nonzero(r['certified'] & ~correct[:s.n]).flatten().cpu().tolist()
        exact=r['exact_mask']
        max_error=float((r['m'][exact]-m[exact]).abs().max()) if bool(exact.any()) else 0.
        full_def=np.maximum(self.K-C,0)
        lower_def=np.maximum(self.K-r['C'],0)
        lower_valid=bool(np.all(r['C']<=C))
        same_def=np.array_equal(full_def,lower_def)
        _,_,chosen=select(r,s.labels,self.K,self.ids[:s.n])
        _,_,truth=choose(m[:s.n].cpu().numpy(),s.labels,self.K,self.ids[:s.n])
        selected_exact=all(bool(exact[i]) for i in selected)
        old_exact=int(exact[:s.n].sum())
        record=dict(step=self.step,point=label,certified=int(r['certified'].sum()),
                    false_safe=false_safe,lower_count_valid=lower_valid,deficits_identical=bool(same_def),
                    choice_identical=chosen==truth,selected_exact=selected_exact,max_margin_difference=max_error,
                    count_lower=r['C'].tolist(),full_counts=C.tolist(),exact_old_rows=old_exact,
                    unexamined_uncertified=int((~exact[:s.n] & ~r['certified']).sum()),
                    candidate_sha256=tensor_hash(state),padded_rows=r['padded_rows'],gemms=r['gemms'])
        self.records.append(record)
        if self.stream:
            self.stream.write(json.dumps(record,separators=(',',':'))+'\n')
        sync();self.wall+=time.perf_counter()-start
        assert not false_safe and max_error==0. and lower_valid and same_def and chosen==truth and selected_exact, record
        return r


def load():
    d=np.load(DATA/'data.npz')
    F=torch.load(INPUT/'FEATURES.pt',weights_only=False,map_location='cpu')['features'].cuda()
    initial=torch.load(INPUT/'v2_210/task9/CHECKPOINT.pt',weights_only=False,map_location='cpu')
    h,y,ids,n=bank(d,F,10)
    K=np.asarray(read(INPUT/'v2_210/task10/CONFIG.json')['K'])
    assert n==10072 and len(K)==72
    return d,F,initial,h,y,ids,n,K


@torch.no_grad()
def tests(h,y,n,before,native,K,ids):
    s=Screen(h,y,n,len(K),80,K);o=Observer(s,K,ids)
    # Identical, moved, repeated, and returned heads exercise cumulative paths.
    for j,state in enumerate((before,native,native,before,native,before)):
        o.evaluate(state,(),f'path{j}')
    # Bounds must support early stopping even when they prove every row.
    toyh=torch.tensor([[1.,0.],[2.,0.],[3.,0.],[1.,0.]],device=h.device)
    toyy=torch.tensor([0,0,0,1],device=h.device)
    toy=Screen(toyh,toyy,3,1,2,np.array([1]))
    state={'weight':torch.tensor([[2.,0.],[-2.,0.]],device=h.device),'bias':torch.zeros(2,device=h.device)}
    a=toy.evaluate(state)
    assert a['C'][0]>=1 and int(a['exact_mask'][:3].sum())==1
    a=toy.evaluate(state)
    assert a['C'][0]>=1 and int(a['exact_mask'][:3].sum())==0
    bad={'weight':-state['weight'],'bias':state['bias']}
    a=toy.evaluate(bad)
    assert a['C'][0]==0 and bool(a['exact_mask'][:3].all())
    a=toy.evaluate(state,{2})
    assert bool(a['exact_mask'][2])
    s=Screen(h,y,n,len(K),80,K);o=Observer(s,K,ids)
    executed,report=screened_projection(before,native,s,K,ids[:n],o.evaluate)
    ref,_,reference=reference_projection(before,native,h,y,n,K,ids[:n],seen=80)
    compare_reports(report,reference,K);assert equal(executed,ref)
    s.timings.resolve()
    return dict(status='PASS',path_checks=6,positive_negative_return_and_selected_tests=True,
                first_update_head_and_decisions_identical=True,early_stop_without_complete_count=True)


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    guard();assert shutil.disk_usage(ROOT.parent).free>3*1024**3
    ROOT.mkdir(exist_ok=False);p.setup()
    assert read(OLD/'results/CHECKS.json')['status']=='PASS'
    paths=[INPUT/q for q in ('FEATURES.pt','TEST_FEATURES.pt','CACHE.json','CHECKS.json',
        'v2_210/task9/CHECKPOINT.pt','v2_210/task9/TRAIN_RESULT.json','v2_210/task10/CONFIG.json')]+[DATA/'data.npz']
    inputs={str(q):sha(q) for q in paths}
    cache=read(INPUT/'CACHE.json')
    assert inputs[str(DATA/'data.npz')]==DATA_SHA
    assert inputs[str(INPUT/'FEATURES.pt')]==cache['artifacts']['FEATURES.pt']
    assert inputs[str(INPUT/'TEST_FEATURES.pt')]==cache['artifacts']['TEST_FEATURES.pt']
    assert inputs[str(INPUT/'v2_210/task9/CHECKPOINT.pt')]==read(INPUT/'v2_210/task9/TRAIN_RESULT.json')['checkpoint_sha256']
    d,F,initial,h,y,ids,n,K=load()
    head,shadow,opt=make_state(initial)
    first=order(d,10)[0]
    ce_grad(head,F[first],torch.from_numpy(d['y'][first]).cuda(),80)
    nn.utils.clip_grad_norm_(head.parameters(),1.)
    before=p.snapshot(head);propose(head,shadow,opt)
    preflight=tests(h,y,n,before,p.snapshot(shadow),K,ids)
    write(ROOT/'NUMERICAL_PREFLIGHT.json',preflight)
    write(ROOT/'PREFLIGHT.json',dict(base=BASE,sources=source_hashes(),inputs=inputs,runtime=runtime(),
          backbone_sha256=cache['backbone_sha256'],numerical_preflight='PASS'))
    head,shadow,opt=make_state(initial)
    assert old.opt_counter(opt)==1890
    cfg=read(INPUT/'v2_210/task10/CONFIG.json')
    assert head_hash(head)==cfg['head_start'] and moments_hash(opt)==cfg['moments_start']
    write(ROOT/'CONFIG.json',dict(K=K.tolist(),bank_ids=ids.tolist(),n=n,updates=210,
          head_start=head_hash(head),moments_start=moments_hash(opt),count_semantics='lower bound, not full count'))
    sync();t=time.perf_counter();s=Screen(h,y,n,72,80,K);sync();setup=time.perf_counter()-t
    rows=[];component={}
    with gzip.open(ROOT/'CALLS.jsonl.gz','xt',encoding='utf8') as calls,gzip.open(ROOT/'STEPS.jsonl.gz','xt',encoding='utf8') as steps:
        o=Observer(s,K,ids,calls)
        for step,current in enumerate(order(d,10),1):
            o.step=step
            sync();t=time.perf_counter()
            _,loss=ce_grad(head,F[current],torch.from_numpy(d['y'][current]).cuda(),80)
            norm=nn.utils.clip_grad_norm_(head.parameters(),1.)
            assert np.isfinite(loss) and torch.isfinite(norm)
            before=p.snapshot(head);propose(head,shadow,opt);native=p.snapshot(shadow)
            sync();common=time.perf_counter()-t;mh=moments_hash(opt)
            sync();t=time.perf_counter();ob=o.wall
            executed,report=screened_projection(before,native,s,K,ids[:n],o.evaluate)
            head.load_state_dict(executed)
            o.evaluate(p.snapshot(head),set(report['selected_controls']),'commit')
            sync();control=time.perf_counter()-t-(o.wall-ob)
            assert moments_hash(opt)==mh and old.opt_counter(opt)==1890+step
            parts=s.timings.resolve()
            for k,v in parts.items():component[k]=component.get(k,0.)+v
            sync();t=time.perf_counter()
            ref,_,reference=reference_projection(before,native,h,y,n,K,ids[:n],seen=80)
            # Match v2 paired timing: additional actual executed-head recheck.
            actual=margins(fixed_affine(ref,h,80),y).cpu().tolist()
            sync();fullwall=time.perf_counter()-t
            assert actual==reference['executed']
            compare_reports(report,reference,K);assert equal(executed,ref)
            assert np.all(np.asarray(o.records[-1]['full_counts'])>=K)
            row=dict(step=step,current_ids=current.tolist(),loss=loss,head_sha256=head_hash(head),moments_sha256=mh,
                 optimizer_step=old.opt_counter(opt),report=report,common_update_wall=common,
                 screened_control_wall=control,reference_control_wall=fullwall,observer_wall=o.wall-ob,
                 cuda_timeline=parts,full_reference_head_identical=True,control_decisions_identical=True,
                 actual_executed_counts=o.records[-1]['full_counts'])
            rows.append(row);steps.write(json.dumps(row,separators=(',',':'))+'\n');steps.flush()
            if step==1 or step%30==0:
                print('V3',step,'sweeps',report['sweep_count'],'exact',s.total_exact,'screen',round(sum(x['screened_control_wall'] for x in rows),3),'full',round(sum(x['reference_control_wall'] for x in rows),3),flush=True)
    pre=read(ROOT/'PREFLIGHT.json')
    assert source_hashes()==pre['sources'] and all(sha(q)==v for q,v in inputs.items());guard()
    save(ROOT/'CHECKPOINT.pt',dict(head=p.cpu_state(p.snapshot(head)),optimizer=opt.state_dict()))
    write(ROOT/'TRAINING_COMPLETE.json',dict(steps=210,head=head_hash(head),moments=moments_hash(opt)))
    test=torch.load(INPUT/'TEST_FEATURES.pt',weights_only=False,map_location='cpu')
    with torch.no_grad():pred=fixed_affine(p.snapshot(head),test['features'].cuda(),80).argmax(1).cpu().tolist()
    ev=dict(indices=test['indices'],predictions=pred,metrics=output_metrics(d,pred,10),readonly=True)
    write(ROOT/'EVALUATION.json',ev)
    v2=read(OLD/'results/EVALUATION.json')
    assert pred==v2['predictions']
    result=dict(status='COMPLETED',gate='PASS',steps=210,old_new_all=ev['metrics'],
        algorithm_wall=setup+sum(v['common_update_wall']+v['screened_control_wall'] for v in rows),
        control_wall=sum(v['screened_control_wall'] for v in rows),
        reference_algorithm_wall=sum(v['common_update_wall']+v['reference_control_wall'] for v in rows),
        reference_control_wall=sum(v['reference_control_wall'] for v in rows),setup_wall=setup,observer_wall=o.wall,
        exact_old_rows=s.total_exact,cache_refreshed_old_rows=s.total_refresh,additional_refresh_forward_rows=0,
        fp64_feature_head_matmul_rows=0,fp64_class_row_norm_elements=s.total_certificate_elements,
        certificate_tensor_max_shape=[80,1537],candidate_calls=s.calls,padded_rows=s.total_padded,gemm_calls=s.total_gemms,
        cuda_timeline=component,false_safe=0,floor_violations=0,full_reference_head_matches=210,decision_matches=210,
        v2_test_predictions_identical=True,total_sweeps=sum(v['report']['sweep_count'] for v in rows),
        skips=sum(v['report']['skip'] for v in rows),unexamined_uncertified_rows=sum(v['unexamined_uncertified'] for v in o.records),
        note='GPU timeline spans include host launch gaps; cache refresh overlaps exact rows; profile overhead included; full observer excluded.')
    write(ROOT/'RESULTS.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        if ROOT.exists():write(ROOT/f'FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
