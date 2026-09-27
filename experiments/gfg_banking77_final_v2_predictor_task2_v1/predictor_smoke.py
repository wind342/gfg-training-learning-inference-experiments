"""Native-only task2 predictor measurement; no parameter control, no test access."""
from .predictor import *
from .preflight import install,verify_inputs

def run(budget,X,y,control):
    state=torch.load(OLD/f'formal/native_{budget}/task1/CHECKPOINT.pt',weights_only=True)
    head=model();head.load_state_dict(state['model']);opt=optimizer(head);opt.load_state_dict(state['optimizer'])
    labels=y.cpu().numpy();ids=np.flatnonzero((labels<7)&control);current=np.flatnonzero((labels//7==1)&~control)
    bank=Bank(X[ids],y[ids],ids,14,Meter());pred=Predictor(bank,read(ROOT/'NUMERICS.json')['predictor_dense_flops_per256'])
    z=bank.exact(params(head));K=bank.counts(z);stats=[];cw_ids=set();start=time.perf_counter()
    with gzip.open(ROOT/f'PREDICTOR_NATIVE_{budget}.jsonl.gz','xt',encoding='utf8') as out:
        for step,batch in enumerate(schedule(current,2,budget),1):
            assert not control[batch].any();before=params(head);bh=phash(before)
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(head(X[batch])[:,:14],y[batch]);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
            native=params(head);moments=opt_hash(opt);predicted=pred.predict(before,native);actual=bank.exact(native)
            r=measure(bank,z,predicted,actual,K);stats.append(r);cw_ids.update(ids[(bank.margins(z)>0)&(bank.margins(actual)<=0)].tolist())
            assert opt_hash(opt)==moments and opt_steps(opt)==[budget+step]*4
            out.write(json.dumps(dict(step=step,batch_ids=batch.tolist(),before_sha256=bh,native_sha256=phash(native),moments_sha256=moments,
                components_sha256=pred.hashes(),predictor=r),separators=(',',':'))+'\n');z=actual
    summary=aggregate(stats);summary.update(budget=budget,K=K.tolist(),distinct_correct_wrong=len(cw_ids),
        predictor_wall=bank.meter.seconds['predictor'],predictor_flops=bank.meter.flops['predictor_dense'],
        wall_with_observers=time.perf_counter()-start,old_ce_backward_examples=0,test_files_opened=0)
    print('native predictor',budget,summary['status'],'MAE',summary['margin_mae'],'false safe',summary['false_safe_updates'],flush=True)
    return summary

def main():
    access=install();setup();guard();verify_inputs()
    for name,digest in read(ROOT/'FREEZE.json')['sources'].items():assert sha(HERE/name)==digest
    X,y,c=load();reports=[run(b,X,y,c) for b in (210,630)]
    write(ROOT/'PREDICTOR_SMOKE.json',dict(status='PASS' if all(r['status']=='PASS' and r['distinct_correct_wrong']>=20 for r in reports) else 'FAIL',
        reports=reports,access=access.report(),scope='Native-only trajectories; controlled trajectories separately measured.'))
    verify_inputs()

if __name__=='__main__':main()
