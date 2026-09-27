"""Post-failure diagnostic only: reconstruct the stopped native candidate.

No screened training restart, no threshold/certificate changes, no test read.
Use unchanged production V2 to reproduce the eleven committed heads, then
inspect the uncommitted step12 native candidate. New create-only output.
"""
from ..core import *


def main():
    p.setup()
    with gzip.open(ROOT/'STEPS.jsonl.gz','rt',encoding='utf8') as f:
        completed = [json.loads(v) for v in f]
    with gzip.open(ROOT/'CALLS.jsonl.gz','rt',encoding='utf8') as f:
        calls = [json.loads(v) for v in f]
    last = calls[-1]
    assert len(completed) == 11 and last['step'] == 12 and last['point'] == 'native'
    d = np.load(DATA/'data.npz')
    F = torch.load(INPUT/'FEATURES.pt',weights_only=False,map_location='cpu')['features'].cuda()
    saved = torch.load(INPUT/'v2_210/task9/CHECKPOINT.pt',weights_only=False,map_location='cpu')
    head, shadow, opt = make_state(saved)
    h,y,ids,n = bank(d,F,10)
    K = np.asarray(read(ROOT/'CONFIG.json')['K'])
    screen = Screen(h,y,n,72,80)
    screen.begin(p.snapshot(head),1)
    for step,current in enumerate(order(d,10)[:12],1):
        ce_grad(head,F[current],torch.from_numpy(d['y'][current]).cuda(),80)
        nn.utils.clip_grad_norm_(head.parameters(),1.)
        before = p.snapshot(head)
        propose(head,shadow,opt)
        native = p.snapshot(shadow)
        if step>1:
            screen.begin(before,step)
        if step<=11:
            executed,_,r = floor_projection(before,native,h,y,n,K,ids[:n],seen=80)
            head.load_state_dict(executed)
            assert head_hash(head)==completed[step-1]['head_sha256']
            assert moments_hash(opt)==completed[step-1]['moments_sha256']
            screen.evaluate(executed,set(r['selected_controls']))
            screen.finish()
        else:
            assert tensor_hash(native)==last['candidate_sha256']
            r=screen.evaluate(native)
            z=p.affine(native,h,80)
            m=margins(z,y)
            mismatch=torch.nonzero(r['correct']!=(m>0)).flatten().cpu().tolist()
            assert mismatch==last['sign_mismatch']
            records=[]
            for i in mismatch:
                t=int(y[i]); full=z[i].clone(); sparse=r['z'][i].clone()
                full[t]=-torch.inf; sparse[t]=-torch.inf
                c=int(full.argmax()); c2=int(sparse.argmax())
                records.append(dict(bank_row=i,dataset_id=int(ids[i]),relation=t,certified=bool(r['certified'][i]),
                    full_target_logit=float(z[i,t]),full_competitor=c,full_competitor_logit=float(z[i,c]),
                    frontier_target_logit=float(r['z'][i,t]),frontier_competitor=c2,frontier_competitor_logit=float(r['z'][i,c2]),
                    full_margin=float(m[i]),frontier_margin=float(r['m'][i]),
                    lower_bound=float(r['lower'][i]),full_batch_rows=len(h),
                    frontier_batch_rows=int(r['exact_mask'].sum())))
            exact64=margins(screen.X@augmented_gpu(native,80).T,y[:n])
            for record in records:
                record['fp64_diagnostic_margin']=float(exact64[record['bank_row']])
            out=dict(status='REPRODUCED',reference_reconstructed_committed_steps=11,
                candidate_sha256=tensor_hash(native),candidate_committed=False,rows=records,
                explanation='Same actual FP32 head/features; selecting fewer batch rows changes GEMM reduction rounding. The discrepancy is in an uncertified frontier row, not a false-safe certificate.',
                no_test_read=True,no_policy_change=True,no_formal_restart=True,
                diagnostic_source_sha256=sha(Path(__file__)))
            write(ROOT/'FAILURE_DIAGNOSIS.json',out)
            print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':
    main()
