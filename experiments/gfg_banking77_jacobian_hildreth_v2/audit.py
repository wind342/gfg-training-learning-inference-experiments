"""Independent in-place AdamW and functional-endpoint replay, task2 both budgets."""
from .cutter import *
from .run import history

@torch.no_grad()
def endpoint(p,bank):
    return torch.cat([functional(p,bank.x[k:k+256]) for k in range(0,bank.padded,256)])[:bank.n,:bank.seen].cpu().numpy()

def counts(z,labels,n):
    good=np.array([all(float(row[y])>float(row[c]) for c in range(len(row)) if c!=y) for row,y in zip(z,labels)])
    return np.bincount(labels[good],minlength=n)

def main():
    setup();guard()
    for name,digest in read(ROOT/'SMOKE_SOURCE_FREEZE.json')['sources'].items():assert sha(HERE/name)==digest
    write(ROOT/'AUDIT_SOURCE_FREEZE.json',dict(sha256=sha(Path(__file__))))
    X,y,control=load();yc=y.cpu().numpy();reports=[]
    for budget in (210,630):
        reference=history(ROOT/'smoke'/f'jacobian_{budget}'/'task2/STEPS.jsonl.gz')
        state=torch.load(OLD/'formal'/f'native_{budget}'/'task1/CHECKPOINT.pt',weights_only=True)
        head=model();head.load_state_dict(state['model']);opt=optimizer(head);opt.load_state_dict(state['optimizer'])
        oldids=np.flatnonzero((yc<7)&control);ids=np.flatnonzero((yc//7==1)&~control)
        bank=Bank(X[oldids],y[oldids],oldids,14,Meter());K=counts(endpoint(params(head),bank),bank.labels,7)
        heads=[]
        for step,batch in enumerate(schedule(ids,2,budget),1):
            before=params(head);opt.zero_grad(set_to_none=True)
            F.cross_entropy(head(X[batch])[:,:14],y[batch]).backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
            native=params(head);mh=opt_hash(opt);assert opt_steps(opt)==[budget+step]*4
            executed,decision=project(before,native,bank,K);commit(head,executed)
            row=reference[step-1];assert decision==row['decision'];assert opt_hash(opt)==mh==row['moments_sha256']
            hh=thash(head.state_dict());assert hh==row['head_sha256'];heads.append(hh)
            actual=counts(endpoint(params(head),bank),bank.labels,7);assert actual.tolist()==row['counts'] and np.all(actual>=K)
            if step%60==0:print('v2 independent audit',budget,step,flush=True)
        reports.append(dict(budget=budget,steps=budget,heads_identical=True,moments_identical=True,counters_identical=True,
            decisions_identical=True,independent_endpoint_counts_identical=True,floor_violations=0,
            trajectory_sha256=hashlib.sha256(''.join(heads).encode()).hexdigest()))
    freeze=read(ROOT/'FREEZE.json')
    for name,digest in freeze['old_evidence_hashes'].items():assert sha(OLD/name)==digest
    for name,digest in freeze['v1_authority_hashes'].items():assert sha(V1/name)==digest
    for name,digest in freeze['v1_source_hashes'].items():assert sha(HERE.parent/'gfg_banking77_jacobian_hildreth_v1'/name)==digest
    write(ROOT/'AUDIT.json',dict(status='PASS',reports=reports,old_and_v1_unchanged=True,test_evaluated=False))
    print('v2 independent replay PASS; all prior evidence unchanged',flush=True)

if __name__=='__main__':main()
