"""Two fixed boundary forwards and a single offline FP32 logit correction."""
from experiments.gfg_banking77_final_v2_predictor_task2_v1.common import (
    Path, np, torch, setup, model, params, thash, sha, read, write, OLD, load)
from experiments.gfg_banking77_final_v2_expansion_audit_v1.observer import full_logits, margin
import sys, os, subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
AUTH=Path('E:/gfg-evidence/banking77-final-v2-expansion-audit-v1')
BASE='ce8124de3bd9549520d2327d6949c6522fe3ee45'


def guard():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True)
    assert subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release^{}'],cwd=REPO,text=True).strip()=='36dab5ce347dbbdac157ef23205f556606d18294'


def opt_identity(state):
    return thash({f'{i}.{k}':v for i,s in state['state'].items() for k,v in s.items() if isinstance(v,torch.Tensor)})


def counts(correct,labels,old):
    return np.bincount(labels[correct],minlength=old)


def main():
    guard(); setup()
    output=HERE/'results';assert not output.exists(),'Create-only; do not overwrite evidence'
    paths=[OLD/'CACHE.json',OLD/'TRAIN_FEATURES.pt',AUTH/'AUDIT.json']
    for budget,task in ((210,7),(630,6)):
        folder=AUTH/f'budget{budget}/task{task}'
        paths.extend(folder/name for name in ('CONFIG.json','EXPANSION.json','EXPANSION_LOGITS.npz'))
        paths.append(Path(read(folder/'CONFIG.json')['source_checkpoint']))
    allowed={p.resolve() for p in paths};opened=set()
    def access(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        if writing and not p.is_relative_to(output):raise PermissionError('Writes restricted to new results: '+str(p))
        if p.is_relative_to(AUTH.parent.resolve()):
            if p not in allowed or 'test' in p.name.lower():raise PermissionError('Evidence outside read allowlist: '+str(p))
            opened.add(str(p))
    sys.addaudithook(access)
    identities={str(p):sha(p) for p in paths}
    assert read(AUTH/'AUDIT.json')['status']=='PASS'
    X,y,control=load();results=[]
    for budget,task,expected in ((210,7,(158,19)),(630,6,(153,0))):
        folder=AUTH/f'budget{budget}/task{task}';cfg=read(folder/'CONFIG.json');audit=read(folder/'EXPANSION.json')
        cp=Path(cfg['source_checkpoint']);assert sha(cp)==cfg['source_checkpoint_sha256']
        state=torch.load(cp,weights_only=True);assert state['task']==task-1
        head=model();head.load_state_dict(state['model']);head.eval()
        headhash=thash(head.state_dict());ohash=opt_identity(state['optimizer'])
        assert headhash==cfg['head_entry'] and ohash==cfg['moments_entry']
        counters=[int(v['step']) for v in state['optimizer']['state'].values()]
        assert counters==cfg['optimizer_steps']
        saved=np.load(folder/'EXPANSION_LOGITS.npz');ids=saved['ids'];labels=saved['labels']
        assert control[ids].all() and np.array_equal(y[ids].cpu().numpy(),labels)
        z=full_logits(params(head),X[ids]);assert z.dtype==np.float32
        assert z.tobytes()==saved['logits'].tobytes()
        old=7*(task-1);seen=7*task;idx=np.arange(len(labels))
        mb=margin(z[:,:old],labels);mr=margin(z[:,:seen],labels)
        before=mb>0;raw=mr>0;flips=before&~raw
        assert (int(before.sum()),int(raw.sum()))==expected
        assert counts(before,labels,old).tolist()==audit['C_before']
        assert counts(raw,labels,old).tolist()==audit['C_after']
        target=z[idx,labels];newmax=z[:,old:seen].max(1)
        gaps=newmax[flips].astype(np.float64)-target[flips].astype(np.float64)
        beta=float(gaps.max()+1e-6);applied=np.float32(beta)
        fixed=z.copy();fixed[:,old:seen]=np.subtract(z[:,old:seen],applied,dtype=np.float32)
        assert fixed[:,:old].tobytes()==z[:,:old].copy().tobytes()
        assert fixed[:,seen:].tobytes()==z[:,seen:].copy().tobytes()
        mf=margin(fixed[:,:seen],labels);correct=mf>0
        Cb=counts(before,labels,old);Cr=counts(raw,labels,old);Cf=counts(correct,labels,old)
        oldbad=~before&~correct
        assert not (correct&~before).any(),'Correction cannot repair old-only mistakes'
        assert thash(head.state_dict())==headhash and opt_identity(state['optimizer'])==ohash
        assert [int(v['step']) for v in state['optimizer']['state'].values()]==counters
        assert all(p.grad is None for p in head.parameters())
        result=dict(boundary=f'T{task-1}->T{task}',budget=budget,old_seen=old,new_seen=seen,old_controls=len(ids),
            correct_before=int(before.sum()),correct_after_raw=int(raw.sum()),correct_after_corrected=int(correct.sum()),
            beta_formula_fp64=beta,beta_applied_fp32=float(applied),gap_min=float(gaps.min()),gap_median=float(np.median(gaps)),gap_max=float(gaps.max()),
            recovered_controls=int((flips&correct).sum()),restored_before_mask=bool(np.array_equal(before,correct)),
            before_to_corrected_correct_to_wrong=int((before&~correct).sum()),before_to_corrected_wrong_to_correct=int((~before&correct).sum()),
            raw_to_corrected_correct_to_wrong=int((raw&~correct).sum()),raw_to_corrected_wrong_to_correct=int((~raw&correct).sum()),
            previously_zero_relations=int((Cb==0).sum()),raw_newly_zero_relations=int(((Cb>0)&(Cr==0)).sum()),
            corrected_newly_zero_relations=int(((Cb>0)&(Cf==0)).sum()),corrected_total_zero_relations=int((Cf==0).sum()),
            corrected_old_competitor_errors=int(oldbad.sum()),new_old_competitor_errors=0,
            min_corrected_margin_on_expansion_flips=float(mf[flips].min()),
            argmax_correct_after_corrected=int((fixed[:,:seen].argmax(1)==labels).sum()),
            checkpoint=str(cp),checkpoint_sha256=sha(cp),head_sha256=headhash,optimizer_sha256=ohash,optimizer_counters=counters,
            parameters_unchanged=True,optimizer_state_unchanged=True,original_logits_byte_identical=True,old_logits_byte_identical=True,
            per_relation=[dict(relation=r,before=int(Cb[r]),raw=int(Cr[r]),corrected=int(Cf[r])) for r in range(old)],
            controls=[dict(id=int(ids[i]),relation=int(labels[i]),before=bool(before[i]),raw=bool(raw[i]),corrected=bool(correct[i]),
                margin_before=float(mb[i]),margin_raw=float(mr[i]),margin_corrected=float(mf[i])) for i in idx])
        results.append(result)
    assert all(sha(p)==v for p,v in identities.items());guard()
    output.mkdir()
    write(output/'RESULTS.json',dict(status='COMPLETED',results=results,training_updates=0,backward_calls=0,optimizer_steps=0,
        controller_calls=0,test_files_opened=0,inputs_sha256=identities,evidence_reads=sorted(opened),
        sources_sha256={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md')},
        scope='Two offline boundaries on old algorithmic controls only; no new-class performance or controller efficacy claim.'))
    for r in results:print({k:v for k,v in r.items() if k not in ('controls','per_relation')})


if __name__=='__main__':main()
