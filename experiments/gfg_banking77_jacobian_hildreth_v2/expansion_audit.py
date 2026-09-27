"""Diagnose seen-class expansion using old controls and frozen checkpoints only."""
from .common import *

@torch.no_grad()
def main():
    setup();assert read(ROOT/'FORMAL.json')['status']=='PASS';X,y,control=load();yc=y.cpu().numpy();result=[]
    for budget in (210,630):
        for task in range(2,12):
            beforetask=task-1
            path=OLD/'formal'/f'native_{budget}'/'task1/CHECKPOINT.pt' if beforetask==1 else ROOT/('smoke' if beforetask==2 else 'formal')/f'jacobian_{budget}'/f'task{beforetask}/CHECKPOINT.pt'
            m=model();m.load_state_dict(torch.load(path,weights_only=True)['model']);headhash=thash(m.state_dict())
            ids=np.flatnonzero((yc<7*(task-1))&control);seen=7*task;bank=Bank(X[ids],y[ids],ids,seen,Meter());z=bank.exact(params(m))
            oldz=z[:,:seen-7];pre=bank.margins(oldz);post=bank.margins(z);loss=(pre>0)&(post<=0)
            incoming=z[:,seen-7:seen].max(1);oldbest=oldz.max(1);labels=bank.labels
            counts_before=np.bincount(labels[pre>0],minlength=seen-7);counts_after=np.bincount(labels[post>0],minlength=seen-7)
            report=read(ROOT/('smoke' if task==2 else 'formal')/f'jacobian_{budget}'/f'task{task}'/'CHECKS.json')
            assert counts_after.tolist()==report['K'] and (counts_before-counts_after).tolist()==report['expansion_losses']
            assert thash(m.state_dict())==headhash
            result.append(dict(budget=budget,task=task,old_controls=len(ids),correct_before_expansion=int((pre>0).sum()),
                correct_after_expansion=int((post>0).sum()),lost_at_expansion=int(loss.sum()),
                zero_K_intents=int((counts_after==0).sum()),old_intents=seen-7,
                lost_rows_won_by_new_classes=int((loss&(incoming>=oldbest)).sum()),
                new_vs_old_max_logit_difference_median=float(np.median(incoming-oldbest)),
                model_unchanged_during_expansion=True,training_updates_during_expansion=0,
                source_checkpoint_sha256=sha(path),counts_before=counts_before.tolist(),counts_after=counts_after.tolist()))
    write(ROOT/'EXPANSION_AUDIT.json',dict(status='PASS',records=result,test_access=False,protocol_unchanged=True,source_sha256=sha(Path(__file__))))
    print([{k:r[k] for k in ('budget','task','lost_at_expansion','correct_after_expansion','zero_K_intents')} for r in result],flush=True)

if __name__=='__main__':main()
