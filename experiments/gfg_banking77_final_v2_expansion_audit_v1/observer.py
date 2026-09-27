"""Pure zero-update readout expansion; no training/controller calls."""
from .common import *


@torch.no_grad()
def full_logits(parameters,features):
    n=len(features);padded=((n+255)//256)*256
    x=features.new_zeros((padded,768));x[:n].copy_(features)
    hidden=features.new_empty((256,256));out=features.new_empty((padded,77))
    for k in range(0,padded,256):
        torch.addmm(parameters[1],x[k:k+256],parameters[0].T,out=hidden)
        q=F.gelu(hidden,approximate='none')
        torch.addmm(parameters[3],q,parameters[2].T,out=out[k:k+256])
    return out[:n].cpu().numpy().copy()


def margin(z,labels):
    other=z.copy();other[np.arange(len(labels)),labels]=-np.inf
    return z[np.arange(len(labels)),labels]-other.max(1)


def summarize(z,labels,ids,old_seen,new_seen):
    assert np.isfinite(z).all() and np.all(labels<old_seen)
    before=z[:,:old_seen].copy();after=z[:,:new_seen].copy()
    assert before.tobytes()==after[:,:old_seen].copy().tobytes()
    mb=margin(before,labels);ma=margin(after,labels)
    cb=mb>0;ca=ma>0;cw=cb&~ca;wc=~cb&ca
    assert np.all(ma<=mb) and not wc.any()
    Cb=np.bincount(labels[cb],minlength=old_seen);Ca=np.bincount(labels[ca],minlength=old_seen)
    ab=before.argmax(1);aa=after.argmax(1);arg_cw=(ab==labels)&(aa!=labels);arg_wc=(ab!=labels)&(aa==labels)
    assert not arg_wc.any()
    true=z[np.arange(len(labels)),labels];new=z[:,old_seen:new_seen];newmax=new.max(1);newwinner=new.argmax(1)+old_seen
    other=before.copy();other[np.arange(len(labels)),labels]=-np.inf;oldmax=other.max(1)
    strict=cw&(newmax>true);tie=cw&(newmax==true)
    assert np.array_equal(cw,strict|tie)
    flips=[]
    for i in np.flatnonzero(cw):
        flips.append(dict(control_id=int(ids[i]),true_relation=int(labels[i]),pre_winning_old_class=int(ab[i]),
            true_class_logit=float(true[i]),strongest_old_competitor_logit=float(oldmax[i]),
            strongest_new_class=int(newwinner[i]),strongest_new_class_logit=float(newmax[i]),
            pre_margin=float(mb[i]),post_margin=float(ma[i]),
            reason='NEW_CLASS_COMPETITION_FLIP' if strict[i] else 'NEW_CLASS_TIE',
            argmax_correct_to_wrong=bool(arg_cw[i])))
    def distribution(a):return dict(n=int(a.size),min=float(a.min()),mean=float(a.mean(dtype=np.float64)),
        std=float(a.std(dtype=np.float64)),p05=float(np.quantile(a,.05)),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))
    losing=Cb>Ca;zero=(Cb>0)&(Ca==0)
    return dict(old_seen=old_seen,new_seen=new_seen,old_controls=len(labels),correct_before=int(cb.sum()),correct_after=int(ca.sum()),
        expansion_loss=int(cb.sum()-ca.sum()),expansion_gain=int(wc.sum()),correct_to_wrong=int(cw.sum()),wrong_to_correct=int(wc.sum()),
        argmax_correct_before=int((ab==labels).sum()),argmax_correct_after=int((aa==labels).sum()),argmax_correct_to_wrong=int(arg_cw.sum()),argmax_wrong_to_correct=int(arg_wc.sum()),
        strict_competition_flips=int(strict.sum()),new_class_ties=int(tie.sum()),relations_losing=int(losing.sum()),
        max_relation_loss=int((Cb-Ca).max(initial=0)),relations_to_zero=int(zero.sum()),post_expansion_K_zero=int((Ca==0).sum()),
        C_before=Cb.tolist(),C_after=Ca.tolist(),delta=(Ca-Cb).tolist(),
        per_relation=[dict(relation=r,before=int(Cb[r]),after=int(Ca[r]),delta=int(Ca[r]-Cb[r]),to_zero=bool(zero[r])) for r in range(old_seen)],
        new_class_logit_distribution=distribution(new),max_new_class_logit=float(newmax.max()),mean_max_new_class_logit=float(newmax.mean(dtype=np.float64)),
        max_new_class_distribution=distribution(newmax),new_class_logit_exceeds_true=int((newmax>true).sum()),
        flips=flips,old_logits_byte_identical=True,old_logits_max_abs_error=0.,margin_monotone=True,
        correctness_definition='strict target margin > 0; argmax ties reported separately')


@torch.no_grad()
def audit(head,shadow,opt,X,y,ids,task,budget,folder):
    old_seen=7*(task-1);new_seen=7*task
    start=state_identity(head,shadow,opt)
    features=X[ids];labels=y[ids].cpu().numpy();parameters=params(head)
    z=full_logits(parameters,features)
    repeated=full_logits(parameters,features)
    assert z.tobytes()==repeated.tobytes(),'Repeated fixed arithmetic output differs'
    report=summarize(z,labels,ids,old_seen,new_seen)
    after=state_identity(head,shadow,opt);assert start==after,'Observer changed state'
    report.update(task=task,budget=budget,boundary=f'T{task-1}->T{task}',state_before=start,state_after=after,
        repeat_logits_byte_identical=True,training_updates=0,backward_calls=0,optimizer_step_calls=0,controller_interventions=0,
        counters_scope='Observer call graph contains only no-grad forward/readout; identities independently checked',test_files_opened=0,
        control_ids=ids.tolist(),logits_sha256=ahash(z),labels_sha256=ahash(labels))
    payload=folder/'EXPANSION_LOGITS.npz'
    with payload.open('xb') as f:np.savez_compressed(f,logits=z,labels=labels,ids=ids)
    report['logits_artifact']=dict(file=payload.name,sha256=sha(payload),bytes=payload.stat().st_size)
    write(folder/'EXPANSION.json',report)
    return report
