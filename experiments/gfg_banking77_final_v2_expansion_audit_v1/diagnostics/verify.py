"""Independent boundary recomputation plus observer-free production replay."""
from ..common import *
from ..run import start_state,make_bank,one_update


@torch.no_grad()
def independent_logits(parameters,features):
    padded=((len(features)+255)//256)*256
    buffer=features.new_zeros((padded,768));buffer[:len(features)].copy_(features)
    outputs=[]
    for start in range(0,padded,256):
        q=F.gelu(F.linear(buffer[start:start+256],parameters[0],parameters[1]),approximate='none')
        outputs.append(F.linear(q,parameters[2],parameters[3]))
    return torch.cat(outputs)[:len(features)].cpu().numpy().copy()


def independent_counts(z,labels,seen,old_seen):
    # Compare each target with every competitor directly, not stored margins.
    chosen=z[:,:seen];target=chosen[np.arange(len(labels)),labels]
    greater=target[:,None]>chosen
    greater[np.arange(len(labels)),labels]=True
    correct=np.all(greater,axis=1)
    return correct,np.bincount(labels[correct],minlength=old_seen)


def verify_boundary(folder,head,X,y,ids,task,budget):
    report=read(folder/'EXPANSION.json');cfg=read(folder/'CONFIG.json')
    assert report['state_before']==report['state_after']
    assert all(report[k]==0 for k in ('training_updates','backward_calls','optimizer_step_calls','controller_interventions','test_files_opened','wrong_to_correct'))
    assert thash(head.state_dict())==report['state_before']['head_sha256']==cfg['head_entry']
    artifact=folder/'EXPANSION_LOGITS.npz';assert sha(artifact)==report['logits_artifact']['sha256']
    stored=np.load(artifact);labels=y[ids].cpu().numpy();z=independent_logits(params(head),X[ids])
    assert np.array_equal(ids,stored['ids']) and np.array_equal(labels,stored['labels'])
    assert z.tobytes()==stored['logits'].tobytes() and ahash(z)==report['logits_sha256']
    old_seen=7*(task-1);new_seen=7*task
    before,Cb=independent_counts(z,labels,old_seen,old_seen)
    after,Ca=independent_counts(z,labels,new_seen,old_seen)
    cw=before&~after;wc=~before&after
    assert not wc.any() and int(cw.sum())==report['correct_to_wrong']==report['expansion_loss']
    assert Cb.tolist()==report['C_before'] and Ca.tolist()==report['C_after']==cfg['K']
    assert report['relations_to_zero']==int(((Cb>0)&(Ca==0)).sum())
    assert report['post_expansion_K_zero']==int((Ca==0).sum())
    assert report['relations_losing']==int((Cb>Ca).sum())
    assert report['max_relation_loss']==int((Cb-Ca).max())
    assert len(report['flips'])==int(cw.sum())
    byid={int(v):i for i,v in enumerate(ids)}
    for flip in report['flips']:
        i=byid[flip['control_id']];target=int(labels[i]);old=z[i,:old_seen].copy();old[target]=-np.inf
        winner=old_seen+int(z[i,old_seen:new_seen].argmax())
        assert before[i] and not after[i]
        assert flip['true_relation']==target and flip['pre_winning_old_class']==target
        assert flip['true_class_logit']==float(z[i,target])
        assert flip['strongest_old_competitor_logit']==float(old.max())
        assert flip['strongest_new_class']==winner and flip['strongest_new_class_logit']==float(z[i,winner])
        assert flip['pre_margin']==float(z[i,target]-old.max())
        assert flip['post_margin']==float(z[i,target]-max(old.max(),z[i,winner]))
        assert flip['reason']==('NEW_CLASS_COMPETITION_FLIP' if z[i,winner]>z[i,target] else 'NEW_CLASS_TIE')
    newmax=z[:,old_seen:new_seen].max(1);true=z[np.arange(len(labels)),labels]
    assert report['new_class_logit_exceeds_true']==int((newmax>true).sum())
    assert report['max_new_class_logit']==float(newmax.max())
    assert report['mean_max_new_class_logit']==float(newmax.mean(dtype=np.float64))
    return dict(budget=budget,task=task,status='PASS',independent_logits_byte_identical=True,
        independent_count_attribution_identical=True,expansion_loss=int(cw.sum()),wrong_to_correct=0,
        relations_to_zero=report['relations_to_zero'])


def replay(budget,X,y,control):
    head,shadow,opt=start_state(budget);yc=y.cpu().numpy();boundaries=[];tasks=[]
    for task in range(3,12):
        folder=ROOT/f'budget{budget}/task{task}';cfg=read(folder/'CONFIG.json')
        ids=np.flatnonzero(control&(yc<7*(task-1)))
        boundaries.append(verify_boundary(folder,head,X,y,ids,task,budget))
        expected=records(folder/'STEPS.jsonl.gz');assert len(expected)==budget
        bank,predictor,bankids,current=make_bank(X,y,control,task,Meter())
        z=bank.exact(params(head));K=bank.counts(z)
        assert K.tolist()==cfg['K'] and opt_hash(opt)==cfg['moments_entry']
        # This entire replay NEVER invokes the production expansion observer.
        for step,batch in enumerate(schedule(current,task,budget),1):
            assert not control[batch].any() and np.all(yc[batch]//7==task-1)
            row,z=one_update(head,shadow,opt,bank,predictor,X,y,batch,K,z)
            reference=expected[step-1]
            for key,value in row.items():
                if key not in ('common_wall','control_wall'):assert value==reference[key],(budget,task,step,key)
            assert opt_steps(opt)==[(task-1)*budget+step]*4
            if step%120==0:print('VERIFY',budget,task,step,flush=True)
        summary=read(folder/'CHECKS.json')
        assert sha(folder/'CHECKPOINT.pt')==summary['checkpoint_sha256']
        stored=torch.load(folder/'CHECKPOINT.pt',weights_only=True)
        assert thash(stored['model'])==thash(head.state_dict())==summary['head_final']
        for key,value in opt.state_dict()['state'].items():
            for name,tensor in value.items():assert torch.equal(tensor.cpu(),stored['optimizer']['state'][key][name].cpu())
        tasks.append(dict(budget=budget,task=task,updates=budget,all_heads_proposals_moments_counters_decisions_identical=True,
            all_predictor_and_endpoint_hashes_identical=True,floor_violations=0,old_ce_backward_examples=0))
        print('VERIFIED TASK',budget,task,flush=True)
    return dict(budget=budget,boundaries=boundaries,tasks=tasks,observer_free_replay_updates=9*budget)


def main():
    setup();guard();access=Access();verify_freeze()
    if len(sys.argv)>1:
        budget=int(sys.argv[1]);assert budget in (210,630)
        assert (ROOT/f'budget{budget}/task11/CHECKS.json').exists()
        X,y,control=load();result=replay(budget,X,y,control);verify_freeze()
        observed={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/f'budget{budget}').rglob('*') if p.is_file()}
        write(ROOT/f'AUDIT_{budget}.json',dict(status='PASS',result=result,verified_artifacts=observed,
            diagnostic_sha256=sha(Path(__file__)),access=access.report(),test_files_opened=0))
        print('ARM AUDIT PASS',budget,flush=True);return
    assert read(ROOT/'TRAINING_COMPLETE.json')['status']=='COMPLETED'
    results=[]
    for budget in (210,630):
        part=read(ROOT/f'AUDIT_{budget}.json');assert part['status']=='PASS'
        assert part['diagnostic_sha256']==sha(Path(__file__))
        assert all(sha(ROOT/p)==digest for p,digest in part['verified_artifacts'].items())
        results.append(part['result'])
    verify_freeze()
    write(ROOT/'AUDIT.json',dict(status='PASS',results=results,independently_recomputed_boundaries=18,
        observer_free_replay_updates=7560,access=access.report(),test_files_opened=0,
        scope='Boundary endpoints/counts/attributions independently implemented; observer-free replay reuses unchanged verified production predictor/cutter, not a second solver implementation.'))
    print('AUDIT PASS: 18 boundaries and 7560 observer-free replay updates',flush=True)


if __name__=='__main__':main()
