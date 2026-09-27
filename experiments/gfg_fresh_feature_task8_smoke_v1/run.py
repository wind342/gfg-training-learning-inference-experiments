"""Exactly two task8 arms, then one sealed read-only evaluation."""
import traceback
from .core import *

def preflight(state,moments,h,R):
    tests={}
    for arm in ARMS:
        q=joint_features(h[:16],R,arm)
        head,shadow,opt=create(state,moments)
        original=nn.functional.linear(h[:16],state['weight'],state['bias'])
        assert torch.equal(head(q),original)
        assert torch.equal(head(q)[:,:SEEN],joint_affine(pack(head),q))
        assert not torch.count_nonzero(head.residual)
        # Joint affine derivatives agree with autograd, including bias.
        head(q)[0,3].backward()
        assert torch.equal(head.weight.grad[3],h[0])
        assert torch.equal(head.residual.grad[3],q[0,DIM:])
        assert float(head.bias.grad[3])==1.
        head.zero_grad(set_to_none=True)
        hd=q.double();w=head.weight.double();u=head.residual.double();b=head.bias.double()
        delta=torch.linspace(-1e-4,1e-4,80*(DIM+EXTRA),device='cuda',dtype=torch.float64).reshape(80,DIM+EXTRA)
        db=torch.ones_like(b)*1e-5
        z=nn.functional.linear(hd[:,:DIM],w,b)+nn.functional.linear(hd[:,DIM:],u)
        endpoint=nn.functional.linear(hd[:,:DIM],w+delta[:,:DIM],b+db)+nn.functional.linear(hd[:,DIM:],u+delta[:,DIM:])
        predicted=z+nn.functional.linear(hd,delta,db)
        assert torch.allclose(endpoint,predicted,atol=1e-12,rtol=0)
        labels=original[:,:SEEN].argmax(1)
        before=pack(head);native={k:v.clone() for k,v in before.items()}
        native['bias'][0]+=10.
        executed,candidate,report=hildreth(before,native,q,labels,len(q),SEEN)
        assert check_safety(before,executed,q,labels,len(q))==0
        assert equal(before,pack(head))
        # Original monitor calls are identical to actual module forward.
        unpack(head,executed)
        assert torch.equal(head(q)[:,:SEEN],joint_affine(executed,q))
        for p in head.parameters():p.grad=torch.ones_like(p)*1e-4
        propose(head,shadow,opt)
        assert [int(opt.state[p]['step']) for p in shadow.parameters()]==[1471,1471,1]
        tests[arm]=dict(initial_exact=True,zero_residual=True,gradient_exact=True,
            affine_endpoint=True,moments_inherited=True,counters_once=True,synthetic_projection_safe=True)
    return tests


def train(arm,state,moments,d,train_h,R,h,y,n,batches):
    folder=ROOT/arm;folder.mkdir()
    head,shadow,opt=create(state,moments)
    q=joint_features(h,R,arm)
    current=joint_features(train_h,R,arm)
    with torch.no_grad():
        assert torch.equal(head(q),nn.functional.linear(h,state['weight'],state['bias']))
        assert torch.equal(head(current),nn.functional.linear(train_h,state['weight'],state['bias']))
    initial_opt=copy.deepcopy(opt.state_dict())
    heads=[proj.cpu_state(pack(head))];natives=[];projected=[];grads=[];compact=[]
    initial_hash=tensor_hash(pack(head))
    features_hash=tensor_hash({'controls':q,'train':current})
    truth=torch.from_numpy(d['y'][batches.flatten()]).cuda()
    torch.cuda.reset_peak_memory_stats();proj.sync(h);start=time.perf_counter()
    with (folder/'STEPS.jsonl').open('x',encoding='utf8') as f:
        for index,ids in enumerate(batches):
            step=index+1;global_step=1470+step
            torch.manual_seed(2021*100000+global_step);torch.cuda.manual_seed_all(2021*100000+global_step)
            head.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(head(current[index*16:(index+1)*16])[:,:SEEN],truth[index*16:(index+1)*16])
            loss.backward();norm=nn.utils.clip_grad_norm_(head.parameters(),1.)
            assert torch.isfinite(loss) and torch.isfinite(norm)
            grad={k:p.grad.detach().cpu().clone() for k,p in head.named_parameters()};grads.append(grad)
            before=pack(head);propose(head,shadow,opt);native=pack(shadow)
            assert [int(opt.state[p]['step']) for p in shadow.parameters()]==[global_step,global_step,step]
            opt_hash=moments_hash(opt)
            executed,candidate,report=hildreth(before,native,q,y,n,SEEN)
            assert opt_hash==moments_hash(opt)
            unpack(head,executed)
            with torch.no_grad():
                actual=margins(head(q)[:,:SEEN],y)
            assert actual.cpu().tolist()==(report['baseline'] if report['skip'] else report['projected'])
            flips=sum(a>0 and b<=0 for a,b in zip(report['baseline'][:n],actual.cpu().tolist()[:n]))
            assert flips==0
            if report['native_violating'] and not report['skip']:
                assert all(float(actual[i])>=proj.EPS for i in report['eligible'])
            report['executed']=actual.cpu().tolist()
            geom=proj.geometry(before,native,executed)
            small=dict(step=step,current_ids=ids.tolist(),loss=float(loss.detach()),gradient_norm_before_clip=float(norm),
                receiving_sha256=tensor_hash(before),native_sha256=tensor_hash(native),executed_sha256=tensor_hash(executed),
                optimizer_sha256=opt_hash,counters=[global_step,global_step,step],
                projection=bool(report['native_violating']),skip=report['skip'],sweeps=report['sweep_count'],
                protected_flips=flips,active_constraints=len(report['final_active']),geometry=geom,
                projection_seconds=report['projection_seconds'])
            f.write(json.dumps(dict(**small,report=report),separators=(',',':'),allow_nan=False)+'\n');f.flush()
            compact.append(small);heads.append(proj.cpu_state(executed));natives.append(proj.cpu_state(native));projected.append(proj.cpu_state(candidate))
            if step==1 or step%30==0:print('TRAIN',arm,step,'skips',sum(v['skip'] for v in compact),'sweeps',report['sweep_count'],flush=True)
    proj.sync(h);wall=time.perf_counter()-start
    assert features_hash==tensor_hash({'controls':q,'train':current})
    save_tensor(folder/'AUDIT.pt',dict(heads=heads,natives=natives,projected=projected,clipped_gradients=grads,
        initial_optimizer=initial_opt,final_optimizer=copy.deepcopy(opt.state_dict())))
    selected=[r for r in compact if r['projection']]
    summary=dict(steps=210,skip=sum(r['skip'] for r in compact),protected_flips=sum(r['protected_flips'] for r in compact),
        projection_triggered=len(selected),norm_ratio=float(np.mean([r['geometry']['norm_ratio'] for r in selected])),
        cosine=float(np.mean([r['geometry']['cosine'] for r in selected])),wall=wall,
        projection_wall=sum(r['projection_seconds'] for r in compact),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        initial_sha256=initial_hash,final_sha256=tensor_hash(pack(head)),
        bank_size=n,old_control_correct=int((actual[:n]>0).sum()),new_audit_correct=int((actual[n:]>0).sum()))
    write(folder/'TRAIN_RESULT.json',dict(status='COMPLETED',summary=summary,
        audit_sha256=sha(folder/'AUDIT.pt'),log_sha256=sha(folder/'STEPS.jsonl'),finished_unix=time.time()))
    write(folder/'COMPACT_STEPS.json',compact)
    return head,summary


@torch.no_grad()
def evaluate(m,heads,R,d):
    write(ROOT/'EVALUATION_BARRIER.json',dict(training_hashes={a:sha(ROOT/a/'TRAIN_RESULT.json') for a in ARMS},time=time.time()))
    before={a:tensor_hash(pack(heads[a])) for a in ARMS}
    ids=scope(d,SEEN)[1];pred={a:[] for a in ARMS};x=torch.from_numpy(d['x']);start=time.perf_counter()
    m.eval()
    for pos in range(0,len(ids),16):
        h=m.features(x[ids[pos:pos+16]].cuda())
        for arm in ARMS:pred[arm]+=heads[arm](joint_features(h,R,arm))[:,:SEEN].argmax(1).cpu().tolist()
    assert before=={a:tensor_hash(pack(heads[a])) for a in ARMS}
    scores={a:summarize_predictions(ids,pred[a],d,TASK) for a in ARMS}
    write(ROOT/'EVALUATION.json',dict(indices=ids.tolist(),predictions=pred,metrics=scores,
        readonly=True,seconds=time.perf_counter()-start,finished_unix=time.time()))
    return scores


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    ROOT.mkdir(exist_ok=False)
    try:
        setup()
        source=old.ROOT
        inputs=[source/'final_architecture_v1/task7/AUDIT.pt',source/'final_architecture_v1/task7/TRAIN_RESULT.json',
            source/'final_architecture_v1/task8/CONFIG.json',source/'final_architecture_v1/task8/TRAIN_RESULT.json',
            source/'CONTROL_CACHE.pt',source/'CHECKS.json',DATA/'data.npz',CHECKPOINT]
        identities={str(p):sha(p) for p in inputs}
        assert identities[str(DATA/'data.npz')]==DATA_SHA and identities[str(CHECKPOINT)]==CKSHA
        meta=json.loads(inputs[1].read_text());assert meta['audit_sha256']==identities[str(inputs[0])]
        config=json.loads(inputs[2].read_text());assert config['control_cache_sha256']==identities[str(source/'CONTROL_CACHE.pt')]
        for path,digest in config['source_hashes'].items():assert sha(REPO/path)==digest,path
        audit=torch.load(inputs[0],map_location='cpu',weights_only=False)
        state={k:v.cuda() for k,v in audit['heads'][-1].items()};moments=audit['final_optimizer']
        assert tensor_hash(state)==meta['head_end']==config['head_start']
        del audit
        cache=torch.load(source/'CONTROL_CACHE.pt',map_location='cpu',weights_only=False)
        d=np.load(DATA/'data.npz');ids,n=old.task_bank(d,TASK);assert ids.tolist()==config['bank_ids'] and n==7833
        iso=old.isolation(d)
        pos={i:j for j,i in enumerate(cache['indices'])};h=cache['features'][[pos[int(i)] for i in ids]].cuda()
        y=torch.from_numpy(d['y'][ids]).cuda();R=random_map().cuda()
        tests=preflight(state,moments,h,R)
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        m=FrozenRelationModel().cuda();m.load_state_dict(saved['model']);freeze(m);m.eval();del saved
        bert_hash=tensor_hash(m.bert.state_dict());assert bert_hash==meta['backbone_sha256']
        batches=order(d,TASK);assert batches.shape==(210,16)
        x=torch.from_numpy(d['x']);cached=[]
        with torch.no_grad():
            for index,batch in enumerate(batches):
                cached.append(m.features(x[batch].cuda()).cpu())
                if index%70==0:print('CACHE',index,flush=True)
            train_h=torch.cat(cached).cuda()
            assert torch.equal(train_h[:16],m.features(x[batches[0]].cuda()))
        # Validate all initial control/training outputs before formal optimization.
        for arm in ARMS:
            head=JointHead(state).cuda()
            with torch.no_grad():
                for features in (h,train_h):
                    assert torch.equal(head(joint_features(features,R,arm)),nn.functional.linear(features,state['weight'],state['bias']))
            del head
        save_tensor(ROOT/'FEATURES.pt',dict(R=R.cpu(),train_features=train_h.cpu(),train_ids=batches.tolist(),bank_ids=ids.tolist()))
        pre=dict(status='PASS',base=BASE,sources=hashes(),inputs=identities,tests=tests,isolation=iso,
            bert_hash=bert_hash,feature_sha256=sha(ROOT/'FEATURES.pt'),initial_logits_exact=True,
            production_projection_bytecode_unchanged=True,created_unix=time.time())
        write(ROOT/'PREFLIGHT.json',pre);guard(pre)
        print('PREFLIGHT PASS',flush=True)
        write(ROOT/'EXECUTION.json',dict(started_unix=time.time(),arms=ARMS,task=TASK))
        heads={};summaries={}
        for arm in ARMS:
            guard(pre);heads[arm],summaries[arm]=train(arm,state,moments,d,train_h,R,h,y,n,batches)
            assert tensor_hash(m.bert.state_dict())==bert_hash
            assert all(p.grad is None and not p.requires_grad for p in m.bert.parameters())
        scores=evaluate(m,heads,R,d)
        assert tensor_hash(m.bert.state_dict())==bert_hash
        guard(pre);assert identities=={str(p):sha(p) for p in inputs}
        original=json.loads((source/'evaluation_task8/EVALUATION.json').read_text())['arms']['final_architecture_v1']['metrics']
        original_summary=json.loads(inputs[3].read_text())['summary']
        write(ROOT/'RESULTS.json',dict(status='COMPLETED',original=original,original_summary=original_summary,
            arms={a:dict(metrics=scores[a],summary=summaries[a]) for a in ARMS},finished_unix=time.time(),
            no_test_tuning=True,inputs_unchanged=True,backbone_unchanged=True))
        print(json.dumps({a:{k:scores[a]['macro_'+k] for k in ('old','new','all')} for a in ARMS}),flush=True)
    except Exception:
        write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise

if __name__=='__main__':main()
