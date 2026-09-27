"""One unchanged-head task8 trajectory under a fixed relation floor."""
import traceback
from .core import *
from .preflight import tests

def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    ROOT.mkdir(exist_ok=False)
    try:
        p.setup();source=old.ROOT
        inputs=[source/'final_architecture_v1/task7/AUDIT.pt',source/'final_architecture_v1/task7/TRAIN_RESULT.json',
            source/'final_architecture_v1/task8/CONFIG.json',source/'CONTROL_CACHE.pt',DATA/'data.npz',CHECKPOINT]
        identities={str(q):sha(q) for q in inputs};assert identities[str(DATA/'data.npz')]==DATA_SHA and identities[str(CHECKPOINT)]==CKSHA
        meta=json.loads(inputs[1].read_text());assert meta['audit_sha256']==identities[str(inputs[0])]
        config=json.loads(inputs[2].read_text());assert config['control_cache_sha256']==identities[str(inputs[3])]
        for path,digest in config['source_hashes'].items():assert sha(REPO/path)==digest,path
        d=np.load(DATA/'data.npz');isolation=old.isolation(d)
        ids,n=old.task_bank(d,8);assert n==7833 and ids.tolist()==config['bank_ids']
        cache=torch.load(inputs[3],map_location='cpu',weights_only=False);pos={i:j for j,i in enumerate(cache['indices'])}
        h=cache['features'][[pos[int(i)] for i in ids]].cuda();y=torch.from_numpy(d['y'][ids]).cuda()
        audit=torch.load(inputs[0],map_location='cpu',weights_only=False)
        state=audit['heads'][-1];moments=audit['final_optimizer'];assert tensor_hash(state)==meta['head_end']==config['head_start']
        del audit
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        m=FrozenRelationModel().cuda();m.load_state_dict(saved['model']);freeze(m);m.head.load_state_dict(state);m.seen=64;del saved
        bert_hash=tensor_hash(m.bert.state_dict());assert bert_hash==meta['backbone_sha256']
        head=m.head;shadow=copy.deepcopy(head);opt=optimizer(shadow.parameters());opt.load_state_dict(moments)
        assert equal(opt.state_dict(),moments) and old.opt_counter(opt)==1470 and opt.param_groups[0]['lr']==5e-5
        entry=p.margins(p.affine(p.snapshot(head),h,64),y).cpu().numpy();labels=y[:n].cpu().numpy()
        K=counts(entry[:n],labels,56);assert K.sum()==6835
        assert entry.tolist()==config['transition']['after']+entry[n:].tolist()
        pre=dict(status='PASS',base=BASE,inputs=identities,sources=sources(),tests=tests(),isolation=isolation,
            backbone_sha256=bert_hash,bank_ids=ids.tolist(),old_count=n,K=K.tolist(),
            head_start=head_hash(head),moments_start=moments_hash(opt),created_unix=time.time())
        write(ROOT/'PREFLIGHT.json',pre);guard(pre);print('PREFLIGHT PASS',K.tolist(),flush=True)
        write(ROOT/'EXECUTION.json',dict(started_unix=time.time(),task=8,steps=210))
        initial_opt=copy.deepcopy(opt.state_dict());heads=[p.cpu_state(p.snapshot(head))];natives=[];projected=[];grads=[];compact=[]
        x=torch.from_numpy(d['x']);truth=torch.from_numpy(d['y']);bank_hash=tensor_hash({'h':h})
        torch.cuda.reset_peak_memory_stats();p.sync(h);start=time.perf_counter()
        with (ROOT/'STEPS.jsonl').open('x',encoding='utf8') as f:
            for step,batch in enumerate(order(d,8),1):
                global_step=1470+step;seed=2021*100000+global_step
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m.train();head.zero_grad(set_to_none=True)
                assert all(not mod.training for mod in m.bert.modules())
                with torch.no_grad():features=m.features(x[batch].cuda())
                loss=nn.functional.cross_entropy(head(features)[:,:64],truth[batch].cuda());loss.backward()
                norm=nn.utils.clip_grad_norm_(head.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm)
                grads.append({name:q.grad.detach().cpu().clone() for name,q in head.named_parameters()})
                before=p.snapshot(head);propose(head,shadow,opt);native=p.snapshot(shadow);mh=moments_hash(opt)
                assert old.opt_counter(opt)==global_step
                executed,candidate,report=floor_projection(before,native,h,y,n,K,ids[:n])
                assert mh==moments_hash(opt)
                with torch.no_grad():
                    for name,q in head.named_parameters():q.copy_(executed[name])
                    actual=margins(head(h)[:,:64],y).cpu().tolist()
                assert actual==report['executed'] and np.all(counts(np.array(actual[:n]),labels,56)>=K)
                assert all(q.grad is None and not q.requires_grad for q in m.bert.parameters())
                geom=p.geometry(before,native,executed)
                small=dict(step=step,current_ids=batch.tolist(),loss=float(loss.detach()),gradient_norm_before_clip=float(norm),
                    before_sha256=tensor_hash(before),native_sha256=tensor_hash(native),executed_sha256=head_hash(head),
                    optimizer_sha256=mh,optimizer_step=global_step,K=K.tolist(),native_counts=report['native_counts'],
                    native_deficits=report['native_deficits'],executed_counts=report['executed_counts'],
                    projection=report['projection_triggered'],skip=report['skip'],sweeps=report['sweep_count'],
                    active_constraints=len(report['final_active']),floor_violations=report['floor_violations'],
                    correct_to_wrong=report['correct_to_wrong'],wrong_to_correct=report['wrong_to_correct'],
                    migration_relations=report['migration_relations'],count_slack_relations=report['count_slack_relations'],
                    geometry=geom,projection_seconds=report['projection_seconds'])
                f.write(json.dumps(dict(**small,report=report),separators=(',',':'),allow_nan=False)+'\n');f.flush()
                compact.append(small);heads.append(p.cpu_state(executed));natives.append(p.cpu_state(native));projected.append(p.cpu_state(candidate))
                if step==1 or step%30==0:print('TRAIN',step,'skip',sum(v['skip'] for v in compact),'CtoW',sum(v['correct_to_wrong'] for v in compact),'sweeps',report['sweep_count'],flush=True)
        p.sync(h);wall=time.perf_counter()-start
        assert tensor_hash(m.bert.state_dict())==bert_hash and tensor_hash({'h':h})==bank_hash
        old.save_tensor(ROOT/'AUDIT.pt',dict(heads=heads,natives=natives,projected=projected,clipped_gradients=grads,
            initial_optimizer=initial_opt,final_optimizer=copy.deepcopy(opt.state_dict())))
        selected=[r for r in compact if r['projection']]
        entry_good=entry[:n]>0;final_good=np.array(actual[:n])>0
        entry_lost=np.bincount(labels[entry_good&~final_good],minlength=56);entry_gained=np.bincount(labels[~entry_good&final_good],minlength=56)
        summary=dict(K=K.tolist(),final_counts=report['executed_counts'],steps=210,skip=sum(r['skip'] for r in compact),
            floor_violation_relation_steps=sum(r['floor_violations'] for r in compact),
            correct_to_wrong=sum(r['correct_to_wrong'] for r in compact),wrong_to_correct=sum(r['wrong_to_correct'] for r in compact),
            same_step_migration_relations=sorted(set(i for r in compact for i in r['migration_relations'])),
            task_entry_to_end_replacement_relations=np.flatnonzero((entry_lost>0)&(entry_gained>0)).tolist(),
            entry_lost=entry_lost.tolist(),entry_gained=entry_gained.tolist(),
            count_slack_relation_steps=sum(len(r['count_slack_relations']) for r in compact),
            projection_triggered=len(selected),active_constraints_median=float(np.median([r['active_constraints'] for r in selected])) if selected else 0,
            norm_ratio=float(np.mean([r['geometry']['norm_ratio'] for r in selected])) if selected else 1.,
            cosine=float(np.mean([r['geometry']['cosine'] for r in selected])) if selected else 1.,
            wall=wall,projection_wall=sum(r['projection_seconds'] for r in compact),peak_cuda_bytes=torch.cuda.max_memory_allocated())
        write(ROOT/'COMPACT_STEPS.json',compact)
        write(ROOT/'TRAIN_RESULT.json',dict(status='COMPLETED',summary=summary,audit_sha256=sha(ROOT/'AUDIT.pt'),
            log_sha256=sha(ROOT/'STEPS.jsonl'),final_head=head_hash(head),finished_unix=time.time()))
        write(ROOT/'EVALUATION_BARRIER.json',dict(train_result_sha256=sha(ROOT/'TRAIN_RESULT.json'),created_unix=time.time()))
        m.eval();test_ids=scope(d,64)[1];pred=[];mh=moments_hash(opt);hh=head_hash(head)
        with torch.no_grad():
            for begin in range(0,len(test_ids),16):pred+=head(m.features(x[test_ids[begin:begin+16]].cuda()))[:,:64].argmax(1).cpu().tolist()
        score=summarize_predictions(test_ids,pred,d,8)
        assert mh==moments_hash(opt) and hh==head_hash(head) and tensor_hash(m.bert.state_dict())==bert_hash
        write(ROOT/'EVALUATION.json',dict(indices=test_ids.tolist(),predictions=pred,metrics=score,readonly=True,finished_unix=time.time()))
        guard(pre);assert identities=={str(q):sha(q) for q in inputs}
        original=json.loads((source/'evaluation_task8/EVALUATION.json').read_text())['arms']['final_architecture_v1']['metrics']
        write(ROOT/'RESULTS.json',dict(status='COMPLETED',original=original,relation_floor=score,summary=summary,
            backbone_unchanged=True,inputs_unchanged=True,no_test_tuning=True))
        print('RESULT',json.dumps({k:score['macro_'+k] for k in ('old','new','all')}),flush=True)
    except Exception:
        write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise

if __name__=='__main__':main()
