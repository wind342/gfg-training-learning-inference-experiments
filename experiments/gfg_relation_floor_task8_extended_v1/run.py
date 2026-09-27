"""Fixed extra two epochs, importing the parent's policy without modifications."""
from experiments.gfg_relation_capability_floor_task8_v1.core import *
from experiments.gfg_relation_capability_floor_task8_v1.core import ROOT as PARENT
from experiments.gfg_relation_capability_floor_task8_v1.check import independent
import traceback
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/relation-floor-task8-extended-v1')
BASE='7ae5104125c2838d456c240f60018b57001ee481'

def sources():
    files=[REPO/x for x in subprocess.check_output(['git','ls-files','experiments'],cwd=REPO,text=True).splitlines() if x.endswith('.py')]
    files+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {x.relative_to(REPO).as_posix():sha(x) for x in sorted(set(files))}

def guard(pre):
    assert sources()==pre['sources']
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)

def bank(d,pre):
    cache=torch.load(old.ROOT/'CONTROL_CACHE.pt',map_location='cpu',weights_only=False)
    ids=pre['bank_ids'];n=pre['old_count'];pos={i:j for j,i in enumerate(cache['indices'])}
    return cache['features'][[pos[i] for i in ids]].cuda(),torch.from_numpy(d['y'][ids]).cuda(),ids,n

def epoch(m,shadow,opt,d,h,y,ids,n,K,number,pre):
    folder=ROOT/f'epoch{number}';folder.mkdir()
    head=m.head;initial_opt=copy.deepcopy(opt.state_dict());start_head=p.snapshot(head)
    assert old.opt_counter(opt)==1470+(number-1)*210
    heads=[p.cpu_state(start_head)];natives=[];projected=[];gradients=[];rows=[]
    x=torch.from_numpy(d['x']);truth=torch.from_numpy(d['y']);labels=y[:n].cpu().numpy()
    torch.cuda.reset_peak_memory_stats();p.sync(h);tick=time.perf_counter()
    with (folder/'STEPS.jsonl').open('x',encoding='utf8') as f:
        for local_step,batch in enumerate(order(d,8),1):
            step=(number-1)*210+local_step;global_step=1470+step;seed=2021*100000+global_step
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m.train();head.zero_grad(set_to_none=True)
            assert all(not mod.training for mod in m.bert.modules())
            with torch.no_grad():features=m.features(x[batch].cuda())
            loss=nn.functional.cross_entropy(head(features)[:,:64],truth[batch].cuda());loss.backward()
            norm=nn.utils.clip_grad_norm_(head.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm)
            gradients.append({name:q.grad.detach().cpu().clone() for name,q in head.named_parameters()})
            before=p.snapshot(head);propose(head,shadow,opt);native=p.snapshot(shadow);mh=moments_hash(opt)
            assert old.opt_counter(opt)==global_step
            executed,candidate,r=floor_projection(before,native,h,y,n,K,ids[:n])
            assert mh==moments_hash(opt)
            with torch.no_grad():
                for name,q in head.named_parameters():q.copy_(executed[name])
                actual=margins(head(h)[:,:64],y).cpu().tolist()
            assert actual==r['executed'] and np.all(counts(np.array(actual[:n]),labels,56)>=K)
            assert all(q.grad is None and not q.requires_grad for q in m.bert.parameters())
            row=dict(step=step,epoch=number,current_ids=batch.tolist(),loss=float(loss.detach()),gradient_norm_before_clip=float(norm),
                before_sha256=tensor_hash(before),native_sha256=tensor_hash(native),executed_sha256=head_hash(head),
                optimizer_sha256=mh,optimizer_step=global_step,K=K.tolist(),native_counts=r['native_counts'],
                native_deficits=r['native_deficits'],executed_counts=r['executed_counts'],projection=r['projection_triggered'],
                skip=r['skip'],sweeps=r['sweep_count'],active_constraints=len(r['final_active']),floor_violations=r['floor_violations'],
                correct_to_wrong=r['correct_to_wrong'],wrong_to_correct=r['wrong_to_correct'],migration_relations=r['migration_relations'],
                count_slack_relations=r['count_slack_relations'],geometry=p.geometry(before,native,executed),projection_seconds=r['projection_seconds'])
            f.write(json.dumps(dict(**row,report=r),separators=(',',':'),allow_nan=False)+'\n');f.flush()
            rows.append(row);heads.append(p.cpu_state(executed));natives.append(p.cpu_state(native));projected.append(p.cpu_state(candidate))
            if local_step==1 or local_step%70==0:print('TRAIN',step,'skip',sum(v['skip'] for v in rows),'sweeps',r['sweep_count'],flush=True)
    p.sync(h);wall=time.perf_counter()-tick
    assert tensor_hash(m.bert.state_dict())==pre['backbone_sha256']
    old.save_tensor(folder/'AUDIT.pt',dict(heads=heads,natives=natives,projected=projected,clipped_gradients=gradients,
        initial_optimizer=initial_opt,final_optimizer=copy.deepcopy(opt.state_dict())))
    selected=[v for v in rows if v['projection']]
    summary=dict(total_task8_steps=number*210,steps_this_epoch=210,K=K.tolist(),final_counts=r['executed_counts'],
        skip=sum(v['skip'] for v in rows),floor_violations=sum(v['floor_violations'] for v in rows),
        correct_to_wrong=sum(v['correct_to_wrong'] for v in rows),wrong_to_correct=sum(v['wrong_to_correct'] for v in rows),
        migration_relations=sorted(set(i for v in rows for i in v['migration_relations'])),projection_triggered=len(selected),
        active_constraints_median=float(np.median([v['active_constraints'] for v in selected])) if selected else 0,
        norm_ratio=float(np.mean([v['geometry']['norm_ratio'] for v in selected])) if selected else 1,
        cosine=float(np.mean([v['geometry']['cosine'] for v in selected])) if selected else 1,
        wall=wall,projection_wall=sum(v['projection_seconds'] for v in rows),peak_cuda_bytes=torch.cuda.max_memory_allocated())
    write(folder/'COMPACT_STEPS.json',rows)
    write(folder/'TRAIN_RESULT.json',dict(status='COMPLETED',summary=summary,audit_sha256=sha(folder/'AUDIT.pt'),
        log_sha256=sha(folder/'STEPS.jsonl'),head_start=tensor_hash(start_head),head_end=head_hash(head),finished_unix=time.time()))
    write(folder/'EVALUATION_BARRIER.json',dict(train_result_sha256=sha(folder/'TRAIN_RESULT.json'),created_unix=time.time()))
    m.eval();test_ids=scope(d,64)[1];pred=[];hh=head_hash(head);mh=moments_hash(opt);rr=rng()
    with torch.no_grad():
        for begin in range(0,len(test_ids),16):pred+=head(m.features(x[test_ids[begin:begin+16]].cuda()))[:,:64].argmax(1).cpu().tolist()
    assert hh==head_hash(head) and mh==moments_hash(opt) and equal(rr,rng())
    score=summarize_predictions(test_ids,pred,d,8)
    write(folder/'EVALUATION.json',dict(indices=test_ids.tolist(),predictions=pred,metrics=score,readonly=True,finished_unix=time.time()))
    print('EVALUATE',number*210,{k:score['macro_'+k] for k in ('old','new','all')},flush=True)
    return dict(metrics=score,summary=summary)

def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    ROOT.mkdir(exist_ok=False)
    try:
        p.setup();parent=json.loads((PARENT/'PREFLIGHT.json').read_text())
        assert json.loads((PARENT/'CHECKS.json').read_text())['status']=='PASS'
        for path,digest in parent['sources'].items():assert sha(REPO/path)==digest
        inputs=[PARENT/name for name in ('PREFLIGHT.json','CHECKS.json','AUDIT.pt','TRAIN_RESULT.json','RESULTS.json')]
        inputs+=[old.ROOT/'CONTROL_CACHE.pt',DATA/'data.npz',CHECKPOINT]
        identities={str(path):sha(path) for path in inputs}
        assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
        assert sha(old.ROOT/'CONTROL_CACHE.pt')==parent['inputs'][str(old.ROOT/'CONTROL_CACHE.pt')]
        meta=json.loads((PARENT/'TRAIN_RESULT.json').read_text());assert sha(PARENT/'AUDIT.pt')==meta['audit_sha256']
        audit=torch.load(PARENT/'AUDIT.pt',map_location='cpu',weights_only=False)
        state=audit['heads'][-1];moments=audit['final_optimizer'];assert tensor_hash(state)==meta['final_head']
        del audit
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);m=FrozenRelationModel().cuda();m.load_state_dict(saved['model']);del saved
        freeze(m);m.head.load_state_dict(state);m.seen=64;head=m.head;shadow=copy.deepcopy(head)
        opt=optimizer(shadow.parameters());opt.load_state_dict(moments)
        assert old.opt_counter(opt)==1680 and equal(opt.state_dict(),moments) and opt.param_groups[0]['lr']==5e-5
        assert tensor_hash(m.bert.state_dict())==parent['backbone_sha256']
        d=np.load(DATA/'data.npz');h,y,ids,n=bank(d,parent);K=np.array(parent['K']);assert len(K)==56 and K.sum()==6835 and n==7833
        now=p.margins(p.affine(p.snapshot(head),h,64),y).cpu().numpy()
        assert counts(now[:n],y[:n].cpu().numpy(),56).tolist()==meta['summary']['final_counts']
        pre=dict(status='PASS',base=BASE,sources=sources(),inputs=identities,bank_ids=ids,old_count=n,K=K.tolist(),
            backbone_sha256=parent['backbone_sha256'],head_start=head_hash(head),moments_start=moments_hash(opt),
            start_counter=1680,end_counter=2100,total_task8_steps=630,created_unix=time.time())
        write(ROOT/'PREFLIGHT.json',pre);guard(pre)
        write(ROOT/'EXECUTION.json',dict(started_unix=time.time(),fixed_endpoints=[420,630]))
        print('PREFLIGHT PASS continuation1680->2100 Ksum6835',flush=True)
        results={}
        for number in (2,3):
            guard(pre);results[str(number*210)]=epoch(m,shadow,opt,d,h,y,ids,n,K,number,pre)
        assert old.opt_counter(opt)==2100 and tensor_hash(m.bert.state_dict())==pre['backbone_sha256']
        guard(pre);assert identities=={str(path):sha(path) for path in inputs}
        parent_result=json.loads((PARENT/'RESULTS.json').read_text())
        write(ROOT/'RESULTS.json',dict(status='COMPLETED',parent210=parent_result['relation_floor'],original210=parent_result['original'],
            continuation=results,backbone_unchanged=True,inputs_unchanged=True,no_test_tuning=True))
    except Exception:
        write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise

def check():
    p.setup();pre=json.loads((ROOT/'PREFLIGHT.json').read_text());guard(pre)
    assert pre['inputs']=={q:sha(q) for q in pre['inputs']}
    d=np.load(DATA/'data.npz');h,y,ids,n=bank(d,pre);labels=y[:n].cpu().numpy();K=np.array(pre['K'])
    parent_pre=json.loads((PARENT/'PREFLIGHT.json').read_text());assert pre['K']==parent_pre['K']
    parent_audit=torch.load(PARENT/'AUDIT.pt',map_location='cpu',weights_only=False)
    previous=parent_audit['heads'][-1];previous_opt=parent_audit['final_optimizer'];del parent_audit
    steps=sweeps=checks=0
    for number in (2,3):
        folder=ROOT/f'epoch{number}';meta=json.loads((folder/'TRAIN_RESULT.json').read_text())
        assert sha(folder/'AUDIT.pt')==meta['audit_sha256'] and sha(folder/'STEPS.jsonl')==meta['log_sha256']
        audit=torch.load(folder/'AUDIT.pt',map_location='cpu',weights_only=False)
        assert equal(previous,audit['heads'][0]) and equal(previous_opt,audit['initial_optimizer'])
        head=nn.Linear(1536,80).cuda();opt=optimizer(head.parameters());opt.load_state_dict(audit['initial_optimizer'])
        for j,line in enumerate((folder/'STEPS.jsonl').open()):
            row=json.loads(line);r=row['report'];before={k:v.cuda() for k,v in audit['heads'][j].items()};head.load_state_dict(before)
            assert row['current_ids']==order(d,8)[j].tolist() and row['K']==K.tolist()
            for name,q in head.named_parameters():q.grad=audit['clipped_gradients'][j][name].cuda()
            opt.step();native=p.snapshot(head);assert equal(native,audit['natives'][j]) and moments_hash(opt)==row['optimizer_sha256']
            e,c,selections,logs,bm,nm,cm,em=independent(before,native,h,y,n,K,ids[:n])
            assert equal(e,audit['heads'][j+1]) and equal(c,audit['projected'][j]) and selections==r['selections'] and logs==r['sweeps']
            assert bm.tolist()==r['baseline'] and nm.tolist()==r['native'] and cm.tolist()==r['projected'] and em.tolist()==r['executed']
            C=np.bincount(labels[em[:n]>0],minlength=56);assert np.all(C>=K) and C.tolist()==r['executed_counts']
            assert old.opt_counter(opt)==1470+row['step'];steps+=1;sweeps+=len(logs);checks+=56
            if (j+1)%70==0:print('CHECK',row['step'],flush=True)
        assert equal(opt.state_dict(),audit['final_optimizer']);previous=audit['heads'][-1];previous_opt=audit['final_optimizer']
        ev=json.loads((folder/'EVALUATION.json').read_text());barrier=json.loads((folder/'EVALUATION_BARRIER.json').read_text())
        assert barrier['train_result_sha256']==sha(folder/'TRAIN_RESULT.json') and barrier['created_unix']<ev['finished_unix']
        assert summarize_predictions(ev['indices'],ev['predictions'],d,8)==ev['metrics']
    guard(pre)
    write(ROOT/'CHECKS.json',dict(status='PASS',native_candidates=steps,independent_sweeps=sweeps,
        relation_step_floor_checks=checks,floor_violations=0,continuous_head_and_moments=True,original_floor_unchanged=True,
        same_batch_order=True,metrics_recomputed=True,source_and_input_identity=True))
    print('CHECKS PASS',steps,sweeps,checks,flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='check':check()
    else:main()
