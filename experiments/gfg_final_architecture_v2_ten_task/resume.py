"""One documented harness-only recovery; reuse completed task2, never retrain it."""
from .core import *
from .run import train_task,evaluate
import gc
import traceback

def main():
    p.setup();original=json.loads((ROOT/'PREFLIGHT.json').read_text())
    assert (ROOT/'FAILURE.json').exists() and not (ROOT/'PREFLIGHT_RESUME.json').exists()
    assert not (ROOT/'COMPLETE.json').exists()
    assert all(sha(q)==v for q,v in original['inputs'].items())
    allowed={f'experiments/{HERE.name}/{name}' for name in ('core.py','run.py','check.py')}
    changed={q for q,v in original['sources'].items() if sha(REPO/q)!=v}
    assert changed==allowed
    for name in ('core.py','run.py','check.py'):
        assert sha(ROOT/'harness_revision1'/f'{name}.txt')==original['sources'][f'experiments/{HERE.name}/{name}']
    d=np.load(DATA/'data.npz');cache=torch.load(old.ROOT/'CONTROL_CACHE.pt',map_location='cpu',weights_only=False)
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    m=FrozenRelationModel().cuda();m.load_state_dict(saved['model']);freeze(m);del saved
    states={};restores={};diagnostics={};h,y,ids,n=bank(d,cache,3)
    for arm in ARMS:
        folder=ROOT/arm/'task2';part=folder/f'epoch{ARMS[arm]}'
        meta=json.loads((part/'TRAIN_RESULT.json').read_text());assert sha(part/'AUDIT.pt')==meta['audit_sha256']
        a=torch.load(part/'AUDIT.pt',map_location='cpu',weights_only=False)
        head=nn.Linear(1536,80).cuda();head.load_state_dict(a['heads'][-1]);shadow=copy.deepcopy(head)
        opt=optimizer(shadow.parameters());opt.load_state_dict(a['final_optimizer'])
        assert old.opt_counter(opt)==counter(arm,3) and head_hash(head)==meta['head_end']
        assert equal(opt.state_dict(),a['final_optimizer'])
        states[arm]=(head,shadow,opt)
        small=old.transition(head,h,y,n,3);full=boundary_same_batch(head,h,y,n,3)
        actual=p.margins(p.affine(p.snapshot(head),h,24),y).cpu().tolist()[:n]
        assert full['after']==actual
        a_small=np.asarray(small['after']);a_full=np.asarray(actual)
        assert np.array_equal(a_small>0,a_full>0)
        diagnostics[arm]=dict(max_abs_roundoff=float(np.max(np.abs(a_small-a_full))),
            different_values=int((a_small!=a_full).sum()),changed_correctness=0)
        restores[arm]=dict(head=head_hash(head),moments=moments_hash(opt),counter=old.opt_counter(opt),
            source_audit_sha256=meta['audit_sha256'])
        del a
    assert tensor_hash(m.bert.state_dict())==original['backbone_sha256']
    frozen={str(q):sha(q) for a in ARMS for q in (ROOT/a/'task2').rglob('*') if q.is_file()}
    frozen.update({str(q):sha(q) for q in (ROOT/'evaluation_task2').iterdir() if q.is_file()})
    record=dict(reason='Task3 bookkeeping compared FP32 margins from different GEMM batch sizes; no correctness differences.',
        production_policy_changed=False,formal_updates_repeated=0,completed_task2_preserved=frozen,
        original_preflight_sha256=sha(ROOT/'PREFLIGHT.json'),original_failure_sha256=sha(ROOT/'FAILURE.json'),
        changed_harness_sources=sorted(changed),diagnostics=diagnostics,restores=restores,
        new_sources=sources(),resumed_task=3,created_unix=time.time())
    write(ROOT/'HARNESS_RECOVERY.json',record)
    pre=copy.deepcopy(original);pre['sources']=sources();pre['harness_recovery_sha256']=sha(ROOT/'HARNESS_RECOVERY.json')
    write(ROOT/'PREFLIGHT_RESUME.json',pre);guard(pre,True)
    try:
        print('HARNESS RECOVERY PASS task2 preserved; resume task3',flush=True)
        for task in range(3,11):
            h,y,ids,n=bank(d,cache,task)
            for arm in ARMS:train_task(m,states[arm],arm,task,d,h,y,ids,n,pre)
            evaluate(m,states,task,d);del h,y;gc.collect()
        assert old.opt_counter(states['v2_210'][2])==2100 and old.opt_counter(states['v2_630'][2])==5880
        assert tensor_hash(m.bert.state_dict())==pre['backbone_sha256'];guard(pre,True)
        assert all(sha(q)==v for q,v in frozen.items())
        write(ROOT/'COMPLETE.json',dict(status='COMPLETED',arms=ARMS,tasks=list(range(2,11)),updates=7560,
            harness_recovery=True,formal_updates_repeated=0,finished_unix=time.time()))
    except Exception:
        write(ROOT/'RESUME_FAILURE.json',dict(traceback=traceback.format_exc(),unix=time.time()));raise

if __name__=='__main__':main()
