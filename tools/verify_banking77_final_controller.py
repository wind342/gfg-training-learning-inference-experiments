"""Small integration check only: 20 task2 updates from each existing checkpoint."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,ast,json,gzip,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from generation_relation_core.banking77_controller import Controller
from generation_relation_core.banking77_controller import runtime
from experiments.gfg_banking77_clean_timing_task2_v1 import common as old
from experiments.gfg_banking77_clean_timing_task2_v1.preflight import install
from experiments.gfg_banking77_clean_timing_task2_v1.audit import endpoint,count

def nodes(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf8')).body if hasattr(n,'name')}

def main():
    access=install();runtime.setup();package=ROOT/'src/generation_relation_core/banking77_controller'
    prior=ROOT/'experiments/gfg_banking77_clean_timing_task2_v1'
    for source,target,names in [('reference.py','geometry.py',('gelu','gelu_prime','inner','Geometry','select')),
        ('cutter.py','solver.py',('ball_hildreth',)),('common.py','runtime.py',('model','optimizer','native_step','Bank','Meter','functional','params','cpu','commit','setup'))]:
        a,b=nodes(prior/source),nodes(package/target)
        for name in names:assert a[name]==b[name],name
    for p in package.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
    X,y,controls=old.load();yc=y.cpu().numpy();ids=old.np.flatnonzero((yc<7)&controls)
    current=old.np.flatnonzero((yc//7==1)&~controls);reports=[]
    for budget in (210,630):
        with gzip.open(old.PREVIOUS/f'firstcheck/budget{budget}/STEPS.jsonl.gz','rt',encoding='utf8') as f:
            refs=[json.loads(next(f)) for _ in range(20)]
        state=old.torch.load(old.OLD/f'formal/native_{budget}/task1/CHECKPOINT.pt',weights_only=True)
        head=runtime.model();head.load_state_dict(state['model']);shadow=old.copy.deepcopy(head)
        opt=runtime.optimizer(shadow);opt.load_state_dict(state['optimizer'])
        controller=Controller(head,X[ids],y[ids],ids,14);events={};first_failed=0;audited=0
        for step,batch in enumerate(old.schedule(current,2,budget)[:20],1):
            ref=refs[step-1];assert batch.tolist()==ref['batch_ids'] and not controls[batch].any()
            assert old.phash(runtime.params(head))==ref['before_sha256']
            capture=step in (4,15,20)
            out=controller.step(head,shadow,opt,X[batch],y[batch],capture_evidence=capture)
            assert out['loss']==ref['current_loss']
            assert old.phash(runtime.params(shadow))==ref['native_sha256']
            assert old.opt_hash(opt)==ref['moments_sha256'] and old.opt_steps(opt)==ref['optimizer_steps']
            assert old.phash(runtime.params(head))==ref['executed_sha256']
            d=ref['decision'];s=out['summary']
            for name in ('native_counts','triggered','rollback','failure_reason','selected_controls','final_active','outer_iterations','inner_sweeps'):assert s[name]==d[name],name
            assert s['rounds']==[dict(outer=q['outer'],status=q['solver']['status'],sweeps=q['solver']['sweeps'],decision=q['decision']) for q in d['rounds']]
            for q in s['rounds']:events[q['decision']]=events.get(q['decision'],0)+1
            first_failed+=sum(not q['first_sweep_check']['safe'] for q in d['rounds'])
            if capture:assert out['evidence']==d;audited+=1
            else:assert 'evidence' not in out
            actual=endpoint(runtime.params(head),controller.bank.x,70);C=count(actual,controller.bank.labels)
            assert old.ahash(actual)==ref['endpoint_sha256'] and C.tolist()==out['counts']==ref['counts']
            assert old.np.all(C>=controller.K)
        reports.append(dict(checkpoint_budget=budget,steps=20,status='PASS',events=events,failed_first_checks=first_failed,
            optional_evidence_checks=audited,default_no_evidence_steps=20-audited,native_executed_moments_identical=True,
            selection_sweep_decisions_identical=True,independent_endpoints_identical=True,floor_violations=0,old_ce_backward_examples=0))
    result=dict(status='PASS',scope='40 updates only; integration verification, no benchmark or experiment',reports=reports,
        numerical_components_ast_identical=True,source_sha256={p.name:old.sha(p) for p in package.glob('*.py')},access=access.report())
    target=package/'VALIDATION.json'
    if target.exists():print('Existing validation artifact preserved; current recheck PASS')
    else:
        with target.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2)
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
