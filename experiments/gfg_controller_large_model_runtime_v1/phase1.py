"""Byte-level architecture bridge against every frozen small-model trajectory."""
from pathlib import Path
from unittest.mock import patch
import gc
import subprocess
import traceback
import time
import numpy as np
import torch
from experiments.gfg_controller_fixed_shrinkage_control_v1 import run as prior
from experiments.gfg_controller_resident_reuse_v1.engine import Controller as OldController, NanoGPTAdapter
from experiments.gfg_controller_batched_coordinates_v1.kernel import boundary_coordinates
from experiments.gfg_nanogpt_functional_online_controller_v1.independent_checker import independently_choose
from .runtime import RuntimeController, ProposalStorage, streamed_boundary, KEYS, ALPHAS

b=prior.b; HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
OUT=Path('E:/gfg-evidence/controller-large-model-runtime-phase1-v1')
BASE='d17f4510adf46ebae4fc391382ca6be1134eec03'


def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def synthetic_tests():
    rng=np.random.default_rng(20260925);receipt=[]
    for n,v,h in [(1,2,3),(7,5,17),(53,11,32),(212,24,64),(7,4097,65)]:
        values=[rng.normal(size=s) for s in [(n,v),(v,h),(v,h),(n,h),(n,h),(n,h)]]
        targets=rng.integers(0,v,n);old=boundary_coordinates(values[0],targets,*values[1:])
        aa,ss,c,mm,cc=streamed_boundary(values[0],targets,*values[1:],retain_coordinates=True)
        assert all(np.array_equal(c[k],old[k]) for k in KEYS),('COORDINATES',n,v,h)
        assert np.array_equal(cc,old['competitors'])
        assert (aa,ss)==independently_choose(old)
        for a in ALPHAS:
            assert np.array_equal(mm[a],(((old['g0']+a*old['first_gap'])+a*a*old['joint_rotation'])+a*a*old['hidden_curvature']).min(1))
        aa2,ss2,c2,_,cc2=streamed_boundary(values[0],targets,*values[1:])
        assert (aa2,ss2)==(aa,ss) and c2 is None and cc2 is None
        receipt.append({'shape':[n,v,h],'exact':True})
    a,s,_,_,_=streamed_boundary(np.zeros((3,4)),np.array([0,1,3]),*[np.zeros(z) for z in [(4,5),(4,5),(3,5),(3,5),(3,5)]])
    assert a==1. and all(z['predicted_correct_count']==0 for z in s.values())
    for alpha in ALPHAS:
        torch.manual_seed(77);m=torch.nn.Linear(17,5).cuda();r=torch.nn.Linear(17,5).cuda();r.load_state_dict(m.state_dict())
        opt=prior.native.optimizer(m,.003);refopt=prior.native.optimizer(r,.003);storage=ProposalStorage(m)
        for t in range(3):
            for p,z in zip(m.parameters(),r.parameters()):
                p.grad=torch.randn_like(p);z.grad=p.grad.clone()
            before=b.params(r);refopt.step();proposal=b.params(r)
            storage.form(opt)
            assert b.phash(b.params(m))==b.phash(before)
            assert b.phash({n:z.cpu() for n,z in storage.proposal.items()})==b.phash(proposal)
            assert b.structure(opt.state_dict())==b.structure(refopt.state_dict())
            with torch.no_grad():
                if alpha!=1.:
                    for n,p in r.named_parameters():p.copy_((before[n]+alpha*(proposal[n]-before[n])).cuda())
            storage.commit(alpha)
            assert b.phash(b.params(m))==b.phash(b.params(r))
        assert storage.allocated_bytes==8*sum(p.numel() for p in m.parameters())
        receipt.append({'proposal_alpha':alpha,'three_steps_exact':True,'control_bytes':storage.allocated_bytes})
    return receipt


def evaluate(model,vx,vy):
    model.eval()
    with torch.no_grad():z=model(vx)[0][:,-1,:].cpu().numpy()
    return z,prior.native.eval_logits(z,vy)


def reset(model,start):
    with torch.no_grad():
        for n,p in model.named_parameters():p.copy_(start[n].cuda())
    model.zero_grad(set_to_none=True)


def execute(row,loaded,ci,mode,rng):
    import copy
    rt,start,batches,vx,vy=loaded;model=rt.model;reset(model,start)
    opt=prior.native.optimizer(model,prior.RATES[ci])
    engine=RuntimeController(model,audit=mode=='audit',hash_parameters=b.phash,check_blockwise=mode=='audit')
    oracle_model=copy.deepcopy(model) if mode=='audit' else None
    oracle_opt=prior.native.optimizer(oracle_model,prior.RATES[ci]) if oracle_model is not None else None
    oracle=prior.native.Controller(oracle_model,NanoGPTAdapter(oracle_model),b.select,b.phash,b.selection_boundary) if oracle_model is not None else None
    ref=prior.record('dynamic','search',row['entry_id'],ci)
    path=OUT/mode/row['entry_id']/str(ci);assert not path.exists();path.mkdir(parents=True)
    initial,acc=evaluate(model,vx,vy);assert b.tensor_sha256(initial)==ref['initial_logits_sha256']
    outputs=[initial];steps=[];coords_out={};hidden_out={};alpha=1.;oldalpha=1.;chain=None;starttime=time.perf_counter()
    for i,expected in enumerate(ref['steps'],1):
        x,y=batches[i-1];seed=b.old.seed_for(row,i-1,rng);lr=expected['lr']
        opt.param_groups[0]['lr']=lr
        proposal_hash=None;oc=None
        if oracle is not None:
            oracle_opt.param_groups[0]['lr']=lr;native_step=oracle_opt.step;capture={}
            def observed_step(*args,**kwargs):
                z=native_step(*args,**kwargs);capture['hash']=b.phash(b.params(oracle_model));return z
            with patch.object(oracle_opt,'step',observed_step):oldalpha,od,oc=oracle.update(oracle_model,oracle_opt,i,x,y,vx,vy,seed,oldalpha)
            proposal_hash=capture['hash']
        alpha,details,c=engine.update(model,opt,i,x,y,vx,vy,seed,alpha)
        assert details==expected['details'],'DETAILS_MISMATCH'
        if oracle is not None:
            assert details==od
            nh=b.phash(b.params(model)) if (i-1)%5!=0 and alpha==1. else b.phash({n:z.cpu() for n,z in engine.buffers.proposal.items()})
            assert nh==proposal_hash,'NATIVE_PROPOSAL_MISMATCH'
        coord_hash=None;block=None
        if mode=='audit' and c is not None:
            assert all(np.array_equal(c[k],oc[k]) for k in KEYS),'COORDINATE_MISMATCH'
            coord_hash=b.identity({k:b.tensor_sha256(torch.from_numpy(c[k])) for k in KEYS})
            assert coord_hash==expected['coordinates_sha256']
            debug=engine.last_debug
            assert all(np.array_equal(debug['hidden'][j],oc[k]) for j,k in enumerate(('hidden','hidden_first','hidden_second'))),'HIDDEN_MISMATCH'
            assert np.array_equal(debug['pre_logits'],oc['pre_logits'])
            for a in ALPHAS:
                margin=(((oc['g0']+a*oc['first_gap'])+a*a*oc['joint_rotation'])+a*a*oc['hidden_curvature']).min(1)
                assert np.array_equal(debug['minima'][a],margin)
            for k in KEYS:coords_out[f'{i}_{k}']=c[k]
            hidden_out[f'{i}_hidden']=debug['hidden'];hidden_out[f'{i}_pre_logits']=debug['pre_logits']
            block={'exact':debug['blockwise_exact'],'max_abs':debug['blockwise_max_abs']}
        post=b.phash(b.params(model));assert post==expected['post_parameters_sha256']
        logits,acc=evaluate(model,vx,vy);assert b.tensor_sha256(logits)==expected['logits_sha256'] and acc==expected['accuracy']
        outputs.append(logits)
        fact={'t':i,'seed':seed,'lr':lr,'details':details,'post_parameters_sha256':post,
            'logits_sha256':b.tensor_sha256(logits),'accuracy':acc,'coordinates_sha256':coord_hash,
            'native_proposal_sha256':proposal_hash,'blockwise':block,'previous_sha256':chain}
        chain=b.identity(fact);steps.append({**fact,'sha256':chain})
    final=b.identity(b.structure({'parameters':b.params(model),'optimizer':opt.state_dict()}))
    assert final==ref['final_state_sha256']
    if oracle is not None:assert b.structure(opt.state_dict())==b.structure(oracle_opt.state_dict())
    result={'mode':mode,'entry':row['entry_id'],'config':ci,'initial_logits_sha256':b.tensor_sha256(initial),
        'steps':steps,'final_state_sha256':final,'control_buffer_bytes':engine.buffers.allocated_bytes,
        'parameter_count':sum(p.numel() for p in model.parameters()),'diagnostic_bridge_seconds':time.perf_counter()-starttime}
    with (path/'OUTPUTS.npz').open('xb') as f:np.savez_compressed(f,logits=np.stack(outputs),targets=vy[:,-1].cpu().numpy())
    if coords_out:
        with (path/'COORDINATES.npz').open('xb') as f:np.savez_compressed(f,**coords_out)
        with (path/'HIDDEN.npz').open('xb') as f:np.savez_compressed(f,**hidden_out)
    result['artifacts']={p.name:b.sha(p) for p in path.iterdir()}
    b.write(path/'TRAJECTORY.json',result)
    print(mode,row['entry_id'],ci,len(steps),'EXACT',flush=True)
    return {'mode':mode,'entry':row['entry_id'],'config':ci,'updates':len(steps),'exact':True,
        'blockwise_nonexact_steps':[s['t'] for s in steps if s['blockwise'] is not None and not all(s['blockwise']['exact'])]}


def main():
    b.old.setup();assert git('rev-parse','HEAD')==BASE and not OUT.exists()
    with patch.object(b,'REPO',prior.native.resident.previous.DEPENDENCIES.parent):oldseal=b.sealcheck()
    names=git('ls-files').splitlines()
    seal={'base_commit':BASE,'utc':b.old.utc(),'sources':{p.name:b.sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md')},
        'baseline_python':{p:b.sha(REPO/p) for p in names if p.endswith('.py')},
        'assignment_sha256':b.sha(b.old.HERE/'RUN_ASSIGNMENT.json'),'source_rng_protocol_sha256':oldseal['source_rng_protocol_sha256'],
        'old_dependency_seal':oldseal,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
        'frozen_tag_object':git('rev-parse',prior.TAG),'frozen_tag_commit':git('rev-parse',prior.TAG+'^{}')}
    assert seal['frozen_tag_commit']==prior.FROZEN
    b.write(HERE/'PHASE1_SEAL.json',seal);OUT.mkdir()
    b.write(HERE/'SYNTHETIC_TESTS.json',synthetic_tests())
    rows=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs'];records=[]
    for row in rows:
        loaded=b.load_row(row)
        try:
            for ci in range(3):
                for mode in ('audit','runtime'):records.append(execute(row,loaded,ci,mode,seal['source_rng_protocol_sha256']))
        finally:loaded[0].close();del loaded;gc.collect();torch.cuda.empty_cache()
    blockfails=[r for r in records if r['blockwise_nonexact_steps']]
    b.write(HERE/'PHASE1_RESULTS.json',{'status':'COMPLETE_PENDING_INDEPENDENT_AUDIT','runtime_equivalence':'PASS',
        'blockwise_component':'NOT_CERTIFIED' if blockfails else 'PASS','records':records,
        'persistent_buffer_reduction':.6,'phase2':'NOT_STARTED','phase3':'NOT_STARTED'})


if __name__=='__main__':
    try:main()
    except Exception:
        with (HERE/f'FAILURE_{time.time_ns()}.txt').open('x',encoding='utf-8') as f:f.write(traceback.format_exc())
        raise
