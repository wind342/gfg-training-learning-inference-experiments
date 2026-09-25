"""External gates against e49; runtime does not import this module."""
import copy
import gc
from datetime import timedelta
from unittest.mock import patch
import numpy as np
import torch
import torch.distributed as dist
from .evidence import HERE,OUT,write,sha,setup
from .controller import FunctionalController
from .adapters import NanoGPTAdapter
from .identity import CanonicalParameterRegistry
from .boundary import calculate,WorkspacePlanner
from experiments.gfg_controller_distributed_runtime_v1.checks import old,b
from experiments.gfg_controller_gpu_boundary_v1.runtime import GPUBlockwiseController,gpu_boundary


def cpu(p):return (p.full_tensor() if hasattr(p,'full_tensor') else p).detach().cpu().clone()
def snapshot(model):return {n:cpu(p) for n,p in model.named_parameters()}


def expanded(values,parameters):
    from torch.distributed.tensor import DTensor
    return {n:cpu(DTensor.from_local(v,device_mesh=parameters[n].device_mesh,placements=parameters[n].placements,
        shape=parameters[n].shape,stride=parameters[n].stride(),run_check=False)) if hasattr(parameters[n],'placements') else cpu(v) for n,v in values.items()}


def one_gate(lr,backend,fsdp=False):
    label=f'{"fsdp1" if fsdp else "single"}-{backend}-{lr}';path=OUT/'gates'/label
    path.mkdir(parents=True);raw={};rows=[];capture={}
    row=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs'][0];rng=b.read(old.HERE/'PHASE1_SEAL.json')['source_rng_protocol_sha256']
    rt,start,batches,vx,vy=b.load_row(row);model=rt.model;old.reset(model,start);oracle=copy.deepcopy(model)
    registry=CanonicalParameterRegistry(model);before_identity=registry.receipt(model);adapter=NanoGPTAdapter(model);units=None
    if fsdp:
        from torch.distributed.device_mesh import init_device_mesh
        dist.init_process_group('gloo',init_method=(path/'init').as_uri(),rank=0,world_size=1,timeout=timedelta(seconds=45))
        units=adapter.wrap_fsdp(model,init_device_mesh('cuda',(1,)))
    after_identity=registry.receipt(model);opt=torch.optim.AdamW(registry.resolve(model).values(),lr=lr,betas=(.9,.98),weight_decay=0,fused=True,foreach=False)
    oracle_opt=old.prior.native.optimizer(oracle,lr);reference=GPUBlockwiseController(oracle,audit=True,hash_parameters=b.phash)
    engine=FunctionalController(model,adapter,registry=registry,units=units,group=dist.group.WORLD if fsdp else None,backend=backend,audit=True)
    native_propose=engine.owner.propose
    def observed_proposal(o):
        native_propose(o);params=engine.owner.parameters()
        capture['candidate']=expanded(engine.owner.proposal.candidate,params);capture['delta']=expanded(engine.owner.proposal.delta,params)
    engine.owner.propose=observed_proposal;alpha=1.
    try:
        raw['targets']=vy[:,-1].cpu().numpy()
        for t in range(1,7):
            pre=snapshot(oracle);seed=b.old.seed_for(row,t-1,rng);candidate={};step=oracle_opt.step
            original_clip=torch.nn.utils.clip_grad_norm_;norms={}
            def observe_clip(which):
                def wrapped(parameters,*args,**kwargs):
                    parameters=list(parameters)
                    for i,p in enumerate(parameters):raw[f'{t}_{which}_raw_gradient_{i}']=cpu(p.grad).numpy()
                    value=original_clip(parameters,*args,**kwargs);norms[which]=cpu(value)
                    raw[f'{t}_{which}_norm']=norms[which].numpy()
                    return value
                return wrapped
            def observed(*args,**kwargs):
                result=step(*args,**kwargs);candidate.update(snapshot(oracle));return result
            with patch.object(oracle_opt,'step',observed),patch.object(torch.nn.utils,'clip_grad_norm_',observe_clip('reference')):
                alpha,info,_=reference.update(oracle,oracle_opt,t,*batches[t-1],vx,vy,seed,alpha)
            with patch.object(torch.nn.utils,'clip_grad_norm_',observe_clip('actual')):
                actual=engine.update(opt,t,*batches[t-1],vx,vy,seed,lr)
            post=snapshot(model);refpost=snapshot(oracle)
            checks={'candidate':True,'delta':True,'moments':True,'gradients':True,'norm':torch.equal(norms['actual'],norms['reference']),'parameters':True,'alpha':actual['alpha']==alpha,'scores':actual.get('scores')==info.get('scores')}
            for i,(n,p) in enumerate(registry.resolve(model).items()):
                refs=dict(oracle.named_parameters());rp=refs[n]
                pairs={'candidate':(capture['candidate'][n],candidate[n]),'delta':(capture['delta'][n],candidate[n]-pre[n]),
                    'parameters':(post[n],refpost[n]),'gradients':(cpu(p.grad),cpu(rp.grad))}
                for k in ('step','exp_avg','exp_avg_sq'):pairs['state_'+k]=(cpu(opt.state[p][k]),cpu(oracle_opt.state[rp][k]))
                for kind,(a,z) in pairs.items():
                    raw[f'{t}_{kind}_{i}']=a.numpy();raw[f'{t}_ref_{kind}_{i}']=z.numpy()
                    key='moments' if kind.startswith('state_') else kind;checks[key]=checks[key] and torch.equal(a,z)
            with torch.no_grad():
                z=model(vx)[0][:,-1,:];oz=oracle(vx)[0][:,-1,:]
            raw[f'{t}_logits']=z.cpu().numpy();raw[f'{t}_ref_logits']=oz.cpu().numpy();checks['logits']=torch.equal(z,oz)
            if engine.last:
                for j in range(3):
                    h=engine.last['hidden'][j];oh=reference.last_debug['hidden'][j]
                    raw[f'{t}_h{j}']=h.cpu().numpy();raw[f'{t}_ref_h{j}']=oh.cpu().numpy();checks[f'h{j}']=torch.equal(h,oh)
                m=engine.last['minima'];rm=reference.last_debug['minima']
                raw[f'{t}_minima']=m.cpu().numpy();raw[f'{t}_ref_minima']=rm.cpu().numpy();checks['minima']=torch.equal(m,rm)
            registry.resolve(model)
            rows.append({'step':t,'checks':checks,'actual':actual,'reference_alpha':alpha,'reference_scores':info.get('scores')})
        with (path/'RAW.npz').open('xb') as f:np.savez_compressed(f,**raw)
        result={'label':label,'backend':backend,'fsdp_world_size':1 if fsdp else None,'rows':rows,'raw_sha256':sha(path/'RAW.npz'),
            'all_exact':all(all(r['checks'].values()) for r in rows),'alpha_exact':all(r['checks']['alpha'] for r in rows),
            'trajectory_exact':all(r['checks']['parameters'] and r['checks']['logits'] for r in rows),
            'identity_before':before_identity,'identity_after':after_identity,'optimizer_parameter_count':len(opt.state),
            'canonical_count':len(registry.entries),'events':engine.owner.events,'packed':engine.owner.packer.events if fsdp else [],
            'parameter_names':list(registry.resolve(model))}
        write(path/'RESULT.json',result);return result
    finally:
        rt.close()
        if fsdp and dist.is_initialized():dist.destroy_process_group()


def synthetic():
    records=[]
    for h in (64,512,1024,2048,4096,8192):
        for seed in (111,222):
            torch.manual_seed(seed);n,v=33,73;device='cuda'
            w=torch.randn(v,h,device=device)*.05;dw=torch.randn_like(w)*.002
            hs=[torch.randn(n,h,device=device)*s for s in (1,.01,.001)]
            z=torch.randn(n,v,device=device)*(1 if seed==111 else 1e-7);y=torch.arange(n,device=device)%7
            if seed==222:
                preliminary=gpu_boundary(z,y,w,dw,*hs,audit=True)[2];ii=torch.arange(n,device=device);competitor=(y+1)%v;a=.25
                correction=a*preliminary['first_gap'][ii,competitor]+a*a*preliminary['joint_rotation'][ii,competitor]+a*a*preliminary['hidden_curvature'][ii,competitor]
                z.fill_(-100.);z[ii,y]=0.;z[ii,competitor]=correction.float()
            ref=gpu_boundary(z,y,w,dw,*hs,audit=True);raw={k:x.cpu().numpy() for k,x in zip(('z','y','w','dw','h','hp','hpp'),(z,y,w,dw,*hs))}
            refcoords=torch.stack([ref[2][k] for k in ('g0','first_gap','joint_rotation','hidden_curvature')]);raw['ref_coordinates']=refcoords.cpu().numpy();raw['ref_minima']=ref[3].cpu().numpy()
            case=[]
            for mode,dedup in [('exact',False),('exact',True),('scalarized',True),('streaming',True)]:
                a,s,m,c,meta=calculate(z,y,w,dw,hs,mode=mode,dedup=dedup,audit=True)
                key=f'{mode}-{dedup}';raw[key+'_coordinates']=c.cpu().numpy();raw[key+'_minima']=m.cpu().numpy()
                case.append({'mode':mode,'dedup':dedup,'coordinates_exact':torch.equal(c,refcoords),'minima_exact':torch.equal(m,ref[3]),'scores_exact':s==ref[1],
                    'alpha_exact':a==ref[0],'coordinate_max_abs':float((c-refcoords).abs().max()),'minima_max_abs':float((m-ref[3]).abs().max()),
                    'selected_alpha':a,'reference_alpha':ref[0],'metadata':meta,'within_workspace_budget':meta['workspace']['measured_peak_increment']<=meta['workspace']['budget']})
            path=OUT/'synthetic'/f'h{h}-seed{seed}';path.mkdir(parents=True)
            with (path/'RAW.npz').open('xb') as f:np.savez_compressed(f,**raw)
            result={'h':h,'seed':seed,'cases':case,'raw_sha256':sha(path/'RAW.npz')};write(path/'RESULT.json',result);records.append(result)
            print('SYNTHETIC',h,seed,[c['minima_exact'] for c in case],flush=True)
    return records
