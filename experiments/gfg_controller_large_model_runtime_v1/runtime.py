"""Proposal/adjudication/commit with two parameter-sized temporary buffers.

The certified path preserves native fused proposal bytes and reference arithmetic.
No model-scale, task, target-count or vocabulary-size constants are embedded here.
"""
from collections import defaultdict
import numpy as np
import torch
from torch.func import functional_call, jvp
from torch.nn.attention import sdpa_kernel, SDPBackend
from experiments.gfg_controller_resident_reuse_v1.engine import shared_derivatives
from experiments.gfg_nanogpt_joint_rotation_branch_v1.analysis import HiddenStateModel

ALPHAS=(.25,.5,.75,1.)
KEYS=('g0','first_gap','joint_rotation','hidden_curvature')


class ProposalStorage:
    def __init__(self,model):
        groups=defaultdict(list)
        for name,p in model.named_parameters():
            assert p.is_floating_point() and p.layout==torch.strided and p.is_contiguous()
            groups[(p.device,p.dtype)].append((name,p))
        self.groups=[];self.proposal={};self.delta={}
        for key,items in groups.items():
            size=sum(p.numel() for n,p in items)
            candidate=torch.empty(size,device=key[0],dtype=key[1]);direction=torch.empty_like(candidate)
            cv=[];dv=[];start=0
            for name,p in items:
                end=start+p.numel();c=candidate[start:end].view_as(p);d=direction[start:end].view_as(p)
                cv.append(c);dv.append(d);self.proposal[name]=c;self.delta[name]=d;start=end
            self.groups.append((items,candidate,direction,cv,dv))
        self.allocated_bytes=sum(c.numel()*c.element_size()+d.numel()*d.element_size() for _,c,d,_,_ in self.groups)

    @torch.no_grad()
    def form(self,opt):
        originals=[]
        try:
            for items,c,d,cv,dv in self.groups:
                torch._foreach_copy_(cv,[p for n,p in items])
                for (n,p),view in zip(items,cv):
                    originals.append((p,p.data));p.data=view
            # Parameter objects, gradients, native recurrence and group order unchanged.
            opt.step()
        finally:
            for p,data in originals:p.data=data
        for items,c,d,cv,dv in self.groups:
            torch._foreach_copy_(dv,cv)
            torch._foreach_sub_(dv,[p for n,p in items])

    @torch.no_grad()
    def commit(self,alpha):
        for items,c,d,cv,dv in self.groups:
            if alpha==1.:
                torch._foreach_copy_([p for n,p in items],cv)
            else:
                # The delta lifetime ends after adjudication: reuse it as scaled endpoint.
                torch._foreach_mul_(dv,alpha)
                torch._foreach_add_(dv,[p for n,p in items])
                torch._foreach_copy_([p for n,p in items],dv)

    def snapshot_model(self):
        # AUDIT-only CPU materialization; no resident GPU audit slab.
        return {n:p.detach().cpu().clone() for items,*_ in self.groups for n,p in items}


def streamed_boundary(logits,targets,w,dw,h,hp,hpp,*,target_tile=32,competitor_tile=8,retain_coordinates=False):
    logits,w,dw,h,hp,hpp=[np.asarray(v,dtype=np.float64) for v in (logits,w,dw,h,hp,hpp)]
    targets=np.asarray(targets);n,v=logits.shape
    assert targets.shape==(n,) and target_tile>0 and competitor_tile>0
    assert all(np.isfinite(z).all() for z in (logits,w,dw,h,hp,hpp))
    assert w.shape==dw.shape and w.shape[0]==v and h.shape==hp.shape==hpp.shape==(n,w.shape[1])
    minima={a:np.full(n,np.inf) for a in ALPHAS}
    coords={k:np.empty((n,v-1),dtype=np.float64) for k in KEYS} if retain_coordinates else None
    # Keep the original competitor order for coordinate identity in AUDIT mode.
    competitors=np.empty((n,v-1),dtype=np.int64) if retain_coordinates else None
    for start in range(0,n,target_tile):
        end=min(n,start+target_tile);yy=targets[start:end];scores=logits[start:end]
        order=np.argsort(scores,axis=1)[:,::-1]
        cs=order[order!=yy[:,None]].reshape(end-start,v-1)
        if retain_coordinates:competitors[start:end]=cs
        for left in range(0,v-1,competitor_tile):
            right=min(v-1,left+competitor_tile);cc=cs[:,left:right]
            diff=w[yy,None,:]-w[cc];dd=dw[yy,None,:]-dw[cc]
            dot=lambda q,z:np.matmul(q,z[start:end,:,None])[...,0]
            rounded=lambda z:z.astype(np.float32).astype(np.float64)
            c={'g0':scores[np.arange(end-start),yy,None]-np.take_along_axis(scores,cc,axis=1),
                'first_gap':rounded(dot(dd,h)+dot(diff,hp)),
                'joint_rotation':rounded(dot(dd,hp)),
                'hidden_curvature':rounded(.5*dot(diff,hpp))}
            assert all(np.isfinite(z).all() for z in c.values())
            for a in ALPHAS:
                gap=((c['g0']+a*c['first_gap'])+a*a*c['joint_rotation'])+a*a*c['hidden_curvature']
                minima[a][start:end]=np.minimum(minima[a][start:end],gap.min(1))
            if retain_coordinates:
                for k in KEYS:coords[k][start:end,left:right]=c[k]
    scores={str(a):{'predicted_correct_count':int(np.count_nonzero(minima[a]>0)),
        'median_predicted_minimum_margin':float(np.median(minima[a]))} for a in ALPHAS}
    alpha=max(ALPHAS,key=lambda a:(scores[str(a)]['predicted_correct_count'],scores[str(a)]['median_predicted_minimum_margin'],-abs(a-1.)))
    return alpha,scores,coords,minima,competitors


def blockwise_hidden(model,parameters,delta,inputs):
    """Experimental second-order jet composition; acceptance requires byte checks."""
    emb_names=('transformer.wte.weight','transformer.wpe.weight')
    ep={n:parameters[n] for n in emb_names};ed={n:delta[n] for n in emb_names}
    positions=torch.arange(inputs.shape[1],device=inputs.device)
    def embed(p):
        return torch.nn.functional.embedding(inputs,p[emb_names[0]])+torch.nn.functional.embedding(positions,p[emb_names[1]])
    h,hp,hpp=shared_derivatives(embed,ep,ed)
    modules=[(f'transformer.h.{i}.',block) for i,block in enumerate(model.transformer.h)]
    modules.append(('transformer.ln_f.',model.transformer.ln_f))
    for prefix,block in modules:
        p={n[len(prefix):]:z for n,z in parameters.items() if n.startswith(prefix)}
        d={n[len(prefix):]:z for n,z in delta.items() if n.startswith(prefix)}
        def first_with_primal(x,pars,tangent):
            out,first=jvp(lambda xx,pp:functional_call(block,pp,(xx,),strict=True),(x,pars),(tangent,d))
            return first,out
        hp,hpp,h=jvp(first_with_primal,(h,p,hp),(hp,d,hpp),has_aux=True)
    return tuple(z[:,-1,:] for z in (h,hp,hpp))


class RuntimeController:
    def __init__(self,model,*,audit=False,hash_parameters=None,cadence=5,check_blockwise=False):
        self.buffers=ProposalStorage(model);self.audit=audit;self.phash=hash_parameters
        self.cadence=cadence;self.identity_steps_skipped=0;self.check_blockwise=check_blockwise
        self.last_debug=None

    def response(self,model,inputs,targets):
        parameters={n:p.detach() for n,p in model.named_parameters()};delta=self.buffers.delta
        prefix={'gpt.'+n:z for n,z in parameters.items()};direction={'gpt.'+n:z for n,z in delta.items()}
        model.eval()
        with torch.no_grad():
            # Preserve reference's full-sequence head call and final-token extraction.
            recorded=functional_call(model,parameters,(inputs,targets),tie_weights=True,strict=True)[0][:,-1,:].cpu().numpy().astype(np.float64)
            wrapper=HiddenStateModel(model)
            with sdpa_kernel(SDPBackend.MATH):
                hs=shared_derivatives(lambda p:functional_call(wrapper,p,(inputs,),tie_weights=True),prefix,direction)
                block_hs=blockwise_hidden(model,parameters,delta,inputs) if self.check_blockwise else None
            arrays=torch.stack(hs).cpu().numpy().astype(np.float64)
            w=parameters['transformer.wte.weight'].cpu().numpy();dw=delta['transformer.wte.weight'].cpu().numpy()
            result=streamed_boundary(recorded,targets[:,-1].cpu().numpy(),w,dw,*arrays,retain_coordinates=self.audit)
            if self.audit:
                self.last_debug={'hidden':arrays,'pre_logits':recorded,'minima':result[3],
                    'blockwise_exact':[torch.equal(a,z) for a,z in zip(hs,block_hs)] if block_hs is not None else None,
                    'blockwise_max_abs':[float((a-z).abs().max()) for a,z in zip(hs,block_hs)] if block_hs is not None else None}
        return result

    def update(self,model,opt,t,x,y,vx,vy,seed,alpha):
        self.last_debug=None
        model.train();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        opt.zero_grad(set_to_none=True);loss=model(x,y)[1];loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        details={'loss':float(loss.detach())}
        if (t-1)%self.cadence!=0 and alpha==1.:
            opt.step();self.identity_steps_skipped+=1;details['alpha']=alpha
            return alpha,details,None
        self.buffers.form(opt)
        coords=None
        if (t-1)%self.cadence==0:
            before_hash=self.phash(self.buffers.snapshot_model()) if self.audit else None
            alpha,scores,coords,_,_=self.response(model,vx,vy)
            if self.audit:assert self.phash(self.buffers.snapshot_model())==before_hash,'SELECTOR_MUTATION'
            details['scores']=scores
        self.buffers.commit(alpha);details['alpha']=alpha
        return alpha,details,coords
