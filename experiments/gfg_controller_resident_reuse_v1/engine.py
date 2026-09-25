"""Resident buffers and shared directional differentiation, not a new rule."""
from collections import defaultdict
import numpy as np
import torch
from torch.func import functional_call, jvp
from torch.nn.attention import sdpa_kernel, SDPBackend
from experiments.gfg_nanogpt_joint_rotation_branch_v1.analysis import HiddenStateModel
from experiments.gfg_controller_batched_coordinates_v1.kernel import boundary_coordinates


class ResidentBuffers:
    """Named parameters grouped by device/dtype; no model-size or task constants.

    Flat slabs are retained on the original device. Foreach performs only copies
    and separate subtract/multiply/add operations, not fused AdamW arithmetic.
    CPU snapshots are immutable independent transfers, shared by named views.
    """
    def __init__(self, model):
        groups=defaultdict(list)
        for name,p in model.named_parameters():
            if not p.is_floating_point() or p.layout!=torch.strided:
                raise ValueError('requires dense real floating-point parameters')
            groups[(p.device,p.dtype)].append((name,p))
        self.groups=[]
        self.named={kind:{} for kind in ('before','delta','proposal','scaled','audit')}
        for (device,dtype),items in groups.items():
            offsets=[]; count=0
            for name,p in items:
                offsets.append((name,count,count+p.numel(),p.shape)); count+=p.numel()
            slabs={kind:torch.empty(count,device=device,dtype=dtype) for kind in self.named}
            views={kind:[slabs[kind][lo:hi].view(shape) for _,lo,hi,shape in offsets] for kind in slabs}
            for kind in views:
                self.named[kind].update({name:v for (name,_),v in zip(items,views[kind])})
            self.groups.append({'parameters':[p for _,p in items],'offsets':offsets,'slabs':slabs,'views':views})

    @torch.no_grad()
    def capture(self, kind):
        for group in self.groups:
            torch._foreach_copy_(group['views'][kind],group['parameters'])

    @torch.no_grad()
    def form_delta(self):
        for group in self.groups:
            torch._foreach_copy_(group['views']['delta'],group['parameters'])
            torch._foreach_sub_(group['views']['delta'],group['views']['before'])

    @torch.no_grad()
    def apply(self,alpha):
        if alpha==1.: return  # preserve native optimizer endpoint bytes
        for group in self.groups:
            torch._foreach_copy_(group['views']['scaled'],group['views']['delta'])
            torch._foreach_mul_(group['views']['scaled'],alpha)
            torch._foreach_add_(group['views']['scaled'],group['views']['before'])
            torch._foreach_copy_(group['parameters'],group['views']['scaled'])

    def snapshot(self,kind):
        result={}
        for group in self.groups:
            slab=group['slabs'][kind]
            # CPU models also require a distinct snapshot, not a view of a slab.
            cpu=slab.detach().cpu() if slab.device.type!='cpu' else slab.detach().clone()
            result.update({name:cpu[lo:hi].view(shape) for name,lo,hi,shape in group['offsets']})
        return result

    def snapshot_model(self):
        self.capture('audit')
        return self.snapshot('audit')


def shared_derivatives(forward, parameters, tangent):
    """Outer JVP returns h' and h''; retain h as its nondifferentiated aux.

    This avoids the separate first JVP formerly executed before the nested JVP.
    No derivative is approximated or omitted. Backend policy belongs to adapter.
    """
    def first_with_hidden(values):
        hidden,first=jvp(forward,(values,),(tangent,))
        return first,hidden
    first,second,hidden=jvp(first_with_hidden,(parameters,),(tangent,),has_aux=True)
    return hidden,first,second


class NanoGPTAdapter:
    """Architecture-specific acquisition, separate from generic boundary core."""
    def __init__(self,model):
        self.wrapper=HiddenStateModel(model)

    def coordinates(self,model,parameters,delta,inputs,targets):
        names=tuple(n for n,_ in model.named_parameters())
        assert set(names)==set(parameters)==set(delta)
        prefixed={'gpt.'+n:parameters[n] for n in names}
        tangent={'gpt.'+n:delta[n] for n in names}
        prior=model.training; model.eval()
        try:
            with torch.no_grad():
                recorded,_=functional_call(model,parameters,(inputs,targets),tie_weights=True,strict=True)
                recorded=recorded[:,-1,:].cpu().numpy().astype(np.float64)
                def forward(values):
                    return functional_call(self.wrapper,values,(inputs,),tie_weights=True)
                with sdpa_kernel(SDPBackend.MATH):
                    hs=shared_derivatives(forward,prefixed,tangent)
                # One transfer for the three equally shaped hidden arrays.
                packed=torch.stack(hs).detach().cpu().numpy().astype(np.float64)
        finally:model.train(prior)
        w=parameters['transformer.wte.weight'].detach().cpu().numpy()
        dw=delta['transformer.wte.weight'].detach().cpu().numpy()
        return boundary_coordinates(recorded,targets[:,-1].cpu().numpy(),w,dw,*packed)


class Controller:
    """Original cadence/selection supplied by caller; unchanged native optimizer."""
    def __init__(self,model,adapter,select,hash_parameters,selection_boundary,cadence=5):
        self.buffers=ResidentBuffers(model)
        self.adapter=adapter; self.select=select; self.phash=hash_parameters
        self.selection_boundary=selection_boundary; self.cadence=cadence
        self.reused_post_snapshot=None

    def update(self,model,opt,t,x,y,vx,vy,seed,alpha):
        self.reused_post_snapshot=None
        model.train(); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        self.buffers.capture('before')
        opt.zero_grad(set_to_none=True)
        loss=model(x,y)[1]; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        details={'loss':float(loss.detach())}
        self.buffers.form_delta()
        coords=None
        if (t-1)%self.cadence==0:
            self.buffers.capture('proposal')
            with self.selection_boundary():
                coords=self.adapter.coordinates(model,self.buffers.named['before'],self.buffers.named['delta'],vx,vy)
                alpha,scores=self.select(coords)
            # Preserve BOTH hashes and the mutation check, with packed copies.
            actual=self.buffers.snapshot_model()
            expected=self.buffers.snapshot('proposal')
            assert self.phash(actual)==self.phash(expected),'SELECTOR_MUTATION'
            details['scores']=scores
            if alpha==1.:self.reused_post_snapshot=actual
        details['alpha']=alpha
        self.buffers.apply(alpha)
        return alpha,details,coords

    def post_snapshot(self):
        # Safe only when no mutation occurred since the internal-audit snapshot.
        if self.reused_post_snapshot is not None:return self.reused_post_snapshot
        return self.buffers.snapshot_model()
