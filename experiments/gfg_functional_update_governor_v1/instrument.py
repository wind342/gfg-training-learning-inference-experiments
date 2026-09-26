"""Measurement only. Reuses v2 ownership, directional adapter and tiled arithmetic.

Candidate count is variable. FOUR coordinate terms are not four actions.
"""
from dataclasses import dataclass, field
import math
import torch
import torch.distributed as dist
from experiments.gfg_controller_distributed_runtime_v2.boundary import WorkspacePlanner,target_payload,same_probes
from experiments.gfg_controller_distributed_runtime_v2.scientific import endpoints
from .governor import ALPHAS


@dataclass
class Measurement:
    alphas: tuple
    baseline_minima: torch.Tensor
    predicted_minima: torch.Tensor
    metadata: dict = field(default_factory=dict)


@torch.no_grad()
def measure_boundaries(z,y,w,dw,hs,*,alphas=ALPHAS,start=0,group=None,mode='exact',planner=None):
    alphas=tuple(float(a) for a in alphas)
    if not alphas or len(set(alphas))!=len(alphas) or any(not math.isfinite(a) or a<=0 for a in alphas):
        raise ValueError('invalid measurement amplitudes')
    if mode not in ('exact','scalarized','streaming'):raise ValueError(mode)
    if group is not None and dist.get_world_size(group)==1:group=None
    if w.dtype!=torch.float32:raise TypeError('FP32 parameter contract')
    n=len(y);v=len(w)
    if not n or z.shape!=(n,v) or w.shape!=dw.shape or len(hs)!=3 or any(h.shape!=(n,w.shape[1]) for h in hs):
        raise ValueError('invalid measurement shapes')
    if group is None and (v<2 or (y<start).any() or (y>=start+v).any()):raise ValueError('invalid targets/vocabulary')
    # Existing planner's reserve dwarfs the small extra candidate array, but still
    # explicitly reserve its additional bytes before asking it for a tile.
    planner=planner or WorkspacePlanner()
    extra=max(0,len(alphas)+1-4)*n*8
    bounded=WorkspacePlanner(planner.max_bytes-extra,planner.free_fraction,planner.reserve+extra)
    config=bounded.plan(n,v,w.shape[1],w.device,streaming=mode=='streaming')
    target,payload,_,communication=target_payload(w,dw,z,y,start,group=group,dedup=True,mode=mode,hs=hs)
    amplitudes=(0.,)+alphas
    minima=torch.full((len(amplitudes),n),math.inf,device=w.device,dtype=torch.float64)
    valid=torch.ones((),device=w.device,dtype=torch.bool)
    bt,bc=config['target_tile'],config['competitor_tile']
    rounded=lambda x:x.float().double()
    for a in range(0,n,bt):
        end=min(n,a+bt);hh=[x[a:end].double() for x in hs]
        if mode!='scalarized':
            rows,inverse=payload;wy=rows[0,inverse[a:end]].double();dy=rows[1,inverse[a:end]].double()
        for c in range(0,v,bc):
            ce=min(v,c+bc);wc=w[c:ce].double();dc=dw[c:ce].double()
            g0=target[a:end,None].double()-z[a:end,c:ce].double()
            if mode=='exact':
                diff=wy[:,None]-wc[None];dd=dy[:,None]-dc[None]
                dot=lambda x,k:torch.bmm(x,hh[k].unsqueeze(-1)).squeeze(-1)
                first=rounded(dot(dd,0)+dot(diff,1));joint=rounded(dot(dd,1));curv=rounded(.5*dot(diff,2))
                del diff,dd
            elif mode=='scalarized':
                first=rounded((payload[0,a:end,None]+payload[1,a:end,None])-(hh[0]@dc.T+hh[1]@wc.T))
                joint=rounded(payload[2,a:end,None]-hh[1]@dc.T)
                curv=rounded(.5*(payload[3,a:end,None]-hh[2]@wc.T))
            else:
                # These FOUR terms are coordinate accumulators, not alpha axes.
                accum=[torch.zeros_like(g0) for _ in range(4)]
                for k in range(0,w.shape[1],config['hidden_chunk']):
                    ke=min(w.shape[1],k+config['hidden_chunk'])
                    diff=wy[:,None,k:ke]-wc[None,:,k:ke];dd=dy[:,None,k:ke]-dc[None,:,k:ke]
                    dot=lambda x,j:torch.bmm(x,hh[j][:,k:ke,None]).squeeze(-1)
                    accum[0].add_(dot(dd,0));accum[1].add_(dot(diff,1))
                    accum[2].add_(dot(dd,1));accum[3].add_(dot(diff,2))
                    del diff,dd
                first=rounded(accum[0]+accum[1]);joint=rounded(accum[2]);curv=rounded(.5*accum[3])
            own=y[a:end,None]==torch.arange(start+c,start+ce,device=y.device)[None]
            values=(g0,first,joint,curv)
            for value in values:valid=valid&torch.isfinite(value).all()
            for j,alpha in enumerate(amplitudes):
                gap=g0 if alpha==0 else endpoints(*values,alpha)
                valid=valid&torch.isfinite(gap).all()
                minima[j,a:end]=torch.minimum(minima[j,a:end],gap.masked_fill(own,math.inf).amin(1))
    if group is not None:
        dist.all_reduce(minima,op=dist.ReduceOp.MIN,group=group)
        flag=valid.int();dist.all_reduce(flag,op=dist.ReduceOp.MIN,group=group);valid=flag.bool()
    if not bool(valid) or not bool(torch.isfinite(minima).all()):raise FloatingPointError('nonfinite measurement; abort update')
    communication.update(min_payload_bytes=minima.numel()*8,min_collectives=2 if group is not None else 0,full_vocab_gather=False)
    return Measurement(alphas,minima[0],minima[1:],{'workspace':config,'communication':communication,'backend':mode})


class FunctionalResponseInstrument:
    def __init__(self,adapter,*,alphas=ALPHAS,backend='exact',budget=64*1024**2,group=None):
        self.adapter=adapter;self.alphas=tuple(alphas);self.backend=backend
        self.planner=WorkspacePlanner(budget);self.group=group;self.bank_identity=None

    @torch.no_grad()
    def measure(self,owner,probes,targets):
        owner.model.eval()
        if self.group is not None:
            identity=(id(probes),id(targets),probes._version,targets._version)
            if identity!=self.bank_identity:same_probes(probes,targets,self.group);self.bank_identity=identity
        owner.events.clear()
        if owner.packer is not None:owner.packer.events.clear()
        try:
            native,hs=self.adapter.jets(owner,probes)
            start,w,dw=owner.readout_shard(self.adapter.readout_name)
            z=torch.nn.functional.linear(native,w)[:,-1,:]
            result=measure_boundaries(z,targets[:,-1],w,dw,hs,alphas=self.alphas,start=start,group=self.group,
                mode=self.backend,planner=self.planner)
            result.metadata['stage_event_count']=len(owner.events)
            return result
        finally:
            # Diagnostic telemetry is bounded to a single measurement.
            owner.events.clear()
            if owner.packer is not None:owner.packer.events.clear()
