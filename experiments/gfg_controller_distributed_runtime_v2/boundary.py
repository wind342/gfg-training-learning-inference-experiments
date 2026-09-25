"""Bounded exact arithmetic, deduplicated targets, and explicitly candidate backends."""
from dataclasses import dataclass
import math
import torch
import torch.distributed as dist
from .scientific import ALPHAS,endpoints,select


@dataclass
class WorkspacePlanner:
    max_bytes:int=64*1024**2
    free_fraction:float=.1
    reserve:int=16*1024**2

    @staticmethod
    def estimate(bt,bc,h,chunk=None):return 8*(4*bt*bc*(h if chunk is None else chunk)+8*(bt+bc)*h+32*bt*bc)+4*1024**2

    def plan(self,n,v,h,device,*,streaming=False):
        free=torch.cuda.mem_get_info(device)[0] if device.type=='cuda' else 2**40
        budget=min(self.max_bytes,int(max(0,free-self.reserve)*self.free_fraction))
        bt=min(n,256);bc=min(max(v,1),128);width=min(h,128) if streaming else h
        payload=2*n*h*4+n*4+4*n*8
        estimate=lambda:self.estimate(bt,bc,h,width)+payload
        while estimate()>budget and (bt>1 or bc>1):
            if bt>=bc and bt>1:bt=max(1,bt//2)
            else:bc=max(1,bc//2)
        predicted=estimate()
        if predicted>budget:raise MemoryError('workspace budget cannot fit the minimum exact tile')
        return {'target_tile':bt,'competitor_tile':bc,'predicted_workspace':predicted,'budget':budget,'hidden_chunk':width}


def same_probes(inputs,targets,group):
    """Small-bank identity check once per bank; never hashes a model."""
    import hashlib
    if dist.get_world_size(group)==1:return
    stamp=(tuple(inputs.shape),tuple(targets.shape),hashlib.sha256(inputs.detach().cpu().contiguous().numpy().tobytes()+targets.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    all_stamps=[None]*dist.get_world_size(group);dist.all_gather_object(all_stamps,stamp,group=group)
    if any(s!=stamp for s in all_stamps):raise ValueError('MIN group requires identical ordered probes on every rank')


@torch.no_grad()
def target_payload(w,dw,z,y,start,*,group=None,dedup=True,mode='exact',hs=None):
    if group is not None and dist.get_world_size(group)==1:group=None
    own=(y>=start)&(y<start+len(w));ids=y[own]-start
    score=z.new_zeros(len(y));score[own]=z[own,ids]
    if mode=='scalarized':
        h,hp,hpp=[x.double() for x in hs];parts=w.new_zeros((4,len(y)),dtype=torch.float64)
        wy=w[ids].double();dy=dw[ids].double()
        parts[0,own]=(dy*h[own]).sum(-1);parts[1,own]=(wy*hp[own]).sum(-1)
        parts[2,own]=(dy*hp[own]).sum(-1);parts[3,own]=(wy*hpp[own]).sum(-1)
        if group is not None:dist.all_reduce(score,group=group);dist.all_reduce(parts,group=group)
        return score,parts,None,{'target_elements':len(y),'target_bytes':score.numel()*score.element_size()+parts.numel()*parts.element_size(),'target_collectives':2 if group is not None else 0}
    unique,inverse=torch.unique(y,sorted=True,return_inverse=True) if dedup else (y,torch.arange(len(y),device=y.device))
    mask=(unique>=start)&(unique<start+len(w));ii=unique[mask]-start
    rows=w.new_zeros((2,len(unique),w.shape[1]));rows[0,mask]=w[ii];rows[1,mask]=dw[ii]
    if group is not None:dist.all_reduce(score,group=group);dist.all_reduce(rows,group=group)
    return score,(rows,inverse),None,{'target_elements':len(unique),'target_bytes':score.numel()*score.element_size()+rows.numel()*rows.element_size(),'target_collectives':2 if group is not None else 0}


@torch.no_grad()
def calculate(z,y,w,dw,hs,*,start=0,group=None,mode='exact',dedup=True,planner=None,audit=False):
    if group is not None and dist.get_world_size(group)==1:group=None
    if mode not in ('exact','scalarized','streaming'):raise ValueError(mode)
    if w.dtype!=torch.float32:raise TypeError('only FP32 parameters / FP64 coordinate arithmetic certified')
    h,hp,hpp=hs;n=len(y);v=len(w);planner=planner or WorkspacePlanner()
    config=planner.plan(n,v,w.shape[1],w.device,streaming=mode=='streaming')
    baseline=torch.cuda.memory_allocated(w.device) if w.is_cuda else 0
    if w.is_cuda and audit:torch.cuda.reset_peak_memory_stats(w.device)
    target,payload,_,communication=target_payload(w,dw,z,y,start,group=group,dedup=dedup,mode=mode,hs=hs)
    minima=torch.full((4,n),math.inf,device=w.device,dtype=torch.float64);valid=torch.ones((),device=w.device,dtype=torch.bool)
    coords=torch.empty((4,n,v),device=w.device,dtype=torch.float64) if audit else None
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
                accum=[torch.zeros_like(g0) for _ in range(4)]
                for k in range(0,w.shape[1],config['hidden_chunk']):
                    ke=min(w.shape[1],k+config['hidden_chunk'])
                    diff=wy[:,None,k:ke]-wc[None,:,k:ke];dd=dy[:,None,k:ke]-dc[None,:,k:ke]
                    dot=lambda x,j:torch.bmm(x,hh[j][:,k:ke,None]).squeeze(-1)
                    accum[0].add_(dot(dd,0));accum[1].add_(dot(diff,1));accum[2].add_(dot(dd,1));accum[3].add_(dot(diff,2))
                    del diff,dd
                first=rounded(accum[0]+accum[1]);joint=rounded(accum[2]);curv=rounded(.5*accum[3])
            own=y[a:end,None]==torch.arange(start+c,start+ce,device=y.device)[None]
            values=(g0,first,joint,curv)
            for value in values:valid=valid&torch.isfinite(value).all()
            for j,alpha in enumerate(ALPHAS):minima[j,a:end]=torch.minimum(minima[j,a:end],endpoints(*values,alpha).masked_fill(own,math.inf).amin(1))
            if audit:coords[:,a:end,c:ce]=torch.stack(values)
    if group is not None:
        dist.all_reduce(minima,op=dist.ReduceOp.MIN,group=group)
        flag=valid.int();dist.all_reduce(flag,op=dist.ReduceOp.MIN,group=group);valid=flag.bool()
    alpha,scores=select(minima,valid)
    communication.update({'min_payload_bytes':minima.numel()*8,'min_collectives':2 if group is not None else 0,'full_vocab_gather':False,'wire_bytes':'NOT_MEASURED'})
    config['measured_peak_increment']=max(0,torch.cuda.max_memory_allocated(w.device)-baseline) if w.is_cuda and audit else None
    return alpha,scores,minima,coords,{'workspace':config,'communication':communication,'backend':mode}
