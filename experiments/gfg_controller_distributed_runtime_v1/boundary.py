"""Vocabulary-local minima plus explicit global MIN; no full-vocabulary gather."""
import math
import torch
import torch.distributed as dist
from experiments.gfg_controller_large_model_runtime_v1.runtime import ALPHAS
from .storage import interval


def select(minima,valid):
    n=minima.shape[1];ordered=minima.sort(1).values
    median=ordered[:,n//2] if n%2 else (ordered[:,n//2-1]+ordered[:,n//2])/2
    payload=torch.cat(((minima>0).sum(1).double(),median,valid.reshape(1).double())).cpu().tolist()
    if payload[-1]!=1 or not all(math.isfinite(x) for x in payload):raise FloatingPointError('nonfinite boundary')
    scores={str(a):{'predicted_correct_count':int(payload[j]),'median_predicted_minimum_margin':payload[j+4]} for j,a in enumerate(ALPHAS)}
    alpha=max(ALPHAS,key=lambda a:(scores[str(a)]['predicted_correct_count'],scores[str(a)]['median_predicted_minimum_margin'],-abs(a-1.)))
    return alpha,scores


@torch.no_grad()
def local_minima(logits,targets,start,w,dw,target_logits,wy,dy,h,hp,hpp,target_tile=256,competitor_tile=128):
    n=len(targets);v=len(w);mins=torch.full((4,n),math.inf,dtype=torch.float64,device=h.device);valid=torch.ones((),dtype=torch.bool,device=h.device)
    for a in range(0,n,target_tile):
        z=min(n,a+target_tile);hs=[x[a:z].double().unsqueeze(-1) for x in (h,hp,hpp)]
        for left in range(0,v,competitor_tile):
            right=min(v,left+competitor_tile);diff=wy[a:z,None].double()-w[None,left:right].double();dd=dy[a:z,None].double()-dw[None,left:right].double()
            dot=lambda p,x:torch.bmm(p,x).squeeze(-1);rounded=lambda x:x.float().double()
            g0=target_logits[a:z,None].double()-logits[a:z,left:right].double()
            first=rounded(dot(dd,hs[0])+dot(diff,hs[1]));joint=rounded(dot(dd,hs[1]));curv=rounded(.5*dot(diff,hs[2]))
            own=targets[a:z,None]==torch.arange(start+left,start+right,device=h.device)[None]
            for x in (g0,first,joint,curv):valid=valid&torch.isfinite(x).all()
            for j,alpha in enumerate(ALPHAS):
                gap=((g0+alpha*first)+alpha*alpha*joint)+alpha*alpha*curv
                mins[j,a:z]=torch.minimum(mins[j,a:z],gap.masked_fill(own,math.inf).amin(1))
    return mins,valid


def collective_min(minima,valid,group=None):
    """Real distributed collectives, only if caller supplies an initialized group."""
    dist.all_reduce(minima,op=dist.ReduceOp.MIN,group=group)
    flag=valid.to(torch.int32);dist.all_reduce(flag,op=dist.ReduceOp.MIN,group=group)
    return minima,flag.bool()


@torch.no_grad()
def emulated_boundary(owner,adapter,inputs,targets,hs,*,audit=False):
    name=adapter.readout_name;v,h=owner.full[name].shape;n=len(targets)
    native=adapter.native_hidden(inputs);y=targets[:,-1];local=[];target_logits=torch.empty(n,device=native.device)
    wy=torch.empty((n,h),device=native.device);dy=torch.empty_like(wy)
    for r in range(owner.world_size):
        start,end=interval(v,r,owner.world_size)
        w=owner.gather(name,start=start*h,end=end*h).view(end-start,h)
        dw=owner.gather(name,'delta',start*h,end*h).view(end-start,h)
        logits=torch.nn.functional.linear(native,w)[:,-1,:]
        mask=(y>=start)&(y<end);ids=y[mask]-start
        target_logits[mask]=logits[mask,ids];wy[mask]=w[ids];dy[mask]=dw[ids]
        local.append((start,w,dw,logits))
    minima=[];flags=[]
    for start,w,dw,z in local:
        m,flag=local_minima(z,y,start,w,dw,target_logits,wy,dy,*hs);minima.append(m);flags.append(flag)
    reduced=torch.stack(minima).amin(0);valid=torch.stack(flags).all();a,s=select(reduced,valid)
    # Estimates, not network measurements: target rows/updates/scores + MIN input.
    communication={'kind':'EMULATED_PAYLOAD_NOT_WIRE_TRAFFIC','min_input_bytes_per_rank':4*n*8,
        'target_row_update_score_payload_bytes':2*n*h*4+n*4,'target_payload_broadcast_recipients':max(0,owner.world_size-1),
        'full_vocabulary_all_gather':False}
    # Audit-only CPU reconstruction is excluded from runtime communication claims.
    recorded=torch.cat([z.cpu() for start,w,dw,z in local],dim=1) if audit else None
    return a,s,minima,reduced,communication,recorded
