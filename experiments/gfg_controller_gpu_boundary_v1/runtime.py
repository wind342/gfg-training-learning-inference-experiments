"""GPU-resident all-competitor boundary reduction; no model CPU snapshots."""
import math
import torch
from torch.func import functional_call
from torch.nn.attention import sdpa_kernel, SDPBackend
from experiments.gfg_controller_large_model_runtime_v1.runtime import ALPHAS, KEYS, blockwise_hidden
from experiments.gfg_controller_large_model_runtime_v1.blockwise_runtime import BlockwiseController


@torch.no_grad()
def gpu_boundary(logits,targets,w,dw,h,hp,hpp,*,target_tile=256,competitor_tile=128,audit=False):
    n,v=logits.shape
    if n<1 or v<2 or targets.shape!=(n,) or target_tile<1 or competitor_tile<1:
        raise ValueError('invalid dimensions or tile size')
    if logits.device.type!='cuda' or any(z.device!=logits.device for z in (targets,w,dw,h,hp,hpp)):
        raise ValueError('all boundary inputs must share a CUDA device')
    if w.shape!=dw.shape or w.shape[0]!=v or h.shape!=hp.shape or h.shape!=hpp.shape or h.shape!=(n,w.shape[1]):
        raise ValueError('incompatible readout/hidden shapes')
    # Only the active head tiles are converted to FP64, not a full FP64 head copy.
    minima=torch.full((4,n),math.inf,dtype=torch.float64,device=logits.device)
    coordinates={k:torch.empty((n,v),dtype=torch.float64,device=logits.device) for k in KEYS} if audit else None
    valid=torch.ones((),dtype=torch.bool,device=logits.device)
    for start in range(0,n,target_tile):
        end=min(n,start+target_tile);y=targets[start:end]
        hs=[z[start:end].to(torch.float64).unsqueeze(-1) for z in (h,hp,hpp)]
        wy=w[y].to(torch.float64).unsqueeze(1);dy=dw[y].to(torch.float64).unsqueeze(1)
        target_logit=logits[start:end].gather(1,y[:,None]).to(torch.float64)
        for left in range(0,v,competitor_tile):
            right=min(v,left+competitor_tile)
            diff=wy-w[left:right].to(torch.float64)[None]
            dd=dy-dw[left:right].to(torch.float64)[None]
            dot=lambda a,z:torch.bmm(a,z).squeeze(-1)
            rounded=lambda z:z.to(torch.float32).to(torch.float64)
            c={'g0':target_logit-logits[start:end,left:right].to(torch.float64),
               'first_gap':rounded(dot(dd,hs[0])+dot(diff,hs[1])),
               'joint_rotation':rounded(dot(dd,hs[1])),
               'hidden_curvature':rounded(.5*dot(diff,hs[2]))}
            own=y[:,None]==torch.arange(left,right,device=y.device)[None]
            for z in c.values():valid=valid & torch.isfinite(z).all()
            for j,a in enumerate(ALPHAS):
                gap=((c['g0']+a*c['first_gap'])+a*a*c['joint_rotation'])+a*a*c['hidden_curvature']
                local=gap.masked_fill(own,math.inf).amin(1)
                minima[j,start:end]=torch.minimum(minima[j,start:end],local)
            if audit:
                for k in KEYS:coordinates[k][start:end,left:right]=c[k]
    counts=(minima>0).sum(1).to(torch.float64)
    ordered=minima.sort(dim=1).values
    median=ordered[:,n//2] if n%2 else (ordered[:,n//2-1]+ordered[:,n//2])/2
    # One 72-byte decision transfer. torch.median alone is wrong for even N.
    payload=torch.cat((counts,median,valid.reshape(1).to(torch.float64))).cpu().tolist()
    if payload[-1]!=1 or not all(math.isfinite(x) for x in payload):raise FloatingPointError('nonfinite boundary')
    scores={str(a):{'predicted_correct_count':int(payload[j]),'median_predicted_minimum_margin':payload[4+j]} for j,a in enumerate(ALPHAS)}
    alpha=max(ALPHAS,key=lambda a:(scores[str(a)]['predicted_correct_count'],scores[str(a)]['median_predicted_minimum_margin'],-abs(a-1.)))
    return alpha,scores,coordinates,minima


class GPUBlockwiseController(BlockwiseController):
    def __init__(self,model,*,target_tile=256,competitor_tile=128,audit=False,hash_parameters=None,**kwargs):
        super().__init__(model,audit=audit,hash_parameters=hash_parameters,**kwargs)
        self.target_tile=target_tile;self.competitor_tile=competitor_tile
        self.decision_transfer_bytes=0

    def response(self,model,inputs,targets):
        parameters={n:p.detach() for n,p in model.named_parameters()};delta=self.buffers.delta
        model.eval()
        with torch.no_grad():
            # Match the reference full-sequence logits call, keeping it on device.
            recorded=functional_call(model,parameters,(inputs,targets),tie_weights=True,strict=True)[0][:,-1,:]
            with sdpa_kernel(SDPBackend.MATH):hs=blockwise_hidden(model,parameters,delta,inputs)
            a,s,c,m=gpu_boundary(recorded,targets[:,-1],parameters['transformer.wte.weight'],delta['transformer.wte.weight'],*hs,
                target_tile=self.target_tile,competitor_tile=self.competitor_tile,audit=self.audit)
            self.decision_transfer_bytes+=72
            self.last_debug={'hidden':hs,'pre_logits':recorded,'coordinates':c,'minima':m} if self.audit else None
        return a,s,c,m,None
