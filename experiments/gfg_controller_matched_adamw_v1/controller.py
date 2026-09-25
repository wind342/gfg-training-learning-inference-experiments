"""Skip only an identity-amplitude cached step's unused pre-state/delta."""
import torch
from experiments.gfg_controller_resident_reuse_v1.engine import Controller as ResidentController


class Controller(ResidentController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.identity_steps_skipped=0

    def update(self,model,opt,t,x,y,vx,vy,seed,alpha):
        if (t-1)%self.cadence==0 or alpha!=1.:
            return super().update(model,opt,t,x,y,vx,vy,seed,alpha)
        self.reused_post_snapshot=None
        self.identity_steps_skipped+=1
        model.train(); torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        opt.zero_grad(set_to_none=True)
        loss=model(x,y)[1];loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        return alpha,{'loss':float(loss.detach()),'alpha':alpha},None
