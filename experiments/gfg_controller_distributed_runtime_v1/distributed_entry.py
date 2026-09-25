"""Explicit EMULATED standalone entry, separate from real FSDP certification."""
import torch
from .storage import EmulatedOwnership
from .adapters import NanoGPTAdapter
from .boundary import emulated_boundary


class EmulatedController:
    def __init__(self,model,optimizer_factory,world_size=2,adapter=None,audit_hook=None):
        self.owner=EmulatedOwnership(model,world_size,optimizer_factory)
        self.adapter=adapter or NanoGPTAdapter(model);self.alpha=1.;self.last=None
        self.audit_hook=audit_hook

    def step(self,t,x,y,probes,targets,seed,lr):
        model=self.owner.model;model.train();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        self.owner.zero_grad();self.owner.set_lr(lr)
        loss=model(x,y)[1];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        self.owner.propose();self.last=None
        if self.audit_hook is not None:self.audit_hook('proposal',self)
        if (t-1)%5==0:
            model.eval();hs=self.adapter.directional(self.owner,probes)
            self.alpha,s,local,minimum,comm,pre_logits=emulated_boundary(self.owner,self.adapter,probes,targets,hs,audit=self.audit_hook is not None)
            self.last={'hidden':hs,'scores':s,'local_minima':local,'minima':minimum,'communication':comm,'pre_logits':pre_logits}
        if self.audit_hook is not None:self.audit_hook('decision',self)
        self.owner.commit(self.alpha)
        return {'loss':float(loss.detach()),'alpha':self.alpha,**({'scores':self.last['scores']} if self.last is not None else {})}


def run_emulated(model,optimizer_factory,batches,seeds,learning_rates,probe_inputs,probe_targets,*,world_size=2,adapter=None,callback=None):
    """Caller owns data/evaluation policy. No downloads, ledgers or model hashes.

    callback is an optional external audit/evaluation hook, excluded from any
    claim that this module itself hashes models. This is not a torchrun entry.
    """
    engine=EmulatedController(model,optimizer_factory,world_size,adapter)
    for t,((x,y),seed,lr) in enumerate(zip(batches,seeds,learning_rates),1):
        details=engine.step(t,x,y,probe_inputs,probe_targets,seed,lr)
        if callback is not None and callback(t,details,engine) is False:break
    return engine
