"""Standalone execution entry: no ledgers, model downloads, snapshots or state hashing."""
import torch
from .identity import CanonicalParameterRegistry
from .ownership import Ownership
from .boundary import calculate,WorkspacePlanner,same_probes
from .clipping import clip


class FunctionalController:
    def __init__(self,model,adapter,*,registry=None,units=None,group=None,backend='exact',cadence=5,budget=64*1024**2,audit=False):
        self.model=model;self.registry=registry or CanonicalParameterRegistry(model)
        self.owner=Ownership(model,self.registry,units=units,group=group);self.adapter=adapter
        self.group=group;self.backend=backend;self.cadence=cadence;self.planner=WorkspacePlanner(budget)
        self.audit=audit;self.alpha=1.;self.last=None;self.bank_identity=None

    def update(self,opt,t,x,y,probes,targets,seed,lr):
        self.last=None;self.model.train();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True);loss=self.model(x,y)[1];loss.backward();norm=clip(self.registry,self.model)
        self.owner.propose(opt)
        if (t-1)%self.cadence==0:
            self.model.eval()
            if self.group is not None:
                # Validate every new bank object before applying cross-rank MIN.
                identity=(id(probes),id(targets),probes._version,targets._version)
                if identity!=self.bank_identity:same_probes(probes,targets,self.group);self.bank_identity=identity
            with torch.no_grad():
                native,hs=self.adapter.jets(self.owner,probes)
                start,w,dw=self.owner.readout_shard(self.adapter.readout_name)
                logits=torch.nn.functional.linear(native,w)[:,-1,:]
                self.alpha,scores,minima,coords,meta=calculate(logits,targets[:,-1],w,dw,hs,start=start,group=self.group,
                    mode=self.backend,planner=self.planner,audit=self.audit)
                self.last={'scores':scores,'metadata':meta}
                if self.audit:self.last.update(hidden=hs,minima=minima,coordinates=coords,pre_logits=logits)
        self.owner.commit(self.alpha)
        return {'alpha':self.alpha,'loss':float(loss.detach()),'norm':float(norm),**({'scores':self.last['scores'],'metadata':self.last['metadata']} if self.last else {})}


def train(model,optimizer,adapter,batches,seeds,learning_rates,probes,targets,*,callback=None,**config):
    engine=FunctionalController(model,adapter,**config)
    for t,((x,y),seed,lr) in enumerate(zip(batches,seeds,learning_rates),1):
        info=engine.update(optimizer,t,x,y,probes,targets,seed,lr)
        if callback is not None and callback(t,info,engine) is False:break
    return engine
