"""Thin execution layer; only measured/non-native steps need proposal buffers."""
import math
import torch
import torch.distributed as dist
from experiments.gfg_controller_distributed_runtime_v2.identity import CanonicalParameterRegistry
from experiments.gfg_controller_distributed_runtime_v2.ownership import Ownership
from experiments.gfg_controller_distributed_runtime_v2.clipping import clip
from .governor import FunctionalUpdateGovernor
from .instrument import FunctionalResponseInstrument
from .scheduler import CheapSentinel,ReleaseToNative


class GovernedTraining:
    def __init__(self,model,adapter,*,registry=None,units=None,group=None,instrument=None,governor=None,
                 scheduler=None,release_steps=0,backend='exact'):
        if group is not None and dist.get_world_size(group)>1:
            raise NotImplementedError('multi-rank scheduler consensus is not validated; this runtime is world1 only')
        self.model=model;self.registry=registry or CanonicalParameterRegistry(model)
        self.owner=Ownership(model,self.registry,units=units,group=group)
        self.instrument=instrument or FunctionalResponseInstrument(adapter,backend=backend,group=group)
        self.governor=governor or FunctionalUpdateGovernor()
        self.scheduler=scheduler or CheapSentinel();self.release=ReleaseToNative(release_steps)
        self.failed=False;self.instrument_calls=0

    def update(self,opt,t,x,y,probes,targets,seed,lr):
        if self.failed:raise RuntimeError('aborted runtime cannot resume: restore a checkpoint in a new runtime')
        if not math.isfinite(lr) or lr<=0:raise ValueError('learning rate must be positive finite')
        try:
            self.model.train();torch.manual_seed(seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad(set_to_none=True)
            loss=self.model(x,y)[1];loss.backward()
            norm=clip(self.registry,self.model)
            loss_value=float(loss.detach());norm_value=float(norm)
            reasons=self.scheduler.observe(t,loss_value,norm_value)
            measured=reasons!=['NONE'];decision=None
            alpha=self.release.at(t)
            if measured:
                self.owner.propose(opt)
                measurement=self.instrument.measure(self.owner,probes,targets)
                decision=self.governor.choose(measurement)
                alpha=decision.alpha
                self.owner.commit(alpha)
                self.release.decided(t,alpha);self.scheduler.measured(t)
                self.instrument_calls+=1
                path='INSTRUMENT'
            elif alpha==1.:
                # No copies, parameter swap, proposal construction or JVP.
                opt.step();path='NATIVE_FAST'
            else:
                # Optional open-loop interpolation, not a new boundary prediction.
                self.owner.propose(opt);self.owner.commit(alpha);path='OPEN_LOOP_RELEASE'
            info={'alpha':alpha,'instrument_called':measured,'instrument_reason':reasons,
                'loss':loss_value,'norm':norm_value,'steps_since_instrument':t-self.scheduler.last_instrument,
                'execution_path':path,'instrument_calls':self.instrument_calls}
            if decision is not None:
                chosen=decision.scores[str(alpha)]
                info.update(baseline_formed_count=decision.formed_count,baseline_unformed_count=decision.unformed_count,
                    scores=decision.scores,selected_alpha=alpha,formation_gain=chosen['formation_gain'],
                    preservation_damage=chosen['preservation_damage'],utility=chosen['utility'],
                    measurement_metadata=measurement.metadata)
            return info
        except Exception:
            # Never silently continue after a partly formed optimizer proposal.
            # On measurement failure parameters are receiving-state bytes, but
            # Adam moments may already have advanced; caller must restore both.
            self.failed=True
            raise
