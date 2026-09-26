"""Reuse v2 execution/failure/timing paths, replace only policy and scheduling."""
from experiments.gfg_functional_update_governor_v2.runtime import GovernedTraining as V2Training
from .governor import GroupGovernor
from .scheduler import GroupSentinel


class GovernedTraining(V2Training):
    def __init__(self, model, adapter, *, mapping, threshold=.9, group_micro_every=1, **kwargs):
        self.mapping = mapping
        super().__init__(model,adapter,scheduler=GroupSentinel(mapping,threshold=threshold,group_micro_every=group_micro_every),
                         governor=GroupGovernor(mapping,threshold=threshold),**kwargs)

    def update(self,*args,**kwargs):
        info = super().update(*args,**kwargs)
        info['runtime_version'] = 'governor_v3'
        info['latched_groups'] = info.pop('latched_targets')
        info['group_observation'] = self.scheduler.snapshot(self.scheduler.last_step)
        info['group_mapping_sha256'] = self.mapping.identity
        if info['instrument_called']:
            info['hierarchy'] = self.governor.summary
            # Retain target counts, but never label them formed capabilities.
            info['current_correct_target_count'] = info.pop('baseline_formed_count')
            info['current_wrong_target_count'] = info.pop('baseline_unformed_count')
        return info
