"""Same v2 execution, with v3 group sensing and v4 historical decision state."""
from experiments.gfg_functional_update_governor_v2.runtime import GovernedTraining as BaseTraining
from experiments.gfg_functional_update_governor_v3.scheduler import GroupSentinel
from .governor import RecoveryGovernor


class GovernedTraining(BaseTraining):
    def __init__(self, model, adapter, *, mapping, threshold=.9, group_micro_every=1, **kwargs):
        self.mapping = mapping
        scheduler = GroupSentinel(mapping,threshold=threshold,group_micro_every=group_micro_every)
        super().__init__(model,adapter,scheduler=scheduler,
            governor=RecoveryGovernor(mapping,scheduler,threshold),**kwargs)

    def update(self,*args,**kwargs):
        info = super().update(*args,**kwargs)
        info['runtime_version'] = 'governor_v4'
        info['latched_groups'] = info.pop('latched_targets')
        info['group_observation'] = self.scheduler.snapshot(self.scheduler.last_step)
        info['group_mapping_sha256'] = self.mapping.identity
        if info['instrument_called']:
            info['hierarchy'] = self.governor.summary
            if info['hierarchy']['scheduler_preview'] != info['group_observation']:
                self.failed=True
                raise RuntimeError('history preview/committed full refresh mismatch')
            info['current_correct_target_count'] = info.pop('baseline_formed_count')
            info['current_wrong_target_count'] = info.pop('baseline_unformed_count')
        return info
