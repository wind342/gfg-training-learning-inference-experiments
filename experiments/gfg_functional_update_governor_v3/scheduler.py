"""One complete group per observation; history and current constraints stay distinct."""
import numpy as np
from experiments.gfg_functional_update_governor_v2.scheduler import FunctionalSentinel


class GroupSentinel(FunctionalSentinel):
    def __init__(self, mapping, threshold=.9, group_micro_every=1, **kwargs):
        super().__init__(micro_every=group_micro_every,**kwargs)
        if not 0 < threshold <= 1:
            raise ValueError('invalid threshold')
        self.mapping = mapping; self.threshold = threshold
        n = len(mapping.labels)
        self.state = ['FORMING']*n
        self.level = np.zeros(n,dtype=np.int8)
        self.anchor = np.zeros(n,dtype=np.float64)
        self.capabilities = np.zeros(n,dtype=np.float64)
        self.q10 = np.zeros(n,dtype=np.float64)
        self.last_observed = np.full(n,-1,dtype=np.int64)

    def bind(self, inputs, targets):
        if len(inputs) != len(self.mapping.ids):
            raise ValueError('bank does not match group map')
        if self.identity is None and self.mapping.target_token_semantics:
            if not np.array_equal(targets[:,-1].detach().cpu().numpy(),self.mapping.ids):
                raise ValueError('task target tokens do not match canonical group identities')
        super().bind(inputs,targets)

    def shard(self, step):
        if step % self.micro_every: return []
        ids = list(self.mapping.members[self.cursor])
        self.cursor = (self.cursor+1)%len(self.mapping.labels)
        return ids

    def _observe_group(self, k, margins, source):
        values = np.asarray(margins,dtype=np.float64)
        if values.shape != (self.mapping.counts[k],) or not np.isfinite(values).all():
            raise FloatingPointError('invalid complete-group observation')
        c = float((values>0).mean()); q = float(np.quantile(values,.1,method='linear'))
        self.capabilities[k] = c; self.q10[k] = q; self.last_observed[k] = self.last_step
        anchor = float(self.anchor[k]); old_level = int(self.level[k])
        def event(kind):
            return [dict(group_id=self.mapping.labels[k],kind=kind,source=source,capability=c,q10=q,anchor_q10=anchor)]
        if self.state[k] == 'FORMING':
            if c >= self.threshold:
                self.state[k] = 'PROTECTED'; self.anchor[k] = q
                return event('GROUP_FORMED')
            return []
        if old_level and c >= self.threshold and (anchor < self.absolute_floor or q >= self.recovery_ratio*anchor):
            self.level[k] = 0; self.state[k] = 'PROTECTED'; self.anchor[k] = q
            return event('GROUP_REARM')
        # q10 <= 0 after positive anchor is also erosion, even when accuracy >=90%.
        severity = 2 if c < self.threshold else 1 if anchor >= self.absolute_floor and q < self.erosion_ratio*anchor else 0
        if c < self.threshold: self.state[k] = 'DEGRADED'
        if severity > old_level:
            self.level[k] = severity
            return event('PROTECTED_GROUP_DEGRADATION' if severity==2 else 'PROTECTED_GROUP_MARGIN_EROSION')
        if old_level or severity:
            return event('PERSISTENT_GROUP_RISK')
        if source == 'full_refresh': self.anchor[k] = q
        return []

    def observe(self, ids, margins, source='micro'):
        if not ids: raise ValueError('empty group observation')
        matches = [k for k,m in enumerate(self.mapping.members) if tuple(ids)==m]
        if len(matches) != 1: raise ValueError('micro must cover exactly one complete group in canonical order')
        return self._observe_group(matches[0],margins,source)

    @staticmethod
    def reasons(events):
        return sorted({e['kind'] for e in events if e['kind'] in ('PROTECTED_GROUP_DEGRADATION','PROTECTED_GROUP_MARGIN_EROSION')})

    def measured(self, step, baseline):
        values = baseline.detach().double().cpu().numpy().copy()
        self.mapping.values(values)  # finite/shape check, same aggregation contract
        events = []
        for k,ids in enumerate(self.mapping.members):
            events.extend(self._observe_group(k,values[list(ids)],'full_refresh'))
        self.baseline = values; self.last_instrument = step
        return events

    def snapshot(self, step):
        return dict(states=list(self.state),capabilities=self.capabilities.tolist(),q10=self.q10.tolist(),
                    anchor_q10=self.anchor.tolist(),levels=self.level.tolist(),
                    steps_since_group_observed=[None if t<0 else int(step-t) for t in self.last_observed],
                    protected_group_count=self.state.count('PROTECTED'),degraded_group_count=self.state.count('DEGRADED'))
