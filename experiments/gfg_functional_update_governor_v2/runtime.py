"""World1 runtime reusing v1 instrument and canonical proposal ownership."""
from contextlib import contextmanager
import math
import time
import torch
import torch.distributed as dist
from experiments.gfg_functional_update_governor_v1.runtime import GovernedTraining as BaseTraining
from experiments.gfg_controller_distributed_runtime_v2.clipping import clip
from .governor import FunctionalUpdateGovernor
from .scheduler import FunctionalSentinel, native_margins


class GovernedTraining(BaseTraining):
    def __init__(self, model, adapter, *, scheduler=None, governor=None, **kwargs):
        group = kwargs.get('group')
        if dist.is_initialized() and dist.get_world_size(group) > 1:
            raise NotImplementedError('v2 functional sentinel is validated for world1 only')
        if kwargs.get('release_steps', 0) != 0:
            raise ValueError('v2 measures and controls the current update only')
        super().__init__(model, adapter, scheduler=scheduler or FunctionalSentinel(),
                         governor=governor or FunctionalUpdateGovernor(), **kwargs)

    def update(self, opt, t, x, y, probes, targets, seed, lr):
        if self.failed:
            raise RuntimeError('aborted runtime cannot resume; restore a full checkpoint')
        timing = {}
        def sync():
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
        @contextmanager
        def timed(name):
            sync(); start = time.perf_counter()
            try:
                yield
            finally:
                sync(); timing[name] = time.perf_counter()-start
        try:
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError('learning rate must be positive finite')
            self.scheduler.bind(probes, targets)
            reasons = self.scheduler.begin(t)
            with timed('training'):
                self.model.train(); torch.manual_seed(seed)
                if x.is_cuda:
                    torch.cuda.manual_seed_all(seed)
                for group in opt.param_groups:
                    group['lr'] = lr
                opt.zero_grad(set_to_none=True)
                loss = self.model(x, y)[1]
                loss.backward()
                norm = clip(self.registry, self.model)
                loss_value, norm_value = float(loss.detach()), float(norm)
                if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                    raise FloatingPointError('nonfinite training signal')
            # Observe RECEIVING parameters, before any proposal/optimizer step.
            ids, events, margins = [], [], []
            if not reasons:
                ids = self.scheduler.shard(t)
                if ids:
                    with timed('micro_forward'):
                        margins = native_margins(self.model, probes[ids], targets[ids]).tolist()
                    with timed('sentinel'):
                        events = self.scheduler.observe(ids, margins)
                        reasons = self.scheduler.reasons(events)
            decision = None
            refresh_events = []
            alpha = 1.
            since = None if self.scheduler.last_instrument is None else t-self.scheduler.last_instrument
            if reasons:
                with timed('proposal'):
                    self.owner.propose(opt)
                with timed('instrument'):
                    measurement = self.instrument.measure(self.owner, probes, targets)
                with timed('governor'):
                    decision = self.governor.choose(measurement)
                    if len(measurement.baseline_minima) != len(probes):
                        raise ValueError('instrument changed probe count')
                    refresh_events = self.scheduler.measured(t, measurement.baseline_minima)
                    alpha = decision.alpha
                with timed('commit'):
                    self.owner.commit(alpha)
                self.instrument_calls += 1
                path = 'FULL_INSTRUMENT'
            else:
                with timed('native_optimizer'):
                    opt.step()
                path = 'MICRO_PROBE' if ids else 'NATIVE_FAST'
            info = dict(alpha=alpha, instrument_called=bool(reasons), instrument_reason=reasons or ['NONE'],
                        loss=loss_value, norm=norm_value, steps_since_instrument=since, execution_path=path,
                        instrument_calls=self.instrument_calls, micro_probe_called=bool(ids), micro_indices=ids,
                        micro_margins=margins, micro_events=events, full_refresh_events=refresh_events,
                        latched_targets=int((self.scheduler.level > 0).sum()), timing=timing)
            if decision is not None:
                chosen = decision.scores[str(alpha)]
                info.update(baseline_formed_count=decision.formed_count, baseline_unformed_count=decision.unformed_count,
                            baseline_minima=measurement.baseline_minima.detach().double().cpu().tolist(),
                            scores=decision.scores, formation_gain=chosen['formation_gain'],
                            preservation_damage=chosen['preservation_damage'], intervention_reason=decision.reason,
                            intervention_objective=decision.objective, measurement_metadata=measurement.metadata)
            return info
        except Exception:
            # Proposal failure may have advanced moments. No silent continuation.
            self.failed = True
            raise
