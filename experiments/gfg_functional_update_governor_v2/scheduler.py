"""Deterministic micro shards; per-target excursion anchors survive full refresh."""
import math
import numpy as np
import torch


@torch.no_grad()
def native_margins(model, inputs, targets):
    """One ordinary eval forward; preserve module modes and RNG, no derivatives.

    Supported forward contract: model(inputs)[0] is [batch, time, vocabulary].
    Eval forwards must not mutate parameters/buffers (true for tested nanoGPT).
    Buffer copies protect persistent buffers even for a stateful custom forward.
    """
    modes = [(m, m.training) for m in model.modules()]
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    versions = [(p, p._version) for p in model.parameters()]
    devices = [inputs.device.index] if inputs.is_cuda else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            z = model(inputs)[0][:, -1, :]
            y = targets[:, -1]
            if z.ndim != 2 or len(z) != len(y) or z.shape[1] < 2 or not torch.isfinite(z).all():
                raise FloatingPointError('invalid/nonfinite micro logits')
            truth = z.gather(1, y[:, None]).squeeze(1)
            other = z.clone().scatter_(1, y[:, None], -torch.inf).amax(1)
            return (truth.double() - other.double()).cpu().numpy()
    finally:
        for n, value in buffers.items():
            actual = model.get_buffer(n)
            if not torch.equal(actual, value):
                actual.copy_(value)
        for module, mode in modes:
            module.training = mode
        if any(p._version != version for p, version in versions):
            raise RuntimeError('unsupported forward mutated parameters; abort runtime')


class FunctionalSentinel:
    def __init__(self, micro_size=8, micro_every=1, max_interval=50,
                 erosion_ratio=.5, recovery_ratio=.75, absolute_floor=1e-6):
        if any(not isinstance(v, int) or v < 1 for v in (micro_size, micro_every, max_interval)):
            raise ValueError('intervals and shard size must be positive integers')
        if not 0 < erosion_ratio < recovery_ratio <= 1 or not math.isfinite(absolute_floor) or absolute_floor <= 0:
            raise ValueError('invalid hysteresis thresholds')
        self.micro_size = micro_size
        self.micro_every = micro_every
        self.max_interval = max_interval
        self.erosion_ratio = erosion_ratio
        self.recovery_ratio = recovery_ratio
        self.absolute_floor = absolute_floor
        self.last_step = 0
        self.last_instrument = None
        self.cursor = 0
        self.identity = None
        self.baseline = self.anchor = self.level = None

    def bind(self, inputs, targets):
        identity = (id(inputs), id(targets), inputs._version, targets._version, tuple(inputs.shape), tuple(targets.shape))
        if not len(inputs) or len(inputs) != len(targets):
            raise ValueError('empty or mismatched control bank')
        if self.identity is None:
            self.identity = identity
        elif identity != self.identity:
            raise ValueError('control bank identity/order changed; create a new sentinel')

    def begin(self, step):
        if step != self.last_step + 1:
            raise ValueError('steps must start at 1 and be consecutive')
        self.last_step = step
        if self.last_instrument is None:
            return ['FIRST_STEP']
        if step-self.last_instrument >= self.max_interval:
            return ['MAX_INTERVAL']
        return []

    def shard(self, step):
        if step % self.micro_every:
            return []
        n = len(self.baseline)
        ids = list(range(self.cursor, min(n, self.cursor+self.micro_size)))
        self.cursor = (ids[-1]+1) % n
        return ids

    def observe(self, ids, margins, source='micro'):
        values = np.asarray(margins, dtype=np.float64)
        if values.shape != (len(ids),) or not np.isfinite(values).all():
            raise FloatingPointError('nonfinite/malformed functional observation')
        if len(set(ids)) != len(ids) or any(i < 0 or i >= len(self.anchor) for i in ids):
            raise ValueError('invalid target identities')
        events = []
        for i, current in zip(ids, values):
            anchor = self.anchor[i]
            level = int(self.level[i])
            if level and current > 0 and current >= self.recovery_ratio * max(anchor, self.absolute_floor):
                self.level[i] = 0
                self.anchor[i] = current
                events.append(dict(probe_index=i, kind='REARM', source=source, anchor=float(anchor), margin=float(current)))
                continue
            severity = 2 if anchor > 0 and current <= 0 else 1 if anchor >= self.absolute_floor and 0 < current < self.erosion_ratio*anchor else 0
            if severity > level:
                kind = 'FORMED_BOUNDARY_CROSSING' if severity == 2 else 'FORMED_MARGIN_EROSION'
                self.level[i] = severity
                events.append(dict(probe_index=i, kind=kind, source=source, anchor=float(anchor), margin=float(current)))
            elif severity:
                events.append(dict(probe_index=i, kind='PERSISTENT_RISK', source=source, anchor=float(anchor), margin=float(current)))
            # A newly observed formed target must not be forgotten until refresh.
            if self.level[i] == 0 and anchor <= 0 and current > 0:
                self.anchor[i] = current
        return events

    @staticmethod
    def reasons(events):
        return sorted({e['kind'] for e in events if e['kind'] in ('FORMED_BOUNDARY_CROSSING', 'FORMED_MARGIN_EROSION')})

    def measured(self, step, baseline):
        values = baseline.detach().double().cpu().numpy().copy()
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise FloatingPointError('invalid full baseline')
        if self.baseline is None:
            self.anchor = values.copy()
            self.level = np.zeros(len(values), dtype=np.int8)
            events = []
        else:
            if values.shape != self.baseline.shape:
                raise ValueError('control baseline size changed')
            events = self.observe(list(range(len(values))), values, 'full_refresh')
            # Refresh current reference, but NEVER overwrite active risk anchors.
            armed = self.level == 0
            self.anchor[armed] = values[armed]
        self.baseline = values
        self.last_instrument = step
        return events
