"""Caller-supplied group identity; no modular arithmetic in the generic policy."""
import hashlib
import numpy as np


class GroupMap:
    def __init__(self, group_ids):
        ids = np.asarray(group_ids)
        if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu':
            raise ValueError('one integer group identity per probe is required')
        self.ids = ids.astype(np.int64, copy=True)
        self.ids.setflags(write=False)
        self.labels = tuple(int(g) for g in np.unique(ids))
        self.members = tuple(tuple(np.flatnonzero(ids == g).tolist()) for g in self.labels)
        self.counts = tuple(len(m) for m in self.members)
        self.identity = hashlib.sha256(self.ids.tobytes()).hexdigest()
        self.target_token_semantics = False

    @classmethod
    def target_tokens(cls, targets, expected=None):
        mapping = cls(targets[:, -1].detach().cpu().numpy())
        mapping.target_token_semantics = True
        if expected is not None and set(mapping.labels) != set(expected):
            raise ValueError('task-specific target group set mismatch')
        return mapping

    def values(self, margins):
        margins = np.asarray(margins, dtype=np.float64)
        if margins.shape != self.ids.shape or not np.isfinite(margins).all():
            raise FloatingPointError('invalid finite bank margins')
        counts = [int((margins[list(m)] > 0).sum()) for m in self.members]
        return dict(correct_counts=counts,
                    capabilities=[c/n for c,n in zip(counts,self.counts)],
                    q10=[float(np.quantile(margins[list(m)],.1,method='linear')) for m in self.members],
                    correct_count=sum(counts),capability=sum(counts)/len(margins))
