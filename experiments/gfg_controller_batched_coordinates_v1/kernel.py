"""Task-independent quadratic boundary coordinates for a linear readout.

Inputs are pre-update logits, target IDs, readout weights and their actual
delta, and hidden state / first / second directional derivatives. No task,
target-count, class-count, hidden-width or model parameter names are assumed.
The adapter must supply derivatives along the SAME actual parameter update.
Bias can be represented by an augmented constant hidden coordinate.
"""
import numpy as np

KEYS = ('g0', 'first_gap', 'joint_rotation', 'hidden_curvature')


def boundary_coordinates(logits, targets, w, dw, h, hp, hpp, *, chunk_size=256):
    arrays = [np.asarray(v, dtype=np.float64) for v in (logits, w, dw, h, hp, hpp)]
    logits, w, dw, h, hp, hpp = arrays
    targets = np.asarray(targets)
    if logits.ndim != 2 or h.ndim != 2:
        raise ValueError('logits and hidden states must be matrices')
    n, classes = logits.shape
    if (n == 0 or classes < 2 or targets.shape != (n,)
            or targets.dtype.kind not in 'iu' or np.any(targets < 0)
            or np.any(targets >= classes) or w.shape != (classes, h.shape[1])
            or dw.shape != w.shape or h.shape[0] != n
            or hp.shape != h.shape or hpp.shape != h.shape
            or not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError('invalid boundary input dimensions, IDs or chunk size')
    if not all(np.isfinite(v).all() for v in arrays):
        raise ValueError('nonfinite boundary inputs')
    out = {k: np.empty((n, classes - 1), dtype=np.float64) for k in KEYS}
    competitors = np.empty((n, classes - 1), dtype=np.int64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        y = targets[start:end]
        scores = logits[start:end]
        # Match the reference's default argsort and descending reversal.
        order = np.argsort(scores, axis=1)[:, ::-1]
        cs = order[order != y[:, None]].reshape(end - start, classes - 1)
        competitors[start:end] = cs
        b = w[y, None, :] - w[cs]
        bp = dw[y, None, :] - dw[cs]
        dot = lambda a, v: np.matmul(a, v[start:end, :, None])[..., 0]
        rounded = lambda v: v.astype(np.float32).astype(np.float64)
        out['g0'][start:end] = scores[np.arange(end-start), y, None] - np.take_along_axis(scores, cs, axis=1)
        out['first_gap'][start:end] = rounded(dot(bp, h) + dot(b, hp))
        out['joint_rotation'][start:end] = rounded(dot(bp, hp))
        out['hidden_curvature'][start:end] = rounded(.5 * dot(b, hpp))
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError('nonfinite boundary output')
    return {**out, 'competitors': competitors, 'pre_logits': logits,
            'hidden': h, 'hidden_first': hp, 'hidden_second': hpp}
