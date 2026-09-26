"""Pure current-step decision policy; no model execution or scheduling."""
from dataclasses import dataclass
import math
import numpy as np

ALPHAS = (.125, .25, .5, 1., 2.)


@dataclass
class Decision:
    alpha: float
    formed_count: int
    unformed_count: int
    scores: dict


class FunctionalUpdateGovernor:
    def __init__(self, preservation_penalty=1.):
        if not math.isfinite(preservation_penalty) or preservation_penalty < 0:
            raise ValueError('preservation_penalty must be finite and nonnegative')
        self.preservation_penalty = float(preservation_penalty)

    def choose(self, measurement):
        alphas = tuple(float(a) for a in measurement.alphas)
        baseline = measurement.baseline_minima.detach().double().cpu().numpy()
        predicted = measurement.predicted_minima.detach().double().cpu().numpy()
        if baseline.ndim != 1 or not len(baseline) or predicted.shape != (len(alphas),len(baseline)):
            raise ValueError('invalid nonempty target/minimum shapes')
        if not np.isfinite(baseline).all() or not np.isfinite(predicted).all():
            raise FloatingPointError('nonfinite boundary; no action selected')
        if 1. not in alphas or len(set(alphas)) != len(alphas) or any(not math.isfinite(a) or a <= 0 for a in alphas):
            raise ValueError('unique positive finite alphas must include native 1')
        formed = baseline > 0
        unformed = ~formed
        scores = {}
        for a, endpoint in zip(alphas,predicted):
            gain = int((unformed & (endpoint > 0)).sum())
            damage = int((formed & (endpoint <= 0)).sum())
            scores[str(a)] = {
                'alpha':a, 'formation_gain':gain, 'preservation_damage':damage,
                'utility':gain-self.preservation_penalty*damage,
                'frontier_progress':float(np.median(endpoint[unformed]-baseline[unformed])) if unformed.any() else None,
                'formed_median_margin':float(np.median(endpoint[formed])) if formed.any() else None,
            }
        if formed.all():
            # Do not amplify an already-formed bank. Shrink only for less damage.
            candidates = [a for a in alphas if a <= 1.]
            least = min(scores[str(a)]['preservation_damage'] for a in candidates)
            if scores['1.0']['preservation_damage'] == least:
                selected = 1.
            else:
                selected = max(candidates,key=lambda a:(-scores[str(a)]['preservation_damage'],
                    scores[str(a)]['formed_median_margin'], -abs(a-1), -a))
        else:
            def key(a):
                s=scores[str(a)]
                return (s['utility'],s['formation_gain'],-s['preservation_damage'],
                    s['frontier_progress'],s['formed_median_margin'] if formed.any() else 0.,-abs(a-1),-a)
            selected=max(alphas,key=key)
        return Decision(selected,int(formed.sum()),int(unformed.sum()),scores)
