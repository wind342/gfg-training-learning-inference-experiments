"""Native-referenced discrete formation/preservation policy, without weighted utility."""
from dataclasses import dataclass
import math
import numpy as np
from experiments.gfg_functional_update_governor_v1.governor import ALPHAS


@dataclass
class Decision:
    alpha: float
    formed_count: int
    unformed_count: int
    scores: dict
    reason: str
    objective: str


class FunctionalUpdateGovernor:
    def choose(self, measurement):
        alphas = tuple(float(a) for a in measurement.alphas)
        b = measurement.baseline_minima.detach().double().cpu().numpy()
        p = measurement.predicted_minima.detach().double().cpu().numpy()
        if b.ndim != 1 or not len(b) or p.shape != (len(alphas), len(b)):
            raise ValueError('invalid nonempty target/minimum shapes')
        if not np.isfinite(b).all() or not np.isfinite(p).all():
            raise FloatingPointError('nonfinite boundary; no action selected')
        if 1. not in alphas or len(set(alphas)) != len(alphas) or any(not math.isfinite(a) or a <= 0 for a in alphas):
            raise ValueError('unique positive finite alphas must include native 1')
        formed = b > 0
        scores = {}
        for a, endpoint in zip(alphas, p):
            scores[str(a)] = {
                'alpha': a,
                'formation_gain': int(((~formed) & (endpoint > 0)).sum()),
                'preservation_damage': int((formed & (endpoint <= 0)).sum()),
                'frontier_progress': float(np.median(endpoint[~formed] - b[~formed])) if (~formed).any() else None,
                'formed_median_margin': float(np.median(endpoint[formed])) if formed.any() else None,
            }
        native = scores['1.0']
        eligible = [a for a in alphas if a <= 1.] if formed.all() else list(alphas)
        # Preservation first, then formation. Frontier is only a tie-breaker
        # among alternatives that strictly improve discrete counts over native.
        least = min(scores[str(a)]['preservation_damage'] for a in eligible)
        eligible = [a for a in eligible if scores[str(a)]['preservation_damage'] == least]
        gain = max(scores[str(a)]['formation_gain'] for a in eligible)
        eligible = [a for a in eligible if scores[str(a)]['formation_gain'] == gain]
        if least == native['preservation_damage'] and gain == native['formation_gain']:
            selected = 1.
        elif formed.all():
            selected = max(eligible, key=lambda a: (-abs(a-1.), scores[str(a)]['formed_median_margin'], -a))
        else:
            selected = max(eligible, key=lambda a: (scores[str(a)]['frontier_progress'], -abs(a-1.), -a))
        objective = 'PRESERVATION' if least < native['preservation_damage'] else 'FORMATION' if gain > native['formation_gain'] else 'NATIVE'
        if selected == 1.:
            reason = 'NO_FORMED_CAPABILITY' if not formed.any() else 'NATIVE_SAFE' if native['preservation_damage'] == 0 else 'NO_BETTER_ACTION'
        else:
            # Direction and purpose are separate: amplification can preserve, too.
            reason = ('AMPLIFY_' if selected > 1 else 'SHRINK_') + objective
        return Decision(selected, int(formed.sum()), int((~formed).sum()), scores, reason, objective)
