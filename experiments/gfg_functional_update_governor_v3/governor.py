"""Group-constrained capability maximization; target losses are diagnostics only."""
from fractions import Fraction
import math
import numpy as np
from experiments.gfg_functional_update_governor_v2.governor import Decision


class GroupGovernor:
    def __init__(self, mapping, threshold=.9):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError('invalid protection threshold')
        self.mapping = mapping
        self.threshold = Fraction(str(threshold))
        self.summary = None

    def choose(self, measurement):
        alphas = tuple(float(a) for a in measurement.alphas)
        b = measurement.baseline_minima.detach().double().cpu().numpy()
        p = measurement.predicted_minima.detach().double().cpu().numpy()
        if p.shape != (len(alphas),len(self.mapping.ids)) or not np.isfinite(p).all():
            raise FloatingPointError('invalid predicted margins')
        if 1. not in alphas or len(set(alphas)) != len(alphas) or any(not math.isfinite(a) or a <= 0 for a in alphas):
            raise ValueError('invalid amplitudes')
        current = self.mapping.values(b)
        ratios = lambda counts: [Fraction(c,n) for c,n in zip(counts,self.mapping.counts)]
        current_ratios = ratios(current['correct_counts'])
        protected = [k for k,c in enumerate(current_ratios) if c >= self.threshold]
        forming = [k for k in range(len(current_ratios)) if k not in protected]
        scores, exact_keys = {}, {}
        for a, endpoint in zip(alphas,p):
            value = self.mapping.values(endpoint)
            fractions = ratios(value['correct_counts'])
            deficits = [max(Fraction(0),self.threshold-fractions[k]) for k in protected]
            violation = (sum(x>0 for x in deficits),max(deficits,default=Fraction(0)),sum(deficits,Fraction(0)))
            macro = sum(fractions,Fraction(0))/len(fractions)
            transitions = dict(correct_to_correct=int(((b>0)&(endpoint>0)).sum()),
                               correct_to_wrong=int(((b>0)&(endpoint<=0)).sum()),
                               wrong_to_correct=int(((b<=0)&(endpoint>0)).sum()),
                               wrong_to_wrong=int(((b<=0)&(endpoint<=0)).sum()))
            progress = sum(fractions[k] >= self.threshold for k in forming)
            q10_progress = float(np.mean(np.asarray(value['q10'])[forming]-np.asarray(current['q10'])[forming])) if forming else 0.
            scores[str(a)] = dict(alpha=a,**value,macro_group_capability=float(macro),
                delta_capability=value['capability']-current['capability'],
                delta_group_capabilities=[x-y for x,y in zip(value['capabilities'],current['capabilities'])],
                newly_protected_groups=progress,q10_progress=q10_progress,transitions=transitions,
                protected_violation_count=violation[0],worst_protected_deficit=float(violation[1]),
                total_protected_deficit=float(violation[2]),
                formation_gain=transitions['wrong_to_correct'],preservation_damage=transitions['correct_to_wrong'])
            exact_keys[a] = (violation,(value['correct_count'],macro),progress,q10_progress)
        native_violations = exact_keys[1.][0]
        preservation = bool(protected) and native_violations[0] > 0
        pool = list(alphas)
        if preservation:
            best_violation = min(exact_keys[a][0] for a in pool)
            pool = [a for a in pool if exact_keys[a][0] == best_violation]
        elif protected:
            pool = [a for a in pool if exact_keys[a][0][0] == 0]
        best_capability = max(exact_keys[a][1] for a in pool)
        pool = [a for a in pool if exact_keys[a][1] == best_capability]
        # Native wins count/macro ties. q10 cannot alone justify intervention.
        if 1. in pool:
            selected = 1.
        elif preservation:
            selected = max(pool,key=lambda a:(-abs(a-1.),-a))
        else:
            selected = max(pool,key=lambda a:(exact_keys[a][2],exact_keys[a][3],-abs(a-1.),-a))
        if selected == 1.:
            reason = 'NO_PROTECTED_GROUP' if not protected else 'NO_BETTER_ACTION' if preservation else 'NATIVE_SAFE'
        elif preservation:
            reason = 'SHRINK_PROTECTED_GROUP' if selected < 1 else 'AMPLIFY_PROTECTED_GROUP'
        else:
            reason = 'AMPLIFY_FORMATION' if selected > 1 else 'FORMATION_PROGRESS'
        self.summary = dict(current=current,current_macro_group_capability=float(sum(current_ratios)/len(current_ratios)),
                            group_labels=list(self.mapping.labels),group_sizes=list(self.mapping.counts),
                            protected_groups=[self.mapping.labels[k] for k in protected],
                            mode='PRESERVATION' if preservation else 'FORMATION',
                            predicted_minima=p.tolist())
        return Decision(selected,int((b>0).sum()),int((b<=0).sum()),scores,reason,self.summary['mode'])
