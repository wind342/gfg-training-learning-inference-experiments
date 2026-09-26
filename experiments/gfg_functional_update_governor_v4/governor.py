"""Current capability constraints with explicit, non-hard historical recovery debt."""
import copy
from fractions import Fraction
from experiments.gfg_functional_update_governor_v2.governor import Decision
from experiments.gfg_functional_update_governor_v3.governor import GroupGovernor as V3Governor


class RecoveryGovernor(V3Governor):
    def __init__(self, mapping, scheduler, threshold=.9):
        super().__init__(mapping, threshold)
        if scheduler.mapping.identity != mapping.identity:
            raise ValueError('scheduler/group identity mismatch')
        self.scheduler = scheduler

    def choose(self, measurement):
        # Preview the SAME full refresh the execution layer will commit below.
        # This includes groups missed by micro and clears debts recovered now.
        preview = copy.deepcopy(self.scheduler)
        refresh = preview.measured(preview.last_step, measurement.baseline_minima)
        inherited = super().choose(measurement)  # unchanged per-target arithmetic
        scores = inherited.scores
        current = self.summary['current']
        debts = [k for k,s in enumerate(preview.state) if s == 'DEGRADED']
        ratios = lambda counts: [Fraction(c,n) for c,n in zip(counts,self.mapping.counts)]
        current_ratios = ratios(current['correct_counts'])
        protected = [k for k,c in enumerate(current_ratios) if c >= self.threshold and k not in debts]
        keys = {}
        for alpha in measurement.alphas:
            alpha = float(alpha); score = scores[str(alpha)]
            cs = ratios(score['correct_counts'])
            deficits = [max(Fraction(0),self.threshold-cs[k]) for k in protected]
            violation = (sum(d>0 for d in deficits), max(deficits,default=Fraction(0)),sum(deficits,Fraction(0)))
            debt_macro = sum((cs[k] for k in debts),Fraction(0))/len(debts) if debts else Fraction(0)
            worsening = sum(cs[k]<current_ratios[k] for k in debts)
            # Predicted capability recovery is not actual scheduler re-arming.
            recovery = sum(cs[k]>=self.threshold for k in debts)
            score.update(protected_violation_count=violation[0],worst_protected_deficit=float(violation[1]),
                total_protected_deficit=float(violation[2]),degraded_macro_capability=float(debt_macro),
                degraded_recovery_count=recovery,degraded_worsening_count=worsening,
                degraded_capabilities={str(self.mapping.labels[k]):float(cs[k]) for k in debts})
            keys[alpha] = (violation,score['correct_count'],sum(cs)/len(cs),debt_macro,-worsening,recovery)
        pool = list(keys)
        if protected:
            mode = 'PRESERVATION' if keys[1.][0][0] else 'SAFE_FORMATION'
            best = min(keys[a][0] for a in pool)
            pool = [a for a in pool if keys[a][0]==best]
            best = max((keys[a][1],keys[a][2]) for a in pool)
            pool = [a for a in pool if (keys[a][1],keys[a][2])==best]
            # Native-safe count/macro ties remain native, including recovery debt.
            if 1. in pool:
                selected = 1.
            else:
                selected = max(pool,key=lambda a:(keys[a][3],keys[a][4],keys[a][5],-abs(a-1.),-a))
        elif debts:
            mode = 'RECOVERY_AWARE_FORMATION'
            best = max(keys[a][1] for a in pool)
            pool = [a for a in pool if keys[a][1]==best]
            # Exact overall-count optimum, no "slightly worse" tolerance/lambda.
            def recovery_key(a):return (keys[a][3],keys[a][4],keys[a][5],keys[a][2])
            best = max(recovery_key(a) for a in pool)
            pool = [a for a in pool if recovery_key(a)==best]
            selected = 1. if 1. in pool else max(pool,key=lambda a:(scores[str(a)]['newly_protected_groups'],
                scores[str(a)]['q10_progress'],-abs(a-1.),-a))
        else:
            mode = 'FORMATION'; selected = inherited.alpha
        if mode == 'RECOVERY_AWARE_FORMATION':
            reason = 'RECOVERY_ASSIST' if selected!=1. and keys[selected][3]>keys[1.][3] else 'RECOVERY_AWARE_FORMATION'
        elif selected==1.:
            reason = 'NATIVE_SAFE' if mode=='SAFE_FORMATION' else 'NO_BETTER_ACTION'
        elif mode=='PRESERVATION':
            reason = 'SHRINK_PROTECTED_GROUP' if selected<1. else 'AMPLIFY_PROTECTED_GROUP'
        else:
            reason = 'AMPLIFY_FORMATION' if selected>1. else 'FORMATION_PROGRESS'
        self.summary.update(protected_groups=[self.mapping.labels[k] for k in protected],
            current_protected_groups=[self.mapping.labels[k] for k in protected],
            degraded_groups=[self.mapping.labels[k] for k in debts],mode=mode,
            scheduler_preview=preview.snapshot(preview.last_step),preview_events=refresh)
        return Decision(selected,inherited.formed_count,inherited.unformed_count,scores,reason,mode)
