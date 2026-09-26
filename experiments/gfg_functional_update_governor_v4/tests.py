"""V3 regressions plus explicit history/selection semantics."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from experiments.gfg_functional_update_governor_v1.tests import measured
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_functional_update_governor_v3.scheduler import GroupSentinel
from experiments.gfg_functional_update_governor_v3 import tests as v3
from .governor import RecoveryGovernor
from .runtime import GovernedTraining


class RecoveryTests(unittest.TestCase):
    def make(self):
        mapping=GroupMap([0]*10+[1]*10)
        s=GroupSentinel(mapping);s.begin(1)
        s.measured(1,torch.tensor([1.]*10+[-1.]*10))
        s.begin(2)
        return RecoveryGovernor(mapping,s),s

    def test_recovery_breaks_overall_tie(self):
        g,s=self.make();b=[-1.]*20
        native=b.copy();native[10:14]=[1.]*4
        recovery=b.copy();recovery[:4]=[1.]*4
        d=g.choose(measured(b,[b,b,recovery,native,native]))
        self.assertEqual(d.alpha,.5);self.assertEqual(d.reason,'RECOVERY_ASSIST')
        self.assertEqual(g.summary['degraded_groups'],[0])
        self.assertEqual(g.summary['protected_groups'],[])
        self.assertEqual(s.state[0],'PROTECTED') # preview did not mutate real history

    def test_debt_not_hard_constraint_or_tradeoff_weight(self):
        g,s=self.make();b=[-1.]*20;recovery=b.copy();recovery[:4]=[1.]*4
        bigger=b.copy();bigger[10:15]=[1.]*5
        d=g.choose(measured(b,[recovery,recovery,recovery,b,bigger]))
        self.assertEqual(d.alpha,2.)

    def test_actual_recovery_clears_debt(self):
        g,s=self.make();s.observe(list(range(10)),[-1.]*10)
        self.assertEqual(s.state[0],'DEGRADED')
        b=[.8]*10+[-1.]*10;g.choose(measured(b,[b]*5))
        self.assertEqual(g.summary['degraded_groups'],[])
        self.assertEqual(g.summary['protected_groups'],[0])

    def test_accuracy_recovery_without_anchor_rearm_still_debt(self):
        g,s=self.make();s.observe(list(range(10)),[-1.]*10)
        b=[.6]*10+[-1.]*10;g.choose(measured(b,[b]*5))
        self.assertEqual(g.summary['degraded_groups'],[0])
        self.assertEqual(g.summary['protected_groups'],[])

    def test_current_protection_overrides_gain(self):
        g,s=self.make();b=[1.]*10+[-1.]*10
        damage=[-1.]*2+[1.]*18
        d=g.choose(measured(b,[b,b,b,damage,damage]));self.assertEqual(d.alpha,.5)

    def test_native_safe_tie_and_target_diagnostics(self):
        g,s=self.make();b=[1.]*10+[-1.]*10
        d=g.choose(measured(b,[b]*5));self.assertEqual(d.alpha,1.)
        self.assertEqual(sum(d.scores['1.0']['transitions'].values()),20)

    def test_real_v3_step76_debt_does_not_invent_recoverable_candidate(self):
        root=Path(__file__).resolve().parents[1]/'gfg_functional_update_governor_v3/results/near-RESULT.json'
        rows=json.loads(root.read_text())['rows']
        from experiments.gfg_functional_update_governor_v1.smoke import banks
        x,y,c,_=banks();s=GroupSentinel(GroupMap.target_tokens(y[c]));g=RecoveryGovernor(s.mapping,s)
        for row in rows:
            i=row['info'];s.begin(row['t'])
            if i['micro_indices']:s.observe(i['micro_indices'],i['micro_margins'])
            if i['instrument_called']:
                m=measured(i['baseline_minima'],i['hierarchy']['predicted_minima'])
                if row['t']==76:
                    d=g.choose(m)
                    self.assertEqual(g.summary['degraded_groups'],[4,13])
                    self.assertEqual(d.alpha,2.)
                    self.assertEqual(d.reason,'RECOVERY_AWARE_FORMATION');break
                s.measured(row['t'],m.baseline_minima)

    def test_v4_real_runtime_uses_history_and_native_path(self):
        # Run v3 native byte/moment equivalence assertions through actual v4.
        class V4Compat(GovernedTraining):
            def update(self,*a,**kw):
                info=super().update(*a,**kw)
                assert info['runtime_version']=='governor_v4'
                info['runtime_version']='governor_v3';return info
        with patch.object(v3,'GovernedTraining',V4Compat):
            v3.RuntimeTests('test_v3_native_exact_and_no_jvp_on_micro').test_v3_native_exact_and_no_jvp_on_micro()


def suite():
    result=unittest.defaultTestLoader.loadTestsFromModule(v3)
    result.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests))
    return result


if __name__=='__main__':
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(suite())
    raise SystemExit(not result.wasSuccessful())
