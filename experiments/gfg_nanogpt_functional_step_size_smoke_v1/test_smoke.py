import subprocess
import sys
import unittest

import numpy as np

from .selector import choose, endpoints
from .adjudicate import metric, summarize
from .independent_checker import independent_metrics
from .common import ALPHAS


class SmokeTests(unittest.TestCase):
    def test_exact_alpha_one(self):
        rng = np.random.default_rng(17)
        g, f, j, h = [rng.normal(size=(212, 23)) for _ in range(4)]
        np.testing.assert_array_equal(endpoints(g, f, j, h, 1.0), g+f+j+h)
        np.testing.assert_array_equal(endpoints(g, f, j, h, 0.0), g)

    def test_lexicographic_priority(self):
        scores = {str(a): {"predicted_correct_count": 10, "median_predicted_minimum_margin": 1.0} for a in ALPHAS}
        self.assertEqual(choose(scores), 1.0)
        scores["0.5"]["median_predicted_minimum_margin"] = 2.0
        self.assertEqual(choose(scores), 0.5)
        scores["0.25"] = {"predicted_correct_count": 11, "median_predicted_minimum_margin": -10.0}
        self.assertEqual(choose(scores), 0.25)
        self.assertNotIn(0.0, ALPHAS)

    def test_transition_accounting(self):
        pre = np.array([1, -1, 1, -1, 0.0])
        base = np.array([-1, -1, 1, 1, 0.0])
        chosen = np.array([1, 1, -1, -1, 0.0])
        v = metric(chosen, base, pre)
        self.assertEqual(v, independent_metrics(chosen, base, pre))
        self.assertEqual([v[k] for k in ("prevented_correct_to_wrong", "additional_wrong_to_correct",
                                         "newly_caused_correct_to_wrong", "lost_wrong_to_correct")], [1,1,1,1])
        self.assertEqual(v["correct_count"], 2)

    def test_oracle_is_section_not_target(self):
        ledger = [{"section_id": "s", "entry_id": "r", "selected_alpha": 1.0},
                  {"section_id": "t", "entry_id": "entry-4ed462761347d6b87e61", "selected_alpha": 1.0}]
        margins = {"0.0": [-1.0]*212, "0.25": [1.0]*106+[-1.0]*106,
                   "0.5": [-1.0]*106+[1.0]*106, "0.75": [-1.0]*212, "1.0": [-1.0]*212}
        result = summarize(ledger, [{"section_id": r["section_id"], "entry_id": r["entry_id"],
                                    "actual_minimum_margins": margins} for r in ledger])
        self.assertEqual(result["pooled_descriptive"]["oracle"]["correct_count"], 212)

    def test_stage_a_outcome_read_blocked(self):
        code = """
from pathlib import Path
from types import SimpleNamespace
from experiments.gfg_nanogpt_functional_step_size_smoke_v1.common import install_read_guard
p = Path.cwd() / 'blocked-test-evidence'
args = SimpleNamespace(reports_root=p, response_root=p, boundary_root=p)
install_read_guard(args, [])
try:
    (p / 'sections' / 'outcome.npz').open('rb')
except RuntimeError as e:
    assert 'STAGE_A_FORBIDDEN_SOURCE' in str(e)
else:
    raise AssertionError('outcome read was not blocked')
"""
        subprocess.run([sys.executable, "-B", "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
