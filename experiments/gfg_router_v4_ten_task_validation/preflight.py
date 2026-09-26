"""Create-only protocol/source/data admission; execute all tests before training."""
import importlib
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .common import HERE,ROOT,REPO,DATA,CHECKPOINT,CKSHA,DATA_SHA,sha,write,unchanged,sources,data_audit
from experiments.gfg_router_v4_partial_capability.regression import StableOldFixture
from experiments.gfg_router_v3_three_task_continuation import tests as oldtests

def main():
    unchanged();assert sha(CHECKPOINT)==CKSHA and sha(DATA/'data.npz')==DATA_SHA
    oldroot=ROOT.parent/'router-v4-partial-capability'
    old=json.loads((oldroot/'CHECKS.json').read_text());assert old['status']=='PASS'
    assert old==json.loads((HERE.parent/'gfg_router_v4_partial_capability/results/CHECKS.json').read_text())
    for arm,row in old['arms'].items():
        for name,h in row['hashes'].items():assert sha(oldroot/f'three-{arm}'/name)==h
    torch.set_num_threads(2)
    names=['gfg_router_v4_ten_task_validation.tests','gfg_router_v4_partial_capability.tests','gfg_router_v3_three_task_continuation.tests',
        'gfg_router_v3_three_task_continuation.audit_tests','gfg_functional_feedback_router_v3.tests',
        'gfg_functional_feedback_router_v2.tests','gfg_functional_feedback_router_v1.tests',
        'gfg_functional_update_governor_v4.tests','gfg_governor_fewrel_v1.tests']
    @unittest.skip('unchanged old object-identity fixture replaced by its existing stable-view equivalent')
    def oldfixture(self):pass
    with patch.object(oldtests.HarnessTests,'test_real_append_preserves_all_old_states',oldfixture):
        suite=unittest.TestSuite()
        for name in names:
            m=importlib.import_module('experiments.'+name)
            suite.addTests(m.suite() if name=='gfg_functional_update_governor_v4.tests' else unittest.defaultTestLoader.loadTestsFromModule(m))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(StableOldFixture))
        result=unittest.TextTestRunner(verbosity=1).run(suite)
    output=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        passed=result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
        skipped=[(str(t),reason) for t,reason in result.skipped],failures=[(str(t),reason) for t,reason in result.failures],
        errors=[(str(t),reason) for t,reason in result.errors],data_audit=data_audit(np.load(DATA/'data.npz')),
        source_hashes=sources(),historical_v4_checks_sha256=sha(oldroot/'CHECKS.json'),
        checkpoint_sha256=CKSHA,data_sha256=DATA_SHA,protocol_sha256=sha(HERE/'PROTOCOL.md'))
    write(ROOT/'PREFLIGHT.json',output);write(HERE/'results/PREFLIGHT.json',output)
    print(json.dumps({k:output[k] for k in ('status','tests','passed','skipped')},indent=2))
    if not result.wasSuccessful():raise RuntimeError('preflight unit tests failed')

if __name__=='__main__':main()
