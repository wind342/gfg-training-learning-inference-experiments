import io
import unittest
from .core import *
from experiments.gfg_head_only_exact_monitor_smoke_v1.tests import Checks

class LRChecks(Checks):
    def test_16_lr_after_load(self):
        initial=copy.deepcopy(self.opt.state_dict())
        for lr in ARMS.values():
            set_lr(self.opt,lr);self.assertEqual(self.opt.param_groups[0]['lr'],lr)
            self.assertTrue(equal(initial['state'],self.opt.state_dict()['state']))

    def test_17_native_at_all_lrs(self):
        for lr in ARMS.values():
            with self.subTest(lr=lr):
                set_lr(self.opt,lr);self.test_12_native_equivalence()

    def test_18_monitor_all_lrs(self):
        for lr in ARMS.values():
            with self.subTest(lr=lr):
                set_lr(self.opt,lr);self.test_15_two_arms()

def main():
    unchanged();ROOT.mkdir(exist_ok=False);setup()
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LRChecks))
    print(stream.getvalue(),flush=True);audit=data_audit(np.load(DATA/'data.npz'))
    oldpre=json.loads((BASE_ROOT/'PREFLIGHT.json').read_text());oldresult=BASE_ROOT/'head_only_exact_monitor'/'RESULT.json'
    baseline=json.loads(oldresult.read_text());assert baseline['status']=='COMPLETED'
    assert sha(BASE_ROOT/'head_only_exact_monitor'/'AUDIT.pt')==baseline['audit_sha256']
    baseline_files={str(p):sha(p) for p in [BASE_ROOT/'PREFLIGHT.json',oldresult,BASE_ROOT/'head_only_exact_monitor'/'AUDIT.pt',BASE_ROOT/'head_only_exact_monitor'/'STEPS.jsonl']}
    report=dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,test_log=stream.getvalue(),source_hashes=sources(),
        baseline_files=baseline_files,data_audit=audit,head_initial_sha256=oldpre['head_initial_sha256'],
        head_optimizer_initial_sha256=oldpre['head_optimizer_initial_sha256'],backbone_initial_sha256=oldpre['backbone_initial_sha256'])
    write(ROOT/'PREFLIGHT.json',report);assert r.wasSuccessful();print('PREFLIGHT PASS',flush=True)

if __name__=='__main__':main()
