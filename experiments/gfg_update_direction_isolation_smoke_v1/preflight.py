import json
import unittest
import numpy as np
import torch
from .core import *

def main():
    unchanged();assert sha(CHECKPOINT)==CKSHA and sha(DATA/'data.npz')==DATA_SHA
    previous=HERE.parent/'gfg_router_v4_ten_task_validation/results/CHECKS.json'
    assert json.loads(previous.read_text())['status']=='PASS'
    torch.set_num_threads(2)
    suite=unittest.defaultTestLoader.loadTestsFromName('experiments.'+HERE.name+'.tests')
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,
        failures=[(str(t),v) for t,v in r.failures],errors=[(str(t),v) for t,v in r.errors],
        data_audit=data_audit(np.load(DATA/'data.npz')),source_hashes=sources(),
        checkpoint_sha256=CKSHA,data_sha256=DATA_SHA,previous_checks_sha256=sha(previous),schedule=SCHEDULE)
    write(ROOT/'PREFLIGHT.json',report);write(HERE/'results/PREFLIGHT.json',report)
    if not r.wasSuccessful():raise RuntimeError('preflight failed; no training authorized')

if __name__=='__main__':main()
