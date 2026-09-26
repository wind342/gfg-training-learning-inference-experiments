"""Read-only result recomputation; optional ordinary check-report creation."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import unittest
import numpy as np
import torch
from . import tests
from .smoke import HERE, REPO
from .scheduler import FunctionalSentinel

BASE = '25ef6dbdde7ad6b31a2807e6323c785a29190dc1'
TAG = 'paper-experiments-cross-system-feedback-release'
TAG_OBJECT = '5a3d12c3de6fce0d9b5f69aa07d009c61e5c189e'
TAG_COMMIT = '36dab5ce347dbbdac157ef23205f556606d18294'


def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    torch.set_num_threads(2)
    assert git('rev-parse',TAG) == TAG_OBJECT
    assert git('rev-parse',TAG+'^{}') == TAG_COMMIT
    # Include staged additions but permit ONLY the new directory. All previously
    # tracked sources remain byte-identical to the base Git objects after filters.
    changes = git('diff','--name-only',BASE).splitlines()
    assert all(p.startswith('experiments/gfg_functional_update_governor_v2/') for p in changes), changes
    existing = git('ls-tree','-r','--name-only',BASE).splitlines()
    for name in existing:
        assert (REPO/name).is_file(), name
    report = dict(base=BASE,tag_object=TAG_OBJECT,tag_commit=TAG_COMMIT,
                  existing_tracked_files=len(existing),previous_tracked_files_unchanged=True,runs={})
    for case in ('safe','near'):
        path = HERE/'results'/f'{case}-RESULT.json'
        d = json.loads(path.read_text())
        rawpath = Path(d['raw_path'])
        assert sha(rawpath) == d['raw_sha256']
        assert sha(Path(d['raw_result_path'])) == d['raw_result_sha256']
        for name, expected in d['source_hashes'].items():
            assert sha(HERE/name) == expected, name
        with np.load(rawpath) as raw:
            y = raw['targets']; first = None
            sentinel = FunctionalSentinel()
            for row in d['rows']:
                z = raw[f"{row['t']}_logits"]; other = z.copy(); other[np.arange(len(y)),y] = -np.inf
                accuracy = float((z[np.arange(len(y)),y].astype(np.float64)-other.max(1).astype(np.float64)>0).mean())
                assert accuracy == row['accuracy']
                if accuracy >= .99 and first is None: first = row['t']
                info = row['info']
                reasons = sentinel.begin(row['t'])
                if not reasons:
                    ids = sentinel.shard(row['t'])
                    assert ids == info['micro_indices']
                    events = sentinel.observe(ids,info['micro_margins']) if ids else []
                    assert events == info['micro_events']
                    reasons = sentinel.reasons(events)
                assert (reasons or ['NONE']) == info['instrument_reason']
                if not info['instrument_called']: assert info['alpha'] == 1.
                else:
                    assert all('utility' not in s for s in info['scores'].values())
                    events = sentinel.measured(row['t'],torch.tensor(info['baseline_minima'],dtype=torch.float64))
                    assert events == info['full_refresh_events']
                assert int((sentinel.level>0).sum()) == info['latched_targets']
        assert first == d['steps_to99']
        prior = HERE/'results'/f'v1-{case}-RESULT.json'
        old = REPO/'experiments/gfg_functional_update_governor_v1/results'/f'{case}-RESULT.json'
        assert sha(prior) == sha(old)
        report['runs'][case] = dict(raw_sha256=d['raw_sha256'],steps_to99=first,
                                   final_accuracy=d['final_accuracy'],copied_v1_report_sha256=sha(prior))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    report.update(tests_run=result.testsRun,test_output=output.getvalue(),
                  failures=len(result.failures),errors=len(result.errors),status='PASS' if result.wasSuccessful() else 'FAIL')
    if not result.wasSuccessful(): raise AssertionError(output.getvalue())
    # Record unchanged imported implementation identities without inventing a
    # new seal/protocol or requiring their existing evidence bundles to change.
    dependencies = [REPO/'experiments/gfg_functional_update_governor_v1'/n for n in ('smoke.py','runtime.py','instrument.py','governor.py')]
    dependencies += [REPO/'experiments/gfg_controller_distributed_runtime_v2'/n for n in ('adapters.py','boundary.py','identity.py','ownership.py','clipping.py','scientific.py','evidence.py')]
    report['imported_source_sha256'] = {str(p.relative_to(REPO)):sha(p) for p in dependencies}
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-report',action='store_true');args=parser.parse_args()
    value=verify()
    if args.write_report:
        with (HERE/'results'/'CHECKS.json').open('x',encoding='utf-8') as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(value,indent=2))
