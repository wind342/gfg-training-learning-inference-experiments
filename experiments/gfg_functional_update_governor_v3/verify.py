"""Small read-only checks: recorded decisions, raw accuracy, sources and tests."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import unittest
import numpy as np
import torch
from experiments.gfg_functional_update_governor_v1.instrument import Measurement
from .groups import GroupMap
from .governor import GroupGovernor
from .scheduler import GroupSentinel
from .smoke import HERE,REPO,harness,write
from . import tests

BASE='ba9790eb212148ae703f93456f5d0d3941e7afca'
TAG='paper-experiments-cross-system-feedback-release'


def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def verify():
    torch.set_num_threads(2)
    assert git('rev-parse',TAG)=='5a3d12c3de6fce0d9b5f69aa07d009c61e5c189e'
    assert git('rev-parse',TAG+'^{}')=='36dab5ce347dbbdac157ef23205f556606d18294'
    assert all(p.startswith('experiments/gfg_functional_update_governor_v3/') for p in git('diff','--name-only',BASE).splitlines())
    result=dict(base=BASE,old_tracked_files_unchanged=True,frozen_tag_unchanged=True,runs={})
    for case in ('safe','near'):
        d=json.loads((HERE/'results'/f'{case}-RESULT.json').read_text())
        assert harness.sha(Path(d['raw_path']))==d['raw_sha256']
        assert harness.sha(Path(d['raw_result_path']))==d['raw_result_sha256']
        for name,expected in d['source_hashes'].items():assert harness.sha(HERE/name)==expected,name
        _,y,control,ev=harness.banks()
        assert control==d['control_ids'] and ev==d['evaluation_ids']
        mapping=GroupMap.target_tokens(y[control],expected=range(23))
        scheduler=GroupSentinel(mapping);governor=GroupGovernor(mapping)
        calls=0;first=None;chosen_delta=[];violations=[]
        with np.load(d['raw_path']) as raw:
            targets=raw['targets']
            for row in d['rows']:
                t=row['t'];info=row['info']
                z=raw[f'{t}_logits'];other=z.copy();other[np.arange(len(z)),targets]=-np.inf
                acc=float((z[np.arange(len(z)),targets].astype(np.float64)-other.max(1).astype(np.float64)>0).mean())
                assert acc==row['accuracy']
                if acc>=.99 and first is None:first=t
                reasons=scheduler.begin(t)
                if not reasons:
                    ids=scheduler.shard(t);assert ids==info['micro_indices']
                    events=scheduler.observe(ids,info['micro_margins']) if ids else []
                    assert events==info['micro_events'];reasons=scheduler.reasons(events)
                assert (reasons or ['NONE'])==info['instrument_reason']
                assert bool(reasons)==info['instrument_called']
                if reasons:
                    calls+=1
                    m=Measurement(tuple(d['config']['alphas']),torch.tensor(info['baseline_minima'],dtype=torch.float64),
                                  torch.tensor(info['hierarchy']['predicted_minima'],dtype=torch.float64))
                    decision=governor.choose(m)
                    assert decision.alpha==info['alpha'] and decision.scores==info['scores']
                    assert governor.summary==info['hierarchy']
                    assert decision.reason==info['intervention_reason']
                    assert scheduler.measured(t,m.baseline_minima)==info['full_refresh_events']
                    chosen_delta.append(info['scores'][str(info['alpha'])]['delta_capability'])
                    violations.append(info['scores'][str(info['alpha'])]['protected_violation_count'])
                    # Separate direct NumPy aggregation of every group and four transitions.
                    for j,a in enumerate(m.alphas):
                        endpoint=m.predicted_minima[j].numpy();before=m.baseline_minima.numpy()>0;after=endpoint>0
                        score=info['scores'][str(a)]
                        assert score['correct_count']==int(after.sum())
                        assert score['capabilities']==[float(after[mapping.ids==k].mean()) for k in mapping.labels]
                        assert sum(score['transitions'].values())==212
                        assert score['transitions']['wrong_to_correct']==int((~before&after).sum())
                        assert score['transitions']['correct_to_wrong']==int((before&~after).sum())
                else:assert info['alpha']==1.
                assert scheduler.snapshot(t)==info['group_observation']
                assert info['latched_groups']==int((scheduler.level>0).sum())
                assert info['group_mapping_sha256']==mapping.identity
        assert first==d['steps_to99'] and calls==d['instrument_calls']
        result['runs'][case]=dict(steps_to99=first,final_accuracy=d['final_accuracy'],instrument_calls=calls,
                                  group_sizes=list(mapping.counts),initial_protected=d['rows'][0]['info']['hierarchy']['protected_groups'],
                                  initial_alpha=d['rows'][0]['info']['alpha'],chosen_predicted_deltas=chosen_delta,
                                  chosen_protected_violations=violations,raw_sha256=d['raw_sha256'])
    output=io.StringIO()
    with contextlib.redirect_stdout(output):
        checked=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    assert checked.wasSuccessful(),output.getvalue()
    result.update(status='PASS',tests_run=checked.testsRun,test_output=output.getvalue(),failures=0,errors=0)
    source_dirs=('gfg_functional_update_governor_v1','gfg_functional_update_governor_v2','gfg_controller_distributed_runtime_v2')
    result['imported_sources']={str(p.relative_to(REPO)):harness.sha(p) for name in source_dirs for p in (REPO/'experiments'/name).glob('*.py')}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-report',action='store_true');args=parser.parse_args()
    report=verify()
    if args.write_report:write(HERE/'results'/'CHECKS.json',report)
    print(json.dumps(report,indent=2))
