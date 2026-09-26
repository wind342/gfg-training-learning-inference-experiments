"""Recompute regression histories, decisions and external raw accuracies."""
import contextlib
import io
import json
from collections import Counter
from pathlib import Path
import unittest
import numpy as np
import torch
from experiments.gfg_functional_update_governor_v1.tests import measured
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_functional_update_governor_v3.scheduler import GroupSentinel
from experiments.gfg_functional_update_governor_v3.smoke import write
from .smoke import HERE,OUT,harness
from .governor import RecoveryGovernor
from .tests import suite


def main():
    dest=HERE/'results';dest.mkdir(exist_ok=True);report={}
    for case in ('safe','near'):
        paths=list((OUT/case).glob('attempt-*/RESULT.json'));assert len(paths)==1
        path=paths[0];d=json.loads(path.read_text());assert d['status']=='COMPLETED'
        raw_path=path.parent/'RAW.npz';assert harness.sha(raw_path)==d['raw_sha256']
        for f,h in d['source_hashes'].items():assert harness.sha(HERE/f)==h
        _,y,control,ev=harness.banks();assert d['control_ids']==control and d['evaluation_ids']==ev
        s=GroupSentinel(GroupMap.target_tokens(y[control]));g=RecoveryGovernor(s.mapping,s)
        changes=[];times=Counter()
        old=json.loads((HERE.parent/'gfg_functional_update_governor_v3/results'/f'{case}-RESULT.json').read_text())
        with np.load(raw_path) as raw:
            target=raw['targets'];first=None
            for row in d['rows']:
                t=row['t'];i=row['info'];assert i['runtime_version']=='governor_v4'
                z=raw[f'{t}_logits'];other=z.copy();other[np.arange(len(z)),target]=-np.inf
                accuracy=float((z[np.arange(len(z)),target].astype('float64')>other.max(1).astype('float64')).mean())
                assert row['accuracy']==accuracy
                if first is None and accuracy>=.99:first=t
                reasons=s.begin(t)
                if not reasons:
                    ids=s.shard(t);assert ids==i['micro_indices']
                    events=s.observe(ids,i['micro_margins']) if ids else []
                    assert events==i['micro_events'];reasons=s.reasons(events)
                assert (reasons or ['NONE'])==i['instrument_reason']
                if reasons:
                    m=measured(i['baseline_minima'],i['hierarchy']['predicted_minima'])
                    decision=g.choose(m)
                    assert decision.alpha==i['alpha'] and decision.scores==i['scores']
                    assert g.summary==i['hierarchy']
                    assert s.measured(t,m.baseline_minima)==i['full_refresh_events']
                    changes.append(dict(step=t,alpha=i['alpha'],mode=i['hierarchy']['mode'],
                        debts=i['hierarchy']['degraded_groups'],protected=i['hierarchy']['protected_groups']))
                assert s.snapshot(t)==i['group_observation']
                times.update(i['timing'])
            assert first==d['steps_to99']
        equal=all(a['accuracy']==b['accuracy'] and a['info']['alpha']==b['info']['alpha'] for a,b in zip(d['rows'],old['rows'])) and len(d['rows'])==len(old['rows'])
        report[case]=dict(steps_to99=first,final_accuracy=d['final_accuracy'],wall=d['wall'],
            instrument_calls=d['instrument_calls'],alpha_histogram=d['alpha_histogram'],
            trigger_histogram=d['trigger_histogram'],timing_totals=dict(times),decisions=changes,
            trajectory_and_alphas_identical_to_v3=equal,peak_allocated=d['peak_allocated'])
        d.update(raw_path=str(raw_path),raw_result_path=str(path),raw_result_sha256=harness.sha(path))
        write(dest/f'{case}-RESULT.json',d)
    output=io.StringIO();torch.set_num_threads(2)
    with contextlib.redirect_stdout(output):
        tested=unittest.TextTestRunner(stream=output,verbosity=2).run(suite())
    assert tested.wasSuccessful(),output.getvalue()
    report.update(status='PASS',tests=tested.testsRun,test_output=output.getvalue())
    write(dest/'CHECKS.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='test_output'},indent=2))


if __name__=='__main__':main()
