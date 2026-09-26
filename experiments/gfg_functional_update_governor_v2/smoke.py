"""Reuse the untouched v1 run harness with explicit scoped dependency injection."""
import argparse
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.gfg_functional_update_governor_v1 import smoke as harness
from experiments.gfg_controller_distributed_runtime_v2.adapters import NanoGPTAdapter
from .runtime import GovernedTraining

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = Path('E:/gfg-evidence/functional-update-governor-v2')
CONFIG = dict(alphas=[.125,.25,.5,1.,2.], release_steps=0, micro_probe_size=8,
              micro_every=1, max_instrument_interval=50, erosion_ratio=.5,
              recovery_ratio=.75, absolute_floor=1e-6, backend='exact',
              count_tie_policy='native; frontier cannot alone justify intervention',
              latch='per-target pre-excursion anchor; soft-to-hard escalation allowed')


def create(lr):
    model = harness.model_for('70',201)
    opt = torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=0,fused=True,foreach=False)
    return model,opt,GovernedTraining(model,NanoGPTAdapter(model))


def run(case):
    # Reuse identical batches, initialization, two discarded warmups, evaluation,
    # first99 stopping and append-only raw storage. No source monkeypatch on disk.
    # `create` is a reserved argument of patch.multiple, NOT an attribute there.
    with patch.multiple(harness, HERE=HERE, OUT=OUT, CONFIG=CONFIG), patch.object(harness,'create',create):
        assert harness.create is create
        harness.run(case)


def summarize():
    dest = HERE/'results'; dest.mkdir(exist_ok=True)
    comparison = {}
    for case in harness.RATES:
        attempts = sorted((OUT/case).glob('attempt-*/RESULT.json'))
        if [p.parent.name for p in attempts] != ['attempt-1','attempt-2']:
            raise RuntimeError('expected preserved harness failure then one corrected run; audit any extra attempt')
        failed = json.loads(attempts[0].read_text())
        assert all('micro_probe_called' not in r['info'] for r in failed['rows'])
        path = attempts[1]; d = json.loads(path.read_text())
        assert all('micro_probe_called' in r['info'] for r in d['rows'])
        assert d['status'] == 'COMPLETED'
        assert d['raw_sha256'] == harness.sha(path.parent/'RAW.npz')
        with np.load(path.parent/'RAW.npz') as raw:
            y = raw['targets']
            for row in d['rows']:
                z = raw[f"{row['t']}_logits"]; other = z.copy(); other[np.arange(len(y)),y] = -np.inf
                accuracy = float((z[np.arange(len(y)),y].astype(np.float64)-other.max(1).astype(np.float64)>0).mean())
                assert accuracy == row['accuracy']
        hit = next((row for row in d['rows'] if row['accuracy'] >= .99),None)
        assert d['steps_to99'] == (hit['t'] if hit else None)
        assert all(r['info']['alpha'] == 1. for r in d['rows'] if not r['info']['instrument_called'])
        assert sum(r['info']['instrument_called'] for r in d['rows']) == d['instrument_calls']
        times = Counter()
        for row in d['rows']: times.update(row['info']['timing'])
        d.update(micro_probe_calls=sum(r['info']['micro_probe_called'] for r in d['rows']),
                 ordinary_native_steps=sum(not r['info']['instrument_called'] for r in d['rows']),
                 timing_totals=dict(times), instrument_wall_fraction=times['instrument']/d['wall'],
                 full_path_wall=sum(r['update_seconds'] for r in d['rows'] if r['info']['instrument_called']),
                 intervention_histogram=dict(Counter(r['info']['intervention_reason'] for r in d['rows'] if r['info']['instrument_called'])),
                 event_histogram=dict(Counter(e['kind'] for r in d['rows'] for e in r['info']['micro_events'])),
                 raw_path=str(path.parent/'RAW.npz'), raw_result_path=str(path),raw_result_sha256=harness.sha(path),
                 excluded_attempt=dict(path=str(attempts[0]),sha256=harness.sha(attempts[0]),
                     reason='Harness injection bug: patch.multiple reserved create argument left v1 constructor active. Not a v2 run.'),
                 accuracy_recomputed_from_raw=True)
        # Publish v1's missing compact reports as byte-identical copies in v2,
        # leaving v1 (including its ignored local files) untouched.
        old = REPO/'experiments/gfg_functional_update_governor_v1/results'/f'{case}-RESULT.json'
        h = json.loads(old.read_text())
        history = json.loads((old.parent/'COMPARISON.json').read_text())[case]['historical_references']
        assert h['control_ids'] == d['control_ids'] and h['evaluation_ids'] == d['evaluation_ids']
        prior = dest/f'v1-{case}-RESULT.json'
        with prior.open('xb') as f: f.write(old.read_bytes())
        assert harness.sha(prior) == harness.sha(old)
        history['governor_v1'] = dict(steps_to99=h['steps_to99'],wall_to99=h['rows'][-1]['wall'],
                                     instrument_calls=h['instrument_calls'], source=str(old), source_sha256=harness.sha(old))
        comparison[case] = dict(steps_to99=d['steps_to99'],wall_to99=hit['wall'] if hit else None,
                               final_accuracy=d['final_accuracy'],total_steps=len(d['rows']),
                               instrument_calls=d['instrument_calls'],instrument_fraction=d['instrument_fraction'],
                               micro_probe_calls=d['micro_probe_calls'],timing_totals=dict(times),
                               full_call_reduction_vs_v1=1-d['instrument_calls']/h['instrument_calls'],
                               decision_alpha_histogram=d['decision_alpha_histogram'],alpha_histogram=d['alpha_histogram'],
                               trigger_histogram=d['trigger_histogram'],intervention_histogram=d['intervention_histogram'],
                               peak_allocated=d['peak_allocated'],historical_references=history,
                               time_comparison='Historical, not contemporaneously rerun. V2 adds segment CUDA synchronization.')
        harness.write(dest/f'{case}-RESULT.json',d)
    harness.write(dest/'COMPARISON.json',comparison)
    print(json.dumps(comparison,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=('safe','near','summary'))
    args = parser.parse_args()
    if args.action == 'summary': summarize()
    else: run(args.action)
