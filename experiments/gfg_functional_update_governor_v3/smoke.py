"""Exactly two fixed runs using the unchanged historical training harness."""
import argparse
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.gfg_functional_update_governor_v1 import smoke as harness
from experiments.gfg_controller_distributed_runtime_v2.adapters import NanoGPTAdapter
from .groups import GroupMap
from .runtime import GovernedTraining

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUT=Path('E:/gfg-evidence/functional-update-governor-v3')
CONFIG=dict(alphas=[.125,.25,.5,1.,2.],release_steps=0,group_micro_every=1,
            max_instrument_interval=50,protected_group_threshold=.9,erosion_ratio=.5,
            recovery_ratio=.75,absolute_floor=1e-6,backend='exact',
            group_mapping='correct target token, task asserts exactly 0..22',
            q10='numpy.quantile(method=linear)',native_tie='overall and macro tie -> native')


def create(lr):
    x,y,control,_=harness.banks()
    mapping=GroupMap.target_tokens(y[control],expected=range(23))
    model=harness.model_for('70',201)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=0,fused=True,foreach=False)
    return model,opt,GovernedTraining(model,NanoGPTAdapter(model),mapping=mapping)


def run(case):
    with patch.multiple(harness,HERE=HERE,OUT=OUT,CONFIG=CONFIG),patch.object(harness,'create',create):
        assert harness.create is create
        harness.run(case)


def write(path,value):
    # Compact JSON: retain per-step evidence without pretty-print inflation.
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,separators=(',',':'),sort_keys=True,allow_nan=False);f.write('\n')


def summarize():
    dest=HERE/'results';dest.mkdir(exist_ok=True);comparison={}
    for case in harness.RATES:
        files=list((OUT/case).glob('attempt-*/RESULT.json'))
        if len(files)!=1:raise RuntimeError('audit missing/multiple attempts explicitly; no best-run selection')
        path=files[0];d=json.loads(path.read_text())
        assert d['status']=='COMPLETED'
        assert all(r['info']['runtime_version']=='governor_v3' for r in d['rows'])
        assert harness.sha(path.parent/'RAW.npz')==d['raw_sha256']
        with np.load(path.parent/'RAW.npz') as raw:
            targets=raw['targets']
            for row in d['rows']:
                z=raw[f"{row['t']}_logits"];other=z.copy();other[np.arange(len(z)),targets]=-np.inf
                acc=float((z[np.arange(len(z)),targets].astype(np.float64)-other.max(1).astype(np.float64)>0).mean())
                assert acc==row['accuracy']
        totals=Counter()
        for row in d['rows']:totals.update(row['info']['timing'])
        d.update(timing_totals=dict(totals),micro_probe_calls=sum(r['info']['micro_probe_called'] for r in d['rows']),
                 intervention_histogram=dict(Counter(r['info']['intervention_reason'] for r in d['rows'] if r['info']['instrument_called'])),
                 protected_group_count_trajectory=[r['info']['group_observation']['protected_group_count'] for r in d['rows']],
                 ordinary_native_steps=sum(not r['info']['instrument_called'] for r in d['rows']),
                 raw_path=str(path.parent/'RAW.npz'),raw_result_path=str(path),raw_result_sha256=harness.sha(path))
        oldroot=REPO/'experiments/gfg_functional_update_governor_v2/results'
        old=json.loads((oldroot/f'{case}-RESULT.json').read_text())
        history=json.loads((oldroot/'COMPARISON.json').read_text())[case]['historical_references']
        assert old['control_ids']==d['control_ids'] and old['evaluation_ids']==d['evaluation_ids']
        history['governor_v2']=dict(steps_to99=old['steps_to99'],final_accuracy=old['final_accuracy'],
                                   wall=old['wall'],instrument_calls=old['instrument_calls'],
                                   source=str(oldroot/f'{case}-RESULT.json'),source_sha256=harness.sha(oldroot/f'{case}-RESULT.json'))
        hit=next((r for r in d['rows'] if r['accuracy']>=.99),None)
        comparison[case]=dict(steps_to99=d['steps_to99'],final_accuracy=d['final_accuracy'],wall=d['wall'],
            wall_to99=hit['wall'] if hit else None,total_steps=len(d['rows']),instrument_calls=d['instrument_calls'],
            instrument_fraction=d['instrument_fraction'],micro_calls=d['micro_probe_calls'],timing_totals=dict(totals),
            full_call_change_vs_v2=d['instrument_calls']-old['instrument_calls'],
            alpha_histogram=d['alpha_histogram'],decision_alpha_histogram=d['decision_alpha_histogram'],
            trigger_histogram=d['trigger_histogram'],intervention_histogram=d['intervention_histogram'],
            peak_allocated=d['peak_allocated'],historical_references=history,
            timing_scope='Historical baselines, not contemporaneously rerun; v3 uses the v2 segment timing path.')
        write(dest/f'{case}-RESULT.json',d)
    harness.write(dest/'COMPARISON.json',comparison)
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('safe','near','summary'));a=p.parse_args()
    summarize() if a.action=='summary' else run(a.action)
