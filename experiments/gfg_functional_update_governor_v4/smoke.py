"""Fixed regression only; historical modular-addition task is not a benchmark."""
import argparse
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.gfg_functional_update_governor_v1 import smoke as harness
from experiments.gfg_functional_update_governor_v3.smoke import CONFIG as V3_CONFIG
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_controller_distributed_runtime_v2.adapters import NanoGPTAdapter
from .runtime import GovernedTraining

HERE=Path(__file__).resolve().parent
OUT=Path('E:/gfg-evidence/functional-update-governor-v4')
CONFIG=dict(V3_CONFIG,recovery='exact overall optimum, debt macro/worsening/recovery, macro; no lambda',
    history='preview current full refresh, then commit identical refresh',scope='regression only')


def create(lr):
    x,y,control,_=harness.banks()
    mapping=GroupMap.target_tokens(y[control],expected=range(23))
    model=harness.model_for('70',201)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=0,fused=True,foreach=False)
    return model,opt,GovernedTraining(model,NanoGPTAdapter(model),mapping=mapping)


def run(case):
    with patch.multiple(harness,HERE=HERE,OUT=OUT,CONFIG=CONFIG),patch.object(harness,'create',create):
        harness.run(case)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=('safe','near'));run(p.parse_args().case)
