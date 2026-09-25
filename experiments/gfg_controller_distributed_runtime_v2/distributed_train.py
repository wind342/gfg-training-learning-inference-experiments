"""Portable torchrun entry. Caller factory supplies model, adapter and training data.

torchrun --standalone --nproc-per-node=2 -m experiments.gfg_controller_distributed_runtime_v2.distributed_train --factory module:function
Factory must return model (CPU), adapter constructed BEFORE wrapping, registry,
optimizer_factory, batches (rank-local slice of fixed global batches), seeds,
learning_rates, identical ordered probes/targets and optional callback.
"""
import argparse
import importlib
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from .controller import train


def main():
    p=argparse.ArgumentParser();p.add_argument('--factory',required=True);p.add_argument('--backend',default='exact',choices=('exact','scalarized','streaming'));a=p.parse_args()
    local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if not torch.cuda.is_available():raise RuntimeError('CUDA FSDP launcher requires a CUDA device')
    torch.cuda.set_device(local)
    if world>torch.cuda.device_count() and os.environ.get('LOCAL_WORLD_SIZE',str(world))==str(world):raise RuntimeError('not enough local CUDA devices')
    backend='nccl' if dist.is_nccl_available() else 'gloo'
    if world>1 and backend!='nccl':raise RuntimeError('real multi-GPU launcher requires NCCL; use CPU collective tests separately')
    dist.init_process_group(backend,timeout=timedelta(seconds=120))
    try:
        module,name=a.factory.split(':');factory=getattr(importlib.import_module(module),name)
        cfg=factory(rank=dist.get_rank(),world_size=world,device=torch.device('cuda',local))
        model=cfg.pop('model').cuda();adapter=cfg.pop('adapter');registry=cfg.pop('registry')
        units=adapter.wrap_fsdp(model,init_device_mesh('cuda',(world,)))
        registry.resolve(model);opt=cfg.pop('optimizer_factory')(registry.resolve(model).values())
        train(model,opt,adapter,units=units,registry=registry,group=dist.group.WORLD,backend=a.backend,**cfg)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
