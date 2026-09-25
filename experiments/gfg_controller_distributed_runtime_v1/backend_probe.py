"""Isolated real-backend probes; CPU Gloo is not CUDA distributed validation."""
import argparse
import traceback
import numpy as np
import torch
import torch.distributed as dist
from .checks import b,HERE,OUT


def worker(rank,path):
    dist.init_process_group('gloo',init_method=(OUT/'gloo-init').as_uri(),rank=rank,world_size=2)
    try:
        with np.load(path,allow_pickle=False) as f:local=f['1_local_minima'];expected=local.min(0)
        value=torch.from_numpy(local[rank].copy());dist.all_reduce(value,op=dist.ReduceOp.MIN)
        result={'rank':rank,'exact':np.array_equal(value.numpy(),expected),'payload_bytes':value.numel()*value.element_size(),
            'backend':'CPU_GLOO','world_size':2,'cuda':False}
        b.write(OUT/'gloo'/f'rank{rank}.json',result)
    finally:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('fsdp','gloo'));a=p.parse_args()
    if a.mode=='gloo':
        torch.multiprocessing.spawn(worker,args=(str(OUT/'gate/w2-lr0.003/RAW.npz'),),nprocs=2,join=True);return
    b.old.setup();stage='init'
    try:
        from torch.distributed.fsdp import fully_shard
        from .storage import FSDPOwnership
        dist.init_process_group('gloo',init_method=(OUT/'fsdp-init').as_uri(),rank=0,world_size=1)
        stage='wrap';model=torch.nn.Sequential(torch.nn.Linear(16,16),torch.nn.Tanh(),torch.nn.Linear(16,4)).cuda()
        fully_shard(model[0]);fully_shard(model[2]);fully_shard(model)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,fused=True,foreach=False)
        owner=FSDPOwnership(model,opt,{'first':model[0]});stage='forward_backward'
        model(torch.randn(8,16,device='cuda')).square().mean().backward()
        before={n:p.clone() for n,p in owner.local.parameters.items()};stage='proposal'
        owner.propose();assert all(torch.equal(p,before[n]) for n,p in owner.local.parameters.items())
        stage='block_materialize'
        with owner.materialize('first',('0.weight','0.bias')) as (params,deltas):
            assert params['0.weight'].shape==(16,16) and deltas['0.weight'].shape==(16,16)
        owner.commit(.5)
        b.write(HERE/'FSDP_PROBE.json',{'status':'PASS_WORLD_SIZE_1_ONLY','backend':'gloo','world_size':1,'stage':stage,
            'local_buffer_bytes':2*owner.local.bytes(),'events':owner.events,'multi_gpu':'NOT_TESTED'})
    except Exception as e:
        b.write(HERE/'FSDP_PROBE.json',{'status':'UNSUPPORTED_OR_FAILED','stage':stage,'type':type(e).__name__,
            'message':str(e),'traceback':traceback.format_exc(),'multi_gpu':'NOT_TESTED'})
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
