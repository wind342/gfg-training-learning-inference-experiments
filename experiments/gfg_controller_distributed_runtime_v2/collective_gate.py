"""Real CPU two-process collectives, separate from CUDA/FSDP certification."""
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from .evidence import OUT,write,sha
from .identity import CanonicalParameterRegistry
from .ownership import StagePackedDelta,row_interval
from .boundary import calculate,same_probes


def worker(rank):
    dist.init_process_group('gloo',init_method=(OUT/'collective-init').as_uri(),world_size=2,rank=rank,timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(310);model=torch.nn.Linear(5,7,bias=True);model.alias=model.weight
        registry=CanonicalParameterRegistry(model);names=tuple(registry.resolve(model));full={n:torch.arange(p.numel()).float().view_as(p)/128 for n,p in registry.resolve(model).items()}
        local={}
        for n,x in full.items():
            a,z=row_interval(x.shape[0],rank,2);local[n]=x[a:z].clone()
        packer=StagePackedDelta(registry,dist.group.WORLD);gathered=packer.gather(names,local)
        packed_exact=all(torch.equal(gathered[n],full[n]) for n in names)
        norm2=sum((x.double().square().sum() for x in local.values()));local_norm2=float(norm2);dist.all_reduce(norm2)
        expected=sum(x.double().square().sum() for x in full.values());norm_exact=torch.equal(norm2,expected)
        n,v,h=5,7,5;w=torch.randn(v,h);dw=torch.randn_like(w)*.001;hs=[torch.randn(n,h)*s for s in (1,.01,.001)]
        y=torch.tensor([0,6,2,6,0]);z=torch.randn(n,v);start,end=row_interval(v,rank,2)
        same_probes(torch.arange(n)[:,None],y,dist.group.WORLD)
        ref=calculate(z,y,w,dw,hs,audit=True)
        got=calculate(z[:,start:end],y,w[start:end],dw[start:end],hs,start=start,group=dist.group.WORLD,audit=True)
        path=OUT/'collectives'/f'rank{rank}';path.mkdir(parents=True)
        raw={'minima':got[2].numpy(),'reference_minima':ref[2].numpy(),'local_norm2':np.array(local_norm2),'global_norm2':norm2.numpy(),'reference_norm2':expected.numpy()}
        for j,nm in enumerate(names):raw[f'packed_{j}']=gathered[nm].numpy();raw[f'reference_{j}']=full[nm].numpy()
        with (path/'RAW.npz').open('xb') as f:np.savez_compressed(f,**raw)
        write(path/'RESULT.json',{'rank':rank,'backend':'CPU_GLOO','world':2,'packed_exact':packed_exact,'norm_exact':norm_exact,
            'minima_exact':torch.equal(got[2],ref[2]),'score_exact':got[1]==ref[1],'alpha_exact':got[0]==ref[0],
            'canonical':registry.receipt(model),'communication':got[4]['communication'],'packed_events':packer.events,'raw_sha256':sha(path/'RAW.npz')})
    finally:dist.destroy_process_group()


if __name__=='__main__':torch.multiprocessing.spawn(worker,nprocs=2,join=True)
