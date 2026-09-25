"""Execution plane: canonical local storage and stage-packed FSDP2 deltas."""
from contextlib import contextmanager
import math
import torch
import torch.distributed as dist


def row_interval(rows,rank,world):
    chunk=(rows+world-1)//world
    return min(rank*chunk,rows),min((rank+1)*chunk,rows)


def local_tensor(p):
    return p.to_local() if hasattr(p,'to_local') else p


class Proposal:
    def __init__(self,parameters):
        self.candidate={n:torch.empty_like(local_tensor(p)) for n,p in parameters.items()}
        self.delta={n:torch.empty_like(local_tensor(p)) for n,p in parameters.items()}

    @torch.no_grad()
    def form(self,parameters,optimizer):
        for n,p in parameters.items():self.candidate[n].copy_(local_tensor(p))
        optimizer.step()
        for n,p in parameters.items():
            p=local_tensor(p);c=self.candidate[n];self.delta[n].copy_(p).sub_(c)
            pp=p.view(-1);cc=c.view(-1)
            for i in range(0,p.numel(),65536):
                tmp=pp[i:i+65536].clone();pp[i:i+65536].copy_(cc[i:i+65536]);cc[i:i+65536].copy_(tmp)

    @torch.no_grad()
    def commit(self,parameters,alpha):
        for n,p in parameters.items():
            p=local_tensor(p)
            if alpha==1.:p.copy_(self.candidate[n])
            else:self.delta[n].mul_(alpha).add_(p);p.copy_(self.delta[n])


class StagePackedDelta:
    def __init__(self,registry,group=None):self.registry=registry;self.group=group;self.events=[]

    def gather(self,names,deltas):
        world=dist.get_world_size(self.group);rank=dist.get_rank(self.group)
        sizes=[]
        for r in range(world):
            sizes.append([math.prod(self.registry.by_name[n].shape[1:])*(row_interval(self.registry.by_name[n].shape[0],r,world)[1]-row_interval(self.registry.by_name[n].shape[0],r,world)[0]) for n in names])
        totals=[sum(s) for s in sizes];length=max(totals);device=deltas[names[0]].device;dtype=deltas[names[0]].dtype
        packed=torch.zeros(length,device=device,dtype=dtype);offset=0
        for n,size in zip(names,sizes[rank]):
            assert deltas[n].numel()==size
            packed[offset:offset+size].copy_(deltas[n].reshape(-1));offset+=size
        gathered=torch.empty(world*length,device=device,dtype=dtype)
        if world==1:gathered.copy_(packed)
        else:dist.all_gather_into_tensor(gathered,packed,group=self.group)
        # One data collective; layouts/sizes are derived from canonical global shapes.
        rows=gathered.view(world,length);output={};offsets=[0]*world
        full=torch.empty(sum(math.prod(self.registry.by_name[n].shape) for n in names),device=device,dtype=dtype);global_offset=0
        for j,n in enumerate(names):
            chunks=[rows[r,offsets[r]:offsets[r]+sizes[r][j]] for r in range(world)]
            count=math.prod(self.registry.by_name[n].shape);view=full[global_offset:global_offset+count];at=0
            for chunk in chunks:view[at:at+len(chunk)].copy_(chunk);at+=len(chunk)
            output[n]=view.view(self.registry.by_name[n].shape);global_offset+=count
            offsets=[offsets[r]+sizes[r][j] for r in range(world)]
        self.events.append({'names':list(names),'world_size':world,'data_collectives':0 if world==1 else 1,'local_packed_bytes':packed.numel()*packed.element_size(),
            'gather_output_bytes':gathered.numel()*gathered.element_size(),'unpacked_bytes':sum(v.numel()*v.element_size() for v in output.values()),
            'padding_elements_per_rank':[length-v for v in totals],'unpack_views_share_storage':True,'wire_bytes':'NOT_MEASURED'})
        return output


class Ownership:
    def __init__(self,model,registry,*,units=None,group=None):
        self.model=model;self.registry=registry;self.units=units;self.group=group
        self.proposal=Proposal(registry.resolve(model));self.events=[];self.live=False
        self.packer=StagePackedDelta(registry,group) if units is not None else None

    def parameters(self):return self.registry.resolve(self.model)

    def propose(self,opt):self.proposal.form(self.parameters(),opt)
    def commit(self,a):self.proposal.commit(self.parameters(),a)

    @contextmanager
    def stage(self,label,names):
        if self.live:raise RuntimeError('overlapping stage materializations')
        self.live=True;unit=self.units[label] if self.units is not None else None
        p={};d={};footprint=0
        try:
            if unit is not None:
                h=unit.unshard(async_op=False)
                if h is not None:h.wait()
            p={n:self.model.get_parameter(n).detach() for n in names}
            d=self.packer.gather(names,self.proposal.delta) if unit is not None else {n:self.proposal.delta[n] for n in names}
            # Count the entire owning FSDP unit, not just the parameters requested.
            if unit is not None:footprint=sum(v.numel()*v.element_size() for v in unit.parameters() if not hasattr(v,'placements'))
            else:footprint=sum(v.numel()*v.element_size() for v in p.values())
            self.events.append({'action':'enter','stage':label,'full_unit_parameter_bytes':footprint,'delta_bytes':sum(v.numel()*v.element_size() for v in d.values())})
            yield p,d
        finally:
            p.clear();d.clear()
            if unit is not None:unit.reshard()
            self.events.append({'action':'exit','stage':label});self.live=False

    def readout_shard(self,name):
        p=self.parameters()[name];w=local_tensor(p).detach();d=self.proposal.delta[name]
        start=0
        if self.units is not None:
            if len(p.placements)!=1 or not p.placements[0].is_shard(0):raise ValueError('only one-dimensional row sharding is implemented')
            start,_=row_interval(p.shape[0],dist.get_rank(self.group),dist.get_world_size(self.group))
        return start,w,d
