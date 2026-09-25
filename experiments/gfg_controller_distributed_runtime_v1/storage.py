"""Local-shard proposal storage; emulation and real FSDP ownership are explicit."""
from contextlib import contextmanager
import torch


def interval(n,r,size):return n*r//size,n*(r+1)//size


class LocalProposal:
    def __init__(self,parameters):
        self.parameters=parameters
        self.candidate={n:torch.empty_like(p) for n,p in parameters.items()}
        self.delta={n:torch.empty_like(p) for n,p in parameters.items()}

    @torch.no_grad()
    def form(self,optimizer):
        for n,p in self.parameters.items():self.candidate[n].copy_(p)
        optimizer.step()
        for n,p in self.parameters.items():
            c=self.candidate[n];d=self.delta[n];d.copy_(p);d.sub_(c)
            # Preserve both receiving and native candidate bytes without rebinding
            # DTensor storage. Only a bounded swap tile is transiently allocated.
            pp=p.reshape(-1);cc=c.reshape(-1)
            for start in range(0,p.numel(),65536):
                end=min(p.numel(),start+65536);tmp=pp[start:end].clone()
                pp[start:end].copy_(cc[start:end]);cc[start:end].copy_(tmp)

    @torch.no_grad()
    def commit(self,alpha):
        for n,p in self.parameters.items():
            if alpha==1.:p.copy_(self.candidate[n])
            else:self.delta[n].mul_(alpha).add_(p);p.copy_(self.delta[n])

    def bytes(self):return sum(p.numel()*p.element_size() for p in self.parameters.values())


class EmulatedOwnership:
    """Virtual ranks share ONE full backward model. Not real FSDP memory savings."""
    def __init__(self,model,world_size,optimizer_factory):
        self.model=model;self.world_size=world_size;self.full=dict(model.named_parameters())
        self.ranks=[];self.optimizers=[];self.events=[];self.live_bytes=0;self.peak_materialized=0
        for r in range(world_size):
            params={}
            for n,p in self.full.items():
                a,z=interval(p.numel(),r,world_size)
                if z>a:params[n]=torch.nn.Parameter(p.detach().view(-1)[a:z])
            owner=LocalProposal(params);self.ranks.append(owner);self.optimizers.append(optimizer_factory(params.values()))

    def set_lr(self,lr):
        for opt in self.optimizers:
            for group in opt.param_groups:group['lr']=lr

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)
        for opt in self.optimizers:opt.zero_grad(set_to_none=True)

    def propose(self):
        for r,owner in enumerate(self.ranks):
            for n,p in owner.parameters.items():
                full=self.full[n];a,z=interval(full.numel(),r,self.world_size)
                p.grad=full.grad.detach().view(-1)[a:z]
            owner.form(self.optimizers[r])

    def commit(self,alpha):
        for owner in self.ranks:owner.commit(alpha)

    def gather(self,name,kind='parameters',start=0,end=None):
        p=self.full[name];end=p.numel() if end is None else end;pieces=[]
        if start==end:return p.new_empty(0)
        for r,owner in enumerate(self.ranks):
            a,z=interval(p.numel(),r,self.world_size);lo=max(a,start);hi=min(z,end)
            if hi>lo:pieces.append(getattr(owner,kind)[name].detach().view(-1)[lo-a:hi-a])
        return torch.cat(pieces)

    @contextmanager
    def materialize(self,label,names):
        if self.live_bytes:raise RuntimeError('overlapping block materializations')
        p={n:self.gather(n).view_as(self.full[n]) for n in names}
        d={n:self.gather(n,'delta').view_as(self.full[n]) for n in names}
        amount=sum(x.numel()*x.element_size() for x in (*p.values(),*d.values()))
        self.live_bytes=amount;self.peak_materialized=max(self.peak_materialized,amount)
        self.events.append({'block':label,'action':'materialize','bytes':amount,'emulated_payload_bytes':amount})
        try:yield p,d
        finally:
            p.clear();d.clear();self.live_bytes=0
            self.events.append({'block':label,'action':'release','bytes':amount})

    def optimizer_state(self,name,key):
        chunks=[]
        for owner,opt in zip(self.ranks,self.optimizers):
            if name in owner.parameters:chunks.append(opt.state[owner.parameters[name]][key])
        return chunks[0] if key=='step' else torch.cat([c.reshape(-1) for c in chunks]).view_as(self.full[name])


class FSDPOwnership:
    """Real FSDP2 binding; requires externally prepared blocks/process mesh.

    This API does not pretend the emulation exercises FSDP hooks or collectives.
    DTensor local shards retain the optimizer's actual storage ownership.
    """
    def __init__(self,model,optimizer,units):
        from torch.distributed.tensor import DTensor
        self.model=model;self.optimizer=optimizer;self.units=units
        self.parameters=dict(model.named_parameters())
        if not all(isinstance(p,DTensor) for p in self.parameters.values()):raise TypeError('requires fully sharded DTensor parameters')
        self.local=LocalProposal({n:p.to_local() for n,p in self.parameters.items()})
        self.events=[];self.live_bytes=0;self.peak_materialized=0

    def propose(self):self.local.form(self.optimizer)
    def commit(self,alpha):self.local.commit(alpha)

    @contextmanager
    def materialize(self,label,names):
        from torch.distributed.tensor import DTensor
        if self.live_bytes:raise RuntimeError('overlapping block materializations')
        unit=self.units[label];handle=unit.unshard(async_op=False)
        if handle is not None:handle.wait()
        p={n:self.model.get_parameter(n).detach() for n in names};d={}
        try:
            for n in names:
                spec=self.parameters[n]
                d[n]=DTensor.from_local(self.local.delta[n],device_mesh=spec.device_mesh,placements=spec.placements,
                    shape=spec.shape,stride=spec.stride(),run_check=False).full_tensor()
            self.live_bytes=sum(x.numel()*x.element_size() for x in (*p.values(),*d.values()))
            self.peak_materialized=max(self.peak_materialized,self.live_bytes)
            self.events.append({'block':label,'action':'materialize','bytes':self.live_bytes,'delta_full_tensor_calls':len(d)})
            yield p,d
        finally:
            p.clear();d.clear();unit.reshard();self.events.append({'block':label,'action':'release','bytes':self.live_bytes});self.live_bytes=0
