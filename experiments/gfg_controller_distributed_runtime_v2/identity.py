"""Logical identity is established BEFORE wrapping, never from a current name alone."""
from dataclasses import dataclass
import hashlib
import torch


@dataclass(frozen=True)
class ParameterIdentity:
    canonical_id:str
    primary:str
    aliases:tuple
    shape:tuple


class CanonicalParameterRegistry:
    def __init__(self,model):
        grouped={};storages={}
        for name,p in model.named_parameters(remove_duplicate=False):
            if id(p) not in grouped:
                if p.device.type!='meta' and p.numel():
                    storage=p.untyped_storage().data_ptr()
                    if storage in storages and storages[storage]!=id(p):
                        raise ValueError('distinct Parameter objects sharing storage are unsupported; use a real tied Parameter')
                    storages[storage]=id(p)
                grouped[id(p)]=[p,[]]
            grouped[id(p)][1].append(name)
        self.entries=[]
        for p,aliases in grouped.values():
            cid=hashlib.sha256(('canonical-parameter-v1:'+','.join(sorted(aliases))).encode()).hexdigest()
            self.entries.append(ParameterIdentity(cid,aliases[0],tuple(aliases),tuple(p.shape)))
        self.by_name={name:e for e in self.entries for name in e.aliases}

    def resolve(self,model):
        result={}
        for e in self.entries:
            params=[model.get_parameter(n) for n in e.aliases]
            if any(p is not params[0] for p in params):raise RuntimeError(f'broken tied identity: {e.aliases}')
            if tuple(params[0].shape)!=e.shape:raise RuntimeError('logical shape changed')
            result[e.primary]=params[0]
        if len({id(p) for p in result.values()})!=len(result):raise RuntimeError('canonical collision')
        return result

    def receipt(self,model):
        self.resolve(model)
        return [{'canonical_id':e.canonical_id,'primary':e.primary,'aliases':list(e.aliases),'shape':list(e.shape)} for e in self.entries]
