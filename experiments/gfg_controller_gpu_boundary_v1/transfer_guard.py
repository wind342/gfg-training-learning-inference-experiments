"""Instrument actual standalone execution; forbid model hash/snapshot helpers."""
from unittest.mock import patch
import torch
from . import validate as run
from .entry import train
from experiments.gfg_controller_large_model_runtime_v1.runtime import ProposalStorage

b=run.b;HERE=run.HERE


def main():
    b.old.setup();assert b.read(HERE/'AUDIT.json')['status']=='PASS'
    row=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs'][0]
    rng=b.read(run.old.HERE/'PHASE1_SEAL.json')['source_rng_protocol_sha256']
    b.write(HERE/'TRANSFER_GUARD_SEAL.json',{'source_sha256':b.sha(HERE/'transfer_guard.py'),
        'runtime_sha256':b.sha(HERE/'runtime.py'),'entry_sha256':b.sha(HERE/'entry.py'),
        'audit_sha256':b.sha(HERE/'AUDIT.json'),'entry':row['entry_id'],'config':1,
        'rule':'forbid snapshot_model/params/phash; every explicit Tensor.cpu transfer <=9 elements'})
    loaded=b.load_row(row)
    try:
        cfg=run.config(row,loaded,1,rng);transfers=[];original=torch.Tensor.cpu
        def guarded_cpu(t,*args,**kwargs):
            if t.numel()>9:raise AssertionError('bulk CPU transfer in standalone runtime')
            transfers.append({'shape':list(t.shape),'bytes':t.numel()*t.element_size()})
            return original(t,*args,**kwargs)
        with patch.object(torch.Tensor,'cpu',guarded_cpu),patch.object(ProposalStorage,'snapshot_model',side_effect=AssertionError('snapshot called')),\
             patch.object(b,'phash',side_effect=AssertionError('hash called')),patch.object(b,'params',side_effect=AssertionError('params copy called')):
            result=train(**cfg,backend='gpu')
        ref=run.old.prior.record('dynamic','search',row['entry_id'],1)
        assert len(result['steps'])==len(ref['steps'])
        assert all(s['alpha']==r['details']['alpha'] and s['accuracy']==r['accuracy'] for s,r in zip(result['steps'],ref['steps']))
        expected=sum((s['t']-1)%5==0 for s in result['steps'])
        assert len(transfers)==expected and all(t=={'shape':[9],'bytes':72} for t in transfers)
        b.write(HERE/'TRANSFER_GUARD.json',{'status':'PASS','updates':len(result['steps']),
            'explicit_tensor_cpu_calls':transfers,'boundary_bytes':sum(t['bytes'] for t in transfers),
            'model_hash_snapshot_helpers_called':False,'seal_sha256':b.sha(HERE/'TRANSFER_GUARD_SEAL.json'),
            'scope':'Tensor.cpu calls and named hash/snapshot helpers guarded; scalar loss and accuracy synchronization still occurs.'})
        print('TRANSFER GUARD PASS',len(result['steps']),sum(t['bytes'] for t in transfers))
    finally:loaded[0].close()


if __name__=='__main__':main()
