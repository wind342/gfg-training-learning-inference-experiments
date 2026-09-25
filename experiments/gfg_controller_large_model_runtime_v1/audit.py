"""Reopen raw evidence and independently check the architecture bridge."""
import math
import numpy as np
import torch
from . import phase1 as run
from experiments.gfg_nanogpt_functional_online_controller_v1.independent_checker import independently_choose

b=run.b; HERE=run.HERE; OUT=run.OUT


def main():
    seal=b.read(HERE/'PHASE1_SEAL.json')
    for name,digest in seal['sources'].items():assert b.sha(HERE/name)==digest,name
    for name,digest in seal['baseline_python'].items():assert b.sha(run.REPO/name)==digest,name
    assert not run.git('diff',run.BASE,'--name-only','--diff-filter=CDMRTUXB'),'OLD_FILES_CHANGED'
    assert run.git('rev-parse',run.prior.TAG)==seal['frozen_tag_object']
    assert run.git('rev-parse',run.prior.TAG+'^{}')==seal['frozen_tag_commit']==run.prior.FROZEN
    assert b.sha(b.old.HERE/'RUN_ASSIGNMENT.json')==seal['assignment_sha256']
    rows={r['entry_id']:r for r in b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs']}
    result=b.read(HERE/'PHASE1_RESULTS.json'); records={}; manifest={};updates=0;decisions=0
    failures=[];maximum=np.zeros(3);crossings=[]
    synthetic=b.read(HERE/'SYNTHETIC_TESTS.json')
    assert len(synthetic)==9 and all(r.get('exact',r.get('three_steps_exact')) is True for r in synthetic)
    for path in sorted(OUT.rglob('TRAJECTORY.json')):
        mode,entry,ci,_=path.relative_to(OUT).parts;ci=int(ci)
        assert mode in ('audit','runtime') and entry in rows and ci in (0,1,2)
        key=(mode,entry,ci);assert key not in records
        r=b.read(path);records[key]=r
        assert (r['mode'],r['entry'],r['config'])==key
        manifest[str(path.relative_to(OUT))]=b.sha(path)
        for name,digest in r['artifacts'].items():
            assert b.sha(path.parent/name)==digest
            manifest[str((path.parent/name).relative_to(OUT))]=digest
        ref=run.prior.record('dynamic','search',entry,ci)
        assert len(r['steps'])==len(ref['steps']) and r['final_state_sha256']==ref['final_state_sha256']
        assert r['control_buffer_bytes']==8*r['parameter_count']
        with np.load(path.parent/'OUTPUTS.npz',allow_pickle=False) as f:logits=f['logits'];targets=f['targets']
        assert logits.dtype==np.float32 and logits.shape[0]==len(r['steps'])+1
        original=run.prior.location('dynamic','search',entry,ci).parent
        with np.load(original/'OUTPUTS.npz',allow_pickle=False) as f:
            assert np.array_equal(targets,f['target']) and np.array_equal(logits,f['logits'])
        comp=logits.copy();comp[:,np.arange(len(targets)),targets]=-np.inf
        gaps=logits[:,np.arange(len(targets)),targets].astype(np.float64)-comp.max(2).astype(np.float64)
        assert np.isfinite(gaps).all();acc=(gaps>0).mean(1)
        assert r['initial_logits_sha256']==b.tensor_sha256(logits[0])==ref['initial_logits_sha256']
        assert (len(r['steps'])==0 if acc[0]>=.99 else (np.all(acc[:-1]<.99) and (acc[-1]>=.99 or len(r['steps'])==200)))
        hits={str(th):int(np.flatnonzero(acc>=th)[0]) if np.any(acc>=th) else None for th in (.95,.99)}
        assert all(hits[str(th)]==run.prior.hit(ref,th) for th in (.95,.99))
        crossings.append({'mode':mode,'entry':entry,'config':ci,'first_hits':hits})
        cp=path.parent/'COORDINATES.npz';hp=path.parent/'HIDDEN.npz'
        coords=np.load(cp,allow_pickle=False) if cp.exists() else None
        hidden=np.load(hp,allow_pickle=False) if hp.exists() else None
        previous=None;alpha=1.;coordkeys=set();hiddenkeys=set()
        try:
            for i,(s,old) in enumerate(zip(r['steps'],ref['steps']),1):
                assert s['t']==i and s['seed']==b.old.seed_for(rows[entry],i-1,seal['source_rng_protocol_sha256'])
                assert s['lr']==run.prior.RATES[ci]*(.1+.9*.5*(1+math.cos(math.pi*(i-1)/199)))
                assert s['accuracy']==float(acc[i]) and s['logits_sha256']==b.tensor_sha256(logits[i])
                for field in ('t','seed','lr','details','accuracy','logits_sha256','post_parameters_sha256'):assert s[field]==old[field]
                if mode=='audit':
                    assert isinstance(s['native_proposal_sha256'],str) and len(s['native_proposal_sha256'])==64
                    if (i-1)%5==0:
                        decisions+=1
                        c={k:coords[f'{i}_{k}'] for k in run.KEYS};coordkeys.update(f'{i}_{k}' for k in run.KEYS)
                        assert s['coordinates_sha256']==old['coordinates_sha256']==b.identity({k:b.tensor_sha256(torch.from_numpy(c[k])) for k in run.KEYS})
                        alpha,scores=independently_choose(c)
                        assert scores==s['details']['scores']
                        hs=hidden[f'{i}_hidden'];pre=hidden[f'{i}_pre_logits']
                        hiddenkeys.update((f'{i}_hidden',f'{i}_pre_logits'))
                        assert hs.shape[:2]==(3,len(targets)) and pre.shape==logits[i-1].shape
                        assert np.isfinite(hs).all() and np.isfinite(pre).all()
                        block=s['blockwise'];assert len(block['exact'])==len(block['max_abs'])==3
                        assert all(math.isfinite(e) and e>=0 for e in block['max_abs'])
                        assert all(ok==(e==0) for ok,e in zip(block['exact'],block['max_abs']))
                        maximum=np.maximum(maximum,block['max_abs'])
                        if not all(block['exact']):failures.append({'entry':entry,'config':ci,'t':i,**block})
                    else:assert s['coordinates_sha256'] is None and s['blockwise'] is None and 'scores' not in s['details']
                    assert alpha==s['details']['alpha']
                else:assert s['native_proposal_sha256'] is None and s['coordinates_sha256'] is None and s['blockwise'] is None
                fact={k:v for k,v in s.items() if k!='sha256'}
                assert fact['previous_sha256']==previous;previous=b.identity(fact);assert previous==s['sha256']
            if coords is not None:assert set(coords.files)==coordkeys and set(hidden.files)==hiddenkeys
            if mode=='runtime':assert coords is None and hidden is None
        finally:
            if coords is not None:coords.close()
            if hidden is not None:hidden.close()
        updates+=len(r['steps'])
    expected={(mode,entry,ci) for mode in ('audit','runtime') for entry in rows for ci in (0,1,2)}
    assert set(records)==expected and len(records)==72
    assert len(result['records'])==72
    for item in result['records']:
        r=records[(item['mode'],item['entry'],item['config'])]
        assert item['exact'] and item['updates']==len(r['steps'])
        assert item['blockwise_nonexact_steps']==[s['t'] for s in r['steps'] if s['blockwise'] is not None and not all(s['blockwise']['exact'])]
    assert result['runtime_equivalence']=='PASS' and result['persistent_buffer_reduction']==.6
    assert result['blockwise_component']==('NOT_CERTIFIED' if failures else 'PASS')
    assert result['phase2']==result['phase3']=='NOT_STARTED'
    report={'status':'PASS','runtime_equivalence':'PASS','trajectories':72,'new_controller_updates':updates,
        'paired_oracle_updates':sum(len(r['steps']) for k,r in records.items() if k[0]=='audit'),
        'audited_decisions':decisions,'blockwise_component':result['blockwise_component'],
        'blockwise_max_abs':maximum.tolist(),'blockwise_failures':failures,'crossings':crossings,
        'seal_sha256':b.sha(HERE/'PHASE1_SEAL.json'),'results_sha256':b.sha(HERE/'PHASE1_RESULTS.json'),
        'synthetic_sha256':b.sha(HERE/'SYNTHETIC_TESTS.json'),'raw_manifest':manifest,
        'scope':'Offline checks reopen logits/coordinates. Native paired runner checks parameter, proposal, hidden and optimizer identity; hashes are receipts, not reconstructed states.'}
    b.write(HERE/'AUDIT.json',report)
    print({k:v for k,v in report.items() if k not in ('raw_manifest','crossings','blockwise_failures')})


if __name__=='__main__':main()
