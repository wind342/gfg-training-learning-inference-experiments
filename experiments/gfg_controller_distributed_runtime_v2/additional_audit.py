"""Supplementary raw-array checks and runtime import-boundary inspection."""
import ast
from pathlib import Path
import numpy as np
from .evidence import OUT,HERE,read,write,sha


def main():
    coordinates=[];clipping=[];collectives=[];imports=[]
    trainer=Path('D:/codex/dependencies/nanoGPT-3adf61e/model.py')
    authority=HERE.parent/'gfg_nanogpt_functional_online_controller_v1'/'EXPERIMENT_CONTEXT.json'
    trainer_digest=sha(trainer)
    assert trainer_digest==read(authority)['trainer_model_sha256']
    for path in sorted((OUT/'synthetic').glob('*/RESULT.json')):
        result=read(path);assert sha(path.parent/'RAW.npz')==result['raw_sha256']
        with np.load(path.parent/'RAW.npz') as raw:
            mask=np.ones(raw['z'].shape,dtype=bool);mask[np.arange(len(raw['y'])),raw['y']]=False
            for case in result['cases']:
                x=raw[f"{case['mode']}-{case['dedup']}_coordinates"];y=raw['ref_coordinates']
                coordinates.append({'h':result['h'],'seed':result['seed'],'mode':case['mode'],'dedup':case['dedup'],
                    'all_coordinate_mismatches':int(np.count_nonzero(x!=y)),
                    'competitor_coordinate_mismatches':int(np.count_nonzero(x[:,mask]!=y[:,mask])),
                    'max_abs_all':float(np.max(np.abs(x-y))),
                    'max_abs_competitors':float(np.max(np.abs(x[:,mask]-y[:,mask])))})
    for path in sorted((OUT/'gates').glob('*/RESULT.json')):
        result=read(path);assert sha(path.parent/'RAW.npz')==result['raw_sha256']
        with np.load(path.parent/'RAW.npz') as raw:
            for row in result['rows']:
                t=row['step']
                for side,prefix in [('actual',''),('reference','ref_')]:
                    norm=np.float32(raw[f'{t}_{side}_norm'])
                    coefficient=np.minimum(np.float32(1)/(norm+np.float32(1e-6)),np.float32(1))
                    exact=True;maximum=0.
                    for i in range(result['canonical_count']):
                        before=raw[f'{t}_{side}_raw_gradient_{i}'];after=raw[f'{t}_{prefix}gradients_{i}']
                        predicted=before*coefficient
                        exact=exact and bool(np.array_equal(predicted,after))
                        maximum=max(maximum,float(np.max(np.abs(predicted.astype(float)-after.astype(float)))))
                    clipping.append({'case':result['label'],'step':t,'side':side,
                        'independent_fp32_coefficient':float(coefficient),'clipped_gradient_exact':exact,'max_abs':maximum})
    local_norms=[]
    for path in sorted((OUT/'collectives').glob('*/RESULT.json')):
        result=read(path);assert sha(path.parent/'RAW.npz')==result['raw_sha256']
        with np.load(path.parent/'RAW.npz') as raw:
            n=len(result['canonical']);exact=all(np.array_equal(raw[f'packed_{j}'],raw[f'reference_{j}']) for j in range(n))
            assert exact==result['packed_exact'];local_norms.append(float(raw['local_norm2']))
            collectives.append({'rank':result['rank'],'packed_reopened_exact':exact,'global_norm2':float(raw['global_norm2'])})
    assert all(sum(local_norms)==r['global_norm2'] for r in collectives)
    names=('scientific.py','identity.py','ownership.py','boundary.py','adapters.py','clipping.py','controller.py','distributed_train.py')
    for name in names:
        tree=ast.parse((HERE/name).read_text(encoding='utf-8'));modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):modules.extend(n.name for n in node.names)
            if isinstance(node,ast.ImportFrom):modules.append('.'*node.level+(node.module or ''))
        assert not any('experiments.' in n or n in ('.evidence','.gates','.scale_job') for n in modules)
        imports.append({'source':name,'sha256':sha(HERE/name),'imports':modules})
    write(OUT/'ADDITIONAL_AUDIT.json',{'candidate_coordinates':coordinates,'clipping_coefficients':clipping,
        'external_trainer_sha256':trainer_digest,'external_trainer_matches_prior_authority':True,
        'all_clipped_gradients_recomputed_exact':all(r['clipped_gradient_exact'] for r in clipping),
        'cpu_collectives':collectives,'local_norm2_sum':sum(local_norms),
        'runtime_imports':imports,'runtime_has_no_old_evidence_imports':True,
        'scope':'FP32 coefficient and multiplication checked independently in NumPy; real CUDA multi-rank DTensor norm remains NOT_TESTED.'})


if __name__=='__main__':main()
