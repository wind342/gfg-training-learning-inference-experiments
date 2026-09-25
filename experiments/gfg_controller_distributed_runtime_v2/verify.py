"""Read-only replay of the sealed audit and supplementary receipts.

GFG_V2_OUT must name the external raw evidence directory. Never regenerates
original outcomes or overwrites archived results.
"""
import numpy as np
from . import audit
from .evidence import OUT,HERE,read,sha


def main():
    original=read(OUT/'FINAL_AUDIT.json')
    for name,identity in original['artifacts'].items():
        assert sha(OUT/name)==identity,('ARTIFACT_CHANGED',name)
    comparisons=[]
    def check_write(path,value):
        expected=read(path)
        if path.name=='FINAL_AUDIT.json':
            value=dict(value);expected=dict(expected)
            value.pop('artifacts');expected.pop('artifacts')
        assert value==expected,('REPORT_CHANGED',str(path))
        comparisons.append(path.name)
    audit.write=check_write
    audit.main()
    for name in ('SUPPLEMENTARY_PROFILE_SEAL.json','DISTRIBUTED_GATE_CORRECTION_SEAL.json'):
        seal=read(OUT/name);assert sha(HERE/seal['source'])==seal['sha256']
    failed=read(OUT/'DISTRIBUTED_GATE_SUPPLEMENT_SEAL.json')
    assert sha(OUT/'supplemental-failed-source'/failed['source'])==failed['sha256']
    count=0
    for path in list((OUT/'distributed-oracle-gate').glob('*/*/RESULT.json'))+list((OUT/'distributed-oracle-gate-v2').glob('*/*/RESULT.json')):
        result=read(path);assert sha(path.parent/'RAW.npz')==result['raw_sha256']
        with np.load(path.parent/'RAW.npz') as raw:
            for step in result['records']:
                t=step['step']
                for key,check in step['checks'].items():
                    left=raw[f'{t}/actual/{key}'];right=raw[f'{t}/reference/{key}']
                    assert bool(np.array_equal(left,right))==check['byte_equal']
                    assert float(np.max(np.abs(left.astype(float)-right.astype(float))))==check['max_abs']
                if f'{t}/actual/minima' in raw:
                    alpha,scores=audit.choose(raw[f'{t}/actual/minima'])
                    assert alpha==step['actual']['alpha'] and scores==step['actual']['scores']
        count+=1
    print('READ_ONLY_REPLAY_PASS',len(comparisons),'reports;',count,'portable oracle cases')


if __name__=='__main__':main()
