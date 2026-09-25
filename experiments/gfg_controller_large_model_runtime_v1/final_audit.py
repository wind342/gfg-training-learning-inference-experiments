"""Recheck both sealed bridges and memory diagnostics without rewriting results."""
from pathlib import Path
from unittest.mock import patch
import statistics
from . import phase1 as run
from . import audit
from . import blockwise_validation as block

b=run.b;HERE=run.HERE


def main():
    original_write=b.write
    def check_existing(path,value):assert b.read(Path(path))==value,('AUDIT_CHANGED',str(path))
    with patch.object(b,'write',check_existing):
        audit.main()
        with patch.object(audit,'HERE',block.DEST),patch.object(audit,'OUT',block.OUT):audit.main()
    first=b.read(HERE/'AUDIT.json');second=b.read(block.DEST/'AUDIT.json')
    seal=b.read(block.DEST/'PHASE1_SEAL.json')
    assert seal['prerequisite_audit_sha256']==b.sha(HERE/'AUDIT.json')
    assert seal['prerequisite_seal_sha256']==b.sha(HERE/'PHASE1_SEAL.json')
    mem=b.read(HERE/'PHASE1_MEMORY.json');ms=b.read(HERE/'MEMORY_SEAL.json')
    assert mem['memory_seal_sha256']==b.sha(HERE/'MEMORY_SEAL.json')
    assert ms['source_sha256']==b.sha(HERE/'memory.py')
    assert ms['blockwise_audit_sha256']==b.sha(block.DEST/'AUDIT.json')
    records=mem['records'];assert len(records)==12
    assert {(r['mode'],r['rep']) for r in records}=={(m,j) for m in ms['modes'] for j in range(3)}
    for r in records:
        ref=run.prior.record('fixed1' if r['mode']=='native' else 'dynamic','search',ms['entry'],1)['steps'][0]
        assert r['post_parameters_sha256']==ref['post_parameters_sha256']
        assert r['alpha']==ref['details']['alpha']
        assert r['peak_allocated']>=r['after_constructor_allocated']>=r['baseline_allocated']>=0
        assert r['peak_reserved']>=r['peak_allocated'] and r['update_seconds_diagnostic']>0
        exact_buffers=r['after_constructor_allocated']-r['baseline_allocated']
        expected=r['parameter_count']*(0 if r['mode']=='native' else 20 if r['mode']=='old' else 8)
        # CUDA allocator alignment is recorded separately; cannot under-allocate.
        assert exact_buffers>=expected
    for mode,stats in mem['median_by_mode'].items():
        for k,v in stats.items():assert v==statistics.median(r[k] for r in records if r['mode']==mode)
    for p,values in mem['lower_bound_estimates_not_measurements'].items():
        p=int(p)
        assert values=={'old_fp32_state_plus_buffers_bytes':36*p,'new_fp32_state_plus_buffers_bytes':24*p,'control_buffers_bytes':8*p}
    assert mem['buffer_formula']=={'old_bytes_per_parameter':20,'new_bytes_per_parameter':8,'reduction_fraction':.6}
    assert first['status']==second['status']=='PASS'
    assert first['blockwise_component']==second['blockwise_component']=='PASS'
    assert not run.git('diff',run.BASE,'--name-only','--diff-filter=CDMRTUXB')
    sources={str(p.relative_to(HERE)):b.sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='FINAL_AUDIT.json'}
    original_write(HERE/'FINAL_AUDIT.json',{'status':'PASS','phase1_runtime_equivalence':'PASS',
        'phase1_blockwise_integration':'PASS','phase2':'NOT_STARTED','phase3':'NOT_STARTED',
        'new_controller_updates':first['new_controller_updates']+second['new_controller_updates'],
        'paired_oracle_updates':first['paired_oracle_updates']+second['paired_oracle_updates'],
        'new_trajectories':first['trajectories']+second['trajectories'],
        'frozen_tag_commit':run.git('rev-parse',run.prior.TAG+'^{}'),
        'old_tracked_files_unchanged':True,'sources_and_reports':sources,
        'scope':'Architecture gate only. No scaled-model or net-speed advantage conclusion.'})
    print('FINAL ARCHITECTURE AUDIT PASS')


if __name__=='__main__':main()
