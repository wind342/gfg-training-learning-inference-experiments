"""Check sealed records again and bind final documentation/source identities."""
from pathlib import Path
from unittest.mock import patch
from . import audit
from . import validate as run

b=run.b;HERE=run.HERE


def main():
    original=b.write
    def verify_existing(path,data):assert b.read(Path(path))==data,('RECHECK_CHANGED',str(path))
    with patch.object(b,'write',verify_existing):audit.main()
    guard=b.read(HERE/'TRANSFER_GUARD.json');seal=b.read(HERE/'TRANSFER_GUARD_SEAL.json')
    assert guard['status']=='PASS' and guard['seal_sha256']==b.sha(HERE/'TRANSFER_GUARD_SEAL.json')
    assert seal['source_sha256']==b.sha(HERE/'transfer_guard.py')
    assert seal['runtime_sha256']==b.sha(HERE/'runtime.py') and seal['entry_sha256']==b.sha(HERE/'entry.py')
    assert seal['audit_sha256']==b.sha(HERE/'AUDIT.json')
    assert guard['model_hash_snapshot_helpers_called'] is False
    assert len(guard['explicit_tensor_cpu_calls'])==10 and guard['updates']==49
    assert all(r=={'shape':[9],'bytes':72} for r in guard['explicit_tensor_cpu_calls'])
    assert guard['boundary_bytes']==720
    a=b.read(HERE/'AUDIT.json');cpu=a['timing_medians']['cpu-32-8']['checked_wall_seconds'];gpu=a['timing_medians']['gpu-256-128']['checked_wall_seconds']
    assert a['all_historical_trajectories_exact'] and a['coordinate_nonexact_decisions']==0
    assert a['score_nonexact_decisions']==a['same_state_alpha_differences']==0
    assert not run.old.git('diff',run.BASE,'--name-only','--diff-filter=CDMRTUXB')
    original(HERE/'FINAL_AUDIT.json',{'status':'PASS','old_files_unchanged':True,
        'frozen_commit':run.old.git('rev-parse',run.old.prior.TAG+'^{}'),
        'observed_byte_equivalence':'PASS','standalone_entry':'PASS','transfer_guard':'PASS',
        'single_case_wall_reduction_fraction':1-gpu/cpu,'scale_training':'NOT_STARTED',
        'manifest':{p.name:b.sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='FINAL_AUDIT.json'}})
    print('FINAL GPU RUNTIME AUDIT PASS',1-gpu/cpu)


if __name__=='__main__':main()
