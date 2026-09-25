"""Final source-boundary and completed-audit receipt; does not run training."""
import subprocess
from . import benchmark as b


def main():
    b.sealcheck()
    result=b.read(b.HERE/'RESULTS.json')
    b.require(result['status']=='PASS' and result['trajectories_checked']==420,'FINAL_AUDIT')
    b.require(b.read(b.HERE/'SOURCE_BRIDGE_CHECK.json')['status']=='PASS','SOURCE_BRIDGE')
    actual=subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release^{}'],cwd=b.REPO,text=True).strip()
    b.require(actual=='36dab5ce347dbbdac157ef23205f556606d18294','FROZEN_TAG_CHANGED')
    changed=subprocess.check_output(['git','diff','--name-only','0dc78f0c509ed9ce09b09b585151e41ecd9a6f71'],cwd=b.REPO,text=True).splitlines()
    b.require(all(p.startswith('experiments/gfg_optimizer_capability_benchmark_v1/') for p in changed),'EXISTING_FILE_CHANGED')
    report={'status':'PASS','utc':b.old.utc(),'frozen_tag_commit':actual,
        'existing_files_changed':False,'new_experiment_files':changed,
        'audit_trajectories':420,'audit_recorded_native_steps':result['native_steps_in_ledgers'],
        'artifact_bytes':result['total_artifact_bytes'],
        'files_sha256':{p.name:b.sha(p) for p in b.HERE.iterdir() if p.is_file() and p.name!='FINAL_CHECK.json'}}
    b.write(b.HERE/'FINAL_CHECK.json',report)
    print('FINAL CHECK PASS: frozen identities unchanged, original files untouched, all selected trajectories replayed three times.')


if __name__=='__main__':main()
