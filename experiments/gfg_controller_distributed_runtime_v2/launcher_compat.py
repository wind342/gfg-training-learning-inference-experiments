"""Optional Windows FileStore rendezvous; does not patch installed PyTorch.

The ordinary portable NCCL torchrun command remains the multi-GPU entry.
This workaround changes rendezvous only, not training data or arithmetic.
"""
import os
import subprocess
import sys
import traceback
from .evidence import OUT,REPO,write,sha


def main():
    env=dict(os.environ,USE_LIBUV='0',TORCH_DISABLE_SHARE_RDZV_TCP_STORE='1')
    commands=[]
    for kind in ('runtime','oracle'):
        for lr in (.001,.003,.01):
            store=OUT/f'file-store-{kind}-{lr}'
            assert not store.exists(),'Never reuse a previous rendezvous store'
            command=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1',
                '--rdzv_backend=c10d',f'--rdzv_endpoint={store.as_posix()}',
                f'--rdzv_id=gfg-v2-{kind}-{lr}','--rdzv_conf=store_type=file',
                '--local_addr=127.0.0.1','-m']
            if kind=='runtime':
                command+=['experiments.gfg_controller_distributed_runtime_v2.distributed_train',
                    '--factory','experiments.gfg_controller_distributed_runtime_v2.launch_fixture:factory']
            else:command+=['experiments.gfg_controller_distributed_runtime_v2.distributed_gate','--lr',str(lr)]
            commands.append({'kind':kind,'lr':lr,'command':command})
    write(OUT/'LAUNCHER_FILESTORE_PROTOCOL.json',{'source_sha256':sha(__file__),'commands':commands,
        'environment':{'USE_LIBUV':'0','TORCH_DISABLE_SHARE_RDZV_TCP_STORE':'1'},
        'world_size':1,'timeout_seconds_per_case':180,
        'reason':'Installed c10d/static TCPStore constructors omit use_libuv; explicit rendezvous option was ignored. FileStore plus disabled store sharing avoids those constructors.'})
    calls=[]
    for row in commands:
        env['GFG_V2_LAUNCH_LR']=str(row['lr'])
        try:
            child=subprocess.run(row['command'],cwd=REPO,env=env,capture_output=True,text=True,timeout=180)
            result={**row,'returncode':child.returncode,'stdout':child.stdout,'stderr':child.stderr}
        except Exception as e:
            result={**row,'status':'FAILED','message':str(e),'traceback':traceback.format_exc()}
        calls.append(result);print('FILESTORE',row['kind'],row['lr'],result.get('returncode'),flush=True)
    write(OUT/'LAUNCHER_FILESTORE_RESULTS.json',{'calls':calls,'real_multigpu':'NOT_TESTED'})


if __name__=='__main__':main()
