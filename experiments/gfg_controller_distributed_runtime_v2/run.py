"""One bounded stage per invocation; all failures are receipts, not suppressed."""
import argparse
import gc
import os
import subprocess
import sys
import traceback
import torch
from .evidence import HERE,OUT,REPO,setup,seal,write,read


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('seal','synthetic','gates','case','collectives','launcher','scale','audit'))
    p.add_argument('--backend',default='exact');p.add_argument('--lr',type=float,default=.003);p.add_argument('--fsdp',action='store_true');a=p.parse_args();setup()
    if a.stage=='seal':print(seal()['hardware']);return
    if a.stage=='synthetic':
        from .gates import synthetic
        write(OUT/'SYNTHETIC.json',{'results':synthetic()});return
    if a.stage=='gates':
        results=[]
        for fsdp,backend in [(False,'exact'),(False,'scalarized'),(False,'streaming'),(True,'exact')]:
            for lr in (.001,.003,.01):
                label=f'{"fsdp1" if fsdp else "single"}-{backend}-{lr}'
                cmd=[sys.executable,'-m','experiments.gfg_controller_distributed_runtime_v2.run','case','--backend',backend,'--lr',str(lr)]+(['--fsdp'] if fsdp else [])
                try:
                    child=subprocess.run(cmd,cwd=REPO,capture_output=True,text=True,timeout=180)
                    write(OUT/'logs'/f'{label}.json',{'command':cmd,'returncode':child.returncode,'stdout':child.stdout,'stderr':child.stderr})
                    result_file=OUT/'gates'/label/'RESULT.json'
                    if result_file.exists() and child.returncode==0:r=read(result_file)
                    else:
                        r={'status':'PROCESS_FAILED','fsdp_world_size':1 if fsdp else None,'backend':backend,'lr':lr,'returncode':child.returncode,'message':child.stderr}
                        write(OUT/'failures'/f'{label}.json',r)
                except Exception as e:
                    r={'status':'FAILED','fsdp_world_size':1 if fsdp else None,'backend':backend,'lr':lr,'message':str(e),'traceback':traceback.format_exc()};write(OUT/'failures'/f'{label}.json',r)
                results.append(r);print('GATE',fsdp,backend,lr,r.get('all_exact'),r.get('message',''),flush=True)
                gc.collect();torch.cuda.empty_cache()
        write(OUT/'GATES.json',{'gates':results});return
    if a.stage=='case':
        from .gates import one_gate
        print(one_gate(a.lr,a.backend,a.fsdp)['all_exact']);return
    if a.stage=='collectives':
        cmd=[sys.executable,'-m','experiments.gfg_controller_distributed_runtime_v2.collective_gate']
    elif a.stage=='launcher':
        env=dict(os.environ,USE_LIBUV='0');calls=[]
        sizes=[1,2] if torch.cuda.device_count()>=2 and torch.distributed.is_nccl_available() else [1]
        for world in sizes:
            for lr in (.001,.003,.01):
                env['GFG_V2_LAUNCH_LR']=str(lr)
                cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}','-m',
                    'experiments.gfg_controller_distributed_runtime_v2.distributed_train','--factory','experiments.gfg_controller_distributed_runtime_v2.launch_fixture:factory']
                try:
                    r=subprocess.run(cmd,cwd=REPO,env=env,capture_output=True,text=True,timeout=150)
                    calls.append({'world':world,'lr':lr,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
                except Exception: calls.append({'world':world,'lr':lr,'traceback':traceback.format_exc()})
        write(OUT/'REAL_MULTIGPU_AUDIT.json',{'real_multigpu':'NOT_TESTED' if 2 not in sizes else 'REQUIRES_RAW_ORACLE_AUDIT','calls':calls});return
    elif a.stage=='scale':cmd=[sys.executable,'-m','experiments.gfg_controller_distributed_runtime_v2.scale_suite']
    else:cmd=[sys.executable,'-m','experiments.gfg_controller_distributed_runtime_v2.audit']
    r=subprocess.run(cmd,cwd=REPO,capture_output=False)
    if r.returncode:write(OUT/'failures'/f'{a.stage}-process.json',{'returncode':r.returncode,'command':cmd})
    raise SystemExit(r.returncode)


if __name__=='__main__':main()
