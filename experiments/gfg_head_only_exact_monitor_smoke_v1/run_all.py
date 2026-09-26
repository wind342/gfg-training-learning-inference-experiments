import subprocess
import sys
from .core import *

def main():
    pre=json.loads((ROOT/'PREFLIGHT.json').read_text());assert pre['status']=='PASS' and sources()==pre['source_hashes']
    out=ROOT/'execution';out.mkdir(exist_ok=False);results=[]
    for suffix,name in [(['run',a],a) for a in ARMS]+[(['check'],'check')]:
        module='experiments.'+HERE.name+'.'+suffix[0]
        command=[sys.executable,'-B','-m',module]+suffix[1:]
        with (out/(name+'.log')).open('x',encoding='utf8') as log:
            p=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(name=name,command=command,returncode=p.returncode,log_sha256=sha(out/(name+'.log'))))
        print(name,p.returncode,flush=True)
        if p.returncode:break
    write(ROOT/'EXECUTION.json',dict(status='PASS' if len(results)==3 and all(r['returncode']==0 for r in results) else 'FAIL',runs=results))
    if any(r['returncode'] for r in results):raise RuntimeError('formal execution failed; artifacts retained')

if __name__=='__main__':main()
