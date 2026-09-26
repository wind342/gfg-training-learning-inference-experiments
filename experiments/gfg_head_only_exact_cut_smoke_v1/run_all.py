import sys
from .core import *

def main():
    guard();logs=ROOT/'execution';logs.mkdir(exist_ok=False);runs=[]
    for name in ('train','evaluate','check'):
        cmd=[sys.executable,'-B','-m','experiments.'+HERE.name+'.'+name]
        with (logs/(name+'.log')).open('x',encoding='utf8') as f:p=subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        runs.append(dict(name=name,returncode=p.returncode,log_sha256=sha(logs/(name+'.log'))));print(name,p.returncode,flush=True)
        if p.returncode:break
    write(ROOT/'EXECUTION.json',dict(status='PASS' if len(runs)==3 and all(r['returncode']==0 for r in runs) else 'FAIL',runs=runs))
    if any(r['returncode'] for r in runs):raise RuntimeError('stopped; failure retained')

if __name__=='__main__':main()
