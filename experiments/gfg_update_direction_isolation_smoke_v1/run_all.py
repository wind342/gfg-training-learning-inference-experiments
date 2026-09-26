import subprocess
import sys
from .core import *

def main():
    logs=ROOT/'execution';logs.mkdir(parents=True);outcomes={}
    for arm in (*ARMS,'check'):
        if arm=='check' and any(v!=0 for v in outcomes.values()):break
        command=[sys.executable,'-B','-m','experiments.'+HERE.name+('.check' if arm=='check' else '.run')]+([] if arm=='check' else [arm])
        with (logs/f'{arm}.log').open('x',encoding='utf8') as f:
            p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
            for line in p.stdout:f.write(line);f.flush();print(line,end='',flush=True)
            outcomes[arm]=p.wait()
    result=dict(status='PASS' if len(outcomes)==4 and all(v==0 for v in outcomes.values()) else 'FAIL',returncodes=outcomes)
    write(ROOT/'EXECUTION.json',result);write(HERE/'results/EXECUTION.json',result)
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
