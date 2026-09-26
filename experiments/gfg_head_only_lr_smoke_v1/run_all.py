import sys
import time
from .core import *

def main():
    guard();logs=ROOT/'execution';logs.mkdir(exist_ok=False);runs=[]
    def run(module,args,name):
        command=[sys.executable,'-B','-m','experiments.'+HERE.name+'.'+module]+args
        with (logs/(name+'.log')).open('x',encoding='utf8') as f:p=subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        runs.append(dict(name=name,returncode=p.returncode,log_sha256=sha(logs/(name+'.log'))));print(name,p.returncode,flush=True)
        if p.returncode:
            write(ROOT/'EXECUTION.json',dict(status='FAIL',runs=runs));raise RuntimeError('stopped, failure retained')
    for arm in ARMS:run('train',[arm],arm)
    hashes={}
    for arm in ARMS:
        assert json.loads((ROOT/arm/'TRAIN_RESULT.json').read_text())['status']=='COMPLETED';hashes[arm]=sha(ROOT/arm/'TRAIN_RESULT.json')
    write(ROOT/'TRAINING_COMPLETE.json',dict(created_unix=time.time(),result_hashes=hashes,source_hashes=sources()))
    run('evaluate',[],'evaluate');run('check',[],'check')
    write(ROOT/'EXECUTION.json',dict(status='PASS',runs=runs))

if __name__=='__main__':main()
