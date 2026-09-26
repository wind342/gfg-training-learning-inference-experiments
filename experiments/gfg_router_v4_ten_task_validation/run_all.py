"""One-shot serial primary experiment driver. Never retries an arm."""
import json
import subprocess
import sys
from .common import ROOT,HERE,ARMS,write,sha

def command(module,args,log):
    with log.open('x',encoding='utf8') as f:
        p=subprocess.Popen([sys.executable,'-B','-m',module,*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
        for line in p.stdout:
            f.write(line);f.flush();print(line,end='',flush=True)
        return p.wait()

def main():
    root=ROOT/'execution';root.mkdir(parents=True)
    outcomes={}
    for arm in ARMS:
        if arm=='random' and not (ROOT/'seed-2021/EXPOSURE_SCHEDULE.json').exists():
            outcomes[arm]=dict(status='NOT_RUN',reason='Router did not produce complete exposure schedule');continue
        code=command('experiments.'+HERE.name+'.run',[arm],root/f'{arm}.log')
        result=ROOT/'seed-2021'/arm/'RESULT.json'
        outcomes[arm]=dict(returncode=code,status=json.loads(result.read_text())['status'] if result.exists() else 'FAILED_BEFORE_RESULT',log_sha256=sha(root/f'{arm}.log'))
    if all(x.get('returncode')==0 for x in outcomes.values()):
        code=command('experiments.'+HERE.name+'.check',[],root/'check.log')
        outcomes['check']=dict(returncode=code,log_sha256=sha(root/'check.log'))
    report=dict(status='PASS' if all(x.get('returncode')==0 for x in outcomes.values()) else 'FAIL',outcomes=outcomes)
    write(ROOT/'EXECUTION.json',report);write(HERE/'results/EXECUTION.json',report)
    print(json.dumps(report,indent=2))
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
