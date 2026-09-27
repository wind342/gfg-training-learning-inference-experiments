from .common import *
from .preflight import install,verify_inputs
from .run import run,warmup
from .compare import compare

def main():
    access=install();setup();verify_inputs();assert read(ROOT/'IDENTITY.json')['status']=='PASS'
    for variant in ('baseline','firstcheck'):assert read(ROOT/variant/'AUDIT.json')['status']=='PASS'
    target=ROOT/'benchmark';target.mkdir(exist_ok=False);warmup();setup();X,y,c=load();reports=[]
    for repeat,order in enumerate(read(ROOT/'FREEZE.json')['benchmark_orders'],1):
        for arm in order:
            mode,variant=arm.split('/');output=target/f'repeat{repeat}'/arm;output.mkdir(parents=True,exist_ok=False)
            for budget in (210,630):
                report=run(budget,X,y,c,variant,output,mode=mode)
                check=compare(output/f'budget{budget}',PREVIOUS/variant/f'budget{budget}',budget)
                reports.append(dict(repeat=repeat,mode=mode,variant=variant,budget=budget,report=report,identity=check))
    verify_inputs();write(target/'SUMMARY.json',dict(status='PASS',reports=reports,access=access.report()))
    print('All 24 benchmark trajectories and exclusive ledgers PASS',flush=True)

if __name__=='__main__':main()
