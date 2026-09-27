from .common import *
from .preflight import install,verify_inputs
from .run import run,warmup
from .compare import compare

def main():
    access=install();setup();verify_inputs()
    assert read(ROOT/'BASELINE_IDENTITY.json')['status']=='PASS'
    for variant in ('baseline','firstcheck'):assert read(ROOT/variant/'AUDIT.json')['status']=='PASS'
    target=ROOT/'benchmark';target.mkdir(exist_ok=False);warmup();setup();X,y,c=load();reports=[]
    for repeat,order in enumerate(read(ROOT/'FREEZE.json')['benchmark_orders'],1):
        for variant in order:
            output=target/f'repeat{repeat}'/variant;output.mkdir(parents=True,exist_ok=False)
            for budget in (210,630):
                report=run(budget,X,y,c,variant,output)
                check=compare(output/f'budget{budget}',ROOT/variant/f'budget{budget}',budget)
                reports.append(dict(repeat=repeat,variant=variant,budget=budget,report=report,identity=check))
    verify_inputs();write(target/'SUMMARY.json',dict(status='PASS',reports=reports,access=access.report()))
    print('All fixed paired benchmark trajectories PASS',flush=True)

if __name__=='__main__':main()
