"""Three prespecified rotated replay repetitions. No best-of or budget extension."""
from .run import *

def main():
    access=install();setup();verify_inputs();freeze=read(ROOT/'FREEZE.json')
    accepted=['baseline','stage1']
    for v in accepted:
        assert read(ROOT/v/'IDENTITY.json')['status']=='PASS' and read(ROOT/v/'AUDIT.json')['status']=='PASS'
    if (ROOT/'stage2/AUDIT.json').exists() and read(ROOT/'stage2/AUDIT.json')['status']=='PASS':accepted.append('stage2')
    warmup();setup();X,y,c=load();reports=[]
    for repetition,order in enumerate(freeze['benchmark_orders'],1):
        for variant in order:
            if variant not in accepted:continue
            folder=ROOT/f'timing/repetition{repetition}/{variant}';folder.mkdir(parents=True,exist_ok=False)
            setup();results=[run(b,X,y,c,variant,folder) for b in (210,630)]
            checks=[compare_arm(folder,b) for b in (210,630)]
            assert all(r['status']=='PASS' for r in checks), 'Timing replay trajectory mismatch'
            reports.append(dict(repetition=repetition,variant=variant,reports=results))
    verify_inputs();write(ROOT/'TIMING.json',dict(status='PASS',accepted_variants=accepted,repetitions=3,reports=reports,access=access.report()))
    print('Three fixed rotated repetitions complete, all trajectories identical',flush=True)

if __name__=='__main__':main()
