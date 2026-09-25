"""Bounded sequential scale work, no claims from missing regimes."""
import math
import subprocess
import sys
from .checks import b,HERE,OUT,REPO
from .scale import CONFIGS


def job(scale,phase,arm,seed,lr,steps):
    cmd=[sys.executable,'-m','experiments.gfg_controller_distributed_runtime_v1.scale','--scale',scale,'--phase',phase,
         '--arm',arm,'--seed',str(seed),'--lr',str(lr),'--steps',str(steps)]
    print('START',scale,phase,arm,seed,lr,flush=True)
    p=subprocess.run(cmd,cwd=REPO,check=False)
    path=OUT/'scale'/scale/phase/f'{arm}-seed{seed}-lr{lr}'/'RESULT.json'
    if not path.exists():b.write(path,{'status':'PROCESS_FAILED','returncode':p.returncode,'scale':scale,'phase':phase,'arm':arm,'seed':seed,'lr':lr})
    r=b.read(path);print('END',r['status'],r.get('final_accuracy'),flush=True)
    return r


def main():
    result={}
    for scale,(_,width,_) in CONFIGS.items():
        factor=math.sqrt(64/width);smoke=[job(scale,'smoke',arm,101,.001*factor,6) for arm in ('native','fixed','dynamic')]
        if not all(r['status']=='COMPLETED' for r in smoke):
            result[scale]={'smoke':'RESOURCE_LIMITED_OR_FAILED','regimes':'NOT_TESTED'};continue
        rates=[x*factor for x in (.0001,.0003,.001,.003)];cal=[]
        for rate in rates:
            pair={arm:job(scale,'calibration',arm,101,rate,80) for arm in ('native','fixed')}
            cal.append(pair)
        hits=lambda r:r['status']=='COMPLETED' and r['final_accuracy']>=.99
        safe=next((i for i,pair in enumerate(cal) if hits(pair['native'])),None)
        near=next((i for i,pair in enumerate(cal) if safe is not None and i>safe and not hits(pair['native']) and hits(pair['fixed'])),None)
        severe=next((i for i,pair in enumerate(cal) if near is not None and i>near and not hits(pair['native']) and hits(pair['fixed'])),None)
        selection={'safe':safe,'near':near,'severe':severe,'rates':rates}
        b.write(OUT/'scale'/scale/'REGIME_SELECTION.json',selection)
        if any(x is None for x in (safe,near,severe)):
            result[scale]={'smoke':'PASS','regimes':'NOT_ESTABLISHED','selection':selection,'transport':'NOT_ASSESSED'};continue
        formal=[]
        for regime,index in [('safe',safe),('near',near),('severe',severe)]:
            for seed in (201,202):
                for arm in ('native','fixed','dynamic'):formal.append(job(scale,'formal-'+regime,arm,seed,rates[index],80))
        result[scale]={'smoke':'PASS','regimes':'ESTABLISHED_IN_CALIBRATION','selection':selection,
            'formal':[{'arm':r['arm'],'phase':r['phase'],'seed':r['seed'],'status':r['status'],'accuracy':r.get('final_accuracy')} for r in formal]}
    b.write(HERE/'SCALE_RESULTS.json',result)


if __name__=='__main__':main()
