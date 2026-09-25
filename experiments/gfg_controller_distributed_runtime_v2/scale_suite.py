"""Frozen calibration, independent comparison, conditional sparse policies."""
import math
import os
import statistics
import subprocess
import sys
from .evidence import HERE,OUT,REPO,read,write


def job(scale,phase,arm,seed,lr,k=5,n=212,u=11):
    path=OUT/'scale'/scale/phase/f'{arm}-s{seed}-lr{lr}-k{k}-n{n}-u{u}'/'RESULT.json'
    if path.exists():return read(path) # explicit resume, never overwrites
    cmd=[sys.executable,'-m','experiments.gfg_controller_distributed_runtime_v2.scale_job','--scale',scale,'--phase',phase,'--arm',arm,'--seed',str(seed),'--lr',str(lr),
        '--cadence',str(k),'--probes',str(n),'--subset-seed',str(u)]
    print('START',scale,phase,arm,seed,lr,k,n,u,flush=True)
    p=subprocess.run(cmd,cwd=REPO,check=False)
    if not path.exists():write(path,{'status':'PROCESS_FAILED','returncode':p.returncode,'scale':scale,'phase':phase,'arm':arm,'seed':seed,'lr':lr})
    result=read(path);print('END',result['status'],result.get('final_accuracy'),flush=True);return result


def reached(r):return r['status']=='COMPLETED' and r.get('final_accuracy',0)>=.99
def steps(r):return next((v['t'] for v in r.get('rows',[]) if v['accuracy']>=.99),201)
def summary(rs):return {'reach':sum(reached(r) for r in rs),'restricted_steps':statistics.mean(steps(r) for r in rs),'wall':statistics.mean(r.get('wall',300) for r in rs),'memory':max(r.get('peak_allocated',0) for r in rs)}
def improves(a,b):return a['reach']>b['reach'] or (a['reach']==b['reach'] and a['restricted_steps']<b['restricted_steps'])


def main():
    gate=read(OUT/'GATES.json');exact_ok=all(r.get('all_exact',False) for r in gate['gates'] if r.get('backend')=='exact' and not r.get('fsdp_world_size'))
    results={};eligible=[]
    for scale,width in [('70',704),('160',1024)]:
        l0=.0001*math.sqrt(64/width);rates=[r*l0 for r in (.125,.25,.5,1,2)]
        cal=[{a:job(scale,'calibration',a,101,lr) for a in ('native','fixed')} for lr in rates]
        safe=next((i for i,p in enumerate(cal) if reached(p['native'])),None)
        near=next((i for i,p in enumerate(cal) if safe is not None and i>safe and not reached(p['native']) and reached(p['fixed'])),None)
        severe=next((i for i,p in enumerate(cal) if near is not None and i>near and not reached(p['native']) and reached(p['fixed'])),None)
        successes=[i for i,p in enumerate(cal) if reached(p['native'])];best=min(successes,key=lambda i:(steps(cal[i]['native']),rates[i])) if successes else None
        selection={'rates':rates,'safe':safe,'near':near,'severe':severe,'tuned_native_index':best,'exact_gate':exact_ok}
        receipt=OUT/'scale'/scale/'REGIME_SELECTION.json'
        if receipt.exists():assert read(receipt)==selection
        else:write(receipt,selection)
        formal={}
        if exact_ok:
            for regime,index in [('safe',safe),('near',near),('severe',severe)]:
                if index is None:continue
                arms={a:[job(scale,'formal-'+regime,a,s,rates[index]) for s in (201,202,203)] for a in ('native','fixed','dynamic')}
                arms['tuned_native']=[job(scale,'formal-'+regime+'-tuned','native',s,rates[best]) for s in (201,202,203)] if best is not None else []
                formal[regime]={a:summary(rs) for a,rs in arms.items() if rs}
                if regime=='near' and improves(formal[regime]['dynamic'],formal[regime]['fixed']):eligible.append((scale,rates[index]))
        results[scale]={'selection':selection,'formal':formal}
    write(OUT/'SCALE_RESULTS.json',results)
    if not eligible:write(OUT/'PHASE3_RESULTS.json',{'status':'NOT_RUN','reason':'no qualifying independent-seed NEAR dynamic benefit','production_preset':'NONE'});return
    scale,lr=eligible[0];configs=[(5,212,11)]+[(k,212,11) for k in (10,20,50)]+[(5,n,u) for n in (106,53,26) for u in (11,29)]
    measured=[]
    for k,n,u in configs:
        rs=[job(scale,'phase3-single','dynamic',s,lr,k,n,u) for s in (301,302)];measured.append({'config':[k,n,u],**summary(rs)})
    def dominates(a,b):return a['reach']>=b['reach'] and all(a[k]<=b[k] for k in ('restricted_steps','wall','memory')) and (a['reach']>b['reach'] or any(a[k]<b[k] for k in ('restricted_steps','wall','memory')))
    frontier=[r for r in measured if not any(dominates(o,r) for o in measured)]
    bestcad=min([r for r in measured if r['config'][1]==212],key=lambda r:(-r['reach'],r['restricted_steps'],r['wall']))
    bestsub=sorted([r for r in measured if r['config'][1]<212],key=lambda r:(-r['reach'],r['restricted_steps'],r['wall']))[:2]
    combinations=sorted(set((bestcad['config'][0],r['config'][1],r['config'][2]) for r in bestsub))
    write(OUT/'PHASE3_SELECTION.json',{'scale':scale,'lr':lr,'single_variable':measured,'pareto':frontier,'combinations':[list(c) for c in combinations],
        'rule':'best cadence plus two best subsets; confirmation independent, no weighted score'})
    confirmed=[]
    for k,n,u in [(5,212,11),*combinations]:
        rs=[job(scale,'phase3-confirm','dynamic',s,lr,k,n,u) for s in (401,402,403)];confirmed.append({'config':[k,n,u],**summary(rs)})
    reference=confirmed[0];good=[r for r in confirmed[1:] if r['reach']>=reference['reach'] and r['restricted_steps']<=reference['restricted_steps'] and r['wall']<reference['wall']]
    write(OUT/'PHASE3_RESULTS.json',{'status':'COMPLETED','scale':scale,'confirmation':confirmed,'production_preset':min(good,key=lambda r:r['wall']) if good else 'NO_RECOMMENDED_PRESET'})


if __name__=='__main__':main()
