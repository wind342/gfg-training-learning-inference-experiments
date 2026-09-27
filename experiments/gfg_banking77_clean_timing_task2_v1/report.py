from .common import *
from .preflight import install,verify_inputs

def stats(v):return dict(median=float(np.median(v)),min=float(min(v)),max=float(max(v)),samples=v)

def main():
    access=install();verify_inputs();benchmark=read(ROOT/'benchmark/SUMMARY.json');reports=[]
    for budget in (210,630):
        arms={}
        for mode in ('recorded','clean'):
            for variant in ('baseline','firstcheck'):
                items=[x for x in benchmark['reports'] if x['budget']==budget and x['mode']==mode and x['variant']==variant]
                timing={key:stats([x['report'][key] for x in items]) for key in ('control_wall','common_wall','entry_wall','total_algorithm_wall','observer_wall','deferred_evidence_wall','training_loop_wall')}
                representative=sorted(items,key=lambda x:x['report']['total_algorithm_wall'])[1]
                # One actual run, so the ledger is additive, unlike independent medians.
                rep=representative['report'];parts=rep['exclusive_seconds']
                for root,key in [('entry_total','entry_wall'),('native_total','common_wall'),('control_total','control_wall')]:
                    assert abs(sum(parts[root].values())-rep[key])<1e-7
                assert abs(sum(sum(p.values()) for p in parts.values())-rep['total_algorithm_wall'])<1e-7
                arms[mode+'/'+variant]=dict(timing=timing,representative_repeat=representative['repeat'],representative=rep)
        comparisons={}
        for left,right in [('recorded/baseline','clean/baseline'),('recorded/firstcheck','clean/firstcheck'),('clean/baseline','clean/firstcheck')]:
            comparisons[left+' -> '+right]={key:dict(relative_reduction=1-arms[right]['timing'][key]['median']/arms[left]['timing'][key]['median'],
                paired_reductions=[1-b/a for a,b in zip(arms[left]['timing'][key]['samples'],arms[right]['timing'][key]['samples'])]) for key in ('control_wall','total_algorithm_wall')}
        reports.append(dict(budget=budget,arms=arms,comparisons=comparisons))
    write(ROOT/'SUMMARY.json',dict(status='PASS',classification='Same per-policy trajectory, evidence deferred outside training timers',reports=reports,access=access.report()))
    results=HERE/'results';results.mkdir(exist_ok=False)
    paths=[ROOT/'FREEZE.json',ROOT/'NUMERICS.json',ROOT/'IDENTITY.json',ROOT/'SUMMARY.json',ROOT/'benchmark/SUMMARY.json']
    for v in ('baseline','firstcheck'):
        paths.append(ROOT/v/'AUDIT.json');paths.extend(ROOT/v/f'budget{b}/CHECKS.json' for b in (210,630))
    for p in paths:
        dst=results/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(p.read_bytes())
    manifest={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in ROOT.rglob('*') if p.is_file()}
    write(HERE/'ARTIFACT_MANIFEST.json',dict(authority=str(ROOT),files=manifest))
    print(json.dumps([dict(budget=r['budget'],comparisons=r['comparisons']) for r in reports],indent=2))

if __name__=='__main__':main()
