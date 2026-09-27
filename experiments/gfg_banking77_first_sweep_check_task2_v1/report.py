from .common import *
from .compare import rows
from .preflight import install,verify_inputs

def stats(v):return dict(median=float(np.median(v)),min=float(min(v)),max=float(max(v)),samples=v)

def main():
    access=install();verify_inputs();benchmark=read(ROOT/'benchmark/SUMMARY.json');out=[]
    for b in (210,630):
        arms={}
        for variant in ('baseline','firstcheck'):
            check=read(ROOT/f'{variant}/budget{b}/CHECKS.json');audit=read(ROOT/variant/'AUDIT.json')
            assert audit['status']=='PASS'
            runs=[x['report'] for x in benchmark['reports'] if x['budget']==b and x['variant']==variant]
            timing={key:stats([x[key] for x in runs]) for key in ('control_wall','total_algorithm_wall','common_wall')}
            regions=sorted(set(k for x in runs for k in x['region_seconds']))
            timing['regions']={k:stats([x['region_seconds'].get(k,0.) for x in runs]) for k in regions}
            arms[variant]=dict(checks=check,timing=timing,audit=next(x for x in audit['reports'] if x['budget']==b))
        original=rows(ROOT/f'baseline/budget{b}/STEPS.jsonl.gz');new=rows(ROOT/f'firstcheck/budget{b}/STEPS.jsonl.gz')
        differences=[a['step'] for a,c in zip(original,new) if a['executed_sha256']!=c['executed_sha256']]
        savings={key:1-arms['firstcheck']['timing'][key]['median']/arms['baseline']['timing'][key]['median'] for key in ('control_wall','total_algorithm_wall')}
        a=arms['baseline']['checks'];c=arms['firstcheck']['checks']
        out.append(dict(budget=b,arms=arms,relative_wall_reduction=savings,sweep_reduction=1-c['inner_sweeps']/a['inner_sweeps'],
            changed_executed_steps=len(differences),first_changed_step=differences[0] if differences else None,
            train_macro_delta_percentage_points={k:100*(c['train_metrics'][k]['macro']-a['train_metrics'][k]['macro']) for k in ('old','current','all')}))
    write(ROOT/'SUMMARY.json',dict(status='PASS',classification='New first-sweep stopping policy; not byte-identical acceleration',reports=out,access=access.report()))
    results=HERE/'results';results.mkdir(exist_ok=False)
    paths=[ROOT/'FREEZE.json',ROOT/'NUMERICS.json',ROOT/'BASELINE_IDENTITY.json',ROOT/'SUMMARY.json',ROOT/'benchmark/SUMMARY.json']
    for v in ('baseline','firstcheck'):
        paths.append(ROOT/v/'AUDIT.json')
        paths.extend(ROOT/v/f'budget{b}/CHECKS.json' for b in (210,630))
    for p in paths:
        dst=results/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(p.read_bytes())
    manifest={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in ROOT.rglob('*') if p.is_file()}
    write(HERE/'ARTIFACT_MANIFEST.json',dict(authority=str(ROOT),files=manifest))
    print(json.dumps([dict(budget=r['budget'],wall_reduction=r['relative_wall_reduction'],train_delta_pp=r['train_macro_delta_percentage_points']) for r in out],indent=2))

if __name__=='__main__':main()
