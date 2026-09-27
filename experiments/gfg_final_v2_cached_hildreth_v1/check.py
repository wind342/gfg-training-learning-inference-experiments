"""Independent artifact cross-check and create-only public packaging."""
from .run import guard,inputs,history
from .core import *


def records(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(x) for x in f]


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    guard();inputs()
    cached=records(ROOT/'cached_validation/STEPS.jsonl.gz')
    transition=records(ROOT/'transition_validation/STEPS.jsonl.gz')
    oldrows=history();assert len(cached)==len(transition)==len(oldrows)==210
    K=np.asarray(read(INPUT/'v2_210/task10/CONFIG.json')['K'])
    for step,(a,b,c) in enumerate(zip(cached,transition,oldrows),1):
        assert a['step']==b['step']==step and a['checks']==b['checks']=='PASS'
        for k in ('head_sha256','moments_sha256'):assert a[k]==b[k]==c[k]
        assert a['report']==b['report']
        assert np.all(np.asarray(a['report']['executed_counts'])>=K)
        for k in ('native_deficits','final_active','final_duals','selected_controls','sweep_count','skip'):
            assert a['report'][k]==c['report'][k]
        for x,y in zip(a['report']['sweeps'],c['report']['sweeps']):
            for k in ('sweep','safe','constraints','primal_sha256','dual_sha256'):assert x[k]==y[k]
    calls=records(ROOT/'transition_validation/CALLS.jsonl.gz')
    assert len(calls)==2152
    for r in calls:
        assert r['false_safe']==r['false_wrong']==r['max_exact_margin_error']==0
        assert r['complete_counts_identical'] and r['deficits_identical'] and r['choices_identical']
        if r['point']=='commit':assert np.all(np.asarray(r['counts'])>=K)
    for step in range(1,211):
        points=[r['point'] for r in calls if r['step']==step]
        assert points==['receiving','native']+['sweep'+str(i) for i in range(1,cached[step-1]['report']['sweep_count']+1)]+['commit']
    evaluations=[read(ROOT/(k+'_validation')/'EVALUATION.json') for k in ('cached','transition')]
    assert evaluations[0]==evaluations[1]
    assert evaluations[0]['predictions']==read(OLD_ROOT/'EVALUATION.json')['predictions']
    paths={}
    for prefix in ('TIMED','PROFILE'):
        for kind in ('full','cached','transition'):
            r=read(ROOT/f'{prefix}_{kind}.json');paths[prefix+'_'+kind]=r
            assert r['steps']==210 and r['sweeps']==1522 and r['final_predictions_identical']
            assert len(r['verification'])==210
            for a,b in zip(r['verification'],cached):
                assert a['head']==b['head_sha256'] and a['moments']==b['moments_sha256']
    summaries={k:{a:b for a,b in v.items() if a not in ('verification','per_step_control')} for k,v in paths.items()}
    result=dict(status='PASS',steps=210,candidate_observer_calls=len(calls),false_safe=0,false_wrong=0,floor_violations=0,
        executed_heads_byte_identical=True,adamw_moments_byte_identical=True,selections_active_pairs_deficits_identical=True,
        sweep_sequence_and_fp64_primal_dual_identical=True,pair_reserves_byte_identical=True,final_predictions_identical=True,
        previous_sources_and_inputs_unchanged=True,banking77_started=False)
    if args.verify_only:
        print(json.dumps(result,indent=2),flush=True);return
    write(ROOT/'CHECKS.json',result)
    write(ROOT/'BENCHMARK.json',dict(order=['full','cached','transition'],single_measurement_per_method=True,
        historical_full_seconds=59.64862279931549,methods=summaries,
        cached_speedup=summaries['TIMED_full']['total_wall']/summaries['TIMED_cached']['total_wall'],
        transition_speedup_vs_cached=summaries['TIMED_cached']['total_wall']/summaries['TIMED_transition']['total_wall'],
        note='TIMED has no observer or component synchronization. PROFILE is a distinct instrumented replay; never substitute its components into standalone wall times. Setup, actual commit/recheck and complete executed-anchor refresh are included. Data loading, external validation hashes and final test are excluded.'))
    sources={q.relative_to(HERE).as_posix():sha(q) for q in HERE.rglob('*') if q.is_file() and q.suffix in ('.py','.md')}
    write(ROOT/'FINAL_SOURCE_MANIFEST.json',dict(base=BASE,sources=sources,
        note='PREFLIGHT records the initial stage-1 implementation. The harness was subsequently extended for gated stage 2; final sources are recorded here.'))
    dest=HERE/'results';dest.mkdir(exist_ok=False)
    selected=['PREFLIGHT.json','TRANSITION_PREFLIGHT.json','CHECKS.json','BENCHMARK.json','FINAL_SOURCE_MANIFEST.json']
    selected += [f'{prefix}_{kind}.json' for prefix in ('TIMED','PROFILE') for kind in ('full','cached','transition')]
    for kind in ('cached','transition'):
        selected += [f'{kind}_validation/{name}' for name in ('STEPS.jsonl.gz','EVALUATION.json','CHECKS.json')]
    selected+=['transition_validation/CALLS.jsonl.gz']
    manifest={}
    for name in selected:
        src=ROOT/name;out=dest/name;out.parent.mkdir(exist_ok=True,parents=True)
        with src.open('rb') as a,out.open('xb') as b:shutil.copyfileobj(a,b)
        assert sha(src)==sha(out)
        manifest[name]=dict(sha256=sha(out),bytes=out.stat().st_size)
    probe=Path('E:/gfg-evidence/final-v2-cached-hildreth-v1-probe')
    for src in sorted(probe.glob('*.json')):
        out=dest/src.name
        with src.open('rb') as a,out.open('xb') as b:shutil.copyfileobj(a,b)
        manifest[src.name]=dict(sha256=sha(out),bytes=out.stat().st_size)
    external={}
    for kind in ('cached','transition'):
        q=ROOT/(kind+'_validation')/'CHECKPOINT.pt';external[str(q)]=dict(sha256=sha(q),bytes=q.stat().st_size)
    write(HERE/'ARTIFACT_MANIFEST.json',dict(status='PASS',authority=str(ROOT),public_files=manifest,external_checkpoints=external,sources=sources))
    print(json.dumps(dict(checks=result,timings={k:v['total_wall'] for k,v in summaries.items()},public_bytes=sum(v['bytes'] for v in manifest.values())),indent=2),flush=True)


if __name__=='__main__':main()
