"""Create compact public records after independent replay succeeds."""
from ..core import *


def main():
    checks=read(ROOT/'CHECKS.json');assert checks['status']=='PASS'
    pre=read(ROOT/'PREFLIGHT.json')
    assert all(sha(REPO/q)==v for q,v in pre['sources'].items())
    assert all(sha(q)==v for q,v in pre['inputs'].items())
    result=read(ROOT/'RESULTS.json');assert result['gate']=='PASS'
    original=read(INPUT/'EVALUATION_TASK10.json')['arms']['v2_210']
    ev=read(ROOT/'EVALUATION.json')
    comparisons=dict(historical=original['metrics'],fixed_arithmetic_screened=ev['metrics'],
        old_new_all_delta_pp={key:100*(ev['metrics']['macro_'+key]-original['metrics']['macro_'+key]) for key in ('old','new','all')},
        changed_test_predictions=sum(a!=b for a,b in zip(original['predictions'],ev['predictions'])),
        fixed_arithmetic_full_reference_identical=True,
        comparison_warning='Changed historical numerical primitive is disclosed; fixed full and screened trajectories match exactly. Timing reference is fixed full, not the historical variable-shape run.')
    write(ROOT/'HISTORICAL_COMPARISON.json',comparisons)
    public=HERE/'results';public.mkdir(exist_ok=False)
    public_names=['NUMERICAL_PREFLIGHT.json','PREFLIGHT.json','CONFIG.json','CALLS.jsonl.gz','STEPS.jsonl.gz',
        'REFRESHES.json','TRAINING_COMPLETE.json','RESULTS.json','CHECKS.json','INDEPENDENT_REPLAY.json',
        'EVALUATION.json','HISTORICAL_COMPARISON.json']
    manifest={}
    for name in public_names:
        source=ROOT/name;destination=public/name
        with source.open('rb') as a,destination.open('xb') as b:
            shutil.copyfileobj(a,b)
        assert sha(source)==sha(destination)
        manifest[name]=dict(sha256=sha(source),bytes=source.stat().st_size)
    write(HERE/'ARTIFACT_MANIFEST.json',dict(status='PASS',authority=str(ROOT),public_files=manifest,
        external_checkpoint=dict(path=str(ROOT/'CHECKPOINT.pt'),sha256=sha(ROOT/'CHECKPOINT.pt'),bytes=(ROOT/'CHECKPOINT.pt').stat().st_size),
        diagnostic_sources={q.relative_to(HERE).as_posix():sha(q) for q in (HERE/'diagnostics').glob('*.py')}))
    print(json.dumps(dict(delta_pp=comparisons['old_new_all_delta_pp'],changed_test_predictions=comparisons['changed_test_predictions'],
        local_bytes=sum(q.stat().st_size for q in ROOT.rglob('*') if q.is_file()),
        public_bytes=sum(v['bytes'] for v in manifest.values())),indent=2))


if __name__=='__main__':
    main()
