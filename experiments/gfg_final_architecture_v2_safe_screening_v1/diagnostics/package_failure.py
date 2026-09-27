"""Preserve the stopped run; never relabel an incomplete experiment PASS."""
from ..core import *


def rows(path):
    with gzip.open(path,'rt',encoding='utf8') as f:
        return [json.loads(line) for line in f]


def main():
    pre=read(ROOT/'PREFLIGHT.json')
    assert all(sha(REPO/q)==v for q,v in pre['sources'].items())
    assert all(sha(q)==v for q,v in pre['inputs'].items())
    rr=rows(ROOT/'STEPS.jsonl.gz');cc=rows(ROOT/'CALLS.jsonl.gz')
    K=np.asarray(read(ROOT/'CONFIG.json')['K']);n=10072
    assert len(rr)==11 and cc[-1]['step']==12 and not (ROOT/'TRAINING_COMPLETE.json').exists()
    assert not (ROOT/'EVALUATION.json').exists()
    assert all(np.all(np.asarray(r['report']['executed_counts'])>=K) for r in rr)
    assert all(r['historical_head_identical'] and r['historical_moments_identical'] for r in rr)
    assert all(not c['false_safe'] for c in cc)
    assert sum(not c['count_match'] for c in cc)==1
    for c in cc:
        mask=np.unpackbits(np.frombuffer(bytes.fromhex(c['certified_bits']),dtype=np.uint8))[:n]
        assert int(mask.sum())==c['certified']
        assert c['frontier']==n-c['certified']
    complete=[c for c in cc if c['step']<=11]
    screening_setup=read(ROOT/'CONFIG.json')['screening_setup_wall']
    screened=screening_setup+sum(r['common_update_wall']+r['screened_control_wall'] for r in rr)
    reference=sum(r['common_update_wall']+r['reference_control_wall'] for r in rr)
    refreshes=[dict(step=1,relations=list(range(72)),rows=n)]+[r['refresh'] for r in rr if r['refresh'] is not None]
    result=dict(status='STOPPED_NUMERICAL_MISMATCH',phase='A',formal_gate='NOT_PASSED',
        scheduled_updates=210,completed_updates=11,stopped_step=12,stopped_point='native',
        full_bank_size=n,observer_calls=len(cc),false_safe=0,
        candidate_count_mismatches=1,executed_relation_floor_violations=0,
        completed_relation_step_checks=11*72,historical_head_identical_steps=11,
        historical_moments_identical_steps=11,reference_head_identical_steps=sum(r['reference_head_identical'] for r in rr),
        reference_full_trace_identical_steps=sum(not r['reference_mismatches'] for r in rr),
        reference_trace_mismatches=[dict(step=r['step'],keys=r['reference_mismatches']) for r in rr if r['reference_mismatches']],
        skipped_completed_steps=sum(r['report']['skip'] for r in rr),
        completed_sweeps=sum(r['report']['sweep_count'] for r in rr),
        all_call_frontier_ratio=dict(median=float(np.median([c['frontier']/n for c in cc])),
                                    p90=float(np.quantile([c['frontier']/n for c in cc],.9)),
                                    max=max(c['frontier']/n for c in cc)),
        completed_full_candidate_rows=len(complete)*n,
        completed_exact_frontier_rows=sum(c['exact_old_rows'] for c in complete),
        completed_anchor_refresh_rows=sum(r['rows'] for r in refreshes),
        completed_relation_refreshes=sum(len(r['relations']) for r in refreshes),
        completed_screened_algorithm_wall=screened,completed_reference_algorithm_wall=reference,
        completed_paired_speedup=reference/screened,screening_setup_wall=screening_setup,
        completed_checker_wall=sum(r['observer_wall'] for r in rr),
        final_old_new_all=None,final_predictions_identical=None,
        test_evaluation_performed=False,banking77_started=False,
        diagnosis=read(ROOT/'FAILURE_DIAGNOSIS.json'),
        conclusion='Safe-set certificates had no observed false-safe, but sparse FP32 GEMM changed one frontier sign and relation count. Exact historical semantics and full-run acceleration are not established. Stop preserved, no automatic restart.')
    result['completed_exact_row_reduction']=1-result['completed_exact_frontier_rows']/result['completed_full_candidate_rows']
    write(ROOT/'PARTIAL_RESULTS.json',result)
    write(ROOT/'FAILURE_AUDIT.json',dict(status='PRESERVED_FAILURE_VERIFIED',formal_gate='NOT_PASSED',
        source_hashes_unchanged=True,input_hashes_unchanged=True,formal_run_retried=False,
        test_not_evaluated=True,eleven_committed_heads_reconstructed_by_original_controller=True,
        all_87_certificate_masks_checked=True,scope='Failure diagnosis and ledger/hash checks; not a successful 210-update validation.'))
    public=HERE/'results';public.mkdir(exist_ok=False)
    names=['NUMERICAL_PREFLIGHT.json','PREFLIGHT.json','CONFIG.json','CALLS.jsonl.gz','STEPS.jsonl.gz',
           'FAILURE_DIAGNOSIS.json','PARTIAL_RESULTS.json','FAILURE_AUDIT.json']
    names += [q.name for q in ROOT.glob('FAILURE_*.json') if q.name not in names]
    manifest={}
    for name in sorted(set(names)):
        source=ROOT/name;destination=public/name
        with source.open('rb') as a,destination.open('xb') as b:
            shutil.copyfileobj(a,b)
        assert sha(source)==sha(destination)
        manifest[name]=dict(sha256=sha(source),bytes=source.stat().st_size)
    write(HERE/'ARTIFACT_MANIFEST.json',dict(authority=str(ROOT),status='PRESERVED_FAILURE',files=manifest))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
