"""Compact auditable artifacts; no data/features/backbone duplication."""
from ..core import *


def main():
    assert read(ROOT/'CHECKS.json')['status']=='PASS'
    pre=read(ROOT/'PREFLIGHT.json')
    assert all(sha(REPO/q)==v for q,v in pre['sources'].items())
    assert all(sha(q)==v for q,v in pre['inputs'].items())
    v2=read(OLD/'results/RESULTS.json');v3=read(ROOT/'RESULTS.json')
    with gzip.open(ROOT/'CALLS.jsonl.gz','rt') as f:calls=[json.loads(x) for x in f]
    K=np.asarray(read(ROOT/'CONFIG.json')['K'])
    slack=np.asarray([c['full_counts'] for c in calls])-K
    analysis=dict(
        v2_displacement_tensor_elements=72*80*1537,v3_displacement_tensor_elements=80*1537,
        tensor_element_reduction_factor=72,
        v2_fp64_refresh_matmul_multiply_add_flops=2*v2['anchor_refresh_rows']*1537*80,
        v3_fp64_refresh_matmul_multiply_add_flops=0,
        v2_fp32_padded_affine_multiply_add_flops=2*v2['padded_candidate_rows']*1536*80,
        v3_fp32_padded_affine_multiply_add_flops=2*v3['padded_rows']*1536*80,
        full_fp32_padded_affine_multiply_add_flops=2*v2['full_fixed_padded_rows']*1536*80,
        flops_scope='Analytical multiply/add counts for dense affine matrix multiplication only; exclude bias, reductions, padding, transfers, certificate vector ops and unchanged Hildreth. Not hardware performance counters.',
        padded_rows_reduction_vs_v2=1-v3['padded_rows']/v2['padded_candidate_rows'],
        padded_rows_reduction_vs_full=1-v3['padded_rows']/v2['full_fixed_padded_rows'],
        relation_candidate_pairs=slack.size,at_floor=int((slack==0).sum()),above_floor=int((slack>0).sum()),
        deficit=int((slack<0).sum()),at_floor_fraction=float((slack==0).mean()),above_floor_fraction=float((slack>0).mean()),
        slack_median=float(np.median(slack)),slack_p90=float(np.quantile(slack,.9)),
        floor_early_stop_unexamined_uncertified_rows=v3['unexamined_uncertified_rows'],
        fp64_cache_refresh_rows=v3['cache_refreshed_old_rows'],additional_refresh_forward_rows=0,
        warning='Unexamined uncertified rows measure early-stop omission in this path; not a separate ablation. Cache refresh rows overlap exact rows and must not be added twice.')
    write(ROOT/'WORK_ANALYSIS.json',analysis)
    destination=HERE/'results';destination.mkdir(exist_ok=False)
    names=['NUMERICAL_PREFLIGHT.json','PREFLIGHT.json','CONFIG.json','CALLS.jsonl.gz','STEPS.jsonl.gz',
           'TRAINING_COMPLETE.json','RESULTS.json','EVALUATION.json','STANDALONE_full.json','STANDALONE_v3.json',
           'STANDALONE_v2.json','BENCHMARK.json','PROFILE.json','CHECKS.json','WORK_ANALYSIS.json']
    manifest={}
    for name in names:
        src=ROOT/name;dst=destination/name
        with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
        assert sha(src)==sha(dst)
        manifest[name]=dict(sha256=sha(src),bytes=src.stat().st_size)
    write(HERE/'ARTIFACT_MANIFEST.json',dict(status='PASS',authority=str(ROOT),public_files=manifest,
        external_checkpoint=dict(path=str(ROOT/'CHECKPOINT.pt'),sha256=sha(ROOT/'CHECKPOINT.pt'),bytes=(ROOT/'CHECKPOINT.pt').stat().st_size),
        diagnostic_sources={q.relative_to(HERE).as_posix():sha(q) for q in (HERE/'diagnostics').glob('*.py')}))
    print(json.dumps(dict(analysis=analysis,public_bytes=sum(v['bytes'] for v in manifest.values())),indent=2))


if __name__=='__main__':main()
