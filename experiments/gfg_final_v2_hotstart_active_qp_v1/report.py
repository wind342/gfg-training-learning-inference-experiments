"""B correctness, cross-update warm/cold attribution and independent report."""
from .run import *
from .tests import gram_check
from experiments.gfg_final_v2_endpoint_kernel_v1.report import records,package


def main():
    guard();inputs();gram_check()
    continuity=read(ROOT/'CONTINUITY.json');audit=records(ROOT/'validate/COLD_AUDIT.jsonl.gz')
    val=read(ROOT/'validate/CHECKS.json');timed=read(ROOT/'timed/CHECKS.json');profile=read(ROOT/'profile/CHECKS.json')
    rows=records(ROOT/'validate/STEPS.jsonl.gz');assert len(rows)==210
    assert all(r['floor_violations']==0 and not r['report']['skip'] for r in rows)
    K=np.asarray(read(INPUT/'v2_210/task10/CONFIG.json')['K'])
    for r in rows:
        assert np.all(np.asarray(r['report']['executed_counts'])>=K)
        assert all(s['status']=='PASS' and s['direct_primal_violation']<=1e-8 for s in r['report']['solves'])
    for mode in ('timed','profile'):
        assert records(ROOT/mode/'STEPS.jsonl.gz')==rows
        assert read(ROOT/mode/'EVALUATION.json')==read(ROOT/'validate/EVALUATION.json')
    assert len(audit)==629 and all(r['solutions_agree'] and r['warm_status']==r['cold_status']=='PASS' for r in audit)
    first=[r for r in audit if r['step']>1 and r['outer']==1];assert len(first)==209
    cross=dict(qps=len(first),warm_pivots=sum(r['warm_iterations'] for r in first),cold_pivots=sum(r['cold_iterations'] for r in first),
        warm_fewer=sum(r['warm_iterations']<r['cold_iterations'] for r in first),
        warm_more=sum(r['warm_iterations']>r['cold_iterations'] for r in first),
        mean_positive_warm=float(np.mean([r['positive_warm'] for r in first])))
    ar=Path('E:/gfg-evidence/final-v2-endpoint-kernel-v1/reference256')
    bt=read(ar/'timed/CHECKS.json');bp=read(ar/'profile/CHECKS.json')
    # This is the original fixed256 control, not any optimized A implementation.
    assert bt['head_matches']==bt['moments_matches']==210 and bt['arm']=='reference256'
    own_continuity=[]
    for a,b in zip(rows,rows[1:]):
        x={tuple(z) for z in a['report']['final_active']};y={tuple(z) for z in b['report']['final_active']}
        own_continuity.append(dict(jaccard=len(x&y)/len(x|y),previous_retained=len(x&y)/len(x)))
    summary=dict(status='PASS',performance_conclusion='NO_SPEEDUP',baseline=bt,baseline_profile=bp,
        warm=timed,warm_profile=profile,validation=val,cross_update_first_qp=cross,
        own_final_active_jaccard_mean=float(np.mean([r['jaccard'] for r in own_continuity])),
        max_warm_cold_dual_error=max(r['max_dual_difference'] for r in audit),
        max_direct_primal_violation=max(s['direct_primal_violation'] for r in rows for s in r['report']['solves']),
        speed_ratio=bt['total_wall']/timed['total_wall'],outer_endpoint_reduction=1-timed['outer_solves']/bt['sweeps'],
        all_endpoint_reduction=1-timed['endpoint_calls']/bt['endpoint_calls'],old_evidence_unchanged=True,
        note='Original fixed256 baseline shared with A reference arm; no optimized endpoint used by B. One standalone replay per method; no confidence interval. No further solver tuning.')
    write(ROOT/'SUMMARY.json',summary)
    oldm=bt['metrics'];newm=timed['metrics']
    metric=lambda m:' / '.join(f'{100*m[k]:.4f}' for k in ('macro_old','macro_new','macro_all'))
    comp=profile['profile_components'];s=continuity['summary']
    lines=['# B: warm-start active-set QP reduces checks but is not faster','',
        'Base `f357dfdb9d13759dc07a5703d6d11301a3adb946`. Separate new actuator; cached Hildreth and all previous evidence remain unchanged. FewRel task10, 210 updates only. No Banking77, old-loss replay or floor relaxation. Original fixed256 endpoint throughout.','',
        '**Safety/solver validation PASS; net speedup NOT ACHIEVED. Stop at this result.**','',
        '## Continuity before implementation','',
        '| Cached Hildreth adjacent updates (209 transitions) | Median | Mean |','|---|---:|---:|']
    for name,label in [('jaccard','Final discovered-set Jaccard'),('previous_retained','Previous constraints retained'),('added','Added constraints'),('deleted','Deleted constraints'),('positive_support_jaccard','Positive-dual support Jaccard'),('dual_pearson','Matched dual Pearson correlation'),('dual_symmetric_relative_change','Matched dual symmetric relative L2 change')]:
        lines.append(f"|{label}|{s[name]['median']:.6f}|{s[name]['mean']:.6f}|")
    lines+=['','These measurements justified a bounded actuator test. Final_active includes discovered zero-dual constraints; it is not synonymous with positive/binding support. CONTINUITY preserves every transition, the source hash, and both set definitions.','',
        '## Standalone result','',
        'Single independent timing per method; no best-of selection or confidence interval. Setup, state commit and actual floor recheck are included. External cold-QP audit, hashes, serialization and final test are excluded. The baseline is the fresh original fixed256 arm, not the buffer/batched implementations in A.','',
        '| Method | Solver iterations / sweeps | Mean discovered constraints | Endpoint calls | Control wall (s) | Total algorithm wall (s) | Old / new / all macro (%) | Floor violations |',
        '|---|---|---:|---:|---:|---:|---|---:|',
        f"|Cached Hildreth|{bt['sweeps']} sweeps|430.05|{bt['endpoint_calls']}|{bt['control_wall']:.4f}|{bt['total_wall']:.4f}|{metric(oldm)}|0|",
        f"|Warm active-set QP|{timed['solver_iterations']} pivots / {timed['outer_solves']} solves|{timed['mean_active_constraints']:.2f}|{timed['endpoint_calls']}|{timed['control_wall']:.4f}|{timed['total_wall']:.4f}|{metric(newm)}|0|",'',
        f"Warm QP is {(timed['total_wall']/bt['total_wall']-1)*100:.2f}% slower in standalone algorithm wall. Solve-followed endpoint checks fall by {summary['outer_endpoint_reduction']*100:.2f}% (1522 -> 629); all endpoint calls fall by {summary['all_endpoint_reduction']*100:.2f}% (2152 -> 1259). Pivots and Hildreth sweeps are different operations and must not be compared as equal work units.",
        '', 'All 210 scheduled updates execute; no failed solves, skips or rollbacks. Final predictions differ at 81 test examples. This is a new actuator and trajectory, not byte-identical acceleration. Final macro changes versus baseline are:',
        '', '| Cohort | Percentage-point change |','|---|---:|']
    for label,k in [('Old','macro_old'),('New','macro_new'),('All','macro_all')]:lines.append(f'|{label}|{100*(newm[k]-oldm[k]):+.6f}|')
    lines+=['','## Does warm-start use adjacent-update continuity?','',
        f"Yes, in the measured iteration sense. For the first QP of each update after step1, inherited duals/support require **{cross['warm_pivots']} pivots versus {cross['cold_pivots']} from zero**, on the identical 209 QPs. There are on average {cross['mean_positive_warm']:.2f} positive inherited multipliers in those starts. These first-QP results isolate cross-update reuse from within-update constraint expansion.",
        '',f"Across all 629 QPs, warm uses 1916 pivots versus 3645 cold; 617 cases take fewer pivots and one takes more. Independent cold solutions agree in objective and dual values; maximum dual difference is {summary['max_warm_cold_dual_error']:.3g}. Cold outcomes never feed the actuator. This supports useful warm starts, not a net wall-time gain.",
        '', '## What is cached and what changes?', '',
        'A persistent constraint registry and FP64 Gram cache survive all updates. For pair (i,c), its normal has +H_i in target class and -H_i in competitor class; Gram entries use the signed class-incidence factor times H_i dot H_j. Only missing requested entries are computed; new pairs incrementally expand the registry. Working submatrices are gathered from cached entries; principal-system factorizations are still recomputed during pivots. No claim of factorization hot updates is made.', '',
        f"Observed cache: {timed['gram_cache']['unique_constraints']} unique constraints, {timed['gram_cache']['computed_nonzero_upper_entries']:,} newly computed nonzero upper-triangle entries, {timed['gram_cache']['reused_entries']:,} cached entry accesses over {timed['gram_cache']['requests']} QPs. Synthetic signed-normal Gram checks verify incremental reuse.", '',
        'At each update, rebuild only the current RHS from the native head and unchanged reserve. Match the previous final pairs/duals into currently selected/discovered constraints. Previously known competitor pairs on newly selected controls can seed the current QP; rows no longer selected do not acquire extra protection merely from history. This avoids changing the current problem into permanent protection of every formerly selected control.', '',
        f"On average {timed['mean_inherited_pairs']:.2f} previous pairs are inherited and {timed['mean_filtered_previous']:.2f} filtered per update after the first. Each active-set solve minimizes 0.5||P-P_native||^2 for its current discovered constraints. The 0.5 factor does not change the requested minimizer. EPS, reserve and relation-floor selection remain unchanged. A block principal-pivot solve checks primal/dual/complementarity residuals, followed by a direct primal check using H and the actual FP32 endpoint.", '',
        '## Separate component profile and remaining cost','',
        'The profile is a separate full replay with region synchronization. Core solver wall compares Hildreth coordinate work with QP pivot/linear-solve work; Gram/RHS and reconstruction are explicitly separate. Profile times are not mixed into standalone totals.', '',
        '| Profiled method | Core solver wall (s) | Gram/RHS (s) | Primal reconstruction (s) | Endpoint wall (s) | Control wall (s) | Total wall (s) |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"|Cached Hildreth|{bp['profile_components']['hildreth_coordinate']:.4f}|N/A|in coordinates|{bp['profile_components']['endpoint_forward']:.4f}|{bp['control_wall']:.4f}|{bp['total_wall']:.4f}|",
        f"|Warm active-set QP|{comp['qp_solver']:.4f}|{comp['gram_and_rhs']:.4f}|{comp['primal_reconstruction']:.4f}|{comp['endpoint_forward']:.4f}|{profile['control_wall']:.4f}|{profile['total_wall']:.4f}|",'',
        'The new solver reduces repeated endpoint checks but introduces principal-system solves, working-matrix handling and explicit primal reconstruction. Those costs prevent a net speedup here. No additional tuning, solver replacement, task expansion or A+B combination was attempted after the negative timing result.', '',
        '## Verification and reproduction','',
        '32 independent SciPy NNLS synthetic cases pass before task10. All 629 actual QPs pass KKT and direct-primal checks; maximum direct violation is '+f"{summary['max_direct_primal_violation']:.3g}."+' Cold solutions independently agree. Timed and profiled replays reproduce every validated head/moment/report, and final predictions. Every executed relation floor passes the unchanged original fixed256 endpoint.', '',
        'Authority: `E:/gfg-evidence/final-v2-hotstart-active-qp-v1`. Compact results and per-QP audits are in `results/`; inputs and checkpoint remain external, identified by hashes. `ARTIFACT_MANIFEST.json` hashes source and evidence files.', '',
        'With `E:/gfg-python/governor-v4-bench/Scripts/python.exe`, set `GFG_HOTSTART_ROOT` to a new nonexistent authority, then run this package modules `analysis`, `tests`, `run validate`, `run timed`, `run profile`, `report` using `python -B -m experiments.gfg_final_v2_hotstart_active_qp_v1.<module>`. Report also reads the declared fresh unoptimized baseline from A reference256; it never imports an optimized endpoint into B training. Existing output directories and packaged results are refused.']
    with (HERE/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    files=['PREFLIGHT.json','CONTINUITY.json','SOLVER_PREFLIGHT.json','GRAM_CHECKS.json','SUMMARY.json','validate/COLD_AUDIT.jsonl.gz']
    for mode in ('validate','timed','profile'):files += [f'{mode}/{name}' for name in ('CHECKS.json','STEPS.jsonl.gz','EVALUATION.json')]
    package(HERE,ROOT,files);print('B independent check/package PASS; no net speedup',flush=True)


if __name__=='__main__':main()
