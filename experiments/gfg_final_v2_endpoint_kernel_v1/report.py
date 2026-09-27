"""Independent A evidence check, separate tables, and byte-preserving package."""
from .core import *


def records(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]


def package(here,root,files):
    dest=here/'results';dest.mkdir(exist_ok=False);manifest={}
    for name in files:
        src=root/name;out=dest/name;out.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as a,out.open('xb') as b:shutil.copyfileobj(a,b)
        assert sha(out)==sha(src)
        manifest[name]=dict(sha256=sha(out),bytes=out.stat().st_size)
    sources={q.name:sha(q) for q in here.iterdir() if q.is_file() and (q.suffix in ('.py','.md') or q.name=='.gitattributes')}
    external={str(q):dict(sha256=sha(q),bytes=q.stat().st_size) for q in root.rglob('CHECKPOINT.pt')}
    write(here/'ARTIFACT_MANIFEST.json',dict(base=BASE,authority=str(root),public_files=manifest,sources=sources,external_checkpoints=external))


def main():
    guard();inputs();baseline=history();allresults={}
    for arm in ARMS:
        val=read(ROOT/arm/'validate/CHECKS.json');timed=read(ROOT/arm/'timed/CHECKS.json');profile=read(ROOT/arm/'profile/CHECKS.json')
        assert read(ROOT/arm/'validate/NUMERICAL_PREFLIGHT.json')['status']=='PASS'
        vr=records(ROOT/arm/'validate/STEPS.jsonl.gz')
        calls=records(ROOT/arm/'validate/CALLS.jsonl.gz');assert len(vr)==210 and len(calls)==val['endpoint_calls']
        for mode in ('timed','profile'):
            rr=records(ROOT/arm/mode/'STEPS.jsonl.gz');assert len(rr)==210
            for a,b in zip(vr,rr):
                assert all(a[k]==b[k] for k in ('step','head_sha256','moments_sha256','report','floor_violations'))
            assert read(ROOT/arm/mode/'EVALUATION.json')==read(ROOT/arm/'validate/EVALUATION.json')
        assert all(r['floor_violations']==0 for r in vr)
        head=sum(a['head_sha256']==b['head_sha256'] for a,b in zip(vr,baseline))
        decision={k:sum(same_report(a['report'],b['report'])[k] for a,b in zip(vr,baseline)) for k in timed['control_matches']}
        assert head==timed['head_matches'] and decision==timed['control_matches']
        total_sign=sum(r['sign_differences'] for r in calls);floor=sum(r['reference_floor_violations'] or 0 for r in calls)
        assert total_sign==val['sign_differences'] and floor==val['fixed256_executed_floor_violations']
        selected_overlap=[];pair_overlap=[];sweep_changes=[]
        for a,b in zip(vr,baseline):
            x=set(a['report']['selected_controls']);y=set(b['report']['selected_controls'])
            selected_overlap.append(len(x&y)/len(x|y) if x|y else 1.)
            x={tuple(z) for z in a['report']['final_active']};y={tuple(z) for z in b['report']['final_active']}
            pair_overlap.append(len(x&y)/len(x|y) if x|y else 1.)
            sweep_changes.append(a['report']['sweep_count']-b['report']['sweep_count'])
        allresults[arm]=dict(timed=timed,profile=profile,validation=val,
            corrected_cached_control_identity=decision,head_matches=head,selected_jaccard_mean=float(np.mean(selected_overlap)),
            active_pair_jaccard_mean=float(np.mean(pair_overlap)),sweep_delta_min=min(sweep_changes),sweep_delta_max=max(sweep_changes),
            reference_floor_steps=[r['step'] for r in calls if r['reference_floor_violations']],
            numerical_classification='byte-identical on this task' if head==210 and total_sign==0 else 'new fixed numerical implementation')
    # All four baddbmm tilings happened to share one trajectory on this task.
    br=records(ROOT/'batch256/validate/STEPS.jsonl.gz')
    for arm in ('batch512','batch1024','batch2048'):
        rr=records(ROOT/arm/'validate/STEPS.jsonl.gz')
        assert all(a['head_sha256']==b['head_sha256'] and a['report']==b['report'] for a,b in zip(br,rr))
    write(ROOT/'SUMMARY.json',dict(status='PASS',arms=allresults,old_evidence_unchanged=True,
        baseline_comparison_note='Initial validation diagnostic historical_control_identity used inherited v3 reports (safe-count lower bounds). This final checker and timed/profile replays recompute against the actual cached full-count log. Raw validation outputs were preserved; no controller trajectory was changed.'))
    lines=['# A: endpoint engineering, independent of QP actuator B','',
        'Base `f357dfdb9d13759dc07a5703d6d11301a3adb946`. FewRel task10, 210 updates only; no Banking77. All previous experiments remain unchanged.','',
        '**Preallocated loop256 preserves the complete trajectory. Batched endpoints are a new fixed numerical implementation.**','',
        'One standalone replay per arm, fixed order shown, on the same RTX 4060 Ti. No best-of selection or confidence interval. Setup, padding and actual commit/recheck are included. External observers, hashes, serialization and final test are excluded.','',
        '| Implementation | Tile | GEMM API calls | Tile matrix products | Padded rows | Control wall (s) | Total algorithm wall (s) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in allresults.items():
        t=r['timed'];lines.append(f"|{arm}|{t['tile']}|{t['gemm_api_calls']:,}|{t['tile_matrix_products']:,}|{t['padded_rows']:,}|{t['control_wall']:.4f}|{t['total_wall']:.4f}|")
    lines+=['','GEMM API calls count Python addmm/baddbmm invocations, not physical GPU kernel launches. A baddbmm call processes multiple constituent tile matrices. No claim of one hardware launch per API call is made.','',
        'The optimized variants pad/copy the immutable feature bank once and preallocate the output. There is no per-tile new_zeros/copy/cat in their Python forward. Library-internal workspaces are not claimed eliminated. Hildreth, reserves, EPS, K, selection and stop rules are unchanged.','',
        '## Numerical and capability results','',
        'Sign differences compare each candidate head with original fixed256 on the same features, including receiving, native, sweep and commit checks. They count observations, not distinct controls. Floor columns distinguish the arm arithmetic from the original arithmetic.','',
        '| Implementation | Sign differences | Executed old sign differences | Own floor violations | Fixed256 observed floor violations | Identical heads / 210 | Old / new / all macro (%) |',
        '|---|---:|---:|---:|---:|---:|---|']
    for arm,r in allresults.items():
        v=r['validation'];m=r['timed']['metrics'];metric=' / '.join(f'{100*m[k]:.4f}' for k in ('macro_old','macro_new','macro_all'))
        lines.append(f"|{arm}|{v['sign_differences']}|{v['executed_old_sign_differences']}|{v['floor_violations']}|{v['fixed256_executed_floor_violations']}|{r['head_matches']}|{metric}|")
    lines+=['','All batched sizes share the same observed trajectory: 1,521 sweeps, 28 changed final test predictions, and head identity only through step29. Their own floor checks pass, but the original fixed256 observer detects one relation-step floor violation at step198. This cannot be reported as zero violations under both numerical contracts or as a byte-identical speedup. No floor was relaxed to hide it.','',
        'Buffer256 matches all 210 heads, moments, deficits, selections, active pairs, primal/dual sweep hashes and final predictions. Both floor checks and all same-head sign comparisons are zero.','',
        '| Batched control comparison to cached baseline | Result |','|---|---:|',
        f"|Mean selected-control Jaccard|{allresults['batch256']['selected_jaccard_mean']:.6f}|",
        f"|Mean active-pair Jaccard|{allresults['batch256']['active_pair_jaccard_mean']:.6f}|",
        f"|Per-update sweep-count change range|{allresults['batch256']['sweep_delta_min']} to {allresults['batch256']['sweep_delta_max']}|",'',
        'Small aggregate metric changes do not prove equivalent control behavior. Exact mismatches and overlaps are preserved in SUMMARY and per-step logs. No screening thresholds, floor adjustments or numerical-path workarounds were tried.','',
        '## Separate instrumented endpoint timing','',
        'These are independent profiled 210-update replays with synchronized region timers; their endpoint wall is not subtracted from the standalone total above. Each replay matches its own validated trajectory.','',
        '| Implementation | Endpoint wall (s) | Profiled control wall (s) | Profiled algorithm wall (s) |','|---|---:|---:|---:|']
    for arm,r in allresults.items():
        pz=r['profile'];lines.append(f"|{arm}|{pz['profile_components']['endpoint_forward']:.4f}|{pz['control_wall']:.4f}|{pz['total_wall']:.4f}|")
    lines+=['','## Evidence and reproduction','',
        'Create-only authority: `E:/gfg-evidence/final-v2-endpoint-kernel-v1`. Compact evidence is in `results/`; checkpoint hashes and artifact/source hashes are in `ARTIFACT_MANIFEST.json`. Inputs/features remain external and read-only.','',
        'Use `E:/gfg-python/governor-v4-bench/Scripts/python.exe`, set `GFG_ENDPOINT_ROOT` to a new nonexistent directory, then run this package with `run validate`, `run timed`, `run profile`, and `report` via `python -B -m experiments.gfg_final_v2_endpoint_kernel_v1.<module>`. Existing output directories are refused. The report package also refuses an existing results directory.','',
        'Bookkeeping correction: the first validation pass inherited a helper comparing historical control fields with v3 lower-bound-count reports. This produced misleading sweep/selection identity flags for the unchanged baseline. The final checker recomputes identities from stored raw reports against the correct cached full-count log; standalone/profile runs use that corrected source. No training result was rerun or altered to obtain a better outcome. Raw validation evidence is preserved.','',
        'A and B are reported separately. No optimized A endpoint is used in the B actuator.']
    with (HERE/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    files=['PREFLIGHT.json','SUMMARY.json']
    for arm in ARMS:
        for mode in ('validate','timed','profile'):
            files += [f'{arm}/{mode}/{name}' for name in ('CHECKS.json','STEPS.jsonl.gz','EVALUATION.json')]
        files += [f'{arm}/validate/{name}' for name in ('CALLS.jsonl.gz','NUMERICAL_PREFLIGHT.json')]
    package(HERE,ROOT,files);print('A independent check/package PASS',flush=True)


if __name__=='__main__':main()
