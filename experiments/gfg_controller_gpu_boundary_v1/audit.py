"""Independent reconstruction of decisions, accuracy, identities and medians."""
import math
import statistics
import numpy as np
from . import validate as run

b=run.b;HERE=run.HERE;OUT=run.OUT


def stats(a,z):
    return {'exact':bool(np.array_equal(a,z)),'close':bool(np.allclose(a,z,rtol=1e-6,atol=1e-8)),
        'different':int((a!=z).sum()),'max_abs':float(abs(a-z).max())}


def choose(c):
    m=np.stack([(((c['g0']+a*c['first_gap'])+a*a*c['joint_rotation'])+a*a*c['hidden_curvature']).min(1) for a in run.ALPHAS])
    s={str(a):{'predicted_correct_count':int((m[j]>0).sum()),'median_predicted_minimum_margin':float(np.median(m[j]))} for j,a in enumerate(run.ALPHAS)}
    a=max(run.ALPHAS,key=lambda a:(s[str(a)]['predicted_correct_count'],s[str(a)]['median_predicted_minimum_margin'],-abs(a-1.)))
    return a,s,m


def numerical(f,prefix,receipt):
    c={side:{k:f[prefix+side+'_'+k] for k in run.KEYS} for side in ('gpu','cpu')}
    g,gs,gm=choose(c['gpu']);a,ss,sm=choose(c['cpu'])
    assert all(stats(c['cpu'][k],c['gpu'][k])==receipt['coordinates'][k] for k in run.KEYS)
    assert np.array_equal(gm,f[prefix+'gpu_minima']) and np.array_equal(sm,f[prefix+'cpu_minima'])
    assert stats(sm,gm)==receipt['minima'] and (gs==ss)==receipt['scores_exact']
    assert (g==a)==receipt['alpha_exact'] and g==receipt['gpu_alpha'] and a==receipt['reference_alpha']
    return g,gs


def main():
    seal=b.read(HERE/'SEAL.json')
    for n,d in seal['sources'].items():assert b.sha(HERE/n)==d,n
    for n,d in seal['old_files'].items():assert b.sha(run.REPO/n)==d,n
    assert not run.old.git('diff',run.BASE,'--name-only','--diff-filter=CDMRTUXB')
    assert run.old.git('rev-parse',run.old.prior.TAG)==seal['frozen_tag']
    assert run.old.git('rev-parse',run.old.prior.TAG+'^{}')==seal['frozen_commit']==run.old.prior.FROZEN
    sy=b.read(HERE/'SYNTHETIC.json');assert len(sy['cases'])==30 and sy['zero_tie']=='PASS'
    for r in sy['cases']:
        assert b.sha(OUT/r['artifact'])==r['artifact_sha256']
        with np.load(OUT/r['artifact'],allow_pickle=False) as f:numerical(f,'',r)
    assert sy['all_alpha_exact']==all(r['alpha_exact'] for r in sy['cases'])
    assert sy['all_coordinates_exact']==all(all(v['exact'] for v in r['coordinates'].values()) for r in sy['cases'])
    rows={r['entry_id']:r for r in b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs']}
    rng=b.read(run.old.HERE/'PHASE1_SEAL.json')['source_rng_protocol_sha256']
    results=b.read(HERE/'RESULTS.json');seen=set();updates=0;decisions=0;badcoords=0;badchoices=0;badscore=0;maxerror=0.;allclose=True
    for r in results['checked']:
        entry,ci=r['entry'],r['config'];key=(entry,ci);assert key not in seen;seen.add(key)
        path=OUT/'checked'/entry/str(ci);rec=b.read(path/'TRAJECTORY.json')
        assert rec['entry']==entry and rec['config']==ci and b.sha(path/'RAW.npz')==rec['raw_sha256']
        ref=run.old.prior.record('dynamic','search',entry,ci)
        with np.load(path/'RAW.npz',allow_pickle=False) as f:
            logits=f['logits'];target=f['targets'];assert len(logits)==len(rec['steps'])+1
            comp=logits.copy();comp[:,np.arange(len(target)),target]=-np.inf
            gaps=logits[:,np.arange(len(target)),target].astype(np.float64)-comp.max(2).astype(np.float64)
            acc=(gaps>0).mean(1);assert np.isfinite(gaps).all()
            assert b.tensor_sha256(logits[0])==rec['initial_logits_sha256']==ref['initial_logits_sha256']
            assert (len(rec['steps'])==0 if acc[0]>=.99 else (all(acc[:-1]<.99) and (acc[-1]>=.99 or len(rec['steps'])==200)))
            chain=None;j=0;alpha=1.
            for t,s in enumerate(rec['steps'],1):
                assert s['t']==t and s['accuracy']==float(acc[t]) and s['logits_sha256']==b.tensor_sha256(logits[t])
                assert s['seed']==b.old.seed_for(rows[entry],t-1,rng)
                assert s['lr']==run.old.prior.RATES[ci]*(.1+.9*.5*(1+math.cos(math.pi*(t-1)/199)))
                old=ref['steps'][t-1] if t<=len(ref['steps']) else None
                assert s['old_post_exact']==(old is not None and s['post_parameters_sha256']==old['post_parameters_sha256'])
                assert s['old_logits_exact']==(old is not None and s['logits_sha256']==old['logits_sha256'])
                if (t-1)%5==0:
                    j+=1;decisions+=1;c=s['numerical_check'];alpha,ss=numerical(f,f'{j}_',c)
                    assert ss==s['details']['scores']
                    badcoords+=not all(v['exact'] for v in c['coordinates'].values());badchoices+=not c['alpha_exact'];badscore+=not c['scores_exact']
                    maxerror=max(maxerror,*(v['max_abs'] for v in c['coordinates'].values()))
                    allclose=allclose and all(v['close'] for v in c['coordinates'].values())
                else:assert s['numerical_check'] is None and 'scores' not in s['details']
                assert s['details']['alpha']==alpha
                fact={k:v for k,v in s.items() if k!='sha256'};assert fact['previous_sha256']==chain
                chain=b.identity(fact);assert chain==s['sha256']
            exact=len(rec['steps'])==len(ref['steps']) and all(s['old_post_exact'] and s['old_logits_exact'] for s in rec['steps']) and rec['final_state_sha256']==ref['final_state_sha256']
            assert exact==rec['trajectory_exact']==r['trajectory_exact'] and len(rec['steps'])==r['updates']
            updates+=len(rec['steps'])
    assert seen=={(e,c) for e in rows for c in range(3)}
    def check_entry(r):
        ref=run.old.prior.record('dynamic','search',r['entry'],r['config'])
        exact=len(r['steps'])==len(ref['steps']) and r['final_state_sha256']==ref['final_state_sha256'] and all(s['alpha']==z['details']['alpha'] and s['accuracy']==z['accuracy'] for s,z in zip(r['steps'],ref['steps']))
        assert r['trajectory_exact']==exact and r['has_per_step_model_hash'] is False and r['cadence']==5
        assert r['control_buffer_bytes']==8*100352 and r['probe_count']==r['evaluation_count']==212
        if r['backend']=='gpu':assert r['boundary_decision_d2h_bytes']==72*sum((s['t']-1)%5==0 for s in r['steps'])
        expected=ref['steps'][-1]['logits_sha256'] if ref['steps'] else ref['initial_logits_sha256']
        if exact:assert r['final_logits_sha256']==expected
        return exact
    assert {(r['entry'],r['config']) for r in results['entry']}==seen
    for summary in results['entry']:
        r=b.read(OUT/'entry'/summary['entry']/f"{summary['config']}.json")
        assert check_entry(r)==summary['trajectory_exact'] and len(r['steps'])==summary['updates']
    timings={}
    for summary in results['timing']:
        mode=summary['backend'];tt,ct=summary['tile'];rep=summary['rep']
        r=b.read(OUT/'timing'/f'{mode}-{tt}-{ct}-{rep}.json');assert check_entry(r)==summary['exact']
        assert r['checked_wall_seconds']==summary['wall'] and r['warmup']==(rep==0)
        if rep:timings.setdefault(f'{mode}-{tt}-{ct}',[]).append(r)
    assert len(results['timing'])==16 and all(len(x)==3 for x in timings.values())
    medians={key:{k:statistics.median(r[k] for r in values) for k in ('checked_wall_seconds','peak_allocated','peak_reserved')} for key,values in timings.items()}
    manifest={str(p.relative_to(OUT)):b.sha(p) for p in OUT.rglob('*') if p.is_file()}
    report={'status':'PASS','checked_trajectories':36,'standalone_trajectories':36,'checked_updates':updates,
        'decisions':decisions,'coordinate_nonexact_decisions':badcoords,'score_nonexact_decisions':badscore,
        'same_state_alpha_differences':badchoices,'coordinate_max_abs_error':maxerror,'coordinates_all_within_declared_tolerance':allclose,
        'synthetic_all_coordinates_exact':sy['all_coordinates_exact'],'synthetic_all_alpha_exact':sy['all_alpha_exact'],
        'all_historical_trajectories_exact':all(r['trajectory_exact'] for group in ('checked','entry') for r in results[group]),
        'timing_medians':medians,'all_timing_trajectories_exact':all(r['exact'] for r in results['timing']),
        'seal_sha256':b.sha(HERE/'SEAL.json'),'results_sha256':b.sha(HERE/'RESULTS.json'),'raw_manifest':manifest,
        'scope':'Audit PASS means records are consistent. Numerical non-equivalence is reported, not relabeled as PASS.'}
    b.write(HERE/'AUDIT.json',report);print({k:v for k,v in report.items() if k!='raw_manifest'})


if __name__=='__main__':main()
