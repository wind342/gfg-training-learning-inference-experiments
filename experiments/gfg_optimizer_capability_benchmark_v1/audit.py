"""Read-only independent boundary/selection/ledger arithmetic audit."""
from pathlib import Path
import json
import hashlib
import statistics
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path('E:/gfg-evidence/optimizer-capability-benchmark-v1')
METHODS = ('adamw','sophia','shampoo','sls','controller')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def filehash(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def hits(r,th):
    if r['initial']['accuracy']>=th: return 0
    for s in r['steps']:
        if s['fact']['evaluation']['accuracy']>=th: return s['fact']['t']
    return None


def check_eval(e,y):
    logits=np.array(e['logits'],dtype=np.float64)
    target=logits[np.arange(len(y)),y].copy()
    logits[np.arange(len(y)),y]=-np.inf
    correct=(target-logits.max(1)>0)
    assert correct.tolist()==e['correct']
    assert float(correct.mean())==e['accuracy']


def check(path):
    r=read(path); y=np.array(r['target_ids'])
    assert filehash(path.parent/'FINAL.pt')==r['final_file_sha256']
    check_eval(r['initial'],y)
    previous=None
    for t,s in enumerate(r['steps'],1):
        f=s['fact']; assert f['t']==t and f['previous_sha256']==previous
        assert digest(f)==s['sha256']; previous=s['sha256']
        assert read(path.parent/('decision-%04d.json'%t))==f['decision']
        check_eval(f['evaluation'],y)
        if r['method']=='controller' and (t-1)%5==0 and '/search/' in path.as_posix():
            with np.load(path.parent/('coords-%04d.npz'%t),allow_pickle=False) as c:
                rankings=[]
                for a in (.25,.5,.75,1.):
                    gaps=c['g0']+a*c['first_gap']+a*a*(c['joint_rotation']+c['hidden_curvature'])
                    margins=gaps.min(1)
                    score={'predicted_correct_count':int((margins>0).sum()),'median_predicted_minimum_margin':float(np.median(margins))}
                    assert score==f['decision']['details']['scores'][str(a)]
                    rankings.append((score['predicted_correct_count'],score['median_predicted_minimum_margin'],-abs(a-1),a))
                assert max(rankings)[-1]==f['decision']['details']['alpha']
    end=(r['steps'][-1]['fact']['evaluation'] if r['steps'] else r['initial'])['accuracy']
    assert end>=.99 or len(r['steps'])==200
    assert all(s['fact']['evaluation']['accuracy']<.99 for s in r['steps'][:-1])
    return r


def main():
    seal=read(HERE/'FREEZE_SEAL.json')
    for name,h in seal['files'].items(): assert filehash(HERE/name)==h
    chosen=read(HERE/'LOO_SELECTION.json')['entries']
    search={}; formal={}; files={}; count=0; steps=0
    for p in sorted(OUT.glob('*/*/*/*/TRAJECTORY.json')):
        r=check(p); count+=1; steps+=len(r['steps'])
        files[str(p.relative_to(OUT))]=filehash(p)
        for child in p.parent.iterdir():
            if child.is_file(): files[str(child.relative_to(OUT))]=filehash(child)
        assert r['protocol_sha256']==seal['files']['PROTOCOL_FREEZE.md']
        key=(r['entry_id'],r['method'],r['config'])
        if p.parts[-5]=='search': search[key]=r
        else: formal.setdefault(key,[]).append(r)
    assert len(search)==240 and sum(map(len,formal.values()))==180
    ids=sorted(chosen)
    for entry in ids:
        initials=[r['initial_parameters_sha256'] for k,r in search.items() if k[0]==entry]
        assert len(set(initials))==1
        for m in METHODS:
            ranking=[]
            for ci in range(4):
                costs=[]
                for th in (.99,.95):
                    v=[hits(search[(other,m,ci)],th) for other in ids if other!=entry]
                    costs.append(statistics.mean(200 if x is None else x for x in v if x!=0))
                ranking.append([*costs,ci])
            assert ranking==chosen[entry][m]['ranks']
            ci=min(ranking)[-1]; assert ci==chosen[entry][m]['config']
            reference=search[(entry,m,ci)]
            reps=formal[(entry,m,ci)]; assert len(reps)==3
            for r in reps:
                assert r['native_replay_exact'] and r['initial']==reference['initial']
                assert r['final_state_sha256']==reference['final_state_sha256']
                assert [s['fact'] for s in r['steps']]==[s['fact'] for s in reference['steps']]
    summaries={}; details=[]
    for th in (.95,.99):
        table=[]
        for m in METHODS:
            eligible=[]
            for entry in ids:
                ci=chosen[entry][m]['config']; r=search[(entry,m,ci)]; hit=hits(r,th)
                times=[]; walls=[]
                for rep in formal[(entry,m,ci)]:
                    n=len(rep['steps']) if hit is None else hit
                    times.append(0. if hit==0 else rep['construction_seconds']+sum(s['compute_seconds'] for s in rep['steps'][:n]))
                d={'entry':entry,'method':m,'threshold':th,'config':ci,'lr':r['lr'],'initial_accuracy':r['initial']['accuracy'],
                    'first_step':hit,'restricted_step':200 if hit is None else hit,'median_compute_seconds':statistics.median(times)}
                details.append(d)
                if hit!=0: eligible.append(d)
            table.append({'method':m,'eligible':len(eligible),'reached':sum(d['first_step'] is not None for d in eligible),
                'mean_restricted_steps':statistics.mean(d['restricted_step'] for d in eligible),
                'mean_restricted_compute_seconds':statistics.mean(d['median_compute_seconds'] for d in eligible)})
        summaries[str(th)]=table
    peaks={k:max(r[k] for rs in formal.values() for r in rs) for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_process_rss_bytes')}
    result={'status':'PASS','trajectories_checked':count,'native_steps_in_ledgers':steps,'summaries':summaries,'per_history':details,'memory_peaks':peaks,
        'total_artifact_bytes':sum((OUT/p).stat().st_size for p in files), 'limitations':['Historical trained-weight continuation with fresh optimizers and zero decay, not original receiving state.',
        'Control and evaluation targets coincide; no held-out-target claim.','Four LR trials, LOO history selection; not best possible tuning.',
        'First hit, not sustained capability. Timing on one small model and one GPU.']}
    for name,obj in [('RESULTS.json',result),('ARTIFACT_MANIFEST.json',{'root':str(OUT),'sha256':files})]:
        with (HERE/name).open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
    lines=['# Optimizer capability benchmark results','','All counts below EXCLUDE histories already at the stated threshold. Unreached runs cost 200 updates; times are construction + synchronized update compute, median of three exact native replays.','']
    for th,table in summaries.items():
        lines += ['## First '+str(round(float(th)*100))+'%','','| Method | Reached / eligible | Mean restricted updates | Mean restricted compute seconds |','|---|---:|---:|---:|']
        for r in table: lines.append(f"| {r['method']} | {r['reached']}/{r['eligible']} | {r['mean_restricted_steps']:.3f} | {r['mean_restricted_compute_seconds']:.6f} |")
        lines.append('')
    lines+=['## Scope','',*['- '+s for s in result['limitations']],'','## Audit','',f'{count} trajectory ledgers, {steps} recorded updates; all selected trajectories repeated three times with exact native hashes, boundaries and choices.','',f'Memory peaks (bytes): {peaks}.',f"Raw artifacts: {result['total_artifact_bytes']} bytes.",'']
    with (HERE/'RESULTS.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines))
    print(json.dumps({'status':'PASS','summaries':summaries,'memory_peaks':peaks,'bytes':result['total_artifact_bytes']},indent=2))


if __name__=='__main__': main()
