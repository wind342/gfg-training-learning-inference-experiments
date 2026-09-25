"""Frozen external audit, numerical candidates are not silently certified."""
import math
import statistics
import numpy as np
import torch
from .evidence import HERE,OUT,REPO,read,write,sha,git,TAG,setup

ALPHAS=(.25,.5,.75,1.)


def choose(m):
    s={str(a):{'predicted_correct_count':int((m[j]>0).sum()),'median_predicted_minimum_margin':float(np.median(m[j]))} for j,a in enumerate(ALPHAS)}
    a=max(ALPHAS,key=lambda a:(s[str(a)]['predicted_correct_count'],s[str(a)]['median_predicted_minimum_margin'],-abs(a-1.)))
    return a,s


def accuracy(z,y):
    q=z.copy();q[np.arange(len(y)),y]=-np.inf
    return float((z[np.arange(len(y)),y].astype(float)-q.max(1).astype(float)>0).mean())


def main():
    setup();seal=read(OUT/'ARCHITECTURE_SEAL.json')
    assert all(sha(REPO/n)==v for n,v in seal['old_files'].items())
    assert all(sha(HERE/n)==v for n,v in seal['sources'].items())
    assert git('rev-parse',TAG)==seal['tag_object'] and git('rev-parse',TAG+'^{}')==seal['tag_commit']
    gate_results=[];identities=[];clipping=[];stage_events=[]
    for p in sorted((OUT/'gates').glob('*/RESULT.json')):
        r=read(p);assert sha(p.parent/'RAW.npz')==r['raw_sha256'];raw=np.load(p.parent/'RAW.npz');count=len(r['parameter_names']);records=[]
        assert r['identity_before']==r['identity_after'] and r['canonical_count']==r['optimizer_parameter_count']
        for row in r['rows']:
            t=row['step'];check={}
            for key in ('candidate','delta','parameters','gradients'):
                check[key]=all(np.array_equal(raw[f'{t}_{key}_{i}'],raw[f'{t}_ref_{key}_{i}']) for i in range(count))
            check['moments']=all(np.array_equal(raw[f'{t}_state_{k}_{i}'],raw[f'{t}_ref_state_{k}_{i}']) for i in range(count) for k in ('step','exp_avg','exp_avg_sq'))
            check['norm']=np.array_equal(raw[f'{t}_actual_norm'],raw[f'{t}_reference_norm'])
            for side in ('actual','reference'):
                expected=math.sqrt(sum(float(np.square(raw[f'{t}_{side}_raw_gradient_{i}'].astype(float)).sum()) for i in range(count)))
                observed=float(raw[f'{t}_{side}_norm']);clipping.append({'case':r['label'],'step':t,'side':side,'fp64_recomputed_norm':expected,'observed_norm':observed,'relative_error':abs(expected-observed)/max(expected,1e-30)})
            check['alpha']=row['actual']['alpha']==row['reference_alpha'];check['scores']=row['actual'].get('scores')==row['reference_scores']
            check['logits']=np.array_equal(raw[f'{t}_logits'],raw[f'{t}_ref_logits'])
            if f'{t}_minima' in raw:
                for j in range(3):check[f'h{j}']=np.array_equal(raw[f'{t}_h{j}'],raw[f'{t}_ref_h{j}'])
                m=raw[f'{t}_minima'];a,s=choose(m);assert a==row['actual']['alpha'] and s==row['actual']['scores']
                check['minima']=np.array_equal(m,raw[f'{t}_ref_minima'])
            assert check==row['checks'];records.append({k:bool(v) for k,v in check.items()})
        all_exact=all(all(x.values()) for x in records);assert all_exact==r['all_exact']
        live=None
        for event in r['events']:
            if event['action']=='enter':assert live is None;live=event['stage']
            else:assert live==event['stage'];live=None
        assert live is None and all(e['data_collectives']==(0 if e['world_size']==1 else 1) and e['unpack_views_share_storage'] for e in r['packed'])
        gate_results.append({'label':r['label'],'backend':r['backend'],'fsdp_world_size':r['fsdp_world_size'],'all_exact':all_exact,
            'alpha_exact':r['alpha_exact'],'trajectory_exact':r['trajectory_exact'],'records':records})
        identities.append({'case':r['label'],'count':r['canonical_count'],'aliases':[e for e in r['identity_after'] if len(e['aliases'])>1]});stage_events.append({'case':r['label'],'lifetime':r['events'],'packed':r['packed']})
    from experiments.gfg_controller_gpu_boundary_v1.runtime import gpu_boundary
    synthetic=[]
    for p in sorted((OUT/'synthetic').glob('*/RESULT.json')):
        r=read(p);assert sha(p.parent/'RAW.npz')==r['raw_sha256'];raw=np.load(p.parent/'RAW.npz')
        args=[torch.from_numpy(raw[k]).cuda() for k in ('z','y','w','dw','h','hp','hpp')];ref=gpu_boundary(*args,audit=True)
        assert np.array_equal(ref[3].cpu().numpy(),raw['ref_minima'])
        for c in r['cases']:
            key=f"{c['mode']}-{c['dedup']}";coords=raw[key+'_coordinates'];m=raw[key+'_minima'];computed=[]
            for a in ALPHAS:
                gap=((coords[0]+a*coords[1])+a*a*coords[2])+a*a*coords[3];gap[np.arange(len(raw['y'])),raw['y']]=np.inf;computed.append(gap.min(1))
            assert np.array_equal(np.stack(computed),m)
            assert bool(np.array_equal(coords,raw['ref_coordinates']))==c['coordinates_exact']
            assert bool(np.array_equal(m,raw['ref_minima']))==c['minima_exact']
            assert (choose(m)[0]==choose(raw['ref_minima'])[0])==c['alpha_exact']
        synthetic.append(r)
    jobs=[]
    for p in sorted((OUT/'scale').rglob('RESULT.json')):
        r=read(p)
        if 'raw_sha256' not in r:
            jobs.append({'path':str(p.relative_to(OUT)),'status':r['status']});continue
        assert sha(p.parent/'RAW.npz')==r['raw_sha256'];raw=np.load(p.parent/'RAW.npz');first=None
        for row in r['rows']:
            assert accuracy(raw[f"{row['t']}_logits"],raw['targets'])==row['accuracy']
            if first is None and row['accuracy']>=.99:first=row
        jobs.append({'path':str(p.relative_to(OUT)),'status':r['status'],'scale':r['scale'],'phase':r['phase'],'arm':r['arm'],'seed':r['seed'],'lr':r['lr'],
            'steps_to99':first['t'] if first else None,'wall_to99':first['wall'] if first else None,'final_accuracy':r.get('final_accuracy'),
            'updates':len(r['rows']),'peak_allocated':r.get('peak_allocated'),'wall':r.get('wall')})
    for p in (OUT/'scale').glob('*/REGIME_SELECTION.json'):
        selection=read(p);scale=p.parent.name;rates=selection['rates'];cal=[j for j in jobs if j.get('scale')==scale and j.get('phase')=='calibration']
        hit=lambda i,arm:any(j['arm']==arm and j['lr']==rates[i] and j['status']=='COMPLETED' and j['steps_to99'] is not None for j in cal)
        safe=next((i for i in range(len(rates)) if hit(i,'native')),None)
        others=[i for i in range(len(rates)) if safe is not None and i>safe and not hit(i,'native') and hit(i,'fixed')]
        assert [selection['safe'],selection['near'],selection['severe']]==[safe,others[0] if others else None,others[1] if len(others)>1 else None]
    collective=[]
    for p in (OUT/'collectives').glob('*/RESULT.json'):
        r=read(p);assert sha(p.parent/'RAW.npz')==r['raw_sha256'];raw=np.load(p.parent/'RAW.npz')
        assert np.array_equal(raw['minima'],raw['reference_minima'])==r['minima_exact']
        assert np.array_equal(raw['global_norm2'],raw['reference_norm2'])==r['norm_exact'];collective.append(r)
    failures=[{'file':str(p.relative_to(OUT)),**read(p)} for p in (OUT/'failures').glob('*.json')]
    write(OUT/'TIED_WEIGHT_AUDIT.json',{'status':'PASS' if identities else 'NOT_TESTED','cases':identities})
    write(OUT/'GLOBAL_CLIP_AUDIT.json',{'status':'WORLD_SIZE1_AND_CPU_COLLECTIVES_ONLY','checks':clipping,'collectives':collective})
    write(OUT/'STAGE_PACKED_COMMUNICATION.json',{'cases':stage_events})
    write(OUT/'BLOCKWISE_LIFETIME_TRACE.json',{'cases':stage_events})
    write(OUT/'WORKSPACE_PLANNER_RESULTS.json',{'cases':synthetic})
    write(OUT/'TARGET_DEDUP_RESULTS.json',{'cases':[{k:r[k] for k in ('h','seed','cases')} for r in synthetic]})
    for mode,name in [('scalarized','SCALARIZATION_AUDIT.json'),('streaming','FLASH_BOUNDARY_AUDIT.json')]:
        write(OUT/name,{'status':'CANDIDATE','trajectory_cases':[r for r in gate_results if r['backend']==mode],
            'synthetic_cases':[{'h':r['h'],'seed':r['seed'],'cases':[c for c in r['cases'] if c['mode']==mode]} for r in synthetic],
            'note':'Streaming chunks write bounded intermediates to HBM; no fused Triton kernel certification' if mode=='streaming' else 'Changed arithmetic, not default certified backend'})
    write(OUT/'FINAL_AUDIT.json',{'integrity':'PASS','old_files_unchanged':True,'frozen_tag_unchanged':True,'source_seal_valid':True,'gates':gate_results,'scale_jobs':jobs,'failures':failures,
        'artifacts':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.npz')},'real_multigpu':read(OUT/'REAL_MULTIGPU_AUDIT.json')['real_multigpu']})
    print('INDEPENDENT_AUDIT_PASS',len(gate_results),len(jobs),flush=True)


if __name__=='__main__':main()
