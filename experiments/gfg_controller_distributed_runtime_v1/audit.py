"""Independent artifact verifier: integrity PASS is not equivalence PASS."""
import json
import math
import statistics
import numpy as np
import torch
from .checks import b,HERE,OUT,REPO,old
from .scale import banks,evaluate_logits
from experiments.gfg_controller_gpu_boundary_v1.runtime import gpu_boundary

ALPHAS=(.25,.5,.75,1.)


def score(m):
    s={str(a):{'predicted_correct_count':int((m[j]>0).sum()),'median_predicted_minimum_margin':float(np.median(m[j]))} for j,a in enumerate(ALPHAS)}
    a=max(ALPHAS,key=lambda a:(s[str(a)]['predicted_correct_count'],s[str(a)]['median_predicted_minimum_margin'],-abs(a-1.)))
    return a,s


def difference(x,y):
    equal=np.array_equal(x,y)
    return {'equal':bool(equal),'mismatch_count':int(np.count_nonzero(x!=y)),
        'max_abs':float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))) if x.size else 0.}


def gates():
    results=[];memory=[];communication=[]
    for path in sorted((OUT/'gate').iterdir()):
        r=b.read(path/'RESULT.json');assert b.sha(path/'RAW.npz')==r['raw_sha256']
        raw=np.load(path/'RAW.npz',allow_pickle=False);names=r['parameter_names'];shape=r['parameter_shapes']
        assert sum(math.prod(s) for s in shape.values())==r['parameter_count']
        count=r['world_size'];expected=[sum((math.prod(s)*(i+1)//count-math.prod(s)*i//count)*4 for s in shape.values()) for i in range(count)]
        assert expected==r['rank_parameter_bytes'] and [2*x for x in expected]==r['rank_controller_bytes']
        assert sum(expected)==r['parameter_count']*4
        live=None
        for e in r['events']:
            if e['action']=='materialize':assert live is None;live=e['block']
            else:assert e['block']==live;live=None
        assert live is None and r['peak_block_materialization']==max(e['bytes'] for e in r['events'])
        changes=[];all_checks=[];crossing=[];ref_crossing=[]
        y=np.array(r['targets']);head=names.index('transformer.wte.weight')
        for receipt in r['receipts']:
            t=receipt['t'];prefix=f'{t}_';checks={}
            for kind in ('candidate','delta','parameters'):
                actual={n:torch.from_numpy(raw[f'{t}_{kind}_{i}']) for i,n in enumerate(names)}
                reference={n:torch.from_numpy(raw[f'{t}_reference_{kind}_{i}']) for i,n in enumerate(names)}
                checks[kind]=all(torch.equal(actual[n],reference[n]) for n in names)
                hash_key='post_state' if kind=='parameters' else kind
                stored='post_sha256' if kind=='parameters' else f'{kind}_sha256'
                assert b.phash(actual)==receipt[stored] and b.phash(reference)==receipt['reference_'+stored]
                if not checks[kind]:changes.append({'step':t,'object':kind,'differing_parameters':[n for n in names if not torch.equal(actual[n],reference[n])]})
            checks['moments']=all(np.array_equal(raw[f'{t}_state_{i}_{k}'],raw[f'{t}_reference_state_{i}_{k}']) for i in range(len(names)) for k in ('step','exp_avg','exp_avg_sq'))
            checks['logits']=np.array_equal(raw[prefix+'logits'],raw[prefix+'reference_logits'])
            crossing.append(evaluate_logits(raw[prefix+'logits'],y));ref_crossing.append(evaluate_logits(raw[prefix+'reference_logits'],y))
            checks['alpha']=receipt['alpha']==receipt['reference_alpha'];checks['scores']=receipt['scores']==receipt['reference_scores']
            if prefix+'minima' in raw:
                m=raw[prefix+'minima'];assert np.array_equal(raw[prefix+'local_minima'].min(0),m)
                a,s=score(m);assert a==receipt['alpha'] and s==receipt['scores']
                checks['minima']=np.array_equal(m,raw[prefix+'reference_minima'])
                for j in range(3):checks[f'h{j}']=np.array_equal(raw[prefix+f'h{j}'],raw[prefix+f'reference_h{j}'])
                # Recompute the complete boundary from archived actual state,
                # actual delta, raw shard-produced logits and jet, not report JSON.
                args=[raw[prefix+'pre_logits'],y,raw[f'{t}_before_{head}'],raw[f'{t}_delta_{head}'],*[raw[prefix+f'h{j}'] for j in range(3)]]
                a2,s2,_,m2=gpu_boundary(*[torch.from_numpy(x).cuda() for x in args])
                assert np.array_equal(m,m2.cpu().numpy()) and a2==a and s2==s,'local boundary transport differs from archived all-competitor recomputation'
                changes.append({'step':t,'minima':difference(m,raw[prefix+'reference_minima']),
                    'hidden':[difference(raw[prefix+f'h{j}'],raw[prefix+f'reference_h{j}']) for j in range(3)]})
            assert {k:bool(v) for k,v in checks.items()}==receipt['checks']
            all_checks.append(checks)
        exact=all(all(v.values()) for v in all_checks);assert exact==r['exact']
        results.append({'case':path.name,'byte_exact':exact,'alpha_exact':all(x['alpha'] for x in all_checks),
            'trajectory_exact':all(x['parameters'] and x['logits'] for x in all_checks),
            'capability_accuracy':crossing,'reference_capability_accuracy':ref_crossing,'differences':changes})
        memory.append({'case':path.name,'parameter_bytes_per_virtual_rank':expected,'controller_bytes_per_virtual_rank':r['rank_controller_bytes'],
            'peak_block_materialization':r['peak_block_materialization'],**r['memory']})
        communication.append({'case':path.name,**r['communication'],'actual_network_bytes':0,'actual_collectives':0,
            'emulated_stage_materializations':sum(e['action']=='materialize' for e in r['events'])})
    return results,memory,communication


def scales():
    x,y,control,evaluation=banks();assert not set(control)&set(evaluation) and len(control)==212 and len(evaluation)==317
    jobs=[]
    for p in sorted((OUT/'scale').rglob('RESULT.json')):
        r=b.read(p)
        if r['status'] in ('FAILED','OOM','PROCESS_FAILED'):
            assert 'traceback' in r or 'returncode' in r
            jobs.append({'path':str(p.relative_to(OUT)),'status':r['status'],'stage':r.get('stage'),'completed_updates':len(r.get('records',[]))});continue
        assert r['control_ids']==control and r['evaluation_ids']==evaluation
        assert b.sha(p.parent/'RAW.npz')==r['raw_sha256'];raw=np.load(p.parent/'RAW.npz',allow_pickle=False)
        assert np.array_equal(raw['targets'],y[evaluation,-1].numpy())
        acc=evaluate_logits(raw['0_logits'],raw['targets']);assert acc==r['initial_accuracy']
        first=0 if acc>=.99 else None;first_wall=0. if first==0 else None
        for row in r['records']:
            acc=evaluate_logits(raw[f"{row['t']}_logits"],raw['targets']);assert acc==row['accuracy']
            assert row['alpha'] in ALPHAS and row['wall']>0 and row['update_seconds']>0
            if first is None and acc>=.99:first=row['t'];first_wall=row['wall']
        assert acc==r['final_accuracy']
        jobs.append({'path':str(p.relative_to(OUT)),'status':r['status'],'scale':r['scale'],'phase':r['phase'],'arm':r['arm'],'seed':r['seed'],'lr':r['lr'],
            'parameters':r['parameters'],'completed_updates':len(r['records']),'final_accuracy':acc,'steps_to_99':first,'seconds_to_99':first_wall,
            'median_update_seconds':statistics.median(z['update_seconds'] for z in r['records']) if r['records'] else None,
            'wall':r['wall'],'peak_allocated':r['peak_allocated'],'peak_reserved':r['peak_reserved'],'control_buffer_bytes':r['control_buffer_bytes'],
            'alpha_histogram':{str(a):sum(z['alpha']==a for z in r['records']) for a in ALPHAS}})
    # Selection must be determined entirely by native/fixed calibration, never dynamic.
    for p in (OUT/'scale').rglob('REGIME_SELECTION.json'):
        selection=b.read(p);scale=p.parent.name
        cal=[j for j in jobs if j.get('scale')==scale and j.get('phase')=='calibration']
        rates=sorted(set(j['lr'] for j in cal))
        reached=lambda lr,arm:any(j['lr']==lr and j['arm']==arm and j['status']=='COMPLETED' and j['steps_to_99'] is not None for j in cal)
        safe=next((lr for lr in rates if reached(lr,'native')),None)
        others=[lr for lr in rates if safe is not None and lr>safe and not reached(lr,'native') and reached(lr,'fixed')]
        expected={'SAFE':safe,'NEAR_BOUNDARY':others[0] if others else None,'SEVERE':others[1] if len(others)>1 else None}
        actual={label:selection['rates'][selection[key]] if selection[key] is not None else None for label,key in [('SAFE','safe'),('NEAR_BOUNDARY','near'),('SEVERE','severe')]}
        assert actual==expected
    return jobs


def main():
    b.old.setup();seal=b.read(HERE/'DISTRIBUTED_SEAL.json')
    assert all(b.sha(HERE/n)==sha for n,sha in seal['sources'].items())
    assert all(b.sha(REPO/n)==sha for n,sha in seal['old_files'].items())
    assert old.git('rev-parse',old.prior.TAG)==seal['frozen_tag'] and old.git('rev-parse',old.prior.TAG+'^{}')==seal['frozen_commit']
    results,memory,communication=gates();jobs=scales()
    gloo=[b.read(p) for p in sorted((OUT/'gloo').glob('rank*.json'))]
    audit={'integrity':'PASS','gate_cases':results,'byte_exact_cases':sum(r['byte_exact'] for r in results),'case_count':len(results),
        'emulated_world_sizes':[1,2,3],'real_multi_gpu':'NOT_TESTED','fsdp_probe':b.read(HERE/'FSDP_PROBE.json'),'cpu_gloo':gloo,
        'scale_jobs':jobs,'old_files_unchanged':True,'frozen_identity_unchanged':True,'source_seal_valid':True}
    b.write(HERE/'MEMORY_DISTRIBUTED.json',{'scope':'emulated audit gates; physical peak includes oracle', 'cases':memory})
    b.write(HERE/'COMMUNICATION_DISTRIBUTED.json',{'cases':communication,'cpu_gloo':gloo,'cuda_wire_traffic':'NOT_MEASURED'})
    b.write(HERE/'DISTRIBUTED_AUDIT.json',audit)
    b.write(HERE/'FINAL_AUDIT.json',{'integrity':'PASS','distributed_equivalence':'PASS' if all(r['byte_exact'] for r in results) else 'PARTIAL',
        'real_multi_gpu':'NOT_TESTED','audit_sha256':b.sha(HERE/'DISTRIBUTED_AUDIT.json'),
        'raw_artifacts':{str(p.relative_to(OUT)):b.sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.npz')},
        'scale_summary_sha256':b.sha(HERE/'SCALE_RESULTS.json')})
    print(json.dumps({'integrity':'PASS','exact_cases':audit['byte_exact_cases'],'total':len(results),'scale_jobs':len(jobs)}))


if __name__=='__main__':main()
