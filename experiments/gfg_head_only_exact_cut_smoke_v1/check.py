"""Independent algebra/affine replay of recorded choices, not new training."""
from .core import *

def main():
    pre=guard();setup();out=ROOT/ARM;r=json.loads((out/'TRAIN_RESULT.json').read_text());cfg=json.loads((out/'CONFIG.json').read_text())
    assert r['status']=='COMPLETED' and r['steps']==210 and cfg['source_hashes']==sources()
    for name,key in [('AUDIT.pt','audit_sha256'),('STEPS.jsonl','log_sha256'),('CONFIG.json','config_sha256')]:assert sha(out/name)==r[key]
    a=torch.load(out/'AUDIT.pt',map_location='cpu',weights_only=False);base=torch.load(LR_ROOT/'lr5e-5'/'AUDIT.pt',map_location='cpu',weights_only=False)
    assert equal(a['initial_optimizer'],base['initial_optimizer']) and torch.equal(a['control_features'],base['control_features'])
    assert tensor_hash(a['heads'][0])==pre['head_initial_sha256'] and tensor_hash(a['heads'][-1])==r['head_sha256']
    assert r['backbone_sha256']==pre['backbone_initial_sha256'] and tensor_hash({'H':a['control_features']})==r['cache_sha256']
    assert all(float(s['step'])==420 for s in a['final_optimizer']['state'].values())
    del base
    d=np.load(DATA/'data.npz');data_audit(d);probes,primary,full=scope(d,16);batches=order(d,2)
    h=a['control_features'].cuda();y=torch.from_numpy(d['y'][probes]).cuda();rows=[json.loads(s) for s in (out/'STEPS.jsonl').read_text().splitlines()]
    assert len(rows)==210 and len(a['heads'])==211 and len(a['cuts'])==len(a['natives'])==210
    checked=0;certificates=0;all_safe=True
    with torch.no_grad():
        for j,row in enumerate(rows):
            assert row['step']==j+1 and row['current_ids']==batches[j].tolist() and row['lr']==LR and row['old_replay']==0
            assert row['seed']==2021*100000+211+j and row['moments_unmodified_by_cut'] and row['bert_frozen_eval']
            b={k:v.cuda() for k,v in a['heads'][j].items()};n={k:v.cuda() for k,v in a['natives'][j].items()};v=row['report']
            assert tensor_hash(b)==row['head_before_sha256'] and tensor_hash(n)==row['native_sha256']
            bm=margins(affine(b,h),y);nm=margins(affine(n,h),y);eligible=bm[:80]>0
            assert bm.cpu().tolist()==v['baseline'] and nm.cpu().tolist()==v['native']
            assert violated(nm,eligible)==v['native_violating'];assert len(v['passes'])<=2
            candidate=n;delta={k:n[k]-b[k] for k in b};z=affine(n,h)
            for number,p in enumerate(v['passes'],1):
                assert p['pass_number']==number and violated(margins(z,y),eligible)
                expected=[]
                for i in range(80):
                    if bool(eligible[i]):
                        t=int(y[i])
                        for c in range(16):
                            if c!=t and bool(z[i,c]>=z[i,t]):expected.append((i,t,c))
                assert expected==[(c['control'],c['target'],c['competitor']) for c in p['certificates']]
                for item,(i,t,c) in zip(p['certificates'],expected):
                    hv=h[i].double();dot=float(torch.dot(hv,delta['weight'][t].double()-delta['weight'][c].double())+delta['bias'][t].double()-delta['bias'][c].double())
                    squared=float(2*(hv.square().sum()+1));scale=dot/squared if dot<0 else 0.
                    assert dot==item['dot_before'] and squared==item['norm2'] and scale==item['coefficient']
                    if scale:
                        delta['weight'][t].add_(h[i],alpha=-scale);delta['weight'][c].add_(h[i],alpha=scale)
                        delta['bias'][t].add_(-scale);delta['bias'][c].add_(scale)
                    after=float(torch.dot(hv,delta['weight'][t].double()-delta['weight'][c].double())+delta['bias'][t].double()-delta['bias'][c].double())
                    assert after==item['dot_after'];certificates+=1
                candidate={k:b[k]+delta[k] for k in b};z=affine(candidate,h);cm=margins(z,y)
                assert cm.cpu().tolist()==p['margins'] and violated(cm,eligible)==p['violating'] and p['safe']==(not p['violating'])
                checked+=160
            assert equal(candidate,a['cuts'][j]);cm=margins(affine(candidate,h),y);skip=bool(violated(cm,eligible))
            assert skip==v['skip'] and (not skip or len(v['passes'])==2)
            assert bool(v['native_violating'])==bool(v['passes']) and v['second_pass']==(len(v['passes'])==2)
            assert v['first_pass_safe']==(v['passes'][0]['safe'] if v['passes'] else None)
            assert v['action']==('CUT_FAILED_SKIP' if skip else 'PROJECTED' if v['passes'] else 'NATIVE')
            execution=b if skip else candidate;assert equal(execution,a['heads'][j+1])
            am=margins(affine(execution,h),y);assert am.cpu().tolist()==v['executed'] and cm.cpu().tolist()==v['cut']
            assert not violated(am,eligible);checked+=480
            assert tensor_hash(candidate)==row['cut_sha256'] and tensor_hash(execution)==row['head_after_sha256']
            for key,st in [('baseline',bm),('native',nm),('cut',cm),('executed',am)]:assert counts(st.cpu().tolist())==v['stats'][key]
            for key in ('native','cut','executed'):assert v['new_gains'][key]==v['stats'][key]['new_correct']-v['stats']['baseline']['new_correct']
            for state,key in ((candidate,'cut_geometry'),(execution,'executed_geometry')):
                g=geometry(b,n,state)
                for k,value in g.items():assert value==v[key][k] or (value is not None and v[key][k] is not None and abs(value-v[key][k])<1e-12)
    ev=json.loads((ROOT/'EVALUATION.json').read_text());assert ev['started_unix']>=r['finished_unix'] and ev['train_result_sha256']==sha(out/'TRAIN_RESULT.json')
    train=ev['result']['training'];ids=np.flatnonzero((d['split']==0)&(d['y']>=8)&(d['y']<16));assert train['indices']==ids.tolist()
    assert train['accuracy16']==float((np.array(train['predictions'])==d['y'][ids]).mean())
    test=ev['result']['test'];assert test['indices']==full.tolist();pred=np.array(test['predictions'])
    assert test['primary']==summarize_predictions(primary,pred[np.isin(full,primary)],d,2)
    assert test['original']==summarize_predictions(full,pred,d,2)
    assert r['final_controls']==rows[-1]['report']['stats']['executed']
    summary=dict(projected_steps=sum(bool(t['report']['native_violating']) for t in rows),first_pass_safe_steps=sum(t['report']['first_pass_safe'] is True for t in rows),
        second_pass_steps=sum(t['report']['second_pass'] for t in rows),skipped_steps=sum(t['report']['skip'] for t in rows))
    assert all(r[k]==n for k,n in summary.items())
    write(ROOT/'CHECKS.json',dict(status='PASS',preflight_tests=25,source_hashes=sources(),old_correct_to_wrong_executed=0,
        reconstructed_endpoint_values=checked,projection_pair_certificates=certificates,initial_moments_preserved=True,
        moments_advanced_once_per_update_including_skips=True,test_after_training=True,summary=summary,
        evaluation_sha256=sha(ROOT/'EVALUATION.json'),limitations='Independent affine/projection replay and metric recomputation, not independent full training.'))
    print('CHECKS PASS',summary,checked,certificates,flush=True)

if __name__=='__main__':main()
