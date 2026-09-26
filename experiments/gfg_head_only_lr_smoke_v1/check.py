"""Independent record and affine-endpoint recalculation; no further training."""
from .core import *

def main():
    pre=guard();setup();d=np.load(DATA/'data.npz');data_audit(d);p,primary,full=scope(d,16);batches=order(d,2)
    barrier=json.loads((ROOT/'TRAINING_COMPLETE.json').read_text());ev=json.loads((ROOT/'EVALUATION.json').read_text())
    assert ev['started_unix']>=barrier['created_unix'] and ev['training_barrier_sha256']==sha(ROOT/'TRAINING_COMPLETE.json')
    count=0;summary={}
    baseaudit=torch.load(BASE_ROOT/'head_only_exact_monitor'/'AUDIT.pt',map_location='cpu',weights_only=False)
    for arm,lr in ARMS.items():
        path=ROOT/arm;r=json.loads((path/'TRAIN_RESULT.json').read_text());cfg=json.loads((path/'CONFIG.json').read_text())
        assert r['status']=='COMPLETED' and r['steps']==210 and r['no_test_inference'] and r['finished_unix']<=barrier['created_unix']
        assert cfg['source_hashes']==sources() and cfg['lr']==r['lr']==lr
        assert sha(path/'TRAIN_RESULT.json')==barrier['result_hashes'][arm]
        for file,key in [('AUDIT.pt','audit_sha256'),('STEPS.jsonl','log_sha256'),('CONFIG.json','config_sha256')]:assert sha(path/file)==r[key]
        a=torch.load(path/'AUDIT.pt',map_location='cpu',weights_only=False)
        assert len(a['heads'])==211 and equal(a['initial_optimizer']['state'],baseaudit['initial_optimizer']['state'])
        assert a['initial_optimizer']['param_groups'][0]['lr']==lr
        assert tensor_hash(a['heads'][0])==pre['head_initial_sha256'] and tensor_hash(a['heads'][-1])==r['head_sha256']
        assert r['backbone_sha256']==pre['backbone_initial_sha256'] and torch.equal(a['control_features'],baseaudit['control_features'])
        for s in a['final_optimizer']['state'].values():assert float(s['step'])==420
        h=a['control_features'].cuda();y=torch.from_numpy(d['y'][p]).cuda();rows=[json.loads(t) for t in (path/'STEPS.jsonl').read_text().splitlines()];assert len(rows)==210
        for j,row in enumerate(rows):
            assert row['current_ids']==batches[j].tolist() and row['step']==j+1 and row['lr']==lr
            assert row['seed']==2021*100000+211+j and row['alpha']==1 and row['old_replay']==0
            assert row['monitor_readonly'] and row['bert_frozen_eval'] and row['executed_candidate_exact']
            assert row['head_before_sha256']==tensor_hash(a['heads'][j]) and row['head_after_sha256']==tensor_hash(a['heads'][j+1])
            v=row['observation'];unsafe=[]
            for head,key in ((a['heads'][j],'baseline'),(a['heads'][j+1],'candidate')):
                z=nn.functional.linear(h,head['weight'].cuda(),head['bias'].cuda())[:,:16];m=margins(z,y).cpu().numpy()
                assert np.array_equal(m,np.array(v[key],dtype=np.float32))
                m=m.astype(np.float64).reshape(16,10);cap=(m>0).mean(1);q=np.quantile(m,.1,axis=1,method='linear')
                assert np.array_equal(cap,v[key+'_stats']['capability']) and np.array_equal(q,v[key+'_stats']['q10'])
                unsafe.append({k for k in range(8) if cap[k]<.9 or (v['anchor'][k]>=1e-6 and q[k]<.5*v['anchor'][k])})
            assert sorted(unsafe[0])==v['already_unsafe_before_update'] and sorted(unsafe[1]-unsafe[0])==v['newly_threatened_by_this_update']
            count+=160
        assert r['final_controls']==ev['arms'][arm]['controls']['final']
        assert all(r['losses'][str(k)]==rows[k-1]['loss'] for k in (1,50,100,150,210))
        summary[arm]=dict(lr=lr,losses=r['losses'],controls=ev['arms'][arm]['controls'],training_wall=r['training_wall'])
    trainids=np.flatnonzero((d['split']==0)&(d['y']>=8)&(d['y']<16))
    for arm,v in ev['arms'].items():
        t=v['training'];assert t['indices']==trainids.tolist();yy=d['y'][trainids]
        assert t['accuracy_16class']==float((np.array(t['predictions'])==yy).mean())
        assert t['accuracy_new_only']==float((np.array(t['new_only_predictions'])==yy).mean())
        test=v['test'];assert test['indices']==full.tolist();pred=np.array(test['predictions'])
        assert test['primary']==summarize_predictions(primary,pred[np.isin(full,primary)],d,2)
        assert test['original']==summarize_predictions(full,pred,d,2)
    write(ROOT/'CHECKS.json',dict(status='PASS',preflight_tests=18,new_arms=3,updates_per_arm=210,source_hashes=sources(),
        original_evidence_unchanged=True,control_endpoint_comparisons=count,all_test_after_all_training=True,
        baseline_retrained=False,initial_moments_preserved=True,arm_checks=summary,evaluation_sha256=sha(ROOT/'EVALUATION.json'),
        boundary='Cached-affine replay and metric recomputation; no independent full retraining.'))
    print('CHECKS PASS',count,flush=True)

if __name__=='__main__':main()
