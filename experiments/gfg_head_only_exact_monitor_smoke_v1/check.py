"""Separate record, cached-affine and native final-checkpoint checker, not retraining."""
from .core import *
from experiments.gfg_governor_fewrel_v1.run import evaluate

def main():
    unchanged();pre=json.loads((ROOT/'PREFLIGHT.json').read_text());assert pre['source_hashes']==sources()
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    d=np.load(DATA/'data.npz');isolation=data_audit(d);p,primary,full=scope(d,16);batches=order(d,2)
    results={};records={};audits={};agreements=0;transitions=0;stats={}
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for arm in ARMS:
        path=ROOT/arm;r=json.loads((path/'RESULT.json').read_text());cfg=json.loads((path/'CONFIG.json').read_text())
        assert r['status']=='COMPLETED' and r['steps']==210 and cfg['source_hashes']==sources()
        for file,key in [('STEPS.jsonl','log_sha256'),('AUDIT.pt','audit_sha256'),('CONFIG.json','config_sha256')]:assert sha(path/file)==r[key]
        rows=[json.loads(t) for t in (path/'STEPS.jsonl').read_text().splitlines()];assert len(rows)==210
        a=torch.load(path/'AUDIT.pt',map_location='cpu',weights_only=False);assert len(a['heads'])==211
        assert tensor_hash({'H':a['control_features']})==r['cache_sha256']
        assert tensor_hash(a['heads'][0])==pre['head_initial_sha256'] and tensor_hash(a['heads'][-1])==r['head_sha256']
        h=a['control_features'].cuda();labels=torch.from_numpy(d['y'][p]).cuda();new=[];already=[]
        for j,row in enumerate(rows):
            assert row['current_ids']==batches[j].tolist() and row['offset']==j+1 and row['seed']==2021*100000+211+j
            assert row['alpha']==1 and row['old_replay']==0 and row['monitor_readonly'] and row['bert_eval'] and row['bert_no_grad']
            assert row['head_before_sha256']==tensor_hash(a['heads'][j]) and row['head_after_sha256']==tensor_hash(a['heads'][j+1])
            if arm==ARMS[1]:
                v=row['observation'];recomputed=[]
                for head,key in ((a['heads'][j],'baseline'),(a['heads'][j+1],'candidate')):
                    z=nn.functional.linear(h,head['weight'].cuda(),head['bias'].cuda())[:,:16]
                    m=margins(z,labels).cpu().numpy();assert np.array_equal(m,np.asarray(v[key],dtype=np.float32));recomputed.append(m)
                    cap=(m.reshape(16,10)>0).mean(1);q=np.quantile(m.astype(np.float64).reshape(16,10),.1,axis=1,method='linear')
                    assert np.array_equal(cap,v[key+'_stats']['capability']) and np.array_equal(q,v[key+'_stats']['q10'])
                def unsafe(m):
                    mm=m.astype(np.float64).reshape(16,10);cap=(mm>0).mean(1);q=np.quantile(mm,.1,axis=1,method='linear')
                    return {k for k in range(8) if cap[k]<.9 or (v['anchor'][k]>=1e-6 and q[k]<.5*v['anchor'][k])}
                u,w=map(unsafe,recomputed)
                assert sorted(u)==v['already_unsafe_before_update'] and sorted(w-u)==v['newly_threatened_by_this_update']
                assert sorted((w-u)&set(v['protected']))==v['newly_threatened_protected']
                assert row['executed_candidate_exact'];agreements+=160
                transitions+=int(np.sum((recomputed[0]>0)!=(recomputed[1]>0)))
                new.append(v['newly_threatened_by_this_update']);already.append(v['already_unsafe_before_update'])
        pred=np.array(r['metrics']['predictions']);assert r['metrics']['indices']==full.tolist()
        assert equal(r['metrics']['primary'],summarize_predictions(primary,pred[np.isin(full,primary)],d,2))
        assert equal(r['metrics']['original'],summarize_predictions(full,pred,d,2))
        results[arm]=r;records[arm]=rows;audits[arm]=a
        if arm==ARMS[1]:stats=dict(newly_unsafe_steps=sum(bool(v) for v in new),newly_unsafe_group_occurrences=sum(map(len,new)),
            already_unsafe_steps=sum(bool(v) for v in already),already_unsafe_group_occurrences=sum(map(len,already)),
            newly_unsafe_per_group={str(k):sum(k in v for v in new) for k in range(8)},
            protected_newly_unsafe_steps=sum(bool(row['observation']['newly_threatened_protected']) for row in rows),
            correct_to_wrong=int(sum(np.sum((np.array(row['observation']['baseline'])[:80]>0)&(np.array(row['observation']['candidate'])[:80]<=0)) for row in rows)))
    aa,bb=[audits[a] for a in ARMS];ra,rb=[results[a] for a in ARMS]
    assert equal(aa['heads'],bb['heads']) and equal(aa['initial_optimizer'],bb['initial_optimizer']) and equal(aa['final_optimizer'],bb['final_optimizer'])
    assert equal(ra['metrics'],rb['metrics']) and equal(ra['initial'],rb['initial'])
    for a,b in zip(records[ARMS[0]],records[ARMS[1]]):
        for k in ('loss','head_grad_norm','head_before_sha256','head_after_sha256','optimizer_sha256','current_ids'):assert a[k]==b[k]
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);model=FrozenRelationModel().cuda();model.load_state_dict(saved['model']);freeze(model);del saved
    assert tensor_hash(model.bert.state_dict())==pre['backbone_initial_sha256']
    x=torch.from_numpy(d['x']);y=torch.from_numpy(d['y'][:,None]);native_matches=0
    for arm in ARMS:
        model.head.load_state_dict(audits[arm]['heads'][-1]);model.seen=16
        ev=evaluate(model,x,y,full);assert ev['predictions']==results[arm]['metrics']['predictions'];native_matches+=len(full)
    out=dict(status='PASS',arms_identical_every_step=True,head_moments_and_final_states_identical=True,old_replay=0,
        cached_endpoint_logits_margins_exact=True,control_endpoint_comparisons=agreements,boundary_transitions=transitions,
        native_final_prediction_matches=native_matches,isolation=isolation['status'],source_hashes=sources(),stats=stats,
        performance=dict(monitor_per_update=rb['monitor_seconds']/210,compute_ratio=rb['compute_update_seconds']/ra['compute_update_seconds'],
            inclusive_update_ratio=rb['inclusive_update_seconds']/ra['inclusive_update_seconds'],training_wall_ratio=rb['training_wall']/ra['training_wall']),
        limitations='Recorded native optimizer trajectory, cached-affine replay and final native inference; no independent full retraining.')
    write(ROOT/'CHECKS.json',out);print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
