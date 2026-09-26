"""Independent record/identity checks, not a native training rerun."""
from collections import Counter
from .core import *

def restore(snapshot):
    s=GroupSentinel(GroupMap(np.repeat(np.arange(16),10)))
    s.state=list(snapshot['states'])
    for field,key in [('anchor','anchor_q10'),('capabilities','capabilities'),('q10','q10'),('level','levels')]:
        getattr(s,field)[:]=snapshot[key]
    return s

def verify(arm,pre,d):
    root=ROOT/arm;c=json.loads((root/'CONFIG.json').read_text());r=json.loads((root/'RESULT.json').read_text())
    assert r['status']=='COMPLETED' and r['steps']==210
    assert c['source_hashes']==pre['source_hashes']==sources() and c['base']==BASE
    assert c['preflight_sha256']==sha(ROOT/'PREFLIGHT.json') and c['checkpoint_sha256']==CKSHA and c['data_sha256']==DATA_SHA
    assert r['log_sha256']==sha(root/'STEPS.jsonl') and r['config_sha256']==sha(root/'CONFIG.json') and r['checkpoint_sha256']==sha(root/'final.pt')
    rows=[json.loads(l) for l in (root/'STEPS.jsonl').read_text().splitlines()];assert len(rows)==210
    batches=order(d,2);previous=None;hit=total=changehit=changes=0;eligibility_losses=[];timing=Counter()
    initial=restore(r['initial_state']);initial.last_step=210
    for i,row in enumerate(rows,1):
        assert row['offset']==i and row['step']==210+i and row['current_ids']==batches[i-1].tolist()
        assert row['dropout_seed']==2021*100000+210+i and row['old_replay']==0
        assert row['detection']==(arm!='native' and i in SCHEDULE)
        timing.update(row['timing'])
        if not row['detection']:
            assert row['alpha']==1 and 'actual' not in row
            continue
        before=row['history_before_refresh'];prev=r['initial_state'] if previous is None else previous['history_after']
        for key in ('states','capabilities','q10','anchor_q10','levels'):assert before[key]==prev[key]
        s=restore(before);s.last_step=row['step'];base=np.asarray(row['baseline']);native=np.asarray(row['native_predicted'])
        assert base.shape==(160,) and native.shape==(4,160)
        s.measured(row['step'],torch.tensor(base));recv=row['receiving_history']
        for key in ('states','capabilities','q10','anchor_q10','levels'):assert s.snapshot(row['step'])[key]==recv[key]
        protected=eligible(s);assert protected==row['protected'];threat=threats(native[-1],s,protected);assert threat==row['threatened']
        losses=[g for g in range(8) if before['states'][g]=='PROTECTED' and g not in protected]
        assert losses==row['lost_protected_since_last_audit'];eligibility_losses+=losses
        expected=[] if previous is None else np.flatnonzero((np.array(previous['actual'])>0)&(base<=0)).tolist()
        assert expected==row['since_last_audit_correct_to_wrong']
        projected=arm=='projection' and bool(threat)
        direction=np.asarray(row['projected_predicted']) if projected else native
        if projected:
            expected_groups=sorted(threat,key=lambda g:(float(native[-1,g*10:g*10+10].min()),g))
            assert [cut['group'] for cut in row['cuts']]==expected_groups
            assert row['second_prediction_delta_sha256']==row['projected_delta_sha256']
            for cut in row['cuts']:
                before_dot=cut['dot_before'];g2=cut['gradient_norm_squared'];coef=before_dot/(g2+EPS) if before_dot<0 else 0.
                assert coef==cut['coefficient'] and bool(coef)==cut['changed'] and g2>=0
                expected_dot=before_dot-coef*g2
                assert abs(cut['dot_after']-expected_dot)<=max(1e-7,2e-5*abs(before_dot))
                probe=cut['probe'];idx=probe['probe_index'];group=cut['group']
                assert group*10<=idx<(group+1)*10 and probe['target']==group and probe['competitor']!=group
                assert np.isclose(base[idx],base[group*10:group*10+10].min(),atol=2e-5,rtol=1e-5)
                assert probe['selected_probe_native_crossing']==bool(base[idx]>0 and native[-1,idx]<=0)
        else:assert row['cuts']==[] and 'projected_predicted' not in row
        a,no,scores=choose(direction,s,protected);assert a==row['alpha'] and no==row['no_safe'] and scores==row['safety_scores']
        chosen=direction[ALPHAS.index(a)];assert chosen.tolist()==row['executed_predicted']
        assert row['moments_identity_versions_unchanged'] is True
        for name,values in [('receiving',base),('native',native[-1]),('projected',direction[-1]),('executed',chosen)]:
            assert aggregates(values,s.mapping,base)==row['aggregates'][name]
        actual=np.array(row['actual']);s.measured(row['step'],torch.tensor(actual));after=row['history_after']
        for key in ('states','capabilities','q10','anchor_q10','levels'):assert s.snapshot(row['step'])[key]==after[key]
        agreement=(chosen>0)==(actual>0);transition=(base>0)!=(actual>0)
        hit+=int(agreement.sum());total+=160;changes+=int(transition.sum());changehit+=int((agreement&transition).sum())
        previous=row
    probes,primary,full=scope(d,16);m=r['metrics'];pred=np.array(m['predictions'])
    assert m['indices']==full.tolist() and m['original']==summarize_predictions(full,pred,d,2)
    assert m['primary']==summarize_predictions(primary,pred[np.isin(full,primary)],d,2)
    measured=[x for x in rows if x['detection']];projected=[x for x in measured if 'projected_predicted' in x]
    assert len(measured)==(0 if arm=='native' else 21)
    stats=dict(status='PASS',primary=m['primary'],original=m['original'],old_replay=0,
        detection_sites=len(measured),full_predictor_calls=len(measured)+len(projected),
        native_threat_steps=sum(bool(x['threatened']) for x in measured),threat_count_histogram=dict(Counter(str(len(x['threatened'])) for x in measured)),
        projection_steps=len(projected),actual_cut_steps=sum(any(c['changed'] for c in x['cuts']) for x in projected),
        projected_alpha1_safe=sum(not x['safety_scores']['1.0'] for x in projected),
        projected_alpha1_safe_fraction=sum(not x['safety_scores']['1.0'] for x in projected)/len(projected) if projected else None,
        projection_still_shrink=sum(x['alpha']<1 for x in projected),shrink_steps=sum(x['alpha']<1 for x in measured),
        no_safe_steps=sum(x['no_safe'] for x in measured),reasons=dict(Counter(x['reason'] for x in measured)),
        boundary_hits=hit,boundary_total=total,boundary_agreement=hit/total if total else None,
        transition_hits=changehit,actual_transitions=changes,
        measured_old_correct_to_wrong=sum(int(((np.array(x['baseline'])[:80]>0)&(np.array(x['actual'])[:80]<=0)).sum()) for x in measured),
        interval_old_correct_to_wrong=sum(sum(j<80 for j in x['since_last_audit_correct_to_wrong']) for x in measured),
        lost_protected_at_refresh=dict(Counter(str(g) for g in eligibility_losses)),
        protected_count_histogram=dict(Counter(str(len(x['protected'])) for x in measured)),
        update_seconds=sum(x['update_seconds'] for x in rows),timing=dict(timing),
        **{k:r[k] for k in ('training_wall','wall','load_seconds','evaluation_seconds','peak_allocated','peak_reserved')},
        hashes={f:sha(root/f) for f in ('CONFIG.json','RESULT.json','STEPS.jsonl','final.pt')})
    return stats

def main():
    unchanged();torch.set_num_threads(2)
    pre=json.loads((ROOT/'PREFLIGHT.json').read_text());assert pre['status']=='PASS'
    assert sha(CHECKPOINT)==CKSHA and sha(DATA/'data.npz')==DATA_SHA
    d=np.load(DATA/'data.npz');arms={a:verify(a,pre,d) for a in ARMS}
    report=dict(status='PASS',base=BASE,arms=arms,source_hashes=sources(),
        boundary='Record-derived consistency and projection algebra certificates; not independent native training or full gradient replay.')
    write(ROOT/'CHECKS.json',report);write(HERE/'results/CHECKS.json',report)
    print('CHECKS_PASS',json.dumps({a:arms[a]['primary']['macro_all'] for a in ARMS}),flush=True)

if __name__=='__main__':main()
