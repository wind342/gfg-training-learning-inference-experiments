"""Independent full-bank production-controller replay of all 210 updates."""
from ..core import *


def rows(path):
    with gzip.open(path,'rt',encoding='utf8') as f:
        return [json.loads(line) for line in f]


def main():
    p.setup()
    pre=read(ROOT/'PREFLIGHT.json')
    assert all(sha(REPO/q)==v for q,v in pre['sources'].items())
    assert all(sha(q)==v for q,v in pre['inputs'].items())
    rr=rows(ROOT/'STEPS.jsonl.gz');cc=rows(ROOT/'CALLS.jsonl.gz')
    assert len(rr)==210
    d=np.load(DATA/'data.npz')
    F=torch.load(INPUT/'FEATURES.pt',weights_only=False,map_location='cpu')['features'].cuda()
    saved=torch.load(INPUT/'v2_210/task9/CHECKPOINT.pt',weights_only=False,map_location='cpu')
    head,shadow,opt=make_state(saved)
    h,y,ids,n=bank(d,F,10);K=np.asarray(read(ROOT/'CONFIG.json')['K'])
    sync();started=time.perf_counter()
    replay=[]
    for step,current in enumerate(order(d,10),1):
        ce_grad(head,F[current],torch.from_numpy(d['y'][current]).cuda(),80)
        nn.utils.clip_grad_norm_(head.parameters(),1.)
        before=p.snapshot(head);propose(head,shadow,opt);native=p.snapshot(shadow)
        executed,_,reference=reference_projection(before,native,h,y,n,K,ids[:n],seen=80)
        head.load_state_dict(executed)
        record=rr[step-1]
        assert record['current_ids']==current.tolist()
        assert head_hash(head)==record['head_sha256']
        assert moments_hash(opt)==record['moments_sha256']
        assert old.opt_counter(opt)==1890+step
        for key in record['report']:
            assert record['report'][key]==reference[key],(step,key)
        assert np.all(np.asarray(reference['executed_counts'])>=K)
        calls=[c for c in cc if c['step']==step]
        assert len(calls)==3+reference['sweep_count']
        for c in calls:
            mask=np.unpackbits(np.frombuffer(bytes.fromhex(c['certified_bits']),dtype=np.uint8))[:n]
            assert int(mask.sum())==c['certified']
            assert not c['false_safe'] and not c['sign_mismatch'] and c['count_match']
            assert c['max_margin_difference']==0.
        assert calls[-1]['counts']==reference['executed_counts']
        replay.append(dict(step=step,head=head_hash(head),moments=moments_hash(opt),all_projection_details_identical=True))
        if step%30==0:
            print('INDEPENDENT FULL REPLAY',step,flush=True)
    sync();replay_wall=time.perf_counter()-started
    final=torch.load(ROOT/'CHECKPOINT.pt',weights_only=False,map_location='cuda')
    assert equal(p.snapshot(head),final['head']) and equal(opt.state_dict(),final['optimizer'])
    test=torch.load(INPUT/'TEST_FEATURES.pt',weights_only=False,map_location='cpu')
    ev=read(ROOT/'EVALUATION.json')
    assert test['indices']==ev['indices']==scope(d,80)[1].tolist()
    with torch.no_grad():
        pred=fixed_affine(p.snapshot(head),test['features'].cuda(),80).argmax(1).cpu().tolist()
        ordinary=p.affine(p.snapshot(head),test['features'].cuda(),80).argmax(1).cpu().tolist()
    assert pred==ev['predictions'] and output_metrics(d,pred,10)==ev['metrics']
    write(ROOT/'INDEPENDENT_REPLAY.json',dict(status='PASS',steps=replay,wall=replay_wall))
    report=dict(status='PASS',scheduled_updates=210,independent_full_controller_replay=210,
        all_executed_heads_moments_and_projection_details_identical=True,
        per_candidate_full_observer_calls=len(cc),observed_false_safe=0,frontier_logit_margin_difference=0.,
        relation_floor_checks=210*72,floor_violations=0,
        actual_final_head_and_test_predictions_recomputed=True,
        same_head_fixed_vs_original_test_prediction_differences=sum(a!=b for a,b in zip(pred,ordinary)),
        original_historical_trajectory_identical=read(ROOT/'RESULTS.json')['historical_head_identical_steps']==210,
        source_hashes_unchanged=True,input_hashes_unchanged=True,
        independent_replay_wall=replay_wall,
        scope='Numerical fixed-tile repair validated against full fixed-tile production controller; not a claim of original full-batch FP32 trajectory identity or formal hardware proof.')
    write(ROOT/'CHECKS.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
