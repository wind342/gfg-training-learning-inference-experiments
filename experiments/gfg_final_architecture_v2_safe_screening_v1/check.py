"""Read-only post-run ledger, identity, and saved-classifier verification."""
from .core import *
from .run import source_hashes, guard


def load_rows(path):
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def main():
    p.setup(); guard()
    pre = read(ROOT/'PREFLIGHT.json')
    assert pre['sources'] == source_hashes()
    assert all(sha(q)==v for q,v in pre['inputs'].items())
    rr = load_rows(ROOT/'STEPS.jsonl.gz')
    cc = load_rows(ROOT/'CALLS.jsonl.gz')
    result = read(ROOT/'RESULTS.json')
    cfg = read(ROOT/'CONFIG.json')
    K = np.asarray(cfg['K'])
    assert len(rr) == result['steps'] == 210
    d = np.load(DATA/'data.npz')
    assert cfg['bank_ids'] == old.task_bank(d, 10)[0].tolist()
    for index, row in enumerate(rr, 1):
        assert row['step'] == index and row['optimizer_step'] == 1890+index
        assert row['current_ids'] == order(d, 10)[index-1].tolist()
        assert np.all(np.asarray(row['report']['executed_counts']) >= K)
        calls = [r for r in cc if r['step'] == index]
        assert [r['point'] for r in calls] == ['receiving','native']+['sweep'+str(s) for s in range(1,row['report']['sweep_count']+1)]+['commit']
        for call in calls:
            mask = np.unpackbits(np.frombuffer(bytes.fromhex(call['certified_bits']),dtype=np.uint8))[:cfg['n']]
            assert int(mask.sum()) == call['certified']
            assert call['frontier'] == cfg['n']-call['certified']
            assert call['frontier'] <= call['exact_old_rows'] <= cfg['n']
            assert not call['false_safe'] and not call['sign_mismatch'] and call['count_match']
        assert calls[-1]['candidate_sha256'] == row['head_sha256']
        assert calls[-1]['counts'] == row['report']['executed_counts']
    assert result['screened_exact_rows'] == sum(r['exact_old_rows'] for r in cc)
    assert result['original_full_rows'] == len(cc)*cfg['n']
    checkpoint = torch.load(ROOT/'CHECKPOINT.pt',map_location='cuda',weights_only=False)
    assert tensor_hash(checkpoint['head']) == rr[-1]['head_sha256']
    assert all(int(v['step']) == 2100 for v in checkpoint['optimizer']['state'].values())
    moments = {str(i)+':'+k:v for i,s in checkpoint['optimizer']['state'].items() for k,v in s.items() if torch.is_tensor(v)}
    assert tensor_hash(moments) == rr[-1]['moments_sha256']
    test = torch.load(INPUT/'TEST_FEATURES.pt',map_location='cpu',weights_only=False)
    ev = read(ROOT/'EVALUATION.json')
    assert ev['indices'] == test['indices'] == scope(d,80)[1].tolist()
    pred = p.affine(checkpoint['head'],test['features'].cuda(),80).argmax(1).cpu().tolist()
    assert pred == ev['predictions'] and output_metrics(d,pred,10) == ev['metrics']
    original = read(INPUT/'EVALUATION_TASK10.json')['arms']['v2_210']
    assert ev['historical_predictions_identical'] == (pred == original['predictions'])
    checks = dict(status='PASS', steps=210, candidate_calls=len(cc), relation_step_floor_checks=210*72,
                  full_observer_scope='Every receiving/native/projected/commit candidate was audited during execution; post-check verifies ledgers and saved classifier, not a second full training replay.',
                  source_identity=True,input_identity=True,final_classifier_recomputed=True,
                  all_certified_bits_counted=True,all_moment_counters=True,
                  scientific_equivalence_gate=result['gate'])
    write(ROOT/'CHECKS.json', checks)
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
