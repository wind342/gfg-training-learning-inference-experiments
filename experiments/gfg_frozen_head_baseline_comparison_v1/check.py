"""Independent record/metric audit plus deterministic first-task reexecution."""
from .core import *
from .run import train_task
import argparse

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def rows(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(line) for line in f]

@torch.no_grad()
def metrics_check(d,F):
    test=torch.load(ROOT/'TEST_FEATURES.pt',weights_only=False,map_location='cpu');TF=test['features'].cuda()
    R=torch.load(ROOT/'ranpac_rp/RANDOM_PROJECTION.pt',weights_only=False,map_location='cuda')
    comparisons=0;historical={}
    for task in range(2,11):
        ev=read(ROOT/f'EVALUATION_TASK{task}.json');ids=np.asarray(ev['indices']);assert ids.tolist()==scope(d,task*8)[1].tolist()
        for arm,value in ev['arms'].items():
            pred=np.asarray(value['predictions']);yy=d['y'][ids]
            per=[float((pred[yy==g]==g).mean()) for g in range(task*8)]
            assert per==list(value['metrics']['per_relation'].values())
            assert float(np.mean(per))==value['metrics']['macro_all']
            if arm in METHODS:
                ck=torch.load(ROOT/arm/f'task{task}/CHECKPOINT.pt',weights_only=False,map_location='cuda')
                logits=nn.functional.linear(TF[:len(ids)],ck['head']['weight'],ck['head']['bias'])
                calculated=logits[:,:task*8].argmax(1).cpu().tolist()
                bank_ids,n=old.task_bank(d,task);target=torch.from_numpy(d['y'][bank_ids]).cuda()
                m=margins(nn.functional.linear(F[bank_ids],ck['head']['weight'],ck['head']['bias'])[:,:task*8],target)[:n].cpu().numpy()
                count=counts(m,d['y'][bank_ids[:n]],8*(task-1)).tolist()
                assert count==read(ROOT/arm/f'task{task}/TRAIN_RESULT.json')['end_counts']
            else:
                ck=torch.load(ROOT/arm/f'task{task}/CLASSIFIER.pt',weights_only=False,map_location='cuda');calculated=[]
                for start in range(0,len(ids),256):
                    features=TF[start:min(start+256,len(ids))]
                    H=torch.relu(features@R) if arm=='ranpac_rp' else features
                    calculated+=(H.double()@ck['weight'])[:,:task*8].argmax(1).cpu().tolist()
            assert calculated==pred.tolist(),(arm,task,'saved classifier/prediction mismatch');comparisons+=1
        hist=read(REPO/f'experiments/gfg_final_architecture_v2_ten_task/results/evaluation_task{task}/EVALUATION.json')
        historical[str(task)]={arm:dict(same_predictions=ev['arms'][arm]['predictions']==hist['arms'][arm]['predictions'],
            same_head=ev['arms'][arm]['head_sha256']==hist['arms'][arm]['head_sha256'],
            all_delta=ev['arms'][arm]['metrics']['macro_all']-hist['arms'][arm]['metrics']['macro_all']) for arm in ('v2_210','v2_630')}
    return dict(classifier_prediction_checks=comparisons,historical_v2=historical)

def audit_logs(d):
    checked=0;summary={}
    initial=torch.load(ROOT/'INITIAL.pt',weights_only=False,map_location='cpu')
    for arm,(method,epochs) in METHODS.items():
        previous=tensor_hash(initial['head']);step=210;total=0
        for task in range(2,11):
            folder=ROOT/arm/f'task{task}';cfg=read(folder/'CONFIG.json');result=read(folder/'TRAIN_RESULT.json')
            rr=rows(folder/'STEPS.jsonl.gz');assert len(rr)==epochs*210
            assert cfg['head_start']==previous and cfg['optimizer_step']==step
            bank_ids,n=old.task_bank(d,task);valid=set(map(int,bank_ids[:n]));K=np.asarray(cfg['K'])
            gen=np.random.default_rng();gen.bit_generator.state=cfg['rng_start']
            for index,r in enumerate(rr):
                current=order(d,task)[index%210].tolist();assert r['current_ids']==current
                expected=[]
                if method in ('er','agem'):expected=gen.choice(bank_ids[:n],16,replace=False).tolist()
                elif method=='gem':
                    for j in range(task-1):expected+=gen.choice(bank_ids[:n][d['y'][bank_ids[:n]]//8==j],16,replace=False).tolist()
                assert r['old_sample_ids']==expected and set(expected)<=valid
                assert r['old_backward_examples']==len(expected)
                C=np.asarray(r['counts']);assert r['deficits']==np.maximum(K-C,0).tolist()
                if method=='v2':assert np.all(C>=K)
                step+=1;assert r['optimizer_step']==step;checked+=1
            ck=torch.load(folder/'CHECKPOINT.pt',weights_only=False,map_location='cpu')
            assert ck['rng']==gen.bit_generator.state
            previous=tensor_hash(ck['head']);assert previous==rr[-1]['head_sha256']==result['head_end']
            assert all(int(s['step'])==step for s in ck['optimizer']['state'].values())
            assert sha(folder/'CHECKPOINT.pt')==result['checkpoint_sha256'] and sha(folder/'STEPS.jsonl.gz')==result['log_sha256']
            assert result['old_backward_examples']==sum(r['old_backward_examples'] for r in rr)
            total+=len(rr)
        summary[arm]=dict(steps=total,continuity=True,sampling_recomputed=True)
    for arm in ('ridge','ranpac_rp'):
        arrived=set()
        for task in range(1,11):
            r=read(ROOT/arm/f'task{task}/TRAIN_RESULT.json')
            assert r['training_indices']==input_ids(d,task).tolist() and r['released_controls']==newly_released(d,task).tolist()
            ids=r['training_indices']+r['released_controls'];assert not arrived.intersection(ids);arrived.update(ids)
            assert len(arrived)==r['arrived_count']
            assert r['lambda_value']==r['cv']['grid'][int(np.argmin(r['cv']['mse']))]
        stats=torch.load(ROOT/arm/'STATISTICS.pt',weights_only=False,map_location='cpu')
        assert stats['arrived']==sorted(arrived)
    return dict(steps=checked,arms=summary,statistical_arrivals=True)

def replay(d,F):
    initial=torch.load(ROOT/'INITIAL.pt',weights_only=False,map_location='cpu');out={}
    for arm in METHODS:
        state=make_state(initial);gen=np.random.default_rng(2021+9000000);folder=ROOT/'REPLAY'/arm/'task2'
        train_task(arm,state,gen,F,d,2,output=folder)
        original=rows(ROOT/arm/'task2/STEPS.jsonl.gz');again=rows(folder/'STEPS.jsonl.gz')
        keys=('head_sha256','moments_sha256','current_ids','old_sample_ids','counts','deficits','optimizer_step')
        assert len(original)==len(again) and all(all(a[k]==b[k] for k in keys) for a,b in zip(original,again))
        out[arm]=dict(steps=len(again),all_step_heads_moments_and_counts_bit_identical=True)
    write(ROOT/'REPLAY_CHECKS.json',dict(status='PASS',arms=out))
    return out

def report():
    evaluations={t:read(ROOT/f'EVALUATION_TASK{t}.json') for t in range(2,11)};arms={}
    for arm in evaluations[10]['arms']:
        tr=[read(ROOT/arm/f'task{t}/TRAIN_RESULT.json') for t in range(2,11)]
        end=evaluations[10]['arms'][arm]['metrics']
        arms[arm]=dict(task10={k:end['macro_'+k] for k in ('old','new','all')},
            mean={k:float(np.mean([evaluations[t]['arms'][arm]['metrics']['macro_'+k] for t in range(2,11)])) for k in ('old','new','all')},
            algorithm_wall=sum(r['algorithm_wall'] for r in tr),old_feature_reads=sum(r['old_feature_reads'] for r in tr),
            old_backward_examples=sum(r['old_backward_examples'] for r in tr),
            observer_wall=sum(r.get('observer_wall',0.) for r in tr),
            relation_floor_violations=sum(r.get('relation_floor_violations',sum(v>0 for v in r.get('deficits',[]))) for r in tr),
            peak_cuda_bytes=max(r['peak_cuda_bytes'] for r in tr))
    write(ROOT/'RESULTS.json',dict(status='COMPLETED',arms=arms,cache=read(ROOT/'CACHE.json')))
    print(json.dumps(arms,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    p.setup();guards(read(ROOT/'PREFLIGHT.json'));assert read(ROOT/'COMPLETED.json')['status']=='COMPLETED'
    d=np.load(DATA/'data.npz');F=torch.load(ROOT/'FEATURES.pt',weights_only=False,map_location='cpu')['features'].cuda()
    logs=audit_logs(d);metrics=metrics_check(d,F)
    if args.replay:replay(d,F)
    assert read(ROOT/'REPLAY_CHECKS.json')['status']=='PASS'
    write(ROOT/'CHECKS.json',dict(status='PASS',logs=logs,metrics=metrics,replay=read(ROOT/'REPLAY_CHECKS.json'),
        scope='All records and saved classifiers checked; task2 of every gradient arm reexecuted. Later steps are NOT claimed independently replayed.'))
    report()

if __name__=='__main__':main()
