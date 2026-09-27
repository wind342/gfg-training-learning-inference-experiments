"""Test reader with a barrier requiring both complete, sealed formal streams."""
from .common import *

@torch.no_grad()
def main():
    setup();guard();assert read(ROOT/'FORMAL.json')['status']=='PASS';assert read(ROOT/'AUDIT.json')['status']=='PASS'
    assert read(ROOT/'SEAL_AUDIT.json')['status']=='PASS'
    for name,digest in read(ROOT/'FORMAL_SOURCE_FREEZE.json')['sources'].items():assert sha(HERE/name)==digest
    seals={str(b):sha(ROOT/'formal'/f'jacobian_{b}'/'SEALED.json') for b in (210,630)}
    for b in (210,630):
        r=read(ROOT/'formal'/f'jacobian_{b}'/'SEALED.json');assert r['status']=='PASS' and len(r['tasks'])==11
        assert sum(q['floor_violations'] for q in r['tasks'][1:])==0
    freeze=read(ROOT/'FREEZE.json');cache=read(OLD/'CACHE.json')
    for name,value in cache['model_files'].items():assert sha(Path('E:/gfg-models/bert-base-uncased-v4')/name)==value['sha256']
    assert sha(OLD/'TEST_FEATURES.pt')==freeze['old_evidence_hashes']['TEST_FEATURES.pt']
    for name,digest in freeze['old_evidence_hashes'].items():assert sha(OLD/name)==digest
    write(ROOT/'TEST_BARRIER.json',dict(status='PASS',sealed_arms=seals,source_sha256=sha(Path(__file__)),
        test_cache_sha256=sha(OLD/'TEST_FEATURES.pt'),opened_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    data=torch.load(OLD/'TEST_FEATURES.pt',weights_only=True);X=data['features'].cuda();labels=data['labels'].cuda()
    assert X.shape==(3080,768);results={}
    for budget in (210,630):
        rows=[];head=model()
        for task in range(1,12):
            path=OLD/'formal'/f'native_{budget}'/'task1/CHECKPOINT.pt' if task==1 else ROOT/'smoke'/f'jacobian_{budget}'/'task2/CHECKPOINT.pt' if task==2 else ROOT/'formal'/f'jacobian_{budget}'/f'task{task}/CHECKPOINT.pt'
            head.load_state_dict(torch.load(path,weights_only=True)['model']);p=params(head)
            ids=torch.nonzero(labels<task*7).flatten();y=labels[ids];n=len(ids);padded=((n+255)//256)*256
            bank=X.new_zeros((padded,768));bank[:n].copy_(X[ids]);sync();clock=time.perf_counter()
            z=torch.cat([functional(p,bank[k:k+256]) for k in range(0,padded,256)])[:n,:task*7];pred=z.argmax(1)
            sync();wall=time.perf_counter()-clock;acc=[float((pred[y==c]==c).float().mean()) for c in range(task*7)]
            rows.append(dict(task=task,old=float(np.mean(acc[:-7])) if task>1 else None,new=float(np.mean(acc[-7:])),all=float(np.mean(acc)),
                per_intent=acc,predictions=pred.cpu().tolist(),row_indices=ids.cpu().tolist(),evaluation_wall=wall,checkpoint_sha256=sha(path)))
        results[f'jacobian_{budget}']=rows;print('final test',budget,{k:rows[-1][k] for k in ('old','new','all')},flush=True)
    previous=read(OLD/'EVALUATION.json');baseline_names=('native_210','er_210','native_630','er_630','ranpac')
    baseline={name:previous['arms'][name] for name in baseline_names}
    write(ROOT/'EVALUATION.json',dict(status='PASS',arms=results,reused_baselines=baseline,
        baseline_source_sha256=sha(OLD/'EVALUATION.json'),frozen_bert_files_unchanged=True,
        previous_test_already_seen=True,new_test_read_only_after_both_streams_sealed=True))

if __name__=='__main__':main()
