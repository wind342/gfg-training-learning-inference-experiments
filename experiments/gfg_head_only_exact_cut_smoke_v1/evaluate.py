"""Read-only full train and test evaluation after the one training run."""
from .core import *

def main():
    pre=guard();setup();r=json.loads((ROOT/ARM/'TRAIN_RESULT.json').read_text());assert r['status']=='COMPLETED' and r['no_test_inference']
    started=time.time();assert started>=r['finished_unix'];assert sha(ROOT/ARM/'AUDIT.pt')==r['audit_sha256']
    a=torch.load(ROOT/ARM/'AUDIT.pt',map_location='cpu',weights_only=False);saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    m=FrozenRelationModel().cuda();m.load_state_dict(saved['model']);m.head.load_state_dict(a['heads'][-1]);freeze(m);m.seen=16;del saved,a
    d=np.load(DATA/'data.npz');x=torch.from_numpy(d['x']);_,primary,full=scope(d,16)
    trainids=np.flatnonzero((d['split']==0)&(d['y']>=8)&(d['y']<16));result={}
    with torch.no_grad():
        for name,ids in (('training',trainids),('test',full)):
            pred=[]
            for b in range(0,len(ids),16):pred+=m(x[ids[b:b+16]].cuda())[0][:,-1].argmax(1).cpu().tolist()
            if name=='training':result[name]=dict(indices=ids.tolist(),predictions=pred,accuracy16=float((np.array(pred)==d['y'][ids]).mean()))
            else:result[name]=dict(indices=ids.tolist(),predictions=pred,primary=summarize_predictions(primary,np.array(pred)[np.isin(full,primary)],d,2),original=summarize_predictions(full,pred,d,2))
            print('EVALUATED',name,flush=True)
    assert tensor_hash(m.bert.state_dict())==pre['backbone_initial_sha256'] and head_hash(m.head)==r['head_sha256']
    baseline=json.loads((LR_ROOT/'EVALUATION.json').read_text())['arms']['lr5e-5']
    write(ROOT/'EVALUATION.json',dict(status='PASS',started_unix=started,train_result_sha256=sha(ROOT/ARM/'TRAIN_RESULT.json'),
        result=result,baseline=baseline,baseline_retrained=False,source_hashes=sources()))
    print('RESULT',result['training']['accuracy16'],result['test']['primary'],flush=True)

if __name__=='__main__':main()
