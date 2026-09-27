"""Feature extraction and numerical preflight, not a parameter-selection run."""
from .core import *

@torch.no_grad()
def main():
    p.setup();guards();ROOT.mkdir(exist_ok=False)
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    d=np.load(DATA/'data.npz');isolation=old.isolation(d)
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=FrozenRelationModel().cuda();model.load_state_dict(saved['model']);freeze(model);model.eval()
    shadow,opt=make_candidate(model,saved['optimizer'])
    for group in opt.param_groups:group['lr']=5e-5
    save(ROOT/'INITIAL.pt',dict(head=p.cpu_state(p.snapshot(model.head)),optimizer=opt.state_dict()))
    before=tensor_hash(model.bert.state_dict());del saved,shadow,opt
    cache=torch.load(old.ROOT/'CONTROL_CACHE.pt',map_location='cpu',weights_only=False)
    manifest=json.loads((old.ROOT/'CACHE_MANIFEST.json').read_text())
    assert sha(old.ROOT/'CONTROL_CACHE.pt')==manifest['sha256']
    assert cache['indices']==old.clean_validation(d).tolist()
    F=torch.zeros((len(d['y']),1536),dtype=torch.float32);available=torch.zeros(len(F),dtype=torch.bool)
    F[cache['indices']]=cache['features'];available[cache['indices']]=True
    x=torch.from_numpy(d['x']);walls={};checks=[]
    for task in range(1,11):
        sync();start=time.perf_counter()
        for batch in order(d,task):
            F[batch]=model.features(x[batch].cuda()).cpu();available[batch]=True
        sync();walls[str(task)]=time.perf_counter()-start
        check=order(d,task)[0]
        observed=model.features(x[check].cuda()).cpu()
        assert torch.equal(F[check],observed)
        checks.append(dict(task=task,indices=check.tolist(),max_error=0.,bit_identical=True))
        print('CACHE train task',task,flush=True)
    # Test features live in a separate evaluator-only artifact.
    test=scope(d,80)[1];TF=torch.empty((len(test),1536));sync();start=time.perf_counter()
    for begin in range(0,len(test),16):
        TF[begin:begin+16]=model.features(x[test[begin:begin+16]].cuda()).cpu()
    sync();test_wall=time.perf_counter()-start
    assert before==tensor_hash(model.bert.state_dict())
    assert not available[test].any() and len(test)==11193
    save(ROOT/'FEATURES.pt',dict(features=F,available=available))
    save(ROOT/'TEST_FEATURES.pt',dict(indices=test.tolist(),features=TF))
    write(ROOT/'CACHE.json',dict(status='PASS',data_sha256=DATA_SHA,checkpoint_sha256=CKSHA,backbone_sha256=before,
        isolation=isolation,raw_forward_checks=checks,train_feature_wall=walls,test_feature_wall=test_wall,
        controls_cache_sha256=sha(old.ROOT/'CONTROL_CACHE.pt'),runtime=runtime(),
        artifacts={n:sha(ROOT/n) for n in ('FEATURES.pt','TEST_FEATURES.pt','INITIAL.pt')},created_unix=time.time()))
    print('CACHE PASS',flush=True)

if __name__=='__main__':main()
