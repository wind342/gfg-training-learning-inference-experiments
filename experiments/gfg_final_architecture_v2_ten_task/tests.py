"""No training: existing policy tests plus generalized independent replay."""
from .core import *
from .independent import independent
from experiments.gfg_relation_capability_floor_task8_v1.preflight import tests as parent_tests
from experiments.gfg_relation_capability_floor_task8_v1.check import independent as original_checker

def tests():
    p.setup();result=parent_tests();seen_tests=[]
    for seen in (16,24,64,80):
        before={'weight':torch.zeros(80,4,device='cuda'),'bias':torch.full((80,),-2.,device='cuda')}
        before['weight'][0,0]=2.;before['weight'][1,0]=-2.
        before['bias'][:2]=0
        native={k:v.clone() for k,v in before.items()};native['weight'][:2].neg_()
        h=torch.tensor([[1.,0,0,0],[-1.,0,0,0]],device='cuda');y=torch.tensor([0,1],device='cuda')
        K=np.zeros(seen-8,dtype=int);K[:2]=1
        e,c,r=floor_projection(before,native,h,y,2,K,[10,11],seen=seen)
        ee,cc,s,l,bm,nm,cm,em=independent(before,native,h,y,2,K,[10,11],seen)
        assert equal(e,ee) and equal(c,cc) and s==r['selections'] and l==r['sweeps']
        assert em.tolist()==r['executed'] and r['projection_triggered'] and not r['skip']
        seen_tests.append(seen)
    # Actual frozen task8 state, not a new optimizer trajectory.
    source=Path('E:/gfg-evidence/relation-capability-floor-task8-v1')
    pre=json.loads((source/'PREFLIGHT.json').read_text());meta=json.loads((source/'TRAIN_RESULT.json').read_text())
    assert sha(source/'AUDIT.pt')==meta['audit_sha256']
    a=torch.load(source/'AUDIT.pt',map_location='cpu',weights_only=False)
    cache=torch.load(old.ROOT/'CONTROL_CACHE.pt',map_location='cpu',weights_only=False)
    d=np.load(DATA/'data.npz');h,y,ids,n=bank(d,cache,8);K=np.asarray(pre['K'])
    for j in (0,104,209):
        b={k:v.cuda() for k,v in a['heads'][j].items()};native={k:v.cuda() for k,v in a['natives'][j].items()}
        expected=original_checker(b,native,h,y,n,K,ids[:n]);actual=independent(b,native,h,y,n,K,ids[:n],64)
        assert equal(expected[:4],actual[:4])
        for x,z in zip(expected[4:],actual[4:]):assert np.array_equal(x,z)
    return dict(status='PASS',parent_policy_tests=result,dynamic_seen=seen_tests,stored_task8_steps=[1,105,210],
                production_policy_unchanged=True)

if __name__=='__main__':print(json.dumps(tests(),indent=2))
