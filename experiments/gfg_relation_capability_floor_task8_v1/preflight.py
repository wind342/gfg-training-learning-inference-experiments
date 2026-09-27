"""Small deterministic tests before any formal optimization."""
from .core import *

def tests():
    labels=np.array([0,0,1,1]);ids=np.array([9,2,5,4])
    C,D,chosen=choose(np.array([-.1,-.1,-.2,-.3]),labels,np.array([1,1]),ids)
    assert chosen=={'0':[1],'1':[2]}
    before={'weight':torch.tensor([[1.],[0.]],device='cuda'),'bias':torch.zeros(2,device='cuda')}
    native={'weight':-before['weight'],'bias':before['bias'].clone()}
    h=torch.tensor([[1.],[-1.]],device='cuda');y=torch.zeros(2,dtype=torch.long,device='cuda')
    e,_,r=floor_projection(before,native,h,y,2,[1],[10,11],seen=2)
    assert r['correct_to_wrong']==r['wrong_to_correct']==1 and not r['projection_triggered']
    assert equal(e,native) and r['migration_relations']==[0]
    # Receiving count2 may fall to fixed K1 without raising the floor.
    good={'weight':torch.zeros_like(before['weight']),'bias':torch.tensor([1.,0.],device='cuda')}
    e,_,r=floor_projection(good,before,h,y,2,[1],[10,11],seen=2)
    assert r['receiving_counts']==[2] and r['executed_counts']==[1] and not r['projection_triggered']
    # A true deficit requires correction, using the original coordinate formula.
    bad={'weight':torch.zeros_like(before['weight']),'bias':torch.tensor([-1.,0.],device='cuda')}
    e,_,r=floor_projection(good,bad,h,y,2,[1],[10,11],seen=2)
    assert r['projection_triggered'] and not r['skip'] and r['executed_counts'][0]>=1
    assert r['selections'][0]['chosen']=={'0':[0]}
    e,_,r=floor_projection(good,bad,h,y,2,[1],[10,11],seen=2,limit=0)
    assert r['skip'] and r['sweep_count']==0 and equal(e,good)
    # No new restriction on relations whose entry count was zero.
    e,_,r=floor_projection(good,bad,h,y,2,[0],[10,11],seen=2)
    assert not r['skip'] and not r['projection_triggered'] and equal(e,bad)
    return dict(status='PASS',checks=['deterministic_nearest_and_tie_selection','native_swap_allowed',
        'fixed_floor_no_ratcheting','true_deficit_recovery','hard_sweep_cap_skip','zero_floor_native'])
