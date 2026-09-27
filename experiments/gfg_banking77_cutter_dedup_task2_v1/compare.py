"""Strict comparison to the sealed 534130d trajectory, outside timing."""
from .common import *

def records(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(line) for line in f]

def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [d for k in a for d in differences(a[k],b.get(k),path+'.'+str(k))]
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):return [dict(path=path,reference_length=len(a),candidate_length=len(b))]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+f'[{i}]')]
    if a==b:return []
    r=dict(path=path,reference=a,candidate=b)
    if isinstance(a,(float,int)) and isinstance(b,(float,int)):r['absolute_difference']=abs(a-b)
    return [r]

def state_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and a.cpu().contiguous().numpy().tobytes()==b.cpu().contiguous().numpy().tobytes()
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(state_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a==b

def compare_arm(folder,budget):
    candidate=folder/f'budget{budget}';reference=AUTH/f'budget{budget}'
    a,b=records(reference/'STEPS.jsonl.gz'),records(candidate/'STEPS.jsonl.gz');assert len(a)==len(b)==budget
    changed=[];first=None;identical_fields={k:0 for k in ('before_sha256','native_sha256','executed_sha256','moments_sha256','optimizer_steps','predictor','endpoint_sha256','counts','decision')}
    for old,new in zip(a,b):
        for k in identical_fields:identical_fields[k]+=int(old[k]==new[k])
        diff=differences(old,new)
        if diff:
            changed.append(old['step'])
            if first is None:first=dict(step=old['step'],differences=diff[:50],difference_count=len(diff))
    x=torch.load(reference/'CHECKPOINT.pt',map_location='cpu',weights_only=True);y=torch.load(candidate/'CHECKPOINT.pt',map_location='cpu',weights_only=True)
    equal=state_equal(x,y)
    result=dict(status='PASS' if not changed and equal else 'FAIL',budget=budget,identical_updates=identical_fields,
        first_divergence=first,changed_steps=changed,final_checkpoint_state_byte_identical=equal,
        serialized_checkpoint_identical=sha(reference/'CHECKPOINT.pt')==sha(candidate/'CHECKPOINT.pt'),
        reference_checkpoint_sha256=sha(reference/'CHECKPOINT.pt'),candidate_checkpoint_sha256=sha(candidate/'CHECKPOINT.pt'))
    write(candidate/'IDENTITY.json',result);return result
