from .common import *
from .preflight import install,verify_inputs

def rows(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]

def compare(left,right,budget):
    a=rows(left/'STEPS.jsonl.gz');b=rows(right/'STEPS.jsonl.gz');assert len(a)==len(b)==budget
    assert a==b,'Trajectory records differ'
    ac=torch.load(left/'CHECKPOINT.pt',weights_only=True);bc=torch.load(right/'CHECKPOINT.pt',weights_only=True)
    assert thash(ac['model'])==thash(bc['model'])
    def hashes(v):
        return thash({f'{i}.{k}':t for i,s in v['optimizer']['state'].items() for k,t in s.items() if isinstance(t,torch.Tensor)})
    assert hashes(ac)==hashes(bc) and ac['optimizer']['param_groups']==bc['optimizer']['param_groups']
    return dict(status='PASS',budget=budget,all_records_identical=True,checkpoint_model_optimizer_identical=True)

def main():
    a=install();setup();verify_inputs()
    reports=[dict(variant=v,**compare(ROOT/v/f'budget{b}',PREVIOUS/v/f'budget{b}',b)) for v in ('baseline','firstcheck') for b in (210,630)]
    write(ROOT/'IDENTITY.json',dict(status='PASS',reports=reports,access=a.report()))
    print('Both clean trajectories match sealed policy records',flush=True)

if __name__=='__main__':main()
