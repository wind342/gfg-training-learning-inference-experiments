"""Long-run harness only; production projection is imported without modification."""
from experiments.gfg_head_only_hildreth_roundoff_smoke_v1.core import *
from experiments.gfg_head_only_hildreth_roundoff_smoke_v1.core import ROOT as PARENT_ROOT
from experiments.gfg_router_v4_ten_task_validation.common import EXCLUDE

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/final-architecture-v1-ten-task')
BASE='3daa90d2572cb50aff09aee3d95997ae4244d799'
ARMS=('head_only_native','final_architecture_v1')
BAD_VALIDATION=(7472,7491,8927,17947,17967,19337,36190,50124,53007,55069)

def unchanged():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)

def sources():
    paths=[REPO/p for p in subprocess.check_output(['git','ls-files','experiments'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    paths+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(set(paths))}

def guard(full=True):
    unchanged();p=json.loads((ROOT/'PREFLIGHT.json').read_text());assert p['status']=='PASS' and p['source_hashes']==sources()
    if full:
        assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
        for path,h in p['dependencies'].items():assert sha(path)==h,path
    return p

def clean_validation(d):
    return np.flatnonzero((d['split']==1)&~np.isin(np.arange(len(d['y'])),BAD_VALIDATION))

def task_bank(d,task):
    v=clean_validation(d);old=v[d['y'][v]<8*(task-1)]
    new=np.concatenate([v[d['y'][v]==g][:10] for g in range(8*(task-1),8*task)])
    return np.concatenate([old,new]),len(old)

def isolation(d):
    h=[hashlib.sha256(x.tobytes()).hexdigest() for x in d['x']]
    tr={h[i] for i in np.flatnonzero(d['split']==0)}
    bad=[int(i) for i in np.flatnonzero(d['split']==1) if h[i] in tr]
    assert bad==list(BAD_VALIDATION)
    ids=clean_validation(d);cv={h[i] for i in ids};assert len(cv)==len(ids)==11190 and not tr&cv
    test=scope(d,80)[1];th={h[i] for i in test};assert not cv&th and not tr&th
    required=[int(i) for i in np.flatnonzero(d['split']==2) if h[i] in tr or h[i] in cv]
    assert required==list(EXCLUDE)
    counts={str(t):task_bank(d,t)[1] for t in range(2,11)};assert counts['2']==1120 and counts['10']==10072
    return dict(status='PASS',excluded_validation=bad,excluded_primary_test=required,bank_sizes=counts,
        criterion='SHA256 of full encoded input, irrespective of labels; not semantic equivalence')

def save_tensor(path,value):
    with Path(path).open('xb') as f:torch.save(value,f)

def opt_counter(opt):
    values=[int(v['step'].item()) for v in opt.state.values()];assert len(values)==2 and len(set(values))==1
    return values[0]

def distribution(values):
    return dict(min=float(min(values)),median=float(np.median(values)),max=float(max(values))) if values else None

def summary(rows,n):
    selected=[r for r in rows if r.get('report',{}).get('native_violating')]
    def geom(key):
        return {k:float(np.mean(v)) if (v:=[r[key][k] for r in selected if r[key][k] is not None]) else None for k in ('norm_ratio','cosine')}
    return dict(protection_bank_size=n,receiving_correct=distribution([len(r['report']['eligible']) for r in rows]),
        receiving_correct_first=len(rows[0]['report']['eligible']),receiving_correct_last=len(rows[-1]['report']['eligible']),
        final_protected_correct=sum(x>0 for x in rows[-1]['report']['executed'][:n]),
        native_safe_steps=len(rows)-len(selected),projection_triggered_steps=len(selected),
        safe_projected_steps=sum(not r['report']['skip'] for r in selected),skip_steps=sum(r['report']['skip'] for r in rows),
        active_constraints=distribution([len(r['report']['final_active']) for r in selected]),
        sweeps=distribution([r['report']['sweep_count'] for r in selected]),total_sweeps=sum(r['report']['sweep_count'] for r in rows),
        executed_old_correct_to_wrong=sum(sum(a>0 and b<=0 for a,b in zip(r['report']['baseline'][:n],r['report']['executed'][:n])) for r in rows),
        projected_geometry=geom('projected_geometry'),executed_geometry=geom('executed_geometry'),
        projection_wall=sum(r['report']['projection_seconds'] for r in rows))

@torch.no_grad()
def transition(head,h,y,n,task):
    before=margins(logits(head,h[:n],8*(task-1)),y[:n]);after=margins(logits(head,h[:n],8*task),y[:n])
    return dict(before=before.cpu().tolist(),after=after.cpu().tolist(),
        correct_before=int((before>0).sum()),correct_after=int((after>0).sum()),
        expansion_correct_to_wrong=torch.nonzero((before>0)&(after<=0)).flatten().cpu().tolist())
