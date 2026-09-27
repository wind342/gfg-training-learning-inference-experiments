"""Continuation harness; import the frozen task2 implementation unchanged."""
from experiments.gfg_banking77_final_v2_predictor_task2_v1.common import *
from experiments.gfg_banking77_final_v2_predictor_task2_v1.cutter import project
from experiments.gfg_banking77_final_v2_predictor_task2_v1.predictor import Predictor,measure,aggregate
import sys, pickle, shutil
from collections import Counter

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/banking77-final-v2-expansion-audit-v1')
PARENT=Path('E:/gfg-evidence/banking77-final-v2-predictor-task2-v1')
PARENT_SOURCE=HERE.parent/'gfg_banking77_final_v2_predictor_task2_v1'
BASE='534130d8995fbf9f6cec176afb915f1a56845702'


def guard():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)
    assert subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release^{}'],cwd=REPO,text=True).strip()=='36dab5ce347dbbdac157ef23205f556606d18294'


def inputs():
    items=[OLD/'CACHE.json',OLD/'TRAIN_FEATURES.pt']
    items += [PARENT/q for q in ('FREEZE.json','NUMERICS.json','FINAL.json','AUDIT.json')]
    items += [PARENT/f'budget{b}/{name}' for b in (210,630) for name in ('CHECKPOINT.pt','CHECKS.json','STEPS.jsonl.gz')]
    return items


class Access:
    """Restrict Python-level evidence reads; reject ALL test files in this run."""
    def __init__(self):
        self.allowed={p.resolve() for p in inputs()};self.evidence=ROOT.parent.resolve()
        self.models=Path('E:/gfg-models').resolve();self.reads=Counter();self.writes=Counter();self.denied=[]
        cache=read(OLD/'CACHE.json')
        self.allowed.update((self.models/'bert-base-uncased-v4'/name).resolve() for name in cache['model_files'])
        sys.addaudithook(self.hook)
    def hook(self,event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        scoped=path.is_relative_to(self.evidence) or path.is_relative_to(self.models)
        forbidden=False
        if scoped:
            forbidden=not path.is_relative_to(ROOT.resolve()) and (writing or path not in self.allowed)
            forbidden=forbidden or 'test' in path.name.lower()
        if path.is_relative_to(REPO/'experiments'):
            own=path.is_relative_to(HERE)
            source=path.parent==PARENT_SOURCE and path.suffix in ('.py','.md')
            forbidden=forbidden or (not own and (writing or not source))
        if forbidden:
            self.denied.append(dict(path=str(path),writing=writing));raise PermissionError('Outside expansion-audit allowlist: '+str(path))
        if scoped:
            (self.writes if writing else self.reads)[str(path)]+=1
    def report(self):return dict(reads=dict(self.reads),writes=dict(self.writes),denied=self.denied,test_files_opened=0,
        scope='Python open hooks for evidence/model/experiment paths; native-library/OS internals excluded; no subprocess data readers.')


def sources():
    return {p.relative_to(REPO).as_posix():sha(p) for folder in (HERE,PARENT_SOURCE) for p in list(folder.glob('*.py'))+[folder/'PROTOCOL.md']}


def verify_freeze():
    f=read(ROOT/'FREEZE.json');assert sources()==f['sources']
    for path,digest in f['inputs'].items():assert sha(path)==digest,path
    guard()


def records(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(line) for line in f]


def state_identity(head,shadow,opt):
    grads={name:p.grad for name,p in head.named_parameters() if p.grad is not None}
    shadowgrads={name:p.grad for name,p in shadow.named_parameters() if p.grad is not None}
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().numpy(),[v.cpu().numpy() for v in torch.cuda.get_rng_state_all()])
    hyper=[{k:v for k,v in group.items() if k!='params'} for group in opt.param_groups]
    return dict(head_sha256=thash(head.state_dict()),shadow_sha256=thash(shadow.state_dict()),
                moments_sha256=opt_hash(opt),optimizer_steps=opt_steps(opt),
                optimizer_hyperparameters_sha256=hashlib.sha256(pickle.dumps(hyper)).hexdigest(),
                head_grad_sha256=thash(grads),shadow_grad_sha256=thash(shadowgrads),
                head_grad_none=[n for n,p in head.named_parameters() if p.grad is None],
                shadow_grad_none=[n for n,p in shadow.named_parameters() if p.grad is None],
                rng_sha256=hashlib.sha256(pickle.dumps(rng)).hexdigest(),
                module_modes=[m.training for m in head.modules()])
