from .common import *
from .access import Access
import ast,scipy

def inputs():
    paths=[OLD/'CACHE.json',OLD/'TRAIN_FEATURES.pt',OLD/'NUMERICS.json',AUTH/'NUMERICS.json']
    for b in (210,630):
        paths.append(OLD/f'formal/native_{b}/task1/CHECKPOINT.pt')
        for v in ('baseline','firstcheck'):
            paths.extend(PREVIOUS/v/f'budget{b}'/n for n in ('CHECKS.json','STEPS.jsonl.gz','CHECKPOINT.pt'))
    return paths

def install():
    a=Access(ROOT,OLD,V1,REPO);a.allowed.update(p.resolve() for p in inputs());a.model_files(read(OLD/'CACHE.json')['model_files']);return a

def identity():
    original=HERE.parent/'gfg_banking77_first_sweep_check_task2_v1'
    for name in ('reference.py','cutter.py','predictor.py','predictor_audit.py','audit.py'):assert sha(HERE/name)==sha(original/name),name
    def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf8')).body if hasattr(n,'name')}
    a,b=nodes(original/'common.py'),nodes(HERE/'common.py')
    for name in a:
        if name!='Meter':assert a[name]==b[name],name
    tree=ast.parse((HERE/'clean.py').read_text(encoding='utf8'));p=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='project')
    forbidden={'ahash','nhash','phash','thash','sha','evidence','tolist','write','dumps'}
    calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(p) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls.intersection(forbidden),calls.intersection(forbidden)
    return dict(original_actuators_audit_predictor_identical=True,common_math_identical_except_profiler=True,no_hash_or_serialization_in_clean_project=True)

def verify_inputs():
    f=read(ROOT/'FREEZE.json')
    for name,digest in f['input_hashes'].items():assert sha(name)==digest,name
    for name,digest in f['sources'].items():assert sha(HERE/name)==digest,name
    guard()

def main():
    a=install();setup();guard();ident=identity();ROOT.mkdir(exist_ok=False);cache=read(OLD/'CACHE.json')
    assert cache['frozen_bert_unchanged'];hashes={str(p):sha(p) for p in inputs()}
    for name,v in cache['model_files'].items():
        p=a.model_root/name;hashes[str(p)]=sha(p);assert hashes[str(p)]==v['sha256']
    write(ROOT/'FREEZE.json',dict(base=BASE,protocol_sha256=sha(HERE/'PROTOCOL.md'),input_hashes=hashes,
        sources={p.name:sha(p) for p in HERE.glob('*.py')},identity=ident,budgets=[210,630],
        benchmark_orders=[['recorded/baseline','clean/baseline','recorded/firstcheck','clean/firstcheck'],
            ['clean/firstcheck','recorded/firstcheck','clean/baseline','recorded/baseline'],
            ['clean/baseline','recorded/baseline','clean/firstcheck','recorded/firstcheck']],
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,gpu=torch.cuda.get_device_name(0),threads=2),
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),access=a.report(),test_files_opened=0))
    print('Protocol/source/runtime/checkpoint/BERT/cache frozen',flush=True)

if __name__=='__main__':main()
