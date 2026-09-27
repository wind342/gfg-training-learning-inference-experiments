from .common import *
from .access import Access
import ast,scipy

def inputs():
    paths=[OLD/'CACHE.json',OLD/'TRAIN_FEATURES.pt',OLD/'NUMERICS.json',AUTH/'NUMERICS.json']
    for b in (210,630):
        paths.append(OLD/f'formal/native_{b}/task1/CHECKPOINT.pt')
        paths.extend(DEDUP/f'stage1/budget{b}'/n for n in ('CHECKS.json','STEPS.jsonl.gz','CHECKPOINT.pt'))
    return paths

def install():
    a=Access(ROOT,OLD,V1,REPO);a.allowed.update(p.resolve() for p in inputs())
    a.model_files(read(OLD/'CACHE.json')['model_files']);return a

def identity():
    original=HERE.parent/'gfg_banking77_cutter_dedup_task2_v1'
    assert sha(HERE/'reference.py')==sha(original/'optimized.py')
    for name in ('predictor.py','predictor_audit.py'):assert sha(HERE/name)==sha(original/name)
    def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf8')).body if hasattr(n,'name')}
    a,b=nodes(original/'common.py'),nodes(HERE/'common.py')
    for name in a:
        if name not in ('guard',):assert a[name]==b[name],name
    a,b=nodes(HERE/'reference.py'),nodes(HERE/'cutter.py')
    for name in ('gelu','gelu_prime','inner','Geometry','select','UniqueGeometry'):assert a[name]==b[name],name
    tree=ast.parse((HERE/'cutter.py').read_text(encoding='utf8'))
    solver=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ball_hildreth')
    solver.args.args.pop();solver.args.defaults.pop()
    loop=next(n for n in solver.body if isinstance(n,ast.For))
    loop.body=[n for n in loop.body if not (isinstance(n,ast.If) and 'first_check' in ast.unparse(n.test))]
    assert ast.dump(solver,include_attributes=False)==a['ball_hildreth']
    return dict(baseline_source_identical=True,common_functions_identical=True,geometry_selection_reserve_identical=True,solver_math_identical_except_first_callback=True)

def verify_inputs():
    f=read(ROOT/'FREEZE.json')
    for name,digest in f['input_hashes'].items():assert sha(name)==digest,name
    for name,digest in f['sources'].items():assert sha(HERE/name)==digest,name
    guard()

def main():
    a=install();setup();guard();ROOT.mkdir(exist_ok=False);cache=read(OLD/'CACHE.json')
    assert cache['frozen_bert_unchanged'];hashes={str(p):sha(p) for p in inputs()}
    for name,v in cache['model_files'].items():
        p=a.model_root/name;hashes[str(p)]=sha(p);assert hashes[str(p)]==v['sha256']
    write(ROOT/'FREEZE.json',dict(base=BASE,protocol_sha256=sha(HERE/'PROTOCOL.md'),input_hashes=hashes,
        sources={p.name:sha(p) for p in HERE.glob('*.py')},identity=identity(),budgets=[210,630],
        benchmark_orders=[['baseline','firstcheck'],['firstcheck','baseline'],['baseline','firstcheck']],
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,gpu=torch.cuda.get_device_name(0),threads=2),
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),access=a.report(),test_files_opened=0))
    print('Frozen protocol, source, checkpoints, cache and BERT hashes',flush=True)

if __name__=='__main__':main()
