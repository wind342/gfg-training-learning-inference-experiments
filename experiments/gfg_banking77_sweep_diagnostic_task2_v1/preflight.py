from .common import *
from .access import Access
import ast

def inputs():
    return [OLD/'CACHE.json',OLD/'TRAIN_FEATURES.pt',OLD/'NUMERICS.json',AUTH/'SUMMARY.json']+[
        p for b in (210,630) for p in (OLD/f'formal/native_{b}/task1/CHECKPOINT.pt',AUTH/f'stage1/budget{b}/STEPS.jsonl.gz',
        AUTH/f'stage1/budget{b}/CHECKS.json',AUTH/f'stage1/budget{b}/CHECKPOINT.pt')]

def install():
    a=Access(ROOT,OLD,V1,REPO);a.allowed.update(p.resolve() for p in inputs())
    a.model_files(read(OLD/'CACHE.json')['model_files']);return a

def verify():
    f=read(ROOT/'FREEZE.json')
    for path,h in f['input_hashes'].items():assert sha(path)==h,path
    for path,h in f['source_hashes'].items():assert sha(REPO/path)==h,path
    guard()

def main():
    access=install();setup();guard();ROOT.mkdir(exist_ok=False)
    original=HERE.parent/'gfg_banking77_cutter_dedup_task2_v1';source=(original/'optimized.py').read_text(encoding='utf8')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='ball_hildreth')
    expected=ast.get_source_segment(source,node).replace('def ball_hildreth(G,b,Ad0,d0norm2,rho,limit=200):','def observe_solver(G,b,Ad0,d0norm2,rho,limit=200,observe=None):')
    expected=expected.replace('for sweep in range(1,limit+1):','for sweep in range(1,limit+1):\n        sweep_start=time.perf_counter()')
    expected=expected.replace('        if violation<=1e-8','        elapsed=time.perf_counter()-sweep_start\n        if observe is not None:observe(sweep,c,t,violation,length,change,elapsed)\n        if violation<=1e-8')
    actual=next(n for n in ast.parse((HERE/'solver.py').read_text()).body if isinstance(n,ast.FunctionDef))
    assert ast.dump(ast.parse(expected).body[0],include_attributes=False)==ast.dump(actual,include_attributes=False)
    cache=read(OLD/'CACHE.json');hashes={str(p):sha(p) for p in inputs()};assert cache['frozen_bert_unchanged']
    assert hashes[str(OLD/'TRAIN_FEATURES.pt')]==cache['feature_file_sha256']
    for name,v in cache['model_files'].items():
        p=access.model_root/name;hashes[str(p)]=sha(p);assert hashes[str(p)]==v['sha256']
    sources=list(HERE.glob('*.py'))+[original/name for name in ('common.py','optimized.py','audit.py','predictor.py','predictor_audit.py','compare.py')]
    write(ROOT/'FREEZE.json',dict(base=BASE,authority=str(AUTH),protocol_sha256=sha(HERE/'PROTOCOL.md'),
        input_hashes=hashes,source_hashes={p.relative_to(REPO).as_posix():sha(p) for p in sources},
        exact_solver_instrumentation_verified=True,budgets=[210,630],task=2,diagnostic_only=True,
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),access=access.report()))
    print('Diagnostic-only protocol and read-only inputs frozen',flush=True)

if __name__=='__main__':main()
