from .common import *
from .access import Access
import ast,scipy

def inputs():
    paths=[OLD/'CACHE.json',OLD/'TRAIN_FEATURES.pt',OLD/'NUMERICS.json',AUTH/'NUMERICS.json']
    for b in (210,630):
        paths.append(OLD/f'formal/native_{b}/task1/CHECKPOINT.pt')
        paths.extend(AUTH/f'budget{b}'/n for n in ('CHECKS.json','STEPS.jsonl.gz','CHECKPOINT.pt'))
    return paths

def install():
    a=Access(ROOT,OLD,V1,REPO);a.allowed.update(p.resolve() for p in inputs())
    a.model_files(read(OLD/'CACHE.json')['model_files']);return a

def identity():
    original=HERE.parent/'gfg_banking77_final_v2_predictor_task2_v1'
    assert sha(HERE/'reference.py')==sha(original/'cutter.py')
    functions=('setup','model','optimizer','params','cpu','opt_hash','opt_steps','schedule','load','native_step','commit','functional','Meter','Bank','metrics')
    def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf8')).body if hasattr(n,'name')}
    a,b=nodes(original/'common.py'),nodes(HERE/'common.py')
    for name in functions:assert a[name]==b[name],name
    assert sha(HERE/'predictor.py')==sha(original/'predictor.py')
    ref,opt=nodes(HERE/'reference.py'),nodes(HERE/'optimized.py')
    for name in ('gelu','gelu_prime','inner','Geometry','select'):assert ref[name]==opt[name],name
    return dict(reference_cutter_source_identical=True,predictor_source_identical=True,common_ast_identical=list(functions),
        geometry_reserve_selection_ast_identical=True)

def verify_inputs():
    f=read(ROOT/'FREEZE.json')
    for name,digest in f['input_hashes'].items():assert sha(name)==digest,name
    for name,digest in f['sources'].items():assert sha(HERE/name)==digest,name
    guard()

def main():
    a=install();setup();guard();ROOT.mkdir(exist_ok=False);cache=read(OLD/'CACHE.json')
    assert cache['frozen_bert_unchanged'];hashes={str(p):sha(p) for p in inputs()}
    assert hashes[str(OLD/'TRAIN_FEATURES.pt')]==cache['feature_file_sha256']
    for name,v in cache['model_files'].items():
        p=a.model_root/name;hashes[str(p)]=sha(p);assert hashes[str(p)]==v['sha256']
    write(ROOT/'FREEZE.json',dict(base=BASE,authority=str(AUTH),protocol_sha256=sha(HERE/'PROTOCOL.md'),input_hashes=hashes,
        sources={p.name:sha(p) for p in HERE.glob('*.py')},identity=identity(),budgets=[210,630],
        validation_order=['baseline','stage1','stage2 only after stage1 identity and audit PASS'],
        benchmark_repetitions=3,benchmark_orders=[['baseline','stage1','stage2'],['stage1','stage2','baseline'],['stage2','baseline','stage1']],
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,gpu=torch.cuda.get_device_name(0),threads=2),
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),access=a.report(),test_files_opened=0))
    print('Protocol, inputs, all three implementations and benchmark orders frozen',flush=True)

if __name__=='__main__':main()
