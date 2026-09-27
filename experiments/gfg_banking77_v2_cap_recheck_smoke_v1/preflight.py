from .common import *
from .access import Access
import ast,sys,scipy

def inputs():
    return [OLD/'CACHE.json',OLD/'TRAIN_FEATURES.pt',OLD/'NUMERICS.json']+[
        p for b in (210,630) for p in (OLD/f'formal/native_{b}/task1/CHECKPOINT.pt',V1/f'smoke/jacobian_{b}/task2/CHECKS.json')]

def install():
    access=Access(ROOT,OLD,V1,REPO);cache=read(OLD/'CACHE.json');access.model_files(cache['model_files']);return access

def identity():
    original=HERE.parent/'gfg_banking77_jacobian_hildreth_v1';results={}
    names={'cutter.py':('gelu','gelu_prime','inner','Geometry','ball_hildreth','select'),
           'common.py':('setup','model','optimizer','params','cpu','opt_hash','opt_steps','schedule','load','native_step','commit','functional','Meter','Bank','metrics')}
    for filename,functions in names.items():
        left={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((original/filename).read_text()).body if hasattr(n,'name')}
        right={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((HERE/filename).read_text()).body if hasattr(n,'name')}
        for name in functions:assert left[name]==right[name],(filename,name)
        results[filename]=list(functions)
    return results

def main():
    access=install();setup();guard();ROOT.mkdir(exist_ok=False);cache=read(OLD/'CACHE.json');hashes={str(p):sha(p) for p in inputs()}
    assert hashes[str(OLD/'TRAIN_FEATURES.pt')]==cache['feature_file_sha256'];assert cache['frozen_bert_unchanged']
    for name,record in cache['model_files'].items():
        p=access.model_root/name;hashes[str(p)]=sha(p);assert hashes[str(p)]==record['sha256']
    write(ROOT/'FREEZE.json',dict(base=BASE,protocol_sha256=sha(HERE/'PROTOCOL.md'),input_hashes=hashes,
        sources={p.name:sha(p) for p in HERE.glob('*.py')},unchanged_v1_ast=identity(),
        budgets=[210,630],scope='task2 only, no formal continuation',frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,gpu=torch.cuda.get_device_name(0),threads=2,openblas_threads=2),
        prior_results_known=True,current_run_test_files_opened=0,access=access.report()))
    print('Two budgets frozen; solver/geometry/settings AST identical to v1; train-only allowlist active',flush=True)

def verify_inputs():
    for name,digest in read(ROOT/'FREEZE.json')['input_hashes'].items():assert sha(Path(name))==digest,name

if __name__=='__main__':main()
