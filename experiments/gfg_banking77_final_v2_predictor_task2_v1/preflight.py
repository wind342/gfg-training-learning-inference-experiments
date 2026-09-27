from .common import *
from .access import Access
import ast,sys,scipy
PREDICTOR_SOURCE=Path('D:/codex/gfg-banking77-nonlinear-v1/experiments/gfg_banking77_nonlinear_v1')

def inputs():
    return [OLD/'CACHE.json',OLD/'TRAIN_FEATURES.pt',OLD/'NUMERICS.json']+[
        p for b in (210,630) for p in (OLD/f'formal/native_{b}/task1/CHECKPOINT.pt',V1/f'smoke/jacobian_{b}/task2/CHECKS.json')]+[OLD/name for name in ('FORMAL_SOURCE_FREEZE.json','smoke_predictor/CHECKS.json','smoke_predictor/STEPS_210.jsonl.gz','smoke_predictor/STEPS_630.jsonl.gz','formal/predictive_210/SEALED.json','formal/predictive_630/SEALED.json')]+[PREDICTOR_SOURCE/name for name in ('common.py','smoke.py','train.py')]

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
    authority=read(OLD/'FORMAL_SOURCE_FREEZE.json')
    for name in ('common.py','smoke.py','train.py'):assert sha(PREDICTOR_SOURCE/name)==authority['sources'][name]
    tree=ast.parse((PREDICTOR_SOURCE/'common.py').read_text(encoding='utf8'));klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Bank');method=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
    historical={str(b):read(OLD/'smoke_predictor/CHECKS.json')['results'][str(b)]['alphas']['1.0'] for b in (210,630)}
    write(ROOT/'PREDICTOR_AUTHORITY.json',dict(source=str(PREDICTOR_SOURCE),verified_sources={k:authority['sources'][k] for k in ('common.py','smoke.py','train.py')},predict_source=ast.get_source_segment((PREDICTOR_SOURCE/'common.py').read_text(encoding='utf8'),method),historical_alpha1_metrics=historical,definition='f0 + Df[delta] + .5 D2f[delta,delta]; delta = native-before; MLP only'))
    write(ROOT/'FREEZE.json',dict(base=BASE,protocol_sha256=sha(HERE/'PROTOCOL.md'),input_hashes=hashes,
        sources={p.name:sha(p) for p in HERE.glob('*.py')},unchanged_v1_ast=identity(),
        budgets=[210,630],scope='task2 only, no formal continuation',frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,gpu=torch.cuda.get_device_name(0),threads=2,openblas_threads=2),
        prior_results_known=True,current_run_test_files_opened=0,access=access.report()))
    print('Two budgets frozen; solver/geometry/settings AST identical to v1; train-only allowlist active',flush=True)

def verify_inputs():
    for name,digest in read(ROOT/'FREEZE.json')['input_hashes'].items():assert sha(Path(name))==digest,name

if __name__=='__main__':main()
