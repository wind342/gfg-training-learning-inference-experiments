from .common import *

def main():
    guard();setup();ROOT.mkdir(exist_ok=False)
    oldhash={str(p.relative_to(OLD)):sha(p) for p in OLD.rglob('*') if p.is_file()}
    cache=read(OLD/'CACHE.json');assert cache['frozen_bert_unchanged']
    for name,v in cache['model_files'].items():assert sha(Path('E:/gfg-models/bert-base-uncased-v4')/name)==v['sha256']
    import scipy
    write(ROOT/'FREEZE.json',dict(base=BASE,protocol_sha256=sha(HERE/'PROTOCOL.md'),frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),old_evidence_hashes=oldhash,
        budgets=[210,630],rho_ratio=2,outer_cap=8,inner_cap=200,smoke_gates=dict(repair_rate_min=.9,rollback_max=.1,new_train_macro_min=.5,new_train_ce_ratio_max=.9,median_norm_ratio_min=.1,unsafe_exposure_min=20),
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,gpu=torch.cuda.get_device_name(0),threads=2,openblas_threads=2),
        previous_test_seen=True,new_test_read=False,sources={p.name:sha(p) for p in HERE.glob('*.py')}))
    print('protocol, authorities and both budgets frozen',flush=True)

if __name__=='__main__':main()
