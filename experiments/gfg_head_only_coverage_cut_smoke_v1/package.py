"""Copy generated JSON/ledgers only; never publish features or model tensors."""
import gzip
from .core import *

def main():
    guard();assert json.loads((ROOT/'CHECKS.json').read_text())['status']=='PASS'
    for name in ('PREFLIGHT.json','TRAINING_COMPLETE.json','EVALUATION.json','CHECKS.json','RESULTS.json'):
        dest=HERE/'results'/name;dest.parent.mkdir(exist_ok=True)
        with dest.open('xb') as f:f.write((ROOT/name).read_bytes())
    for arm in ARMS:
        dest=HERE/'records'/arm;dest.mkdir(parents=True,exist_ok=False)
        for name in ('CONFIG.json','TRAIN_RESULT.json'):
            with (dest/name).open('xb') as f:f.write((ROOT/arm/name).read_bytes())
        with (dest/'STEPS.jsonl.gz').open('xb') as f:
            with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write((ROOT/arm/'STEPS.jsonl').read_bytes())
    paths=sorted(list((HERE/'results').glob('*.json'))+list((HERE/'records').glob('*/*')))
    write(HERE/'results'/'PUBLIC_RECORDS_MANIFEST.json',dict(base=BASE,files={p.relative_to(HERE).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths},
        local_only=['CACHE.pt','*/AUDIT.pt'],original_data_published=False))

if __name__=='__main__':main()
