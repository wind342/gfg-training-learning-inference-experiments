import gzip
from .core import *

def main():
    guard();assert json.loads((ROOT/'CHECKS.json').read_text())['status']=='PASS'
    result=HERE/'results';record=HERE/'records';result.mkdir(exist_ok=False);record.mkdir(exist_ok=False)
    for name in ('PREFLIGHT.json','NUMERICAL_CHECK.json','TRAINING_COMPLETE.json','EVALUATION.json','CHECKS.json','RESULTS.json'):
        with (result/name).open('xb') as f:f.write((ROOT/name).read_bytes())
    for name in ('CONFIG.json','TRAIN_RESULT.json'):
        with (record/name).open('xb') as f:f.write((ROOT/ARM/name).read_bytes())
    with (record/'STEPS.jsonl.gz').open('xb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write((ROOT/ARM/'STEPS.jsonl').read_bytes())
    paths=sorted(list(result.glob('*.json'))+list(record.glob('*')))
    write(result/'PUBLIC_RECORDS_MANIFEST.json',dict(base=BASE,files={p.relative_to(HERE).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths},
        local_only=[ARM+'/AUDIT.pt'],reused_cache=str(COVER_ROOT/'CACHE.pt'),original_data_published=False))

if __name__=='__main__':main()
