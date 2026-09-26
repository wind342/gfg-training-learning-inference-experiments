import gzip
from .core import *

def transfer(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('xb') as f:f.write(src.read_bytes())

def main():
    guard();assert json.loads((ROOT/'CHECKS.json').read_text())['status']=='PASS'
    for name in ('PREFLIGHT.json','CACHE_MANIFEST.json','EXECUTION.json','COMPLETE.json','CHECKS.json','RESULTS.json'):
        transfer(ROOT/name,HERE/'results'/name)
    for task in range(2,11):
        for name in ('BARRIER.json','EVALUATION.json'):
            transfer(ROOT/f'evaluation_task{task}'/name,HERE/'records'/f'evaluation_task{task}'/name)
        for arm in ARMS:
            src=ROOT/arm/f'task{task}';dst=HERE/'records'/arm/f'task{task}'
            for name in ('CONFIG.json','TRAIN_RESULT.json'):transfer(src/name,dst/name)
            with (dst/'STEPS.jsonl.gz').open('xb') as f:
                with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
                    with (src/'STEPS.jsonl').open('rb') as inp:
                        while block:=inp.read(1024*1024):z.write(block)
    files=sorted(p for top in (HERE/'results',HERE/'records') for p in top.rglob('*') if p.is_file())
    write(HERE/'results/PUBLIC_RECORDS_MANIFEST.json',dict(base=BASE,files={p.relative_to(HERE).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in files},
        local_only='CONTROL_CACHE.pt and per-arm/task AUDIT.pt; no BERT checkpoint duplication',original_data_published=False))

if __name__=='__main__':main()
