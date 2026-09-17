"""Generate a compact distributable ledger and full raw-evidence manifest."""
import gzip
from .common import *

def main():
    verify_seal()
    require(read(HERE / 'INDEPENDENT_CHECK.json')['status']=='PASS','CHECK_REQUIRED')
    files={str(p.relative_to(OUTPUT)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
    target=HERE / 'EXECUTION_LEDGER.jsonl.gz'
    with target.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as archive:
            for row in read(HERE / 'RUN_ASSIGNMENT.json')['runs']:
                for branch in BRANCHES:
                    root=OUTPUT / row['entry_id'] / branch
                    for t in range(1,HORIZON+1):
                        record={'entry_id':row['entry_id'],'branch':branch,
                            'step':read(root / 'steps' / f'{t:04d}.json'),
                            'selection':read(root / 'selections' / f'{t:04d}.json')}
                        if t%CADENCE==0: record['evaluation']=read(root / 'evaluations' / f'{t:04d}.json')
                        archive.write(canonical_bytes(record)+b'\n')
    write_new(HERE / 'MANIFEST.json',{'schema':SCHEMA+'-archive','external_root':str(OUTPUT),
        'external_files':files,'external_bytes':sum(v['bytes'] for v in files.values()),
        'compact_ledger':{'path':target.name,'sha256':sha(target),'bytes':target.stat().st_size},
        'files':{p.name:sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='MANIFEST.json'},
        'existing_research_archive_modified':False})
    print('PACKAGE_COMPLETE',len(files),sum(v['bytes'] for v in files.values()),flush=True)

if __name__=='__main__':main()
