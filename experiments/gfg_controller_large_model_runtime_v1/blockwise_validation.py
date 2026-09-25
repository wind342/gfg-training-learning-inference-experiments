"""Integrate the independently checked blockwise backend without altering v1."""
from pathlib import Path
from unittest.mock import patch
import gc
import time
import traceback
import torch
from . import phase1 as run
from . import audit
from .blockwise_runtime import BlockwiseController

HERE=run.HERE; DEST=HERE/'blockwise_validation'
OUT=Path('E:/gfg-evidence/controller-large-model-runtime-blockwise-v1')
b=run.b


def main():
    run.b.old.setup()
    first=b.read(HERE/'AUDIT.json')
    assert first['status']=='PASS' and first['blockwise_component']=='PASS'
    assert not DEST.exists() and not OUT.exists()
    seal=b.read(HERE/'PHASE1_SEAL.json')
    for n,d in seal['sources'].items():assert b.sha(HERE/n)==d
    newseal={**seal,'utc':b.old.utc(),'sources':{'../'+p.name:b.sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md')},
        'prerequisite_audit_sha256':b.sha(HERE/'AUDIT.json'),'prerequisite_seal_sha256':b.sha(HERE/'PHASE1_SEAL.json'),
        'backend':'blockwise_only_in_runtime'}
    DEST.mkdir();OUT.mkdir();b.write(DEST/'PHASE1_SEAL.json',newseal)
    b.write(DEST/'SYNTHETIC_TESTS.json',b.read(HERE/'SYNTHETIC_TESTS.json'))
    rows=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs'];records=[]
    with patch.object(run,'OUT',OUT),patch.object(run,'RuntimeController',BlockwiseController):
        for row in rows:
            loaded=b.load_row(row)
            try:
                for ci in range(3):
                    for mode in ('audit','runtime'):records.append(run.execute(row,loaded,ci,mode,seal['source_rng_protocol_sha256']))
            finally:loaded[0].close();del loaded;gc.collect();torch.cuda.empty_cache()
    failures=[r for r in records if r['blockwise_nonexact_steps']]
    b.write(DEST/'PHASE1_RESULTS.json',{'status':'COMPLETE_PENDING_INDEPENDENT_AUDIT','runtime_equivalence':'PASS',
        'blockwise_component':'NOT_CERTIFIED' if failures else 'PASS','records':records,
        'persistent_buffer_reduction':.6,'phase2':'NOT_STARTED','phase3':'NOT_STARTED'})
    with patch.object(audit,'HERE',DEST),patch.object(audit,'OUT',OUT):audit.main()


if __name__=='__main__':
    try:main()
    except Exception:
        with (HERE/f'BLOCKWISE_FAILURE_{time.time_ns()}.txt').open('x',encoding='utf-8') as f:f.write(traceback.format_exc())
        raise
