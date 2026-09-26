"""Re-evaluate the saved final checkpoints; no training or controller decisions."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .prepare import ROOT, sha
from .model import RelationModel
from .run import evaluate


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=np.load(ROOT/'data.npz');x=torch.from_numpy(data['x']);y=torch.from_numpy(data['y'][:,None])
    checks={}
    for arm in ('native','governor'):
        root=ROOT/f'10tasks-{arm}'/'attempt-1'
        result=json.loads((root/'RESULT.json').read_text())
        assert result['status']=='COMPLETED'
        expected=result['metrics'][-1];record=result['checkpoints'][-1]
        path=Path(record['path']);assert sha(path)==record['sha256']
        start=time.perf_counter();saved=torch.load(path,map_location='cpu',weights_only=False)
        model=RelationModel().cuda();model.load_state_dict(saved['model']);model.seen=saved['seen']
        assert model.seen==80 and saved['step']==2100
        indices=np.flatnonzero((data['split']==2)&(data['y']<model.seen))
        actual=evaluate(model,x,y,indices)
        for key in ('indices','predictions','accuracy','per_relation'):
            assert actual[key]==expected[key],(arm,key)
        control=evaluate(model,x,y,np.array(expected['probe_ids']))
        differences=[abs(control['per_relation'][str(k)]-actual['per_relation'][str(k)]) for k in range(model.seen)]
        checks[arm]=dict(status='PASS',checkpoint_sha256=record['sha256'],
            test_count=len(indices),accuracy=actual['accuracy'],predictions_exact=True,
            control_accuracy=control['accuracy'],control_per_relation=control['per_relation'],
            control_test_mean_absolute_group_gap=float(np.mean(differences)),
            seconds=time.perf_counter()-start)
        print(arm,checks[arm],flush=True)
        del model,saved;torch.cuda.empty_cache()
    with (ROOT/'CHECKPOINT_REPLAY.json').open('x',encoding='utf8') as f:json.dump(checks,f,indent=2)


if __name__=='__main__':main()
