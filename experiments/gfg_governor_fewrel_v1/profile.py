"""One real BERT update plus bounded functional measurement, no benchmark tuning."""
import json
import time
import traceback
import numpy as np
import torch
from experiments.gfg_controller_distributed_runtime_v2.identity import CanonicalParameterRegistry
from experiments.gfg_controller_distributed_runtime_v2.ownership import Ownership
from .model import RelationModel,BertInstrument
from .prepare import ROOT


def main():
    torch.set_num_threads(2);torch.manual_seed(2021)
    data=np.load(ROOT/'data.npz');x=torch.from_numpy(data['x'][:16]).cuda();y=torch.from_numpy(data['y'][:16,None]).cuda()
    report={'kind':'one-update resource profile, no policy selection'}
    try:
        model=RelationModel().cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-6,weight_decay=0,fused=True)
        owner=Ownership(model,CanonicalParameterRegistry(model));instrument=BertInstrument(model,chunk=4)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        model(x,y)[1].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        torch.cuda.synchronize();report['forward_backward_seconds']=time.perf_counter()-start
        start=time.perf_counter();owner.propose(opt);torch.cuda.synchronize();report['proposal_seconds']=time.perf_counter()-start
        start=time.perf_counter();m=instrument.measure(owner,x[:4],y[:4]);torch.cuda.synchronize()
        report.update(status='PASS',four_probe_instrument_seconds=time.perf_counter()-start,
            peak_allocated=torch.cuda.max_memory_allocated(),finite=bool(torch.isfinite(m.predicted_minima).all()),
            predicted_shape=list(m.predicted_minima.shape),parameters=sum(p.numel() for p in model.parameters()))
    except Exception:report.update(status='FAILED',traceback=traceback.format_exc(),peak_allocated=torch.cuda.max_memory_allocated())
    with (ROOT/'PROFILE.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
