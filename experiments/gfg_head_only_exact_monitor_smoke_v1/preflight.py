import io
import unittest
from .core import *
from .tests import Checks

def main():
    unchanged();ROOT.mkdir(exist_ok=False)
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(stream.getvalue(),flush=True)
    audit=data_audit(np.load(DATA/'data.npz'))
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=FrozenRelationModel().cuda();model.load_state_dict(saved['model']);freeze(model)
    shadow,opt=make_candidate(model,saved['optimizer'])
    assert equal(model.head.state_dict(),{n:saved['model']['head.'+n] for n in ('weight','bias')})
    assert equal(model.head.state_dict(),shadow.state_dict())
    report=dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,test_log=stream.getvalue(),
        data_audit=audit,checkpoint_sha256=CKSHA,data_sha256=DATA_SHA,source_hashes=sources(),
        head_initial_sha256=head_hash(model.head),head_optimizer_initial_sha256=moments_hash(opt),
        backbone_initial_sha256=tensor_hash(model.bert.state_dict()),torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    write(ROOT/'PREFLIGHT.json',report);assert report['status']=='PASS';print('PREFLIGHT PASS',flush=True)

if __name__=='__main__':main()
