"""Explicit first-step resource retry; no scientific settings are changed."""
import json
from pathlib import Path
import torch
from . import run as original

def main():
    failed=original.ROOT/'replay_governor/RESULT.json'
    result=json.loads(failed.read_text())
    assert result['status']=='FAILED' and result['steps']==0
    assert 'workspace budget cannot fit the minimum exact tile' in result['traceback']
    base=original.BertInstrument
    class ReleaseUnusedCache(base):
        def measure(self,*args,**kwargs):
            # The planner queries driver free memory, which excludes unused
            # PyTorch reservations. Free only unused cache; preserve live tensors.
            torch.cuda.empty_cache()
            return super().measure(*args,**kwargs)
    original.BertInstrument=ReleaseUnusedCache
    original.ROOT=original.ROOT/'resource_retry'
    raw_write=original.write
    def annotate(path,obj):
        if path.name=='CONFIG.json':
            obj=dict(obj,resource_retry=dict(wrapper_sha256=original.sha(Path(__file__)),
                failed_result_sha256=original.sha(failed),
                change='torch.cuda.empty_cache immediately before full measurement only'))
        return raw_write(path,obj)
    original.write=annotate
    original.run('replay_governor')

if __name__=='__main__':main()
