"""I/O-only staging wrapper; frozen selection and training functions unchanged.

Avoid parsing each large trajectory eleven times during LOO selection. Cache
only fields read by crossing(); check compact vs full threshold answers before
discarding logits. The independent final audit uses original complete files.
"""
from functools import lru_cache
from unittest.mock import patch
from . import benchmark as b

original_selections=b.selections
original_read=b.read


def cached_selections(rows):
    @lru_cache(maxsize=256)
    def selection_read(path):
        r=original_read(path)
        compact={'initial':{'accuracy':r['initial']['accuracy']},'steps':[
            {'fact':{'t':s['fact']['t'],'evaluation':{'accuracy':s['fact']['evaluation']['accuracy']}}} for s in r['steps']]}
        for th in (.95,.99):
            b.require(b.crossing(r,th)==b.crossing(compact,th),'COMPACT_CROSSING_PARITY')
        return compact
    with patch.object(b,'read',selection_read):
        return original_selections(rows)


if __name__=='__main__':
    sealpath=b.HERE/'TIMING_RUNNER_SEAL.json'
    if not sealpath.exists():
        b.write(sealpath,{'sha256':b.sha(b.Path(__file__)),'utc':b.old.utc(),
            'scope':'I/O cache only. Original frozen selections() and update() unchanged. Full-vs-compact crossings checked on every input.'})
    else:
        b.require(original_read(sealpath)['sha256']==b.sha(b.Path(__file__)),'TIMING_WRAPPER_CHANGED')
    b.selections=cached_selections
    b.sys.argv=[__file__,'timing']
    b.main()
