"""Read allowlist for experiment authorities; no recursive evidence scans."""
import os,sys,json
from pathlib import Path
from collections import Counter

class Access:
    def __init__(self,root,old,v1,repo):
        self.root=Path(root).resolve();self.old=Path(old).resolve();self.v1=Path(v1).resolve();self.repo=Path(repo).resolve()
        self.evidence=self.old.parent;self.allowed={self.old/'CACHE.json',self.old/'TRAIN_FEATURES.pt',self.old/'NUMERICS.json'}
        for b in (210,630):
            self.allowed.add(self.old/f'formal/native_{b}/task1/CHECKPOINT.pt')
            self.allowed.add(self.v1/f'smoke/jacobian_{b}/task2/CHECKS.json')
        self.allowed.update(self.old/name for name in ('FORMAL_SOURCE_FREEZE.json','smoke_predictor/CHECKS.json','smoke_predictor/STEPS_210.jsonl.gz','smoke_predictor/STEPS_630.jsonl.gz','formal/predictive_210/SEALED.json','formal/predictive_630/SEALED.json'))
        self.reads=Counter();self.denied=[];self.model_root=Path('E:/gfg-models/bert-base-uncased-v4').resolve();self.models=set()
        sys.addaudithook(self.hook)
    def hook(self,event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        forbidden=False
        if p.is_relative_to(self.evidence) and not p.is_relative_to(self.root):
            forbidden=writing or p not in self.allowed
            if not forbidden:self.reads[str(p)]+=1
        if p.is_relative_to(self.repo/'experiments') and not p.is_relative_to(self.repo/'experiments/gfg_banking77_final_v2_predictor_task2_v1'):
            # Only inspect the frozen v1 implementation/protocol, never old results or test artifacts.
            allowed_source=p.parent in (self.repo/'experiments/gfg_banking77_jacobian_hildreth_v1',self.repo/'experiments/gfg_banking77_v2_cap_recheck_smoke_v1') and p.suffix in ('.py','.md')
            forbidden=forbidden or writing or not allowed_source
        if p.is_relative_to(self.model_root):
            forbidden=forbidden or writing or p not in self.models
            if not forbidden:self.reads[str(p)]+=1
        if forbidden:
            self.denied.append(dict(path=str(p),writing=writing));raise PermissionError('Outside smoke input allowlist: '+str(p))
    def model_files(self,names):self.models={self.model_root/name for name in names}
    def report(self):return dict(allowed_external_reads=dict(self.reads),denied=self.denied,test_files_opened=0,
        scope='Python open audit hook for evidence/model/previous-experiment paths; excludes interpreter/OS/native-library internals. No subprocess experiment data readers.')
