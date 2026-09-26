"""Independent V4 fork, inheriting the fixed three-task V3 harness."""
import argparse
import json
from pathlib import Path
import subprocess
from experiments.gfg_router_v3_three_task_continuation import run as prior
from experiments.gfg_router_v3_three_task_continuation import gate as prior_gate
from experiments.gfg_governor_fewrel_v1 import run as fewrel
from .policy import Router,PartialSentinel,complete_old

HERE=Path(__file__).resolve().parent
ROOT=Path('E:/gfg-evidence/router-v4-partial-capability')
BASE='c67de1af5ca0379b0db9f2fe20e624344e324451'

def unchanged():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=HERE.parents[1],check=True,capture_output=True)

def admit():
    unchanged();prior_gate.unchanged_base=unchanged
    gate=prior_gate.audit_smoke()
    old=prior_gate.ROOT/'CHECKS.json'
    authority=json.loads(old.read_text())
    public=json.loads((prior.HERE/'results/CHECKS.json').read_text())
    frozen=json.loads(subprocess.check_output(['git','show',BASE+':experiments/gfg_router_v3_three_task_continuation/results/CHECKS.json'],cwd=HERE.parents[1]))
    assert authority==public==frozen and authority['status']=='PASS'
    for arm,a in authority['arms'].items():
        for f,h in a['hashes'].items():assert prior_gate.v3check.base.sha(prior_gate.ROOT/f'three-{arm}'/f)==h
    gate.update(v4_base=BASE,prior_three_checks_sha256=prior_gate.v3check.base.sha(old))
    ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/'SMOKE_CHECKS.json'
    if path.exists():assert json.loads(path.read_text())==gate
    else:prior_gate.v3check.base.write(path,gate)
    return gate

def setup(gate):
    base=prior.setup(gate);base.HERE=HERE;base.CORE=HERE;base.ROOT=ROOT;base.Router=Router
    fewrel.GroupSentinel=PartialSentinel
    transitions=[];original_extend=base.extend_bank;original_write=base.write
    def extend(engine,*args):
        completed=set(engine.scheduler.mapping.labels)
        original_extend(engine,*args)
        converted=complete_old(engine.scheduler,completed)
        transitions.append(dict(step=engine.scheduler.last_step,completed=sorted(completed),converted=converted,
            states=list(engine.scheduler.state)))
    base.extend_bank=extend
    def write(path,obj):
        if path.name=='CONFIG.json':obj=dict(obj,v4_base=BASE,partial_rule='completed FORMING -> PARTIAL; even global steps one rotated partial sharing anchor slot',policy_sha256=base.sha(HERE/'policy.py'))
        if path.name=='RESULT.json':obj=dict(obj,partial_boundary_transitions=transitions)
        return original_write(path,obj)
    base.write=write
    return base

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=('native','replay','router'));a=p.parse_args()
    g=admit();print('V3_SMOKE_AND_THREE_TASK_ADMISSION_PASS',flush=True)
    setup(g).run(a.arm,'three')
