"""Read-only V3 smoke authority validation and derived continuation admission."""
import json
import subprocess
from pathlib import Path
from experiments.gfg_functional_feedback_router_v3 import run as v3
from experiments.gfg_functional_feedback_router_v3 import check as v3check

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/router-v3-three-task-continuation')
BASE='62a658ca07060f7be76436b6ba10072233b25456'

def normalize(x):return json.loads(json.dumps(x))

def validate_summary(checks,result,rows):
    if checks.get('status')!='PASS' or 'router_v3' not in checks:
        raise ValueError('V3 smoke CHECKS must be PASS with router_v3 schema')
    if result.get('status')!='COMPLETED' or result.get('steps')!=60 or len(rows)!=60:
        raise ValueError('V3 smoke must contain exactly 60 completed updates')
    if [r['step'] for r in rows]!=list(range(211,271)) or any(r['task']!=2 for r in rows):
        raise ValueError('unexpected smoke task/step identities')
    if checks['router_v3']['metrics'][-1]['offset']!=60:
        raise ValueError('smoke endpoint missing')

def unchanged_base():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)

def audit_smoke():
    unchanged_base()
    raw=v3.ROOT
    checks=json.loads((raw/'CHECKS.json').read_text())
    public=json.loads((v3.HERE/'results/CHECKS.json').read_text())
    committed=json.loads(subprocess.check_output(['git','show',BASE+':experiments/gfg_functional_feedback_router_v3/results/CHECKS.json'],cwd=REPO))
    if checks!=public or checks!=committed:raise ValueError('V3 CHECKS authority changed')
    result=json.loads((raw/'smoke-router/RESULT.json').read_text())
    rows=list(map(json.loads,(raw/'smoke-router/STEPS.jsonl').read_text().splitlines()))
    validate_summary(checks,result,rows)
    base=v3check.base;base.ROOT=raw;base.HERE=v3.HERE;base.CORE=v3.HERE;base.Router=v3check.Router
    summary,_=base.verify('router','smoke')
    for key,value in summary.items():
        if normalize(value)!=checks['router_v3'][key]:raise ValueError('smoke recomputation mismatch: '+key)
    c=json.loads((raw/'smoke-router/CONFIG.json').read_text())
    for field in ('inherited_sources','router_v2_sources'):
        for rel,h in c[field].items():
            if base.sha(HERE.parent/rel)!=h:raise ValueError('inherited source changed: '+rel)
    for row in rows:
        i=row['info'];nc=len(row['new_ids']);no=len(i['feedback_example_ids'])
        if i['old_loss_weight']!=no/(nc+no) or i['current_loss_weight']!=nc/(nc+no):
            raise ValueError('invalid V3 smoke dose')
    return dict(status='PASS',three_task_gate=True,base_commit=BASE,
        authority=str(raw/'CHECKS.json'),authority_sha256=base.sha(raw/'CHECKS.json'),
        smoke_hashes=summary['hashes'],smoke_steps=60,
        rule='V3 CHECKS authority, completed 60-step records, hashes, routing, gate, dose and metrics rechecked; no policy or efficacy threshold added')

def admit():
    gate=audit_smoke();ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'SMOKE_CHECKS.json'
    if path.exists():
        if json.loads(path.read_text())!=gate:raise ValueError('continuation admission identity changed')
    else:v3check.base.write(path,gate)
    return gate
