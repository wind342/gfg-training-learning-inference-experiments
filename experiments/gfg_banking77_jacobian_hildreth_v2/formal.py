"""Continue the two successfully audited fixed smoke trajectories, no tuning."""
from .run import *

def main():
    guard();assert read(ROOT/'SMOKE.json')['status']=='PASS';assert read(ROOT/'AUDIT.json')['status']=='PASS'
    for name,digest in read(ROOT/'SMOKE_SOURCE_FREEZE.json')['sources'].items():assert sha(HERE/name)==digest
    write(ROOT/'FORMAL_SOURCE_FREEZE.json',dict(sources={p.name:sha(p) for p in HERE.glob('*.py')},protocol_sha256=sha(HERE/'PROTOCOL.md')))
    reports=[]
    for budget in (210,630):
        rr=[read(OLD/'formal'/f'native_{budget}'/'task1/CHECKS.json'),read(ROOT/'smoke'/f'jacobian_{budget}'/'task2/CHECKS.json')]
        for task in range(3,12):rr.append(run('jacobian',budget,'formal',task))
        write(ROOT/'formal'/f'jacobian_{budget}'/'SEALED.json',dict(status='PASS',budget=budget,tasks=rr,test_evaluated=False));reports.append(rr)
    write(ROOT/'FORMAL.json',dict(status='PASS',streams=reports,test_evaluated=False))

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(ROOT/f'FORMAL_FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
