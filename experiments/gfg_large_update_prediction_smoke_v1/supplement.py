"""Three extra baseline-low cases; initial sweep stays sealed and unchanged."""
import time
import traceback
from . import run as r


def main():
    r.old.setup()
    original=r.old.read(r.OUT/'SEAL.json')
    rows=r.old.read(r.old.HERE/'RUN_ASSIGNMENT.json')['runs']
    ranking=[]
    for row in rows:
        p=r.Path('E:/gfg-evidence/optimizer-capability-benchmark-v1/search')/row['entry_id']/'controller-0/0/TRAJECTORY.json'
        d=r.old.read(p)
        ranking.append({'entry':row['entry_id'],'initial_accuracy':d['initial']['accuracy'],'source':str(p),'sha256':r.sha(p)})
    selected=sorted([d for d in ranking if d['entry'] not in original['entries']],key=lambda d:(d['initial_accuracy'],d['entry']))[:3]
    r.OUT=r.OUT.parent/'supplement-low-baseline'
    assert not r.OUT.exists()
    r.OUT.mkdir()
    r.write(r.OUT/'SEAL.json',{'time_ns':time.time_ns(),'base':r.git('rev-parse','HEAD'),
        'sources':{str(p.relative_to(r.REPO)):r.sha(p) for p in r.HERE.glob('*') if p.is_file()},
        'old_files':original['old_files'],'ranking':ranking,'entries':[d['entry'] for d in selected],
        'alphas':r.ALPHAS,'tag_object':original['tag_object'],'tag_commit':original['tag_commit'],
        'scope':'exploratory post-initial low-baseline addition'})
    for chosen in selected:
        row=next(row for row in rows if row['entry_id']==chosen['entry'])
        try:
            r.case(row)
        except Exception:
            r.write(r.OUT/(row['entry_id']+'-FAILURE.json'),{'traceback':traceback.format_exc()})
            raise


if __name__=='__main__':main()
