"""Export compact audit records only; never publish datasets or checkpoints."""
import json
from pathlib import Path
from .prepare import ROOT, sha
from .report import summarize
from .audit_data import audit


def main():
    out=Path(__file__).resolve().parent/'results';out.mkdir(exist_ok=True)
    with (out/'DATA_EQUIVALENCE.json').open('x',encoding='utf8') as f:json.dump(audit(),f,indent=2)
    for tasks in (3,10):
        report=summarize(tasks)
        for arm in ('native','governor'):
            for checkpoint in report[arm]['checkpoints']:
                assert sha(Path(checkpoint['path']))==checkpoint['sha256']
        with (out/f'{tasks}tasks-CHECKS.json').open('x',encoding='utf8') as f:
            json.dump(report,f,indent=2,allow_nan=False)
    data=json.loads((ROOT/'DATA_MANIFEST.json').read_text())
    compact={k:v for k,v in data.items() if k!='sample_ids'}
    compact['full_manifest_sha256']=sha(ROOT/'DATA_MANIFEST.json')
    compact['sample_identity_count']=len(data['sample_ids'])
    with (out/'DATA_AUTHORITY.json').open('x',encoding='utf8') as f:json.dump(compact,f,indent=2)
    profile=json.loads((ROOT/'PROFILE.json').read_text())
    with (out/'PROFILE.json').open('x',encoding='utf8') as f:json.dump(profile,f,indent=2)
    replay=json.loads((ROOT/'CHECKPOINT_REPLAY.json').read_text())
    assert all(row['status']=='PASS' for row in replay.values())
    with (out/'CHECKPOINT_REPLAY.json').open('x',encoding='utf8') as f:json.dump(replay,f,indent=2)
    print('PASS: full logs, decisions, test metrics, checkpoint hashes; exported compact records only')


if __name__=='__main__':main()
