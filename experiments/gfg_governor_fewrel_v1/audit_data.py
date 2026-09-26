"""Check locally used marker data against the pinned original FewRel records."""
from collections import Counter
import json
import numpy as np
from .prepare import ROOT, sha


def audit():
    root=ROOT/'sources'
    official={}
    for name in ('FewRel-data-train_wiki.json','FewRel-data-val_wiki.json'):
        part=json.loads((root/name).read_text());assert not(set(part)&set(official));official.update(part)
    marked=json.loads((root/'CRL-datasets-data_with_marker.json').read_text())
    assert set(official)==set(marked) and len(official)==80
    checked=[]
    for relation,rows in marked.items():
        observed=[]
        for row in rows:
            tokens=[];spans={};active={}
            for token in row['tokens']:
                if token in ('[E11]','[E21]'):
                    key=token[2];assert key not in active;active[key]=len(tokens)
                elif token in ('[E12]','[E22]'):
                    key=token[2];spans[key]=tuple(range(active.pop(key),len(tokens)))
                else:tokens.append(token)
            assert not active and set(spans)=={'1','2'}
            observed.append((tuple(tokens),spans['1'],spans['2']))
        expected=[(tuple(r['tokens']),tuple(r['h'][2][0]),tuple(r['t'][2][0])) for r in official[relation]]
        assert Counter(observed)==Counter(expected),relation
        checked.append(dict(relation=relation,count=len(rows)))
    data=np.load(ROOT/'data.npz');x,y,split=data['x'],data['y'],data['split']
    test={}
    for i in np.flatnonzero(split==2):test.setdefault((int(y[i]),x[i].tobytes()),[]).append(int(i))
    overlap=[]
    for k in range(80):
        for i in np.flatnonzero((y==k)&(split==1))[:10]:
            matches=test.get((int(y[i]),x[i].tobytes()),[])
            if matches:overlap.append(dict(probe_index=int(i),test_indices=matches,target=k))
    return dict(status='PASS_WITH_DISCLOSED_DUPLICATE',check='exact multiset of tokens + head/tail spans, all 80 relations',
        probe_test_content_overlap=overlap,
        duplicate_policy='original public split retained; secondary test metrics exclude matching probe content, never tune/retrain',
        relations=checked,sources={p.name:sha(p) for p in root.glob('*.json')})


if __name__=='__main__':print(json.dumps(audit(),indent=2))
