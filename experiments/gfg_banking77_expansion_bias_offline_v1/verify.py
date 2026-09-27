"""Independent scalar readout verification; no model or training imports."""
from pathlib import Path
import json,hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
AUTH=Path('E:/gfg-evidence/banking77-final-v2-expansion-audit-v1')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    path=HERE/'results/RESULTS.json';data=json.loads(path.read_text(encoding='utf8'))
    assert all(sha(p)==v for p,v in data['inputs_sha256'].items())
    assert all(sha(HERE/p)==v for p,v in data['sources_sha256'].items())
    checks=[]
    for r in data['results']:
        old=r['old_seen'];seen=r['new_seen'];task=seen//7
        saved=np.load(AUTH/f"budget{r['budget']}/task{task}/EXPANSION_LOGITS.npz")
        z=saved['logits'];labels=saved['labels'];ids=saved['ids'];gaps=[]
        before=[];raw=[]
        for row,y in zip(z,labels):
            b=all(row[y]>row[c] for c in range(old) if c!=y)
            a=all(row[y]>row[c] for c in range(seen) if c!=y)
            before.append(b);raw.append(a)
            if b and not a:gaps.append(float(max(row[old:seen]))-float(row[y]))
        beta=max(gaps)+1e-6;assert beta==r['beta_formula_fp64']
        assert float(np.float32(beta))==r['beta_applied_fp32']
        assert min(gaps)==r['gap_min'] and float(np.median(gaps))==r['gap_median'] and max(gaps)==r['gap_max']
        corrected=[];per=np.zeros((old,3),dtype=int)
        for i,(row,y) in enumerate(zip(z,labels)):
            new=[np.float32(v-np.float32(beta)) for v in row[old:seen]]
            competitors=[row[c] for c in range(old) if c!=y]+new
            c=bool(all(row[y]>v for v in competitors));corrected.append(c)
            rec=r['controls'][i]
            assert rec['id']==int(ids[i]) and rec['relation']==int(y)
            assert (rec['before'],rec['raw'],rec['corrected'])==(bool(before[i]),bool(raw[i]),c)
            assert rec['margin_corrected']==float(np.float32(row[y]-max(competitors)))
            per[y]+=np.array([before[i],raw[i],c],dtype=int)
        b=np.array(before);a=np.array(raw);c=np.array(corrected)
        assert (int(b.sum()),int(a.sum()),int(c.sum()))==(r['correct_before'],r['correct_after_raw'],r['correct_after_corrected'])
        assert int((b&~a&c).sum())==r['recovered_controls']
        assert bool(np.array_equal(b,c))==r['restored_before_mask']
        for j,rec in enumerate(r['per_relation']):
            assert [rec['before'],rec['raw'],rec['corrected']]==per[j].tolist()
        for key,value in dict(before_to_corrected_correct_to_wrong=int((b&~c).sum()),before_to_corrected_wrong_to_correct=int((~b&c).sum()),
            raw_to_corrected_correct_to_wrong=int((a&~c).sum()),raw_to_corrected_wrong_to_correct=int((~a&c).sum()),
            previously_zero_relations=int((per[:,0]==0).sum()),raw_newly_zero_relations=int(((per[:,0]>0)&(per[:,1]==0)).sum()),
            corrected_newly_zero_relations=int(((per[:,0]>0)&(per[:,2]==0)).sum()),corrected_total_zero_relations=int((per[:,2]==0).sum()),
            corrected_old_competitor_errors=int((~b&~c).sum())).items():assert r[key]==value,key
        checks.append(dict(boundary=r['boundary'],budget=r['budget'],status='PASS',independent_scalar_recalculation=True))
    out=dict(status='PASS',checks=checks,results_sha256=sha(path),verifier_sha256=sha(__file__),all_inputs_unchanged=True)
    with (HERE/'results/CHECKS.json').open('x',encoding='utf8',newline='\n') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
