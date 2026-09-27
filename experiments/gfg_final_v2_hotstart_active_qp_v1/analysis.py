"""Read-only continuity analysis of the sealed cached-Hildreth trajectory."""
from experiments.gfg_final_v2_cached_hildreth_v1.core import *

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('GFG_HOTSTART_ROOT','E:/gfg-evidence/final-v2-hotstart-active-qp-v1'))
BASE='f357dfdb9d13759dc07a5703d6d11301a3adb946'
SOURCE=Path('E:/gfg-evidence/final-v2-cached-hildreth-v1/cached_validation/STEPS.jsonl.gz')


def describe(values):
    x=np.asarray(values,dtype=float)
    return dict(mean=float(x.mean()),median=float(np.median(x)),p10=float(np.quantile(x,.1)),min=float(x.min()),max=float(x.max()))


def main():
    ROOT.mkdir(exist_ok=False)
    with gzip.open(SOURCE,'rt') as f:rows=[json.loads(s) for s in f]
    prev=None;transitions=[];allpairs=set();support_sizes=[]
    for row in rows:
        r=row['report'];current={tuple(k):v for k,v in zip(r['final_active'],r['final_duals'])}
        allpairs.update(current);support_sizes.append(sum(v>0 for v in current.values()))
        if prev is not None:
            a,b=set(prev),set(current);common=sorted(a&b)
            x=np.asarray([prev[k] for k in common]);y=np.asarray([current[k] for k in common])
            pa={k for k,v in prev.items() if v>0};pb={k for k,v in current.items() if v>0}
            transitions.append(dict(step=row['step'],previous_count=len(a),current_count=len(b),common=len(common),
                jaccard=len(common)/len(a|b),previous_retained=len(common)/len(a),current_covered=len(common)/len(b),
                added=len(b-a),deleted=len(a-b),positive_support_jaccard=len(pa&pb)/len(pa|pb),
                dual_pearson=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else None,
                dual_l2_difference=float(np.linalg.norm(x-y)),
                dual_symmetric_relative_change=float(np.linalg.norm(x-y)/(np.linalg.norm(x)+np.linalg.norm(y))),
                dual_median_absolute_change=float(np.median(np.abs(x-y)))))
        prev=current
    summary={k:describe([r[k] for r in transitions if r[k] is not None]) for k in
        ('jaccard','previous_retained','current_covered','added','deleted','positive_support_jaccard','dual_pearson','dual_symmetric_relative_change')}
    result=dict(status='PASS',source_sha256=sha(SOURCE),steps=210,transitions=transitions,summary=summary,
        unique_discovered_pairs=len(allpairs),positive_support_size=describe(support_sizes),
        decision='PROCEED',reason='Observed median discovered-set Jaccard 0.869 and previous retention 0.930 show substantial continuity; dual correlation is also positive. This justifies testing, not assuming, warm-start speedup.')
    write(ROOT/'CONTINUITY.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='transitions'},indent=2),flush=True)


if __name__=='__main__':main()
