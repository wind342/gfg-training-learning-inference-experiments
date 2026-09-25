"""Read-only independent NumPy arithmetic/statistics and identity audit."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUT=Path('E:/gfg-evidence/large-update-prediction-smoke-v1/attempt-3')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metrics(pred,actual,baseline):
    pc=pred>0; ac=actual>0; bc=baseline>0
    transition_true=2*bc.astype(int)+ac
    transition_pred=2*bc.astype(int)+pc
    recalls=[float((transition_pred[transition_true==i]==i).mean()) for i in range(4) if np.any(transition_true==i)]
    balanced=[float((pc[ac==i]==i).mean()) for i in (False,True) if np.any(ac==i)]
    finite=np.isfinite(actual)&np.isfinite(pred)
    return {'n':len(pc),'predicted_accuracy':float(pc.mean()),'actual_accuracy':float(ac.mean()),
        'agreement':float((pc==ac).mean()),'balanced_recall_observed_classes':float(np.mean(balanced)),
        'transition_macro_recall_observed_classes':float(np.mean(recalls)),
        'gained':int((~bc&ac).sum()),'lost':int((bc&~ac).sum()),
        'false_positive':int((pc&~ac).sum()),'false_negative':int((~pc&ac).sum()),
        'mean_abs_gap_error_finite':float(np.abs(pred[finite]-actual[finite]).mean()) if finite.any() else None,
        'finite_endpoint_count':int(finite.sum())}


def verify():
    seal=read(OUT/'SEAL.json');alphas=seal['alphas'];reports=[];aggregated={}
    for n,d in {**seal['old_files'],**seal['sources']}.items():assert sha(REPO/n)==d,n
    for suffix,key in [('', 'tag_object'),('^{}','tag_commit')]:
        assert subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release'+suffix],cwd=REPO,text=True).strip()==seal[key]
    for entry in seal['entries']:
        p=OUT/entry;s=read(p/'PREDICTION_SEAL.json');e=read(p/'EXECUTION.json')
        assert sha(p/'PREDICTIONS.npz')==s['predictions_sha256']
        assert sha(p/'ACTUAL.npz')==e['actual_sha256']
        assert all(r['time_ns']>s['time_ns'] for r in e['receipts'])
        for r in e['receipts']:assert r['state']['optimizer']==s['proposal_state']['optimizer']
        with np.load(p/'PREDICTIONS.npz') as raw:
            c=raw['coordinates'];y=raw['targets'];pred=raw['predicted_minima'];z0=raw['baseline_logits']
        with np.load(p/'ACTUAL.npz') as raw:z=raw['logits']
        mask=np.arange(z.shape[-1])[None,:]==y[:,None]
        def gaps(logits):
            g=logits[np.arange(len(y)),y].astype(np.float64)[:,None]-logits.astype(np.float64)
            g[mask]=np.inf
            return g.min(1)
        baseline=gaps(z0)
        actual=np.stack([gaps(zz) for zz in z])
        for j,a in enumerate(alphas):
            endpoint=((c[0]+a*c[1])+a*a*c[2])+a*a*c[3]
            endpoint[mask]=np.inf
            assert np.array_equal(endpoint.min(1),pred[j]),(entry,a)
        assert np.array_equal(z[0],z0)
        assert e['restoration_exact'] and e['alpha1_native_exact']
        order=lambda j:(int((pred[j,:212]>0).sum()),float(np.median(pred[j,:212])),-abs(alphas[j]-1),-alphas[j])
        chosen=max(range(1,len(alphas)),key=order)
        assert alphas[chosen]==s['selected_alpha']
        assert alphas[max((1,2,3),key=order)]==s['restricted_alpha']
        report={'entry':entry,'selected_alpha':s['selected_alpha'],'restricted_alpha':s['restricted_alpha'],'subsets':{}}
        for label,sl in [('control',slice(0,212)),('selection_disjoint',slice(212,None)),('all',slice(None))]:
            rows=[]
            for j,a in enumerate(alphas):
                rows.append({'alpha':a,**metrics(pred[j,sl],actual[j,sl],baseline[sl])})
                aggregated.setdefault((label,j),[]).append((pred[j,sl],actual[j,sl],baseline[sl]))
            report['subsets'][label]={'baseline_accuracy':float((baseline[sl]>0).mean()),'rows':rows,
                'selected_actual_accuracy':rows[chosen]['actual_accuracy'],
                'oracle_best_actual_alpha':max(range(1,len(alphas)),key=lambda j:rows[j]['actual_accuracy'])}
            report['subsets'][label]['oracle_best_actual_alpha']=alphas[report['subsets'][label]['oracle_best_actual_alpha']]
        reports.append(report)
    aggregate={}
    for (label,j),values in aggregated.items():
        pred,actual,baseline=[np.concatenate(v) for v in zip(*values)]
        aggregate.setdefault(label,[]).append({'alpha':alphas[j],**metrics(pred,actual,baseline)})
    return {'integrity':'PASS','cases':reports,'aggregate':aggregate,
        'limitations':['Three historical small-model states, not 70M/160M','One-step forks, not time-to-capability','No generalization or statistical confirmation claim']}


if __name__=='__main__':
    print(json.dumps(verify(),indent=2,sort_keys=True,allow_nan=False))
