"""Post-outcome aggregation only; no model execution or old-file writes."""
from pathlib import Path
import json
import shutil
import numpy as np
from . import verify as v


def write(p,d):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main():
    dest=v.HERE/'verification';dest.mkdir(exist_ok=False)
    base=v.OUT.parent;reports=[];raw=[]
    for name in ['attempt-3','supplement-low-baseline']:
        v.OUT=base/name;r=v.verify();reports.append(r)
        write(dest/(name+'-RESULTS.json'),r)
        shutil.copy2(v.OUT/'SEAL.json',dest/(name+'-SEAL.json'))
        for case in r['cases']:
            p=v.OUT/case['entry'];prefix=name+'-'+case['entry']
            for f in ['PREDICTION_SEAL.json','EXECUTION.json']:
                shutil.copy2(p/f,dest/(prefix+'-'+f))
            raw.extend({'path':str(p/f),'sha256':v.sha(p/f),'bytes':(p/f).stat().st_size} for f in ['PREDICTIONS.npz','ACTUAL.npz'])
    for attempt in ['attempt-1','attempt-2']:
        for p in (base/attempt).glob('*FAILURE.json'):
            shutil.copy2(p,dest/(attempt+'-'+p.name))
    cases=[c for r in reports for c in r['cases']]
    low=[c for c in cases if c['subsets']['control']['baseline_accuracy']<.5]
    summary={'integrity':'PASS','initial_cases':3,'supplementary_cases':3,'low_baseline_cases':len(low),
        'scope':'100352-parameter historical model; one-step fixed-direction forks; no full training speed claim',
        'all_six_predicted_choices_match_actual_best_control':all(c['selected_alpha']==c['subsets']['control']['oracle_best_actual_alpha'] for c in cases),
        'low_baseline_mean_accuracy':float(np.mean([c['subsets']['control']['baseline_accuracy'] for c in low])),
        'low_baseline_selected_mean_accuracy':float(np.mean([c['subsets']['control']['selected_actual_accuracy'] for c in low])),
        'low_baseline_table':[], 'raw_artifacts':raw}
    for j,a in enumerate([0,.25,.5,1,2,4,8,16]):
        summary['low_baseline_table'].append({'alpha':a,**{k:float(np.mean([c['subsets']['control']['rows'][j][k] for c in low])) for k in ['predicted_accuracy','actual_accuracy','agreement','balanced_recall_observed_classes']}})
    write(dest/'SUMMARY.json',summary)
    print(json.dumps({k:val for k,val in summary.items() if k!='raw_artifacts'},indent=2))


if __name__=='__main__':main()
