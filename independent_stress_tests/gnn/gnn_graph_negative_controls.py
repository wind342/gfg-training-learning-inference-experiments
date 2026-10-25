"""Exploratory AFTER-FREEZE graph/query counterfactual tests.
Not part of the original frozen 9 gates. Retrains frozen source exactly,
confirms final state SHA, then evaluates graph and marker counterfactuals.
"""
import json,hashlib
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
import gnn_tlir_counterexample as e
ROOT=Path('/mnt/data')
formal=json.loads((ROOT/'gnn_tlir_formal_results.json').read_text())
output=[]

def predict(model,a,x):
 model.eval()
 with torch.no_grad():return model(a,x).argmax(-1)

def actual_twohop_labels(adjs,features):
 result=[]
 for a,x in zip(adjs,features):
  marker=int(torch.where(x[:,1]>0.5)[0].item())
  edges=(a>0).to(torch.float32)
  yes=(edges[0,marker]>0) or ((edges@edges)[0,marker]>0)
  result.append(int(yes))
 return torch.tensor(result,dtype=torch.int64)

for row in formal['runs']:
 seed=row['seed'];e.seed_all(seed)
 train=e.build_data(seed+100,e.TRAIN_N);test=e.build_data(seed+200,e.TEST_N)
 model=e.GraphNet();opt=torch.optim.RMSprop(model.parameters(),lr=e.LR,alpha=.95,eps=1e-8,momentum=0)
 rng=np.random.default_rng(seed+300)
 for step in range(1,e.STEPS+1):
  idx=torch.tensor(rng.integers(0,len(train[2]),size=e.BATCH),dtype=torch.long)
  model.train();opt.zero_grad(set_to_none=True)
  loss=F.cross_entropy(model(train[0][idx],train[1][idx]),train[2][idx]);loss.backward();opt.step()
  if step in (1,25,100,200,350,e.STEPS):e.accuracy(model,test)
 final_sha=e.state_sha(e.clone_state(model))
 assert final_sha==row['identities']['final_sha256'],f'RECONSTRUCTED_FINAL_CHECKPOINT_MISMATCH:{seed}'
 base=predict(model,test[0],test[1]);acc=float((base==test[2]).float().mean())
 # Graph structure shuffle: use another held-out graph adjacency but retain each query's anchor/marker, then recompute the TRUE topological label.
 order=np.random.default_rng(seed+666).permutation(len(test[2]))
 changed_adj=test[0][torch.tensor(order)]
 changed_y=actual_twohop_labels(changed_adj,test[1]);changed_pred=predict(model,changed_adj,test[1]);
 changed_count=int((changed_y!=test[2]).sum())
 # Counterfactual marker identity: relocate each marker to a distinct non-anchor vertex while retaining its edges.
 fx=test[1].clone()
 rng_marker=np.random.default_rng(seed+777)
 for x in fx:
  old=int(torch.where(x[:,1]>.5)[0].item())
  candidates=[n for n in range(1,e.N) if n!=old]
  x[old,1]=0; x[int(rng_marker.choice(candidates)),1]=1
 new_y=actual_twohop_labels(test[0],fx);new_pred=predict(model,test[0],fx)
 o={'seed':seed,'checkpoint_sha256':final_sha,'original_accuracy':acc,
 'shuffled_adjacency_recomputed_truth_accuracy':float((changed_pred==changed_y).float().mean()),
 'shuffled_adjacency_stale_truth_accuracy':float((changed_pred==test[2]).float().mean()),
 'shuffled_adjacency_label_flips':changed_count,
 'relocated_marker_recomputed_truth_accuracy':float((new_pred==new_y).float().mean()),
 'relocated_marker_stale_truth_accuracy':float((new_pred==test[2]).float().mean()),
 'relocated_marker_label_flips':int((new_y!=test[2]).sum())}
 output.append(o)
 print(json.dumps(o))
report={'kind':'EXPLORATORY_POSTHOC_NEGATIVE_CONTROLS_NOT_ORIGINAL_FORMAL_GATES','source_sha256':hashlib.sha256((ROOT/'gnn_tlir_counterexample.py').read_bytes()).hexdigest(),'runs':output}
p=ROOT/'gnn_graph_negative_controls.json';p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('saved',p)
