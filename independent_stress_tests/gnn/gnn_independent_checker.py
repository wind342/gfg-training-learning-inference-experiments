"""Read-only secondary verification of frozen GNN stress-test results.
Scientific boundaries: checks output structure, frozen contract and recomputes
aggregate verdicts; byte-identical native rerun is separately verified by SHA.
"""
import hashlib,json,sys
from pathlib import Path

ROOT=Path('/mnt/data')
def hash_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
 p=ROOT/'gnn_falsification_frozen.json';s=ROOT/'gnn_tlir_counterexample.py'
 f=ROOT/'gnn_tlir_formal_results.json';r=ROOT/'gnn_tlir_formal_replay.json'
 protocol=json.loads(p.read_text());result=json.loads(f.read_text());replay=json.loads(r.read_text())
 errors=[]
 def require(condition,label):
  if not condition:errors.append(label)
 require(hash_file(s)==protocol['source_sha256'],'SOURCE_CHANGED_AFTER_FREEZE')
 require(hash_file(ROOT/'gnn_tlir_pilot.json')==protocol['pilot_sha256'],'PILOT_RECORD_MISMATCH')
 require(hash_file(f)==hash_file(r) and result==replay,'FRESH_REEXECUTION_NOT_BYTE_IDENTICAL')
 require(result['kind']=='FORMAL' and result['source_sha256']==protocol['source_sha256'],'SOURCE_AUTHORITY_MISMATCH')
 require(result['seeds']==protocol['formal_seeds'],'FORMAL_SEEDS_CHANGED')
 require(len(result['runs'])==3,'RUN_COUNT_CHANGED')
 seed_results=[]
 for run in result['runs']:
  k=run['seed'];ex=run['receiving_state_exchanges'];curves=run['finite_amplitude'];support=run['support'];backs=run['rollbacks'];got=run['gates']
  require([x['later'] for x in ex]==[100,200,350] and [x['earlier'] for x in ex]==[25,100,200],f'{k}:RECEIVER_SELECTION_CHANGED')
  require([x['step'] for x in curves]==[25,100,200,350],f'{k}:UPDATE_AMPLITUDE_SAMPLE_CHANGED')
  require([x['step'] for x in support]==[25,100,200,350],f'{k}:SUPPORT_SAMPLE_CHANGED')
  require([x['component'] for x in backs]==[0,1,2,3],f'{k}:ROLLBACK_SET_CHANGED')
  n=sum(x['nonlinear_count'] for x in curves)
  total=sum(x['number'] for x in curves)
  require(total==256,f'{k}:TARGET_COUNT_CHANGED')
  recalculated={
   'training_capability':run['training_accuracy']>=0.85,
   'frozen_execution_integrity':run['execution_integrity']['frozen_inference_exact'],
   'same_update_different_receiver':sum(x['nrmse']>0.05 and max(x['received_rms'],x['alternate_rms'])>1e-4 for x in ex)>=2,
   'finite_amplitude_nonlinearity':n>=0.1*total,
   'distributed_support':sum(x['active_fraction'] for x in support)/4>=0.5,
   'persistent_support_reallocation':sum(x['reallocation_fraction'] for x in support)/4>=0.25,
   'frozen_query_conditioning':all(x['max_query_profile_l1']>0.05 for x in support),
   'nonadditive_complete_logits':all(x['max_complete_logit_pair_interaction_rms']>1e-6 for x in support),
   'exact_learned_version_rollback':all(x['maximum_full_output_difference']>1e-6 and x['exact_restoration'] for x in backs)
  }
  require(got==recalculated,f'{k}:VERDICT_RECOMPUTATION_MISMATCH')
  require(run['identities']['model_initial_sha256']!=run['identities']['final_sha256'],f'{k}:NO_LEARNED_STATE_CHANGE')
  require(len(run['identities']['updates'])==4,f'{k}:UPDATE_IDENTITY_MISSING')
  seed_results.append({'seed':k,'trained_accuracy':run['training_accuracy'],'nonlinear_count':n,'nonlinear_total':total,'nonlinear_fraction':n/total,'support_reallocation_fraction_mean':sum(x['reallocation_fraction'] for x in support)/4,'all_gates':got,'pass_count':sum(got.values())})
 return {'status':'PASS' if not errors else 'FAIL','errors':errors,'authority':'FROZEN_SOURCE_PLUS_FRESH_BYTE_IDENTICAL_REEXECUTION_AND_AGGREGATE_RECOMPUTATION','frozen_protocol_sha256':hash_file(p),'source_sha256':hash_file(s),'formal_results_sha256':hash_file(f),'fresh_replay_sha256':hash_file(r),'seed_results':seed_results,'notice':'Not a GFG Core v3 validation; does not independently verify absent raw tensor receipts.'}
if __name__=='__main__':
 out=check(); p=ROOT/'gnn_falsification_audit.json';p.write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='seed_results'},ensure_ascii=False,indent=2));print('seed_summary',[(x['seed'],x['pass_count'],x['nonlinear_count'],x['support_reallocation_fraction_mean']) for x in out['seed_results']]);
 sys.exit(0 if out['status']=='PASS' else 1)
