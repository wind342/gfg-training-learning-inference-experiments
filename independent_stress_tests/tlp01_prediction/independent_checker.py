"""Read-only post-seal results and identity audit of TL-P01 cross-system run.

Independent of experiment's metric implementation; checks hashes, source
checkpoint identities, predictions-to-label bijection, native endpoint logits,
all recorded transition counts and false/true label outcomes.
"""
import collections,hashlib,json,sys
from pathlib import Path
import torch
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/data');sys.path.insert(0,str(SOURCE))
import gnn_tlir_counterexample as g
import snn_tlir_adversarial as s


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(cond,desc):
 if not cond:raise AssertionError(desc)


def check(path):
 path=Path(path)
 frozen=json.loads((ROOT/'FROZEN_PROTOCOL.json').read_text())
 require(frozen['predictor_source_sha256']==sha(ROOT/'tlp01_cross_system.py'),'code_hash')
 for key,name in [('gnn_source_sha256','gnn_tlir_counterexample.py'),('snn_source_sha256','snn_tlir_adversarial.py')]:require(frozen[key]==sha(SOURCE/name),name)
 prepared=json.loads((path/'PREPARED_MANIFEST.json').read_text())
 sealed=json.loads((path/'PREDICTIONS_SEALED.json').read_text())
 receipt=json.loads((path/'PREDICTIONS_SEAL_RECEIPT.json').read_text())
 report=json.loads((path/'ADJUDICATION_REPORT.json').read_text())
 original=json.loads((path/'ADJUDICATION.json').read_text())
 require(receipt['prediction_sha256']==sha(path/'PREDICTIONS_SEALED.json')==original['pretarget_seal_sha256'],'SEAL_HASH')
 require(prepared['frozen_protocol_sha256']==sha(ROOT/'FROZEN_PROTOCOL.json')==report['frozen_protocol_sha256'],'FROZEN_HASH')
 require(sealed['prepared_manifest_sha256']==sha(path/'PREPARED_MANIFEST.json'),'PREPARED_HASH')
 require(report['full_adjudication_sha256']==sha(path/'ADJUDICATION.json'),'RESULT_HASH')
 recorded={(e['system'],e['seed'],e['step']):e for e in sealed['events']}
 require(len(recorded)==len(sealed['events'])==28,'EVENT_COUNT')
 original_ledger={r['record_id']:r for r in original['ledger']}
 require(len(original_ledger)==len(original['ledger'])==10272,'LEDGER_ID_UNIQUENESS')
 # Four-way metrics independently recomputed without calling experiment metrics().
 summary={}
 for ref in prepared['entries']:
  sysid,seed=ref['system'],ref['seed']
  file=path/'pretarget'/ref['pretarget_file']
  require(sha(file)==ref['pretarget_sha256'],'PRETARGET_HASH')
  require(sha(ref['native_endpoint_reference'])==ref['native_endpoint_sha256'],'POST_NATIVE_HASH')
  pre=torch.load(file,map_location='cpu',weights_only=False)
  labels=pre['labels']
  if sysid=='gnn':
   original_results=json.loads((SOURCE/'gnn_tlir_formal_results.json').read_text())
   native_sha=next(r['identities'] for r in original_results['runs'] if r['seed']==seed)
   targets=pre['inputs']['adj'],pre['inputs']['x']
   model=g.GraphNet().eval()
   native=torch.load(ref['native_endpoint_reference'],map_location='cpu',weights_only=False)
  else:
   model=s.HardSpikingNetwork().eval()
   native=torch.load(ref['native_endpoint_reference'],map_location='cpu',weights_only=False)['snapshots']
   require(pre['original_native_sha256']==sha(ref['native_endpoint_reference']),'ORIGINAL_ARCHIVE')
  for step,p_event in sorted(pre['events'].items()):
   seal=recorded[(sysid,seed,step)]
   hash_pre=(g.state_sha if sysid=='gnn' else s.state_hash)(p_event['pre'])
   hash_delta=(g.state_sha if sysid=='gnn' else s.state_hash)(p_event['delta'])
   require(hash_pre==seal['state_hash']==p_event['pre_sha'],'PRE_STATE_HASH')
   require(hash_delta==seal['delta_hash']==p_event['delta_sha'],'ACTUAL_DELTA_HASH')
   if sysid=='gnn':
    orig_row=next(x for x in native_sha['updates'] if x['step']==step)
    require(orig_row['pre_sha']==hash_pre and orig_row['update_sha']==hash_delta,'GNN_ORIGINAL_MANIFEST')
    state=native[step]['post'];g.restore_state(model,state)
    with torch.no_grad():forward=model(*targets)
   else:
    state=native[step]['post'];s.restore(model,state)
    with torch.no_grad():forward=model(pre['inputs']['images'])
   hash_post=(g.state_sha if sysid=='gnn' else s.state_hash)(state)
   require(hash_post==p_event['post_sha'],'POST_STATE_HASH')
   pred={r['target_id']:r for r in seal['predictions']}
   require(len(pred)==len(labels)==seal['record_count'],'TARGETS_UNCHANGED')
   for i,target_id in enumerate(pre['ids']):
    name=str(target_id)
    require(name in pred,'TARGET_MATCH')
    claim=pred[name];require(claim['target_class']==int(labels[i]),'CLASS_MATCH')
    truth=int(labels[i]); true_val=float(forward[i,truth]); rivals=forward[i].clone();rivals[truth]=-float('inf')
    observed=bool(true_val>float(rivals.max()))
    r=original_ledger[claim['record_id']]
    require(r['actual']==observed,'READOUT_TRUTH_DISAGREEMENT')
    require(r['pre_correct']==claim['pre_correct'],'PRECORRECT_DISAGREEMENT')
    require(r['methods']==claim['methods'],'SEALED_PRED_DISAGREEMENT')
    require(r['true_transition']==('MAINTAIN_CORRECT' if r['pre_correct'] and observed else 'CORRECT_TO_WRONG' if r['pre_correct'] else 'WRONG_TO_CORRECT' if observed else 'MAINTAIN_WRONG'),'TRANSITION_WRONG')
    summary.setdefault(sysid,[]).append((observed,claim['methods']['quadratic_complete']['correct'],r['true_transition']))
 for sysid,records in summary.items():
  d=report['system_summary'][sysid]['metrics']['quadratic_complete']
  require(d['count']==len(records),'COUNT')
  require(d['correct']==sum(x[0]==x[1] for x in records),'CORRECT_COUNT')
  counts=collections.Counter(r[2] for r in records)
  for name,support in d['transition_supports'].items():
   require(support==counts[name],'TRANSITION_SUPPORT')
   rate=sum(x[0]==x[1] for x in records if x[2]==name)/support if support else None
   if rate is not None:require(abs(rate-d['transition_recall'][name])<1e-12,'TRANSITION_RECALL')
  require(abs(d['accuracy']-d['correct']/d['count'])<1e-12,'ACCURACY')
  print(sysid,'COUNT',len(records),'CORRECT',d['correct'],'ACC',d['accuracy'],'MACRO',d['four_way_macro_recall'])
 print('INDEPENDENT_AUDIT_PASS')
 return {'status':'PASS','checked_targets':sum(len(v) for v in summary.values()),'checked_events':len(recorded),
         'prediction_sha256':sha(path/'PREDICTIONS_SEALED.json'),'adjudication_sha256':sha(path/'ADJUDICATION.json')}

if __name__=='__main__':
 r=check(Path(sys.argv[1] if len(sys.argv)>1 else ROOT/'artifacts'))
 (ROOT/'INDEPENDENT_CHECK_RESULT.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
