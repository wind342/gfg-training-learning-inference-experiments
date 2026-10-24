"""Cross-system execution of the original TL-P01 quadratic_complete boundary formula.

GNN: exact nested PyTorch forward-mode JVP for h, h', h''.
SNN: true directional derivatives of the hard-spike hidden output are zero at
non-threshold points. The training-only surrogate gradient is NOT substituted.

The frozen stage order is PREPARE -> SEAL -> ADJUDICATE. The SEAL command never
loads an evaluation endpoint or the original saved post-state tensor file.
The affine output head is represented by appending a constant-one hidden
coordinate, so the original weight-only TL-P01 equations remain unchanged.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=Path('/mnt/data')
sys.path.insert(0,str(SOURCE_ROOT))
import gnn_tlir_counterexample as gnn
import snn_tlir_adversarial as snn

METHODS=('unchanged','linear','joint_rotation','hidden_curvature','quadratic_complete')
STEPS_GNN=(25,100,200,350)
STEPS_SNN=(50,200,400,700)
GNN_SEEDS=(20261024,20261025,20261026)
SNN_SEEDS=(71001,71002,71003,71004)


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')


def require(ok,msg):
    if not ok:raise RuntimeError(msg)


def validate_frozen():
    c=json.loads((ROOT/'FROZEN_PROTOCOL.json').read_text('utf8'))
    require(c['predictor_source_sha256']==sha_file(__file__),'PREDICTOR_SOURCE_HAS_CHANGED')
    require(c['gnn_source_sha256']==sha_file(SOURCE_ROOT/'gnn_tlir_counterexample.py'),'GNN_SOURCE_HAS_CHANGED')
    require(c['snn_source_sha256']==sha_file(SOURCE_ROOT/'snn_tlir_adversarial.py'),'SNN_SOURCE_HAS_CHANGED')
    require(c['original_tlp01_method']=='quadratic_complete','PRIMARY_METHOD_CHANGED')
    return c


class GNNHidden(torch.nn.Module):
    def __init__(self,network):
        super().__init__()
        self.network=network
    def forward(self,adj,node_features):
        h=self.network.embed(node_features)
        for block in self.network.blocks:
            h=block(h,adj,1.0)
        return self.network.readout[1](self.network.readout[0](h[:,0]))


def get_snn_hidden_and_threshold_distance(model,images):
    # The exact h(theta,x) of the original hard-spike forward. No autograd or
    # approximation enters this procedure; track distance to the spike boundary.
    drive_base=F.relu(model.stem(images))
    membrane=[torch.zeros((len(images),snn.HIDDEN),dtype=images.dtype) for _ in range(snn.N_COMPONENTS)]
    prior=[torch.zeros_like(membrane[0]) for _ in range(snn.N_COMPONENTS)]
    pooled=torch.zeros_like(membrane[0])
    smallest=float('inf')
    for _ in range(snn.TIMESTEPS):
        drive=drive_base
        for j,block in enumerate(model.blocks):
            membrane[j]=0.65*membrane[j]+block.feed(drive)+0.2*block.recurrent(prior[j])
            voltage=membrane[j]-0.6
            smallest=min(smallest,float(voltage.detach().abs().min()))
            spike=(voltage>=0).to(voltage.dtype)
            membrane[j]=membrane[j]-0.6*spike
            prior[j]=spike
            drive=spike
        pooled=pooled+drive
    return pooled/snn.TIMESTEPS,smallest


def append_constant(h,first,second):
    one=torch.ones((h.shape[0],1),dtype=h.dtype)
    zero=torch.zeros_like(one)
    return torch.cat((h,one),1),torch.cat((first,zero),1),torch.cat((second,zero),1)


def pred_for_event(system,seed,step,pre,delta,inputs,labels,ids):
    # This function is intentionally forbidden to read model post states,
    # alpha>0 forward responses, or any previously observed post labels/logits.
    torch.set_grad_enabled(True)
    model=(gnn.GraphNet() if system=='gnn' else snn.HardSpikingNetwork()).eval()
    (gnn.restore_state if system=='gnn' else snn.restore)(model,pre)
    checkpoint=(gnn.state_sha(pre) if system=='gnn' else snn.state_hash(pre))
    tangent_id=(gnn.state_sha(delta) if system=='gnn' else snn.state_hash(delta))
    if system=='gnn':
        wrapper=GNNHidden(model).eval()
        def calc(a,x):
            base={name:v.detach().clone() for name,v in wrapper.named_parameters()}
            tangent={name:delta[name.removeprefix('network.')].detach().clone() for name in base}
            def forward(p):return functional_call(wrapper,p,(a,x),strict=True)
            def first(p):return jvp(forward,(p,),(tangent,))[1]
            h,hp=jvp(forward,(base,),(tangent,))
            _,hpp=jvp(first,(base,),(tangent,))
            return h.detach(),hp.detach(),hpp.detach(),None
        head_key='readout.2'
        logits=lambda a,x:model(a,x)
    else:
        def calc(x):
            with torch.no_grad():h,min_threshold=get_snn_hidden_and_threshold_distance(model,x)
            return h,torch.zeros_like(h),torch.zeros_like(h),min_threshold
        head_key='head'
        logits=lambda x:model(x)
    W=pre[head_key+'.weight']; dW=delta[head_key+'.weight']
    b=pre[head_key+'.bias']; db=delta[head_key+'.bias']
    head=torch.cat((W,b[:,None]),dim=1)
    dhead=torch.cat((dW,db[:,None]),dim=1)
    nclasses=head.shape[0]
    ledger=[];max_reconstruction=0.;min_voltage=float('inf')
    B=64 if system=='gnn' else 90
    for aidx in range(0,len(labels),B):
        sl=slice(aidx,min(aidx+B,len(labels)))
        ys=labels[sl]
        if system=='gnn':
            a,x=inputs['adj'][sl],inputs['x'][sl]
            h,hp,hpp,_=calc(a,x)
            with torch.no_grad():real0=logits(a,x)
        else:
            x=inputs['images'][sl]
            h,hp,hpp,dist=calc(x)
            min_voltage=min(min_voltage,dist)
            with torch.no_grad():real0=logits(x)
        recon=h@W.T+b[None,:]
        error=float((recon-real0).abs().max())
        max_reconstruction=max(max_reconstruction,error)
        require(error<1e-4, f'HEAD_DECOMPOSITION_MISMATCH:{system}:{seed}:{step}:{error}')
        h,hp,hpp=append_constant(h,hp,hpp)
        # b_c = weight_y - weight_c; exactly TL-P01 using constant bias dimension.
        b_c=head[ys,None,:]-head[None,:,:]
        db_c=dhead[ys,None,:]-dhead[None,:,:]
        gap=(b_c*h[:,None,:]).sum(-1)
        first=(db_c*h[:,None,:]).sum(-1)+(b_c*hp[:,None,:]).sum(-1)
        joint=(db_c*hp[:,None,:]).sum(-1)
        hidden=.5*(b_c*hpp[:,None,:]).sum(-1)
        allgap={'unchanged':gap,'linear':gap+first,'joint_rotation':gap+first+joint,
               'hidden_curvature':gap+first+hidden,
               'quadratic_complete':gap+first+joint+hidden}
        others=torch.arange(nclasses)[None,:]!=ys[:,None]
        for local in range(len(ys)):
            i=aidx+local
            result={}
            for k,arr in allgap.items():
                values=arr[local][others[local]]
                result[k]={'correct':bool((values>0).all()),'minimum_gap':float(values.min())}
            ledger.append({'record_id':f'{system}:{seed}:{step}:{ids[i]}',
                           'target_id':str(ids[i]),'target_class':int(ys[local]),
                           'pre_correct':bool((gap[local][others[local]]>0).all()),
                           'methods':result})
    require(len({x['record_id'] for x in ledger})==len(ledger),'DUPLICATE_TARGET_IDENTITIES')
    return {'system':system,'seed':seed,'step':step,'state_hash':checkpoint,'delta_hash':tangent_id,
            'record_count':len(ledger),'head_reconstruction_max_abs_error':max_reconstruction,
            'minimum_hard_threshold_distance':min_voltage if system=='snn' else None,
            'derivative_semantics':('nested_exact_forward_mode_JVP' if system=='gnn'
                                    else 'piecewise_constant_hard_spike_true_local_derivatives_zero'),
            'predictions':ledger}


def train_gnn_prepare(seed):
    gnn.seed_all(seed)
    train=gnn.build_data(seed+100,gnn.TRAIN_N)
    test=gnn.build_data(seed+200,gnn.TEST_N)
    model=gnn.GraphNet();opt=torch.optim.RMSprop(model.parameters(),lr=gnn.LR,alpha=.95,eps=1e-8,momentum=0.)
    rng=np.random.default_rng(seed+300)
    pret={};post={}
    for k in range(1,gnn.STEPS+1):
        idx=torch.tensor(rng.integers(0,len(train[2]),size=gnn.BATCH),dtype=torch.long)
        model.train();before=gnn.clone_state(model) if k in STEPS_GNN else None
        opt.zero_grad(set_to_none=True)
        out=model(train[0][idx],train[1][idx]);loss=F.cross_entropy(out,train[2][idx]);loss.backward();opt.step()
        if k in STEPS_GNN:
            after=gnn.clone_state(model)
            delta={name:after[name]-before[name] for name in before}
            pret[k]={'pre':before,'delta':delta,'step':k,
                     'pre_sha':gnn.state_sha(before),'delta_sha':gnn.state_sha(delta),
                     'post_sha':gnn.state_sha(after)}
            post[k]={'post':after,'step':k,'post_sha':gnn.state_sha(after)}
    expected=json.loads((SOURCE_ROOT/'gnn_tlir_formal_results.json').read_text())
    ref=next(r for r in expected['runs'] if r['seed']==seed)['identities']
    for row in ref['updates']:
        k=row['step'];r=pret[k]
        require(row['pre_sha']==r['pre_sha'] and row['post_sha']==r['post_sha'] and row['update_sha']==r['delta_sha'],f'GNN_FROZEN_NATIVE_IDENTITY_FAILED:{seed}:{k}')
    require(ref['final_sha256']==gnn.state_sha(gnn.clone_state(model)),f'GNN_FINAL_SHA_FAILED:{seed}')
    inputvals={'adj':test[0],'x':test[1]}
    return {'seed':seed,'inputs':inputvals,'labels':test[2], 'ids':test[3],
            'events':pret,'original_results_sha256':sha_file(SOURCE_ROOT/'gnn_tlir_formal_results.json')},post


def prepare(output,which):
    validate_frozen()
    target=output/'pretarget';native=output/'native_endpoints'
    require(not target.exists() and not native.exists(),'REFUSE_TO_OVERWRITE_PREPARED_STATES')
    target.mkdir(parents=True);native.mkdir(parents=True)
    manifest=[]
    for system in (('gnn','snn') if which=='both' else (which,)):
        seeds=GNN_SEEDS if system=='gnn' else SNN_SEEDS
        for seed in seeds:
            if system=='gnn':
                data,post=train_gnn_prepare(seed)
                nativefile=native/f'{system}_{seed}_post.pt'
                torch.save(post,nativefile)
            else:
                src=SOURCE_ROOT/f'SNN_TLIR_formal/seed_{seed}/NATIVE_STATES_AND_OUTCOMES.pt'
                saved=torch.load(src,map_location='cpu',weights_only=False)
                _,test,_=snn.digits_dataset(seed)
                require(saved['seed']==seed and saved['dataset_digest']==_,'SNN_DATASET_MISMATCH')
                require(torch.equal(test[1],saved['heldout_labels']) and np.array_equal(test[2],saved['heldout_indices']),'SNN_TARGET_IDENTITIES_MISMATCH')
                events={}
                for k in STEPS_SNN:
                    rec=saved['snapshots'][k]
                    require(snn.state_hash(rec['pre'])==rec['pre_sha256'] and snn.state_hash(rec['delta'])==rec['delta_sha256'],f'SNN_NATIVE_STATE_MISMATCH:{seed}:{k}')
                    events[k]={'pre':rec['pre'],'delta':rec['delta'],'step':k,
                               'pre_sha':rec['pre_sha256'],'delta_sha':rec['delta_sha256'],'post_sha':rec['post_sha256']}
                data={'seed':seed,'inputs':{'images':test[0]},'labels':test[1],'ids':[str(i) for i in test[2]],
                      'events':events,'original_native_sha256':sha_file(src)}
                nativefile=src
            path=target/f'{system}_{seed}_pre.pt'
            torch.save(data,path)
            manifest.append({'system':system,'seed':seed,'pretarget_file':path.name,
                             'pretarget_sha256':sha_file(path),'native_endpoint_reference':str(nativefile),
                             'native_endpoint_sha256':sha_file(nativefile),
                             'targets':len(data['labels']),'checkpoints':sorted(data['events'])})
            print(f'PREPARED {system} {seed} {len(data["labels"])} targets',flush=True)
    save_json(output/'PREPARED_MANIFEST.json',{'schema':'tlp01-transport-pretarget-v1','entries':manifest,
                                             'frozen_protocol_sha256':sha_file(ROOT/'FROZEN_PROTOCOL.json')})


def seal(output):
    validate_frozen()
    require((output/'PREPARED_MANIFEST.json').is_file(),'PREPARE_FIRST')
    path=output/'PREDICTIONS_SEALED.json'
    require(not path.exists() and not (output/'ADJUDICATION.json').exists(),'PREDICTIONS_ALREADY_SEALED')
    manifest=json.loads((output/'PREPARED_MANIFEST.json').read_text())
    result=[]
    for entry in manifest['entries']:
        p=output/'pretarget'/entry['pretarget_file']
        require(sha_file(p)==entry['pretarget_sha256'],'PRETARGET_ARTIFACT_CHANGED')
        # No path under native_endpoints or original .pt archive is read here.
        data=torch.load(p,map_location='cpu',weights_only=False)
        for k,r in sorted(data['events'].items()):
            pred=pred_for_event(entry['system'],entry['seed'],int(k),r['pre'],r['delta'],data['inputs'],data['labels'],data['ids'])
            require(pred['state_hash']==r['pre_sha'] and pred['delta_hash']==r['delta_sha'],'IDENTITY_MISMATCH')
            result.append(pred)
            print(f'SEALED {entry["system"]} {entry["seed"]} {k}: {len(pred["predictions"])} targets',flush=True)
    save_json(path,{'schema':'tlp01-cross-system-sealed-predictions-v1','method':'quadratic_complete',
                    'frozen_protocol_sha256':sha_file(ROOT/'FROZEN_PROTOCOL.json'),
                    'prepared_manifest_sha256':sha_file(output/'PREPARED_MANIFEST.json'),
                    'no_positive_amplitude_forward_in_predictor':True,
                    'no_post_update_outcomes_accessed_in_sealing':True,'events':result})
    save_json(output/'PREDICTIONS_SEAL_RECEIPT.json',{'prediction_sha256':sha_file(path),'event_count':len(result),
                                                    'record_count':sum(x['record_count'] for x in result)})
    print('SEALED_SHA256',sha_file(path),flush=True)


def transition(pre,post):
    return ('MAINTAIN_CORRECT' if pre and post else 'CORRECT_TO_WRONG' if pre else 'WRONG_TO_CORRECT' if post else 'MAINTAIN_WRONG')


def metrics(rows,method):
    # All known truth groups counted; undefined recalls are not imputed as 1.
    tr=['MAINTAIN_CORRECT','CORRECT_TO_WRONG','MAINTAIN_WRONG','WRONG_TO_CORRECT']
    truth=[row['actual'] for row in rows];pred=[row['methods'][method]['correct'] for row in rows]
    supports={k:sum(row['true_transition']==k for row in rows) for k in tr}
    recall={k:(sum(row['methods'][method]['correct']==row['actual'] and row['true_transition']==k for row in rows)/supports[k] if supports[k] else None) for k in tr}
    classes={str(k):(sum(p==t for t,p in zip(truth,pred) if t==k)/sum(t==k for t in truth) if sum(t==k for t in truth) else None) for k in (False,True)}
    observed=[v for v in recall.values() if v is not None]
    return {'count':len(rows),'correct':sum(t==p for t,p in zip(truth,pred)),'accuracy':sum(t==p for t,p in zip(truth,pred))/len(rows),
            'four_way_macro_recall':sum(observed)/len(observed) if observed else None,
            'four_way_complete':all(supports.values()),'transition_supports':supports,'transition_recall':recall,
            'binary_balanced_accuracy':sum(v for v in classes.values() if v is not None)/sum(v is not None for v in classes.values()),
            'binary_class_recall':classes}


def adjudicate(output):
    validate_frozen()
    require(not (output/'ADJUDICATION.json').exists(),'RESULT_ALREADY_ADJUDICATED')
    receipt=json.loads((output/'PREDICTIONS_SEAL_RECEIPT.json').read_text())
    raw=output/'PREDICTIONS_SEALED.json'
    require(sha_file(raw)==receipt['prediction_sha256'],'SEALED_PREDICTION_CHANGED')
    sealed=json.loads(raw.read_text()); require(len(sealed['events'])==receipt['event_count'],'SEALED_EVENT_COUNT_CHANGED')
    manifest=json.loads((output/'PREPARED_MANIFEST.json').read_text())
    src={(e['system'],e['seed']):e for e in manifest['entries']}
    ledger=[]; per_event=[]
    for event in sealed['events']:
        system,seed,step=event['system'],event['seed'],event['step']
        meta=src[(system,seed)]
        require(sha_file(meta['native_endpoint_reference'])==meta['native_endpoint_sha256'],'NATIVE_ENDPOINT_SOURCE_CHANGED')
        p=output/'pretarget'/meta['pretarget_file'];require(sha_file(p)==meta['pretarget_sha256'],'PRETARGET_CHANGED')
        data=torch.load(p,map_location='cpu',weights_only=False)
        if system=='gnn':
            post=torch.load(meta['native_endpoint_reference'],map_location='cpu',weights_only=False)[step]['post']
            model=gnn.GraphNet().eval();gnn.restore_state(model,post)
            with torch.no_grad():logits=model(data['inputs']['adj'],data['inputs']['x'])
            observed_logits=logits.detach();sha=gnn.state_sha(post)
        else:
            data_snn=torch.load(meta['native_endpoint_reference'],map_location='cpu',weights_only=False)
            post=data_snn['snapshots'][step]['post']
            model=snn.HardSpikingNetwork().eval();snn.restore(model,post)
            with torch.no_grad():observed_logits=model(data['inputs']['images'])
            sha=snn.state_hash(post)
        require(sha==data['events'][step]['post_sha'],'POSTSTATE_NATIVE_IDENTITY_MISMATCH')
        targets=data['labels'];observed=(observed_logits.argmax(-1)==targets)
        # Strict >0 target-all-competitor for predicted as original, check ties.
        others=observed_logits.clone();others.scatter_(1,targets[:,None],-float('inf'))
        post_margin=(observed_logits.gather(1,targets[:,None]).flatten()-others.max(1).values)
        true_correct=post_margin>0
        require(bool((true_correct==observed).all()),'POST_TIE_AMBIGUITY_IN_READOUT')
        lookup={row['target_id']:row for row in event['predictions']}
        ids=[str(i) for i in data['ids']]
        require(len(set(ids))==len(ids)==len(lookup),'RECORD_IDENTITY_COVERAGE_MISMATCH')
        perrows=[]
        for idx,i in enumerate(ids):
            pred=lookup[i]
            row={'record_id':pred['record_id'],'system':system,'seed':seed,'step':step,'target_id':i,
                 'target_class':int(targets[idx]),'pre_correct':pred['pre_correct'],
                 'actual':bool(true_correct[idx]),'true_transition':transition(pred['pre_correct'],bool(true_correct[idx])),
                 'actual_post_margin':float(post_margin[idx]),'methods':pred['methods']}
            perrows.append(row);ledger.append(row)
        per_event.append({'system':system,'seed':seed,'step':step,'metrics':{k:metrics(perrows,k) for k in METHODS}})
        print(f'ADJUDICATED {system} {seed} {step}: quadratic accuracy {per_event[-1]["metrics"]["quadratic_complete"]["accuracy"]:.4%}',flush=True)
    bysystem={}
    for system in ('gnn','snn'):
        data=[r for r in ledger if r['system']==system]
        if not data:continue
        methods={k:metrics(data,k) for k in METHODS}
        ref=methods['linear'];primary=methods['quadratic_complete']
        fixed=sum(r['methods']['linear']['correct']!=r['actual'] and r['methods']['quadratic_complete']['correct']==r['actual'] for r in data)
        broken=sum(r['methods']['linear']['correct']==r['actual'] and r['methods']['quadratic_complete']['correct']!=r['actual'] for r in data)
        bysystem[system]={'metrics':methods,'repairs_vs_linear':{'fixed':fixed,'broken':broken,'net':fixed-broken},
                           'verdict':('FOUR_WAY_NOT_EVALUABLE' if not primary['four_way_complete']
                                      else 'PREDICTION_TRANSPORT_SUPPORTED' if primary['accuracy']>=.90 and primary['four_way_macro_recall']>=.90
                                      else 'PREDICTION_TRANSPORT_NOT_SUPPORTED')}
    save_json(output/'ADJUDICATION.json',{'schema':'tlp01-cross-system-adjudication-v1',
                 'pretarget_seal_sha256':sha_file(raw),'prepared_manifest_sha256':sha_file(output/'PREPARED_MANIFEST.json'),
                 'original_tlp01_method':'quadratic_complete','system_summary':bysystem,
                 'per_event':per_event,'ledger':ledger})
    save_json(output/'ADJUDICATION_REPORT.json',{'schema':'tlp01-cross-system-report-v1',
                 'frozen_protocol_sha256':sha_file(ROOT/'FROZEN_PROTOCOL.json'),'predictions_sha256':sha_file(raw),
                 'full_adjudication_sha256':sha_file(output/'ADJUDICATION.json'),
                 'system_summary':bysystem,
                 'per_seed':{system:{str(seed):metrics([r for r in ledger if r['system']==system and r['seed']==seed],'quadratic_complete')
                                     for seed in (GNN_SEEDS if system=='gnn' else SNN_SEEDS)} for system in bysystem},
                 'limitations':['GNN replay of existing fixed experiment not genuinely fresh model confirmation',
                                'SNN archival pre/post saved beforehand, but prediction computations sealed before endpoints read in this analysis',
                                'No native GFG Core v3 capture',
                                'SNN discontinuous hidden dynamics make its true local directional derivatives zero away from threshold; surrogate gradient not used']})
    print('FINISHED',json.dumps({k:{'verdict':v['verdict'],'accuracy':v['metrics']['quadratic_complete']['accuracy'],'macro':v['metrics']['quadratic_complete']['four_way_macro_recall']} for k,v in bysystem.items()}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','seal','adjudicate'))
    ap.add_argument('--output',default=str(ROOT/'artifacts'))
    ap.add_argument('--which',choices=('gnn','snn','both'),default='both')
    args=ap.parse_args()
    out=Path(args.output)
    if args.stage=='prepare':
        require(not out.exists(),'REFUSE_OVERWRITE_OUTPUT_ROOT');out.mkdir(parents=True)
        prepare(out,args.which)
    elif args.stage=='seal':seal(out)
    elif args.stage=='adjudicate':adjudicate(out)

if __name__=='__main__':main()
