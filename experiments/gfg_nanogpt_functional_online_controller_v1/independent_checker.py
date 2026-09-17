"""Full native re-execution, independent endpoint scoring and graph/byte checks."""
import ast
import time
import zlib
from .common import *
from .evidence import load_ref, check_references
from .selector import coordinates
from .analysis import summarize
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import validate_support_gfg, SupportGFG

def independently_choose(c):
    scores={}
    for a in (.25,.5,.75,1.):
        gap=((c['g0']+a*c['first_gap'])+a*a*c['joint_rotation'])+a*a*c['hidden_curvature']
        margin=np.min(gap,axis=1)
        scores[str(a)]={'predicted_correct_count':int(np.count_nonzero(margin>0)),
                        'median_predicted_minimum_margin':float(np.median(margin))}
    a=max((.25,.5,.75,1.),key=lambda a:(scores[str(a)]['predicted_correct_count'],scores[str(a)]['median_predicted_minimum_margin'],-abs(a-1.)))
    return a,scores

def check_graph(root,row):
    report=validate_support_gfg(root / 'execution.sqlite3',source_database_path=Path(row['bundle_path']) / 'participant_gfg.sqlite3',
        tensor_directory=root / 'tensor-objects',expected_schema=SCHEMA,require_checkpoint_grid=False)
    graph=SupportGFG(root / 'execution.sqlite3')
    previous=None
    try:
        for record,block in graph.blocks():
            material={'block_ordinal':record['block_ordinal'],'optimizer_step':block['optimizer_step'],
                'payload_sha256':record['payload_sha256'],'prior_block_sha256':previous,
                'scope_id':graph.metadata('scope_id'),'stage':block['stage']}
            require(identity(material)==record['block_sha256'],'GRAPH_CHAIN_IDENTITY')
            previous=record['block_sha256']
            for obj in block['objects']:
                require(obj['object_id']=='obj_'+identity({'content_sha256':obj['content_sha256'],
                    'domain_scope_id':graph.metadata('scope_id'),'semantic_key':obj['semantic_key']}),'OBJECT_IDENTITY')
            for fact in block['fact_blocks']:
                material={k:v for k,v in fact.items() if k!='fact_block_id'}
                require(fact['fact_block_id']=='factblock_'+identity(material),'FACT_IDENTITY')
            if block['optimizer_step']>0:
                types=[o['occurrence_type'] for o in block['occurrences']]
                require(types[:3]==['native_adamw_proposal','sealed_amplitude_selection','apply_parameter_displacement'],'STAGE_ORDER')
                t=block['optimizer_step']
                expected=3+(1 if t%CADENCE==0 else 0)
                require(len(types)==expected,'UNDECLARED_OCCURRENCE')
                payloads={obj['role']:obj['payload'] for obj in block['objects']}
                selection=read(root / 'selections' / f'{t:04d}.json')
                step=read(root / 'steps' / f'{t:04d}.json')
                require(payloads['sealed_amplitude_selection']==selection,'GFG_SELECTION_BINDING')
                require(payloads['native_adamw_proposal']==step['proposal'],'GFG_PROPOSAL_BINDING')
                require(payloads['apply_parameter_displacement']==step['application'],'GFG_APPLIED_BINDING')
                if t%CADENCE==0:
                    require(payloads['capability_evaluation']==read(root / 'evaluations' / f'{t:04d}.json'),'GFG_EVALUATION_BINDING')
    finally:
        graph.close()
    return report

def check_evaluation(rt,root,t,seen):
    e=read(root / 'evaluations' / f'{t:04d}.json')
    check_references(root,e,seen)
    rt.model.eval()
    with torch.no_grad(): logits=rt.base.forward().numpy()
    require(np.array_equal(logits,load_ref(root,e['logits'])),'REPLAY_EVALUATION_LOGITS')
    y=rt.base.validation_targets[:,-1].numpy()
    competitors=logits.copy()
    competitors[np.arange(len(y)),y]=-np.inf
    margins=logits[np.arange(len(y)),y].astype(np.float64)-competitors.max(axis=1).astype(np.float64)
    require(np.array_equal(margins,load_ref(root,e['margins'])),'EVALUATION_MARGINS')
    require((margins>0).tolist()==e['correct'],'EVALUATION_CORRECTNESS')
    require(float((margins>0).mean())==e['accuracy'],'EVALUATION_ACCURACY')
    require(float(margins.mean())==e['mean_margin'] and float(np.median(margins))==e['median_margin'],'EVALUATION_MARGIN_METRICS')
    require(e['state_sha256']==state_hash(rt.snapshot()),'EVALUATION_STATE_BINDING')

def check_branch(row,branch,protocol_hash):
    root=OUTPUT / row['entry_id'] / branch
    complete=read(root / 'COMPLETE.json')
    require(complete['status']=='COMPLETE' and complete['horizon']==HORIZON,'FORMAL_COMPLETENESS')
    require(sha(root / 'execution.sqlite3')==complete['graph']['database_sha256'],'GRAPH_FILE_HASH')
    seen=set()
    initial=read(root / 'INITIAL.json')
    check_references(root,initial,seen)
    source=load_source(row)
    require(initial['state']['commitment']==source.commitment(),'SOURCE_INITIAL_BYTES')
    rt=runtime(row)
    try:
        rt.set_rng(row['initial_rng']['seed'])
        require(rt.rng_commitment(row['initial_rng']['seed'])==initial['rng']['commitment']==row['initial_rng'],'INITIAL_RNG_BYTES')
        require(np.array_equal(torch.get_rng_state().numpy(),load_ref(root,initial['rng']['cpu'])),'CPU_RNG_PAYLOAD')
        for v,ref in zip(torch.cuda.get_rng_state_all(),initial['rng']['cuda']):
            require(np.array_equal(v.numpy(),load_ref(root,ref)),'CUDA_RNG_PAYLOAD')
        require(ast.literal_eval(initial['rng']['python_state'])==random.getstate(),'PYTHON_RNG_PAYLOAD')
        check_evaluation(rt,root,0,seen)
        for t in range(1,HORIZON+1):
            step=read(root / 'steps' / f'{t:04d}.json')
            selection_path=root / 'selections' / f'{t:04d}.json'
            sel=read(selection_path)
            check_references(root,step,seen)
            check_references(root,sel,seen)
            require(sha(selection_path)==step['selection_file_sha256']==step['application']['selection_file_sha256'],'STAGE_SEAL')
            require(sel['created_utc']<=step['application']['created_utc'],'STAGE_WALL_ORDER')
            require(not sel['forbidden_evidence_reads'],'OUTCOME_LEAKAGE')
            require(sel['allowed_information']==['own_pre_parameters','own_formed_delta','fixed_validation_inputs_and_labels'],'SELECTOR_INPUT_BOUNDARY')
            require(sel['entry_id']==row['entry_id'] and sel['branch']==branch,'BRANCH_LINEAGE')
            require(sel['protocol_sha256']==protocol_hash,'SELECTION_PROTOCOL')
            pre=rt.snapshot()
            require(state_hash(pre)==step['proposal']['pre_state_sha256']==sel['pre_state_sha256'],'PRE_STATE_LINEAGE')
            b=batch(rt,row['source_step']+t-1)
            o=row['opportunities'][t-1]
            require(step['batch_identity']==identity(o),'BATCH_IDENTITY')
            require(hashlib.sha256(b.inputs.numpy().tobytes()).hexdigest()==o['batch_inputs_sha256'],'BATCH_BYTES')
            require(hashlib.sha256(b.targets.numpy().tobytes()).hexdigest()==o['batch_targets_sha256'],'TARGET_BATCH_BYTES')
            seed=seed_for(row,t-1,protocol_hash)
            require(step['seed']==seed,'RNG_SCHEDULE')
            select_batch(rt,b)
            # This is the existing, unmodified native trainer, on this replay's
            # own current state; not a stored or other branch's gradient.
            actual=rt.train_step(execute_optimizer=True,seed=seed)
            proposed=rt.snapshot()
            p=step['proposal']
            require(p['proposed_state']['commitment']==proposed.commitment(),'NATIVE_PROPOSAL_AND_MOMENT_RECURRENCE_BYTE_EXACT')
            require(p['rng_before']==actual.rng_before and p['rng_after']==actual.rng_after,'RNG_REPLAY')
            require(p['loss']==actual.loss and p['gradient_norm']==actual.total_gradient_norm,'TRAINING_REPLAY')
            for key,values in [('raw_gradients',actual.raw_gradients),('clipped_gradients',actual.clipped_gradients),('delta',actual.parameter_updates)]:
                saved=load_named_array(root,p[key])
                require(all(np.array_equal(saved[n],v.numpy()) for n,v in values.items()),'OWN_NATIVE_REPLAY:'+key)
            require(norm(actual.parameter_updates)==p['proposed_norm'],'PROPOSED_NORM')
            require(sel['proposal_delta_sha256']==p['delta']['raw_tensor_sha256'],'SELECTION_UPDATE_BINDING')
            if branch=='controller':
                c={k:load_ref(root,v) for k,v in sel['coordinates'].items()}
                a,scores=independently_choose(c)
                require(a==sel['selected_alpha'] and scores==sel['scores'],'INDEPENDENT_SELECTION_RECOMPUTATION')
                rebuilt=coordinates(rt.model,pre.parameters,actual.parameter_updates,rt.base.validation_inputs.to(rt.device),rt.base.validation_targets.to(rt.device))
                require(all(np.array_equal(c[k],v) for k,v in rebuilt.items()),'FULL_CURRENT_STATE_COORDINATE_REEXECUTION')
            else:
                a=1. if branch=='native' else .25
                require(sel['selected_alpha']==a and sel['coordinates'] is None and sel['scores'] is None,'FIXED_NO_CONTROLLER_INFORMATION')
            # Independent application, preserving all native moments and steps.
            if a!=1.:
                with torch.no_grad():
                    for n,param in rt.named_parameters.items():
                        value=pre.parameters[n]+a*(proposed.parameters[n]-pre.parameters[n])
                        param.copy_(value.to(rt.device))
            post=rt.snapshot()
            require(post.commitment()==step['application']['state']['commitment'],'FULL_BRANCH_REPLAY_BYTE_EXACT')
            require(all(torch.equal(post.optimizer[n][k],proposed.optimizer[n][k]) for n in post.optimizer for k in ('step','exp_avg','exp_avg_sq')),'NO_MOMENT_SCALING')
            displacement={n:post.parameters[n]-pre.parameters[n] for n in post.parameters}
            saved=load_named_array(root,step['application']['applied_displacement'])
            require(all(np.array_equal(v.numpy(),saved[n]) for n,v in displacement.items()),'APPLIED_DISPLACEMENT')
            require(norm(displacement)==step['application']['applied_norm'],'APPLIED_NORM')
            if t%CADENCE==0: check_evaluation(rt,root,t,seen)
            if t%50==0: print('CHECK',row['entry_id'],branch,t,flush=True)
        graph=check_graph(root,row)
        return {'status':'PASS','entry_id':row['entry_id'],'branch':branch,'replayed_steps':HORIZON,
            'native_trainer_byte_exact':True,'full_gradient_proposal_optimizer_and_applied_state_reexecution':True,
            'controller_coordinates_reexecuted':HORIZON if branch=='controller' else 0,
            'validated_tensor_references':len(seen),'graph':graph}
    finally:
        rt.close()

def main():
    setup()
    verify_seal()
    for path,digest in read(HERE / 'SOURCE_MANIFEST.json')['files'].items():
        require(sha(Path(path))==digest,'SOURCE_CHANGED:'+path)
    protocol_hash=sha(HERE / 'PROTOCOL_FREEZE.md')
    rows=read(HERE / 'RUN_ASSIGNMENT.json')['runs']
    checks=[]
    start=time.perf_counter()
    try:
        for row in rows:
            for branch in BRANCHES:
                checks.append(check_branch(row,branch,protocol_hash))
        # Equality is explicitly verified independently across all three branches.
        for row in rows:
            initials=[read(OUTPUT / row['entry_id'] / b / 'INITIAL.json') for b in BRANCHES]
            require(all(x['state']==initials[0]['state'] and x['rng']==initials[0]['rng'] for x in initials),'BRANCH_INITIAL_BYTE_EQUALITY')
            for t in range(1,HORIZON+1):
                steps=[read(OUTPUT / row['entry_id'] / b / 'steps' / f'{t:04d}.json') for b in BRANCHES]
                require(len({s['batch_identity'] for s in steps})==len({s['seed'] for s in steps})==1,'CROSS_BRANCH_OPPORTUNITY_ALIGNMENT')
                require(all(s['proposal']['rng_before']==steps[0]['proposal']['rng_before'] for s in steps),'CROSS_BRANCH_RNG_ALIGNMENT')
        rebuilt=summarize()
        require(rebuilt==read(HERE / 'RESULTS.json'),'METRICS_AND_VERDICT_RECOMPUTATION')
        independent_deltas={b:[] for b in ('native','fixed')}
        for row in rows:
            areas={}
            for b in BRANCHES:
                events=[read(OUTPUT / row['entry_id'] / b / 'evaluations' / f'{t:04d}.json') for t in range(0,HORIZON+1,CADENCE)]
                proportions=[sum(e['correct'])/len(e['correct']) for e in events]
                areas[b]=sum((left+right)*.5*CADENCE for left,right in zip(proportions,proportions[1:]))/HORIZON
            for b in ('native','fixed'):
                independent_deltas[b].append(areas['controller']-areas[b])
        independent_supported=all(sum(ds)/12 >= .01 and sum(d>0 for d in ds)>=9 for ds in independent_deltas.values())
        require(independent_supported==(rebuilt['verdict']=='ONLINE_FUNCTIONAL_CONTROLLER_SUPPORTED'),'INDEPENDENT_PRIMARY_VERDICT')
        for b,ds in independent_deltas.items():
            require(abs(sum(ds)/12-rebuilt['comparisons'][b]['mean_auc_delta'])<1e-14,'INDEPENDENT_AUC')
        verify_seal()
        write_new(HERE / 'INDEPENDENT_CHECK.json',{'status':'PASS','checks':checks,'formal_steps':12*3*HORIZON,
            'native_equivalence_steps':12*HORIZON,'controller_selection_and_coordinate_recomputations':12*HORIZON,
            'protocol_sha256':protocol_hash,'verdict':rebuilt['verdict'],'elapsed_seconds':time.perf_counter()-start,
            'outcome_leakage_assurance':'Frozen source audit + pure selector input boundary + denied external evidence reads + durable per-step stage seals + complete replay; not an adversarial OS isolation proof.'})
        print('INDEPENDENT_CHECK PASS',flush=True)
    except Exception as e:
        write_new(HERE / ('INDEPENDENT_FAILURE-'+str(time.time_ns())+'.json'),{'status':'FAIL','error':repr(e),'completed_branch_checks':checks})
        raise

if __name__=='__main__': main()
