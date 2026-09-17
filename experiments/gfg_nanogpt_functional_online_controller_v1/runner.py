import argparse
import contextlib
import gzip
import sys
import time
from .common import *
from .selector import coordinates, select
from .evidence import Evidence

SELECTION_ACTIVE = False
SELECTION_READS = []

def guard(event, args):
    if not SELECTION_ACTIVE or event != 'open' or not isinstance(args[0], (str,bytes,os.PathLike)):
        return
    p = str(Path(os.fsdecode(args[0])).resolve()).lower()
    if 'gfg-evidence' in p or 'data_private' in p or p.startswith(str(OUTPUT).lower()):
        SELECTION_READS.append(p)
        raise RuntimeError('SELECTION_EVIDENCE_READ_FORBIDDEN:' + p)

@contextlib.contextmanager
def selection_boundary():
    global SELECTION_ACTIVE
    SELECTION_ACTIVE = True
    try:
        yield
    finally:
        SELECTION_ACTIVE = False

def evaluate(rt, store, t):
    rt.model.eval()
    before = state_hash(rt.snapshot())
    with torch.no_grad():
        logits = rt.base.forward().numpy()
    y = rt.base.validation_targets[:, -1].numpy()
    masked = logits.copy()
    masked[np.arange(len(y)), y] = -np.inf
    margins = logits[np.arange(len(y)),y].astype(np.float64) - masked.max(axis=1).astype(np.float64)
    require(state_hash(rt.snapshot()) == before, 'EVALUATION_MUTATED_STATE')
    return {'t':t, 'state_sha256':before,
            'logits':store.put(logits,representation='native_capability_logits'),
            'margins':store.put(margins,representation='target_minimum_competitor_margin'),
            'correct':(margins > 0).tolist(), 'accuracy':float((margins > 0).mean()),
            'mean_margin':float(margins.mean()), 'median_margin':float(np.median(margins))}

def run_branch(row, branch, protocol_hash):
    root = OUTPUT / row['entry_id'] / branch
    require(not root.exists(), 'BRANCH_ALREADY_EXISTS_NO_OVERWRITE:' + str(root))
    rt = runtime(row)
    ev = Evidence(root, row, branch, protocol_hash)
    store, writer = ev.store, ev.writer
    timing = dict(training=0., selection=0., application=0., evaluation=0., evidence=0., batch_loading=0.)
    start = time.perf_counter()
    try:
        rt.set_rng(row['initial_rng']['seed'])
        require(rt.rng_commitment(row['initial_rng']['seed']) == row['initial_rng'], 'INITIAL_RNG')
        initial = rt.snapshot()
        require(state_hash(initial) == row['initial_state_sha256'], 'INITIAL_STATE')
        writer.start_block('initialization', 0)
        origin = ev.object('source_checkpoint', {'origin_kind':'GeneratedOrigin',
            'source_object_id':read(row['receiver_state_path'])['state']['state_id'],
            'source_content_sha256':sha(Path(row['receiver_state_path'])),
            'source_state_sha256':state_hash(initial),
            'source_locator':row['receiver_state_path']}, 0, 'GeneratedOrigin')
        initial_doc = {'state':pack_state(store,initial),'rng':pack_rng(store,rt,row['initial_rng']['seed']),
                       'source_step':row['source_step'], 'branch':branch,'entry_id':row['entry_id']}
        pre_ref, _ = ev.event('branch_initial_state', initial_doc,[origin],0)
        write_new(root / 'INITIAL.json', initial_doc)
        validation_origin = ev.object('validation_definition',{'targets':row['evaluation_identities'],
            'inputs':store.put(rt.base.validation_inputs,representation='validation_inputs'),
            'targets_tensor':store.put(rt.base.validation_targets,representation='validation_targets')},0)
        before_eval=time.perf_counter()
        evaluation = evaluate(rt,store,0)
        timing['evaluation'] += time.perf_counter()-before_eval
        ev.event('capability_evaluation',evaluation,[pre_ref,validation_origin],0)
        write_new(root / 'evaluations' / '0000.json',evaluation)
        writer.flush_block()
        writer.connection.commit()
        for offset in range(HORIZON):
            t = offset + 1
            step_start=time.perf_counter()
            pre = rt.snapshot()
            writer.start_block('online_opportunity',t)
            receiving_origin = ev.generated(pre_ref,t)
            batch_start=time.perf_counter()
            b = batch(rt,row['source_step']+offset)
            opportunity = row['opportunities'][offset]
            require(hashlib.sha256(b.inputs.numpy().tobytes()).hexdigest() == opportunity['batch_inputs_sha256'], 'BATCH_INPUT')
            require(hashlib.sha256(b.targets.numpy().tobytes()).hexdigest() == opportunity['batch_targets_sha256'], 'BATCH_TARGET')
            batch_origins = [writer.origin(s) for s in b.source_rows.values()]
            select_batch(rt,b)
            seed = seed_for(row,offset,protocol_hash)
            rt.set_rng(seed)
            rng = pack_rng(store,rt,seed)
            rng_ref = ev.object('rng_opportunity',rng,t)
            timing['batch_loading'] += time.perf_counter()-batch_start
            torch.cuda.synchronize()
            train_start=time.perf_counter()
            native = rt.train_step(execute_optimizer=True,seed=seed)
            proposed = rt.snapshot()
            torch.cuda.synchronize()
            training_seconds=time.perf_counter()-train_start
            timing['training'] += training_seconds
            proposal_doc = {'pre_state_sha256':state_hash(pre), 'proposed_state':pack_state(store,proposed),
                'delta':store.put_named(native.parameter_updates,representation='own_native_proposed_delta'),
                'raw_gradients':store.put_named(native.raw_gradients,representation='own_raw_gradients'),
                'clipped_gradients':store.put_named(native.clipped_gradients,representation='own_clipped_gradients'),
                'loss':native.loss, 'gradient_norm':native.total_gradient_norm,
                'proposed_norm':norm(native.parameter_updates),'rng_before':native.rng_before,'rng_after':native.rng_after}
            proposal_ref, proposal_occ = ev.event('native_adamw_proposal',proposal_doc,[receiving_origin,rng_ref,*batch_origins],t)
            torch.cuda.synchronize()
            select_start=time.perf_counter()
            coords = None
            scores = None
            with selection_boundary():
                if branch == 'controller':
                    coords = coordinates(rt.model,pre.parameters,native.parameter_updates,
                        rt.base.validation_inputs.to(rt.device),rt.base.validation_targets.to(rt.device))
                    alpha,scores = select(coords)
                else:
                    alpha = 1.0 if branch == 'native' else 0.25
            torch.cuda.synchronize()
            selection_seconds=time.perf_counter()-select_start
            timing['selection'] += selection_seconds
            require(state_hash(rt.snapshot()) == state_hash(proposed),'COORDINATES_MUTATED_MODEL')
            selection_doc = {'t':t, 'pre_state_sha256':state_hash(pre),
                'proposal_delta_sha256':proposal_doc['delta']['raw_tensor_sha256'],
                'selected_alpha':alpha,'scores':scores, 'branch':branch, 'entry_id':row['entry_id'],
                'coordinates':None if coords is None else {k:store.put(v,representation=k) for k,v in coords.items()},
                'forbidden_evidence_reads':list(SELECTION_READS),
                'allowed_information':['own_pre_parameters','own_formed_delta','fixed_validation_inputs_and_labels'],
                'selection_rule':'import smoke v1 endpoints and choose; no fit',
                'created_utc':utc(), 'protocol_sha256':protocol_hash}
            selection_path=root / 'selections' / f'{t:04d}.json'
            write_new(selection_path,selection_doc)
            selection_ref, selection_occ = ev.event('sealed_amplitude_selection',selection_doc,
                [receiving_origin,proposal_ref,validation_origin],t)
            # The durable selection exists before any scaled endpoint is applied/evaluated.
            app_start=time.perf_counter()
            applied = scaled_state(pre,proposed,native.parameter_updates,alpha)
            if alpha != 1.0:
                with torch.no_grad():
                    for n,p in rt.named_parameters.items():
                        p.copy_(applied.parameters[n].to(rt.device))
            post = rt.snapshot()
            require(state_hash(post) == state_hash(applied),'APPLIED_STATE_MISMATCH')
            timing['application'] += time.perf_counter()-app_start
            displacement={n:post.parameters[n]-pre.parameters[n] for n in pre.parameters}
            application_doc={'t':t,'state':pack_state(store,post),
                'selection_file_sha256':sha(selection_path), 'selection_occurrence_id':selection_occ,
                'applied_displacement':store.put_named(displacement,representation='actual_applied_displacement'),
                'applied_norm':norm(displacement),'created_utc':utc()}
            pre_ref, applied_occ = ev.event('apply_parameter_displacement',application_doc,[receiving_origin,proposal_ref,selection_ref],t)
            evaluation=None
            if t % CADENCE == 0:
                eval_start=time.perf_counter()
                evaluation=evaluate(rt,store,t)
                timing['evaluation'] += time.perf_counter()-eval_start
                ev.event('capability_evaluation',evaluation,[pre_ref,validation_origin],t)
                write_new(root / 'evaluations' / f'{t:04d}.json',evaluation)
            record={'t':t,'offset':offset,'source_step':row['source_step']+offset,'seed':seed,
                'batch_identity':identity(opportunity),'proposal':proposal_doc,'selection_file_sha256':sha(selection_path),
                'application':application_doc,'occurrences':[proposal_occ,selection_occ,applied_occ],
                'timing_seconds':{'training':training_seconds,'selection':selection_seconds}}
            write_new(root / 'steps' / f'{t:04d}.json',record)
            writer.flush_block()
            writer.connection.commit()
            if t % CADENCE == 0:
                print('FORMAL',row['entry_id'],branch,t,'/',HORIZON,flush=True)
        graph_manifest=writer.close()
        timing['wall_total']=time.perf_counter()-start
        timing['evidence']=timing['wall_total']-sum(v for k,v in timing.items() if k not in ('evidence','wall_total'))
        write_new(root / 'COMPLETE.json',{'status':'COMPLETE','entry_id':row['entry_id'],'branch':branch,
            'horizon':HORIZON,'graph':graph_manifest,'timing_seconds':timing,'protocol_sha256':protocol_hash})
    except Exception as e:
        write_new(root / ('FAILURE-'+str(time.time_ns())+'.json'),{'status':'FAIL','error':repr(e),'created_utc':utc()})
        raise
    finally:
        rt.close()

def main():
    setup()
    sys.addaudithook(guard)
    verify_seal()
    protocol_hash=sha(HERE / 'PROTOCOL_FREEZE.md')
    rows=read(HERE / 'RUN_ASSIGNMENT.json')['runs']
    for row in rows:
        for branch in BRANCHES:
            root=OUTPUT / row['entry_id'] / branch
            if (root / 'COMPLETE.json').exists():
                require(read(root / 'COMPLETE.json')['protocol_sha256']==protocol_hash,'RESUME_PROTOCOL')
                continue
            run_branch(row,branch,protocol_hash)

if __name__ == '__main__':
    main()
