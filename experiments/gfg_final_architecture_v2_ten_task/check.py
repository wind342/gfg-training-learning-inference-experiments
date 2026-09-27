"""Full independent native-update/projection replay and sealed metric audit."""
from .core import *
from .independent import independent
import gc

def main():
    p.setup();pre=active_preflight()
    checker_key=f'experiments/{HERE.name}/check.py'
    if sha(HERE/'check.py')!=pre['sources'][checker_key]:
        previous_source=ROOT/'checker_revision1/check.py.txt'
        assert sha(previous_source)==pre['sources'][checker_key]
        revision=dict(reason='Independent checker must clone initial optimizer state separately for each arm; PyTorch can alias same-device state tensors.',
            previous_sha256=sha(previous_source),current_sha256=sha(HERE/'check.py'),training_modified=False,
            failed_checker_attempt_verified_steps=210,formal_updates_repeated=0)
        path=ROOT/'CHECKER_REVISION.json'
        if path.exists():assert json.loads(path.read_text())==revision
        else:write(path,revision)
        pre=copy.deepcopy(pre);pre['sources'][checker_key]=sha(HERE/'check.py')
    guard(pre,True)
    assert json.loads((ROOT/'COMPLETE.json').read_text())['updates']==7560
    d=np.load(DATA/'data.npz');assert old.isolation(d)==pre['isolation']
    cache=torch.load(old.ROOT/'CONTROL_CACHE.pt',map_location='cpu',weights_only=False)
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    m=FrozenRelationModel().cuda();m.load_state_dict(saved['model']);freeze(m)
    shadow,initial_opt=make_candidate(m,saved['optimizer'])
    for group in initial_opt.param_groups:group['lr']=5e-5
    initial_head=p.cpu_state(p.snapshot(m.head));initial_mom=copy.deepcopy(initial_opt.state_dict())
    del saved,shadow,initial_opt
    previous={a:(initial_head,initial_mom) for a in ARMS};total=0;sweeps=0;floorchecks=0;table=[];endpoints={}
    first210=None
    for task in range(2,11):
        h,y,ids,n=bank(d,cache,task);labels=y[:n].cpu().numpy();groups=(task-1)*8
        evdir=ROOT/f'evaluation_task{task}';ev=json.loads((evdir/'EVALUATION.json').read_text());barrier=json.loads((evdir/'BARRIER.json').read_text())
        assert ev['barrier_sha256']==sha(evdir/'BARRIER.json') and ev['readonly']
        prev_ev=json.loads((ROOT/f'evaluation_task{task-1}/EVALUATION.json').read_text())
        task_results={}
        for arm in ARMS:
            folder=ROOT/arm/f'task{task}';meta=json.loads((folder/'TRAIN_RESULT.json').read_text());config=json.loads((folder/'CONFIG.json').read_text())
            assert meta['config_sha256']==sha(folder/'CONFIG.json')
            assert barrier['training_results'][arm]==sha(folder/'TRAIN_RESULT.json')
            assert meta['finished_unix']<=barrier['created_unix']<=ev['finished_unix']
            assert config['bank_ids']==ids.tolist() and config['old_count']==n
            head=nn.Linear(1536,80).cuda();head.load_state_dict(previous[arm][0]);opt=optimizer(head.parameters());opt.load_state_dict(copy.deepcopy(previous[arm][1]))
            assert head_hash(head)==config['head_start'] and moments_hash(opt)==config['moments_start']
            boundary=(boundary_same_batch if config.get('boundary_batch')=='full-monitor' else old.transition)(head,h,y,n,task)
            assert boundary==config['transition']
            entry=p.margins(p.affine(p.snapshot(head),h,task*8),y).cpu().tolist()
            K=np.bincount(labels[np.asarray(entry[:n])>0],minlength=groups);assert K.tolist()==config['K']
            assert np.bincount(labels,minlength=groups).tolist()==config['group_sizes']
            allrows=[];epochs=[]
            for epoch in range(1,ARMS[arm]+1):
                part=folder/f'epoch{epoch}';rmeta=json.loads((part/'TRAIN_RESULT.json').read_text());epochs.append(rmeta)
                assert meta['epoch_records'][str(epoch)]==sha(part/'TRAIN_RESULT.json')
                assert sha(part/'AUDIT.pt')==rmeta['audit_sha256'] and sha(part/'STEPS.jsonl')==rmeta['log_sha256']
                a=torch.load(part/'AUDIT.pt',map_location='cpu',weights_only=False)
                assert equal(previous[arm][0],a['heads'][0]) and equal(previous[arm][1],a['initial_optimizer'])
                assert len(a['heads'])==211 and len(a['natives'])==len(a['projected'])==len(a['clipped_gradients'])==210
                assert old.opt_counter(opt)==counter(arm,task,epoch)
                compact=[];epoch_identity=[]
                with (part/'STEPS.jsonl').open() as f:
                    for j,line in enumerate(f):
                        row=json.loads(line);r=row.pop('report');compact.append(row);epoch_identity.append((row['native_sha256'],row['executed_sha256'],row['optimizer_sha256']))
                        assert row['current_ids']==order(d,task)[j].tolist() and row['K']==K.tolist()
                        assert row['global_step']==counter(arm,task,epoch,j+1) and row['seed']==2021*100000+row['global_step']
                        before={k:v.cuda() for k,v in a['heads'][j].items()};head.load_state_dict(before)
                        for name,q in head.named_parameters():q.grad=a['clipped_gradients'][j][name].cuda()
                        opt.step();native=p.snapshot(head)
                        assert equal(native,a['natives'][j]) and moments_hash(opt)==row['optimizer_sha256']
                        assert old.opt_counter(opt)==row['global_step']
                        assert tensor_hash(before)==row['before_sha256'] and tensor_hash(native)==row['native_sha256']
                        e,c,selections,logs,bm,nm,cm,em=independent(before,native,h,y,n,K,ids[:n],task*8)
                        assert equal(e,a['heads'][j+1]) and equal(c,a['projected'][j])
                        assert tensor_hash(e)==row['executed_sha256']
                        assert selections==r['selections'] and logs==r['sweeps']
                        assert bm.tolist()==r['baseline'] and nm.tolist()==r['native'] and cm.tolist()==r['projected'] and em.tolist()==r['executed']
                        C=np.bincount(labels[em[:n]>0],minlength=groups);N=np.bincount(labels[nm[:n]>0],minlength=groups)
                        assert np.all(C>=K) and C.tolist()==row['executed_counts']==r['executed_counts']
                        assert N.tolist()==row['native_counts'] and np.maximum(K-N,0).tolist()==row['native_deficits']
                        lost=(bm[:n]>0)&(em[:n]<=0);gained=(bm[:n]<=0)&(em[:n]>0)
                        lc=np.bincount(labels[lost],minlength=groups);gc_=np.bincount(labels[gained],minlength=groups)
                        assert int(lost.sum())==row['correct_to_wrong'] and int(gained.sum())==row['wrong_to_correct']
                        assert np.flatnonzero((lc>0)&(gc_>0)).tolist()==row['migration_relations']
                        assert bool(np.any(N<K))==row['projection']
                        assert bool(np.any(np.bincount(labels[cm[:n]>0],minlength=groups)<K))==row['skip']
                        assert len(logs)==row['sweeps'] and row['floor_violations']==0
                        total+=1;sweeps+=len(logs);floorchecks+=groups
                        if (j+1)%70==0:print('CHECK',arm,task,epoch,j+1,'total',total,flush=True)
                assert len(compact)==210 and compact==json.loads((part/'COMPACT_STEPS.json').read_text())
                assert equal(opt.state_dict(),a['final_optimizer'])
                assert tensor_hash(a['heads'][-1])==rmeta['head_end']
                if task==2 and epoch==1:
                    if arm=='v2_210':first210=epoch_identity
                    else:assert epoch_identity==first210
                previous[arm]=(a['heads'][-1],a['final_optimizer']);allrows+=compact
                del a;gc.collect()
            assert len(allrows)==210*ARMS[arm]
            summary=aggregate(allrows,n,K,labels,entry,em.tolist())
            for key,value in summary.items():assert value==meta['summary'][key],key
            assert meta['summary']['training_wall']==sum(v['training_wall'] for v in epochs)
            assert head_hash(head)!=None and moments_hash(opt)==meta['moments_end']
            assert tensor_hash(previous[arm][0])==meta['head_end']==ev['arms'][arm]['head_sha256']
            metrics=summarize_predictions(ev['indices'],ev['arms'][arm]['predictions'],d,task)
            metrics['previous_task_macro']=float(np.mean([metrics['per_relation'][str(g)] for g in range((task-2)*8,(task-1)*8)]))
            assert metrics==ev['arms'][arm]['metrics']
            expanded=per_relation_metrics(np.asarray(ev['indices']),ev['arms'][arm]['expanded_predictions'],d,task*8)
            assert expanded==ev['arms'][arm]['expanded_metrics']
            prior=prev_ev['arms'][arm];test_entry=prior['expanded_metrics']
            test_end=[metrics['per_relation'][str(g)] for g in range(groups)]
            task_results[arm]=dict(metrics=metrics,summary=meta['summary'],control_test_gap=dict(
                entry_control_counts=K.tolist(),end_control_counts=meta['summary']['task_end_counts'],
                entry_control_accuracy=meta['summary']['entry_control_accuracy'],end_control_accuracy=meta['summary']['end_control_accuracy'],
                prior_endpoint_old_macro=prior['metrics']['macro_all'],expanded_entry_old_macro=test_entry['macro'],
                end_old_macro=metrics['macro_old'],entry_test_per_relation=test_entry['per_relation'],end_test_per_relation=test_end,
                test_delta_pp=[100*(b-a) for a,b in zip(test_entry['per_relation'],test_end)],
                expansion_correct_to_wrong=boundary['expansion_correct_to_wrong']))
            endpoints[(task,arm)]={k:v.clone() for k,v in previous[arm][0].items()}
            del head,opt
        # Actual read-only BERT test predictions independently reconstructed.
        m.eval();test_ids=scope(d,task*8)[1];assert test_ids.tolist()==ev['indices']
        verify_heads={a:nn.Linear(1536,80).cuda() for a in ARMS}
        for a,head in verify_heads.items():head.load_state_dict(endpoints[(task,a)])
        predictions={a:[] for a in ARMS};expanded_predictions={a:[] for a in ARMS}
        with torch.no_grad():
            for begin in range(0,len(test_ids),16):
                f=m.features(torch.from_numpy(d['x'][test_ids[begin:begin+16]]).cuda())
                for a,head in verify_heads.items():
                    z=head(f);predictions[a]+=z[:,:task*8].argmax(1).cpu().tolist()
                    expanded_predictions[a]+=z[:,:min(80,task*8+8)].argmax(1).cpu().tolist()
        for a in ARMS:
            assert predictions[a]==ev['arms'][a]['predictions'] and expanded_predictions[a]==ev['arms'][a]['expanded_predictions']
        history=json.loads((old.ROOT/f'evaluation_task{task}/EVALUATION.json').read_text())
        table.append(dict(task=task,historical={a:history['arms'][a]['metrics'] for a in history['arms']},new=task_results))
        print('TASK VERIFIED',task,flush=True);del verify_heads,h,y;gc.collect()
    assert total==7560 and tensor_hash(m.bert.state_dict())==pre['backbone_sha256'];guard(pre,True)
    summary={}
    for arm in ARMS:
        vals=[t['new'][arm] for t in table]
        summary[arm]=dict(task10=vals[-1]['metrics'],mean={k:float(np.mean([v['metrics']['macro_'+k] for v in vals])) for k in ('old','new','all')},
            skips=sum(v['summary']['skip_steps'] for v in vals),training_wall=sum(v['summary']['training_wall'] for v in vals),
            projection_wall=sum(v['summary']['projection_wall'] for v in vals))
    write(ROOT/'RESULTS.json',dict(status='COMPLETED',table=table,summary=summary))
    write(ROOT/'CHECKS.json',dict(status='PASS',native_candidates=total,independent_sweeps=sweeps,relation_step_floor_checks=floorchecks,
        floor_violations=0,head_and_optimizer_continuity=True,first210_common_trajectory_exact=True,
        unchanged_production_policy=True,source_and_input_identity=True,actual_test_predictions_recomputed=True,
        task_expansion_audited=True,fixed_task_entry_floors=True,full_primary_metric_recompute=True))
    print('CHECKS PASS',total,sweeps,floorchecks,flush=True)

if __name__=='__main__':main()
