"""Seal compact public evidence after boundary and observer-free replay checks."""
from ..common import *


def main():
    access=Access();verify_freeze();audit=read(ROOT/'AUDIT.json');assert audit['status']=='PASS'
    arms={};files=['FREEZE.json','PREFLIGHT.json','TRAINING_COMPLETE.json','AUDIT_210.json','AUDIT_630.json','AUDIT.json']
    for budget in (210,630):
        rows=[];ids=set()
        for task in range(3,12):
            base=f'budget{budget}/task{task}';b=read(ROOT/base/'EXPANSION.json');c=read(ROOT/base/'CHECKS.json')
            assert b['state_before']==b['state_after'] and b['wrong_to_correct']==0
            assert c['floor_violations']==0 and c['old_ce_backward_examples']==0
            rows.append({k:b[k] for k in ('boundary','task','old_seen','new_seen','old_controls','correct_before','correct_after',
                'expansion_loss','correct_to_wrong','wrong_to_correct','relations_losing','relations_to_zero','post_expansion_K_zero',
                'max_relation_loss','strict_competition_flips','new_class_ties')})
            ids.update(f['control_id'] for f in b['flips'])
            files.extend(f'{base}/{name}' for name in ('EXPANSION.json','EXPANSION_LOGITS.npz','CONFIG.json','CHECKS.json','STEPS.jsonl.gz'))
        arms[str(budget)]=dict(boundaries=rows,total_expansion_correct_to_wrong=sum(r['correct_to_wrong'] for r in rows),
            total_expansion_wrong_to_correct=sum(r['wrong_to_correct'] for r in rows),unique_flipped_control_ids=len(ids),
            maximum_boundary_loss=max(r['expansion_loss'] for r in rows),maximum_relation_loss=max(r['max_relation_loss'] for r in rows),
            relation_to_zero_events=sum(r['relations_to_zero'] for r in rows),zero_loss_boundaries=sum(r['expansion_loss']==0 for r in rows),
            strict_competition_flips=sum(r['strict_competition_flips'] for r in rows),tie_events=sum(r['new_class_ties'] for r in rows))
    result=dict(status='PASS',arms=arms,total_scheduled_continuation_updates=7560,
        production_controller_source_unchanged=True,observer_does_not_modify_state=True,all_boundary_old_logits_byte_identical=True,
        all_flips_attributed_to_new_class_competition_or_explicit_tie=True,all_training_task_floors_held=True,
        old_ce_backward_examples=0,test_files_opened=0,access=access.report(),
        scope='Old algorithmic controls only; not population test accuracy. Cumulative counts are events across boundaries, not necessarily distinct controls. Shared hidden causality and predictor-prevention claims not made.')
    write(ROOT/'FINAL.json',result);files.append('FINAL.json')
    public=HERE/'results';public.mkdir(exist_ok=False);manifest={}
    for name in files:
        src=ROOT/name;dst=public/name;dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
        assert sha(src)==sha(dst)
        manifest[name]=dict(sha256=sha(src),bytes=src.stat().st_size)
    checkpoints={f'budget{b}/task{t}/CHECKPOINT.pt':dict(sha256=sha(ROOT/f'budget{b}/task{t}/CHECKPOINT.pt'),
        bytes=(ROOT/f'budget{b}/task{t}/CHECKPOINT.pt').stat().st_size) for b in (210,630) for t in range(3,12)}
    write(HERE/'ARTIFACT_MANIFEST.json',dict(status='PASS',authority=str(ROOT),public_files=manifest,external_checkpoints=checkpoints,
        diagnostic_sources={p.relative_to(HERE).as_posix():sha(p) for p in (HERE/'diagnostics').glob('*.py')}))
    print(json.dumps(result,indent=2));print('PUBLIC BYTES',sum(r['bytes'] for r in manifest.values()),flush=True)


if __name__=='__main__':main()
