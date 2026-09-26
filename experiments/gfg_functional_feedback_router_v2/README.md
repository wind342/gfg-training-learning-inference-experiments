# Router V2: sparse stable anchors, no blanket replay from one risk

This fixes the V1 allocation defect without clearing any real recovery state.
Base41a9724e1d0f32a7ffbdf07fd360779823c26cac and all V1 evidence remain unchanged.

`policy.py` changes one allocation rule: stable groups contribute at most ONE
rotating anchor, not enough anchors to fill all remaining slots. Each selected
risk/anchor group still contributes2 examples. One risk group therefore uses at
most4 old examples even with80 old groups. Genuine multiple risks may use more,
up to16. Existing state, observer, native predictor, emergency gate, optimizer,
memory/data split and gradients are reused by `run.py`, with inherited source
hashes recorded. No gradient projection, new weighting or learned router.

## Up-front and early stop checks

Seven new allocation tests passed before execution. Runtime invariants apply
before backward, and the first10-step allocation gate passed before continuation:
first batch6 examples for inherited risk groups0/4 plus one anchor, then4 for
group0 plus one rotating anchor. No need to run60 steps to detect the prior
structural error. The fixed short run proceeded only with valid allocations;
there were no failed training attempts or selected reruns.

## Same-checkpoint60-step result

All arms start at97.1429% old-test accuracy,0% new. All see the same960 new-task
training occurrences. References are the hash-checked previously completed runs,
not newly tuned models. Test scores never select training feedback.

| Method | Old accuracy | New accuracy | All accuracy | Old exposures | Wall |
|---|---:|---:|---:|---:|---:|
| New-only native AdamW |74.1071%|84.5536%|79.3304%|0|70.803s|
| Fixed replay native |93.3929%|79.0179%|86.2054%|960|85.873s|
| Router V1 |93.6607%|79.0179%|86.3393%|960|131.262s|
| Router V2 |96.0714%|69.6429%|82.8571%|282|118.433s|

Allocation repair worked:70.625% fewer old-example occurrences than fixed replay
or V1. Counts are30 batches of4,27 of6,3 with no old feedback. No16-example batch
occurred. Modes:44 RECOVERY,13 BALANCE,3 FORMATION; feedback active57/60=95%.
One RELEASE at offset19, then renewed risk activated feedback at offset22.
Persistent or renewed real risk was not hidden or reset. Thus the savings come
mainly from smaller targeted coalitions, not keeping feedback off for most steps.

Old per-relation occurrences:0=92,1=18,2=18,3=84,4=18,5=18,6=18,7=16.
This is no longer equal feedback to all old groups at every step.

Compared with fixed replay, old accuracy is2.6786 percentage points higher,
but new accuracy is9.3750 points lower; balanced total accuracy is3.3482 points
lower. Compared with new-only, old accuracy is21.9643 points higher, new14.9107
points lower. The repair improves old-capability preservation and reduces old
exposure, but does NOT establish a better joint stability/plasticity result.
Resource saving is not treated as success irrespective of capability.

The unchanged loss is .5 current MEAN + .5 old MEAN when feedback is active.
Fewer old examples do NOT reduce the total old-loss weight; selected groups get
a larger share of it. This is an important interpretation boundary, not a proven
unique cause of the new-learning slowdown. No post-result weight/threshold sweep.

## Predictor, integrity and scope

Three full measurements; all gates NATIVE_SAFE; alpha1 for all60 steps; no shrink.
Selected endpoint agreement480/480, with4 real correct/incorrect transitions,
all predicted. This is a small selected-endpoint check, not a new global claim.
Peak allocation6,167,233,024 bytes. Wall includes monitoring/audit and evaluation,
excludes initial loading. The historical fixed replay remains faster in wall.

Checker PASS: routing/cursors, source/model-checkpoint identities, original/new
batch identities, train/control/test role separation, recorded test predictions,
actual endpoint aggregation and predicted gate decisions. The checker consumes
recorded micro histories, not independent native reruns. The inherited FewRel
probe/test duplicate at7561 remains disclosed; exclusion sensitivity does not
change the conclusions. Memory excludes encoded control/test duplicates.

This fix's protocol stops at60 steps; no three-task run was performed. No original
result was edited. Raw create-only artifacts:
E:/gfg-evidence/functional-feedback-router-v2/.
Compact results and hashes: results/CHECKS.json. Source/record/test checks are
recorded separately; no dataset/model/checkpoint is committed.

```text
python -m experiments.gfg_functional_feedback_router_v2.tests
python -m experiments.gfg_functional_feedback_router_v2.run
python -m experiments.gfg_functional_feedback_router_v2.check
```

These commands reject overwriting the completed output. The runner/checker reuse
the pinned local V1 dependencies; this is a scoped fix, not an independent package.
