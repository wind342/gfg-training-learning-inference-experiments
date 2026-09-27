# Relation-floor task8: fixed continuation to630 updates

Base7ae5104125c2838d456c240f60018b57001ee481. Continue the actual relation-floor
step210 head and AdamW states,not a restart. Exactly two additional210-step epochs.
Task8 only; no new architecture,policy,LR,seed,bank or optimizer changes.

## Accuracy

Same primary test split; relation-macro percentages. The420 and630 checkpoints
were declared before training and evaluated read-only. Neither was used to tune
or stop training. The210 row references the existing parent result.

| Total task8 updates | Old56 | New8 | All64 | Skip in this epoch |
|---|---:|---:|---:|---:|
|210(parent)|76.2139|68.2143|75.2139|0|
|420|75.0917|76.6964|75.2923|0|
|630|74.3389|80.2679|75.0800|0|

At630 versus210:new+12.0536pp,old-1.8750pp,all-0.1339pp. New capability continues
to improve,so210 steps was not the ceiling of this protected trajectory. However,
more training does not improve all-seen test accuracy here and old test retention
drifts downward. Preserved finite validation-control floors do not imply unchanged
population/test capability. Extra epochs explicitly spend more examples and time;
this is not an equal-budget efficiency comparison or a statistical-significance
claim. Continued improvement beyond630 is untested; do not automatically extend.

## Safety and dynamics

All56 ORIGINAL task-entry floors remain fixed,sum6835. They were NOT recomputed
from the210 or420 states. Old bank7833 plus80 new audit-only controls is unchanged.
Both new epochs have0 skips and0 relation-floor violations,with210 projection
calls each. Counters/heads/moments continue1680->2100 without reset.

| Added epoch | Correct->wrong events | Wrong->correct events | Active constraints median | Norm ratio | Cosine | Training loop s |
|---|---:|---:|---:|---:|---:|---:|
|211-420|169|168|465|0.527250|0.510024|55.2761|
|421-630|203|205|495|0.551649|0.535309|54.2417|

Projection/check times29.9205s and29.1311s;total added training-loop time109.5179s,
excluding loading,serialization,test and independent checking. New local evidence
about1.03GiB. Original experiment artifacts are neither overwritten nor deleted.
Across all630 steps(including parent):592 correct->wrong and594 wrong->correct
events; repeated flips can involve the same sample. Control correct totals at
210/420/630 are6836/6835/6837. These are control-set correctness changes,not direct
measurements of neural support identity.

## Checks and reproduction

The exact parent floor_projection function is imported unchanged,including
Hildreth coordinate/reserve/100-sweep cap/first-safe behavior. No fresh-feature
branch or extended-stopping micro-experiment logic is used. Frozen BERT stays
byte-identical and receives no gradients. LR5e-5,batch16,original batch order is
repeated each epoch; scheduled RNG seed follows the continuous update counter.

Independent reconstruction checks all420 new native AdamW candidates,selections,
Hildreth primal/dual sweeps,FP32 endpoints and23520 relation-step floors. It verifies
parent->epoch2->epoch3 head/moment continuity,fixed K,batch identities,test metrics
and source/input hashes. See [CHECKS.json](results/CHECKS.json),
[RESULTS.json](results/RESULTS.json) and [protocol](PROTOCOL.md).

```text
python -B -m experiments.gfg_relation_floor_task8_extended_v1.run
python -B -m experiments.gfg_relation_floor_task8_extended_v1.run check
```

Full tensor/selection evidence remains at
E:/gfg-evidence/relation-floor-task8-extended-v1; compact machine records are under
results/. Source/input and large local artifact hashes are recorded in PREFLIGHT
and per-epoch TRAIN_RESULT. Public generated JSON preserves original bytes.
The runner refuses to overwrite outputs. One formal execution,no outcome-driven
rerun,extra task or parameter adjustment.
