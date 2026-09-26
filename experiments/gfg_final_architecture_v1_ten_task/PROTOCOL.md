# Final Architecture v1: fixed first ten-task validation

Base: 3daa90d2572cb50aff09aee3d95997ae4244d799. Only this new experiment directory
may change. No old experiment, main or frozen tag changes. Exactly two arms:
head_only_native and final_architecture_v1. One formal run, no automatic restart,
parameter search, result-driven revision or additional arm/seed.

## Frozen architecture and training

Frozen/eval/gradient-free task1 BERT; existing 80-row linear head; actual fused
AdamW candidate; exact affine boundary prediction; Hildreth active-set projection;
FP32 roundoff reserve; actual FP32 recheck; commit/skip. Production projection is
IMPORTED UNCHANGED from gfg_head_only_hildreth_roundoff_smoke_v1.core.hildreth,
passing only the existing old_count and seen parameters. Reserve, coordinate
updates, pair order, first-safe stopping and maximum100 sweeps are unchanged.

Native-safe: all receiving-correct old controls remain margin>0. Triggered
projection: all receiving-correct old controls must have actual FP32 margin>=1e-6.
Do not strengthen the fast path or weaken projected acceptance. Already incorrect
controls are not newly protected. No scalar shrink, replay, Router, old loss,
distillation, LoRA, adapter or trainable BERT. The receiving/native-envelope
roundoff reserve is the same conservative estimate, not a universal error bound.

80 relations,10 tasks of8; original fixed task order; seed2021; LR5e-5; batch16;
210 scheduled updates/task; tasks2-10 continuously from the same verified task1
checkpoint. Batches use the existing order(data,task,2021). Gradient clip1, AdamW
betas(.9,.999),weight_decay0,fused implementation inherited unchanged. One epoch
per task. All80 head rows retained. Loss/readout uses first8*t classes. Head AdamW
moments inherited at step210 and never reset: ending at2100, including skips.
Projection cannot modify moments. Step RNG seed=2021*100000+global_scheduled_step.

Native baseline uses identical BERT/head/optimizer/batches and direct native
updates, without any per-step boundary prediction or protection. Both arms retain
their own live head and optimizer continuously. Shared BERT is immutable.

## Data correction approved before implementation

Hash full encoded inputs, regardless of labels. Exclude validation IDs
7472,7491,8927,17947,17967,19337,36190,50124,53007,55069 because they duplicate
training inputs. Do not replace, replicate or move samples; all training batches
remain unchanged. The first8 relations' bank is unchanged. At task10 old bank is
10072, not10080. Remaining clean validation samples are all retained per old
relation. Historical primary test exclusions stay exactly
7561,12566,18193,29275,29338,47545,55297; no additional exclusions are needed.
The bank and primary test must be hash-disjoint from train and each other.

Cache BERT features once for clean validation inputs, reusing the byte-identical
1200-row parent control cache where available. Append first10 clean new-relation
validation samples as read-only audit rows, never protection constraints. This
retains the exact task2 control matrix shape/order. Future cached validation
features/labels are not passed into an earlier task's protection or loss.

At each task boundary, before any update, record old-control correctness under
previous and expanded seen-class readouts. Expansion-induced losses are reported
separately from executed-update correct-to-wrong. Do not add a protection rule
for expansion losses. No thresholds are learned from these diagnostics.

## Evaluation and audit

After both arms finish each task and their TRAIN_RESULT records are hashed,
evaluate both heads on the same primary all-seen test examples, through frozen
BERT. Test is absent from training, caches used by projection, or parameter choice.
Compute BERT features once per evaluation batch and apply both heads; evaluation
is read-only and restores RNG/modes. Report old/new/all macro, immediately previous
task macro, per-relation and cohort accuracies. Task2-10 means are unweighted means
of the nine corresponding macro scores. No stop/revision based on test results.

Record full current-batch identity, clipped head gradients, receiving/native/
projected/executed head states, moment hashes and counters. Save a separate local
audit per task to bound memory, not repeated BERT checkpoints. Record full Hildreth
reports including primal/dual hashes, pair reserves and margins. Stream JSONL and
retain every failure/skip. Record training-loop wall, projection-call wall, GPU
peak allocation, and task execution wall separately from test/audit/cache time.

Independent checker reconstructs native AdamW candidates from saved clipped
gradients and inherited moments; separately replays Hildreth dual/primal updates
with dynamic bank/seen sizes, verifies every head/moment/endpoint, and recomputes
metrics from saved predictions. Check task-to-task exact head/moment continuity,
unchanged BERT, source hashes, data/cache identities and test-after-training barriers.
Task2 Final trajectory must match the parent saved head/native/projected history;
check this before proceeding to task3. No formal rerun if a mismatch occurs.

Report protection sizes, receiving-correct counts, native-safe/projection/safe/
skip counts, constraint/sweep min/median/max, executed flips, norm ratios/cosines,
projection time, training time and exact-state checks. Do not infer global minimum
distance from first-safe projection, infeasibility from100 sweeps, or general test
safety from control safety. Keep all failed outcomes as scalability evidence.

All code/protocol frozen by preflight SHA256 before training. Unit tests and
small stored-state replay are permitted before the formal run, no new training.
Model/data/cache/audit tensors stay local; package generated machine records and
checks with hashes. Expected disk budget is under15 GiB; no old data deletion.
