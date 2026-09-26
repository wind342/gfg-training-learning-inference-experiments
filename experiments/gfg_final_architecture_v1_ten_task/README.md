# Final Architecture v1: first FewRel ten-task validation

Base `3daa90d2572cb50aff09aee3d95997ae4244d799`. New isolated experiment;
all earlier results, main and frozen tags are unchanged. The production Hildreth
function is imported unchanged. See [PROTOCOL.md](PROTOCOL.md) for the exact
frozen rule and the pre-approved data-isolation correction.

Frozen BERT + existing plastic linear head + actual AdamW candidate + exact
functional-boundary prediction + Hildreth multi-constraint projection + FP32
roundoff reserve + actual FP32 recheck + commit/skip.

## Primary result

Both arms completed tasks2-10 continuously from the same task1 checkpoint.
Final Architecture v1 executed all1890 scheduled updates, with zero skips and
zero executed correct-to-wrong occurrences on receiving-correct protected controls.
It preserved substantially more old test capability than native head-only training,
but learned less of several new tasks. Control safety is not test-set safety.
Final uses labeled validation controls as safety constraints, whereas native does
not. Thus the comparison fixes current training data and optimizer settings, not
total access to labeled historical examples. No control cross-entropy or replay
loss is used.

All entries below are percentages; old/new/all are relation-macro accuracies.

| Task | Native old | Native new | Final old | Final new | Final all | Final skips |
|---|---:|---:|---:|---:|---:|---:|
|2|55.98|90.00|92.32|85.00|88.66|0|
|3|23.01|97.68|83.88|94.82|87.53|0|
|4|21.06|95.26|85.71|84.37|85.38|0|
|5|14.74|96.16|84.08|81.34|83.53|0|
|6|19.18|91.01|81.30|76.30|80.47|0|
|7|16.89|90.80|78.53|77.14|78.33|0|
|8|14.32|89.11|76.44|65.98|75.14|0|
|9|17.49|96.61|74.22|78.92|74.74|0|
|10|11.81|92.59|73.09|65.41|72.32|0|

| Summary | Native old | Native new | Native all | Final old | Final new | Final all |
|---|---:|---:|---:|---:|---:|---:|
|Task10:72 old /8 new /80 all|11.8147|92.5874|19.8920|73.0879|65.4079|72.3199|
|Unweighted task2-10 mean|21.6100|93.2467|35.5522|81.0645|78.8101|80.6775|

Final task10 advantage: old +61.2732 percentage points and all +52.4279 points;
new -27.1795 points. These are single-seed descriptive comparisons, not estimates
of population performance or statistical significance. The task mean averages
nine different task-end macro scores, not a pooled prediction accuracy.

## Protection and scalability

| Task | Old bank | Receiving-correct at first update | Correct at task end | Active pairs min/median/max | Sweeps min/median/max | Training s | Projection/check s |
|---|---:|---:|---:|---|---|---:|---:|
|2|1120|1085|1094|1 /39 /58|1 /1 /4|26.48|4.15|
|3|2237|2045|2064|11 /129 /172|1 /4 /6|31.32|8.74|
|4|3357|3148|3158|41 /161.5 /213|2 /4 /6|35.73|12.58|
|5|4474|4079|4110|47 /207.5 /271|2 /5 /7|40.87|17.18|
|6|5594|5036|5070|89 /330 /375|2 /7 /10|47.24|22.93|
|7|6714|5928|5962|94 /333 /416|3 /6 /11|51.37|26.56|
|8|7833|6835|6860|88 /353 /458|2 /7 /10|57.86|32.18|
|9|8953|7590|7627|152 /413 /463|3 /7 /12|63.83|37.61|
|10|10072|8546|8575|146 /451.5 /594|2 /7 /13|69.47|42.59|

Constraint/sweep distributions above cover triggered updates only. Full per-step
receiving-correct counts, projected/native norm ratios, cosine, per-task summaries
and all head-state identities are in the machine records. There were3 native-safe
updates and1887 safe projected updates, totaling9973 sweeps. No sweep-cap failure.
No expansion-only old-control correct-to-wrong event occurred at task boundaries.
This was measured before each task's first update; it was not added as a new gate.

Final training-loop wall totals424.15 s versus190.59 s for native (about2.23x).
Projection-call time totals204.53 s, including endpoint checks and bookkeeping;
it is not isolated solver-only time. Training-loop measurements include logging
and per-step audit work, exclude model/cache setup, per-task audit serialization,
full test evaluation and independent checking. Shared read-only test evaluation
totaled354.62 s. Timing is one local run, not a replicated performance benchmark.
Final-arm peak PyTorch CUDA allocated memory ranged675.42-728.46 MiB; this is not
whole-device memory usage including driver/context/display allocations.

The increasing projection work is real overhead. Zero skips does not mean zero
restriction: new-task test accuracy at tasks8 and10 was around65-66%, compared
with native89-93%. This experiment supports continuous execution and substantially
improved old retention in this setting, not unchanged plasticity or no forgetting.

Final per-cohort test macro (task1 through10):86.25%,73.03%,85.80%,72.66%,71.16%,
68.98%,66.96%,57.95%,74.99%,65.41%. Each cohort has8 relations. Complete relation
and immediately previous-task scores are included in every evaluation record.

## Fixed execution and isolation

Seed2021,80 relations,10 tasks of8, LR5e-5,210 updates/task,batch16, original
current-task batch order, no scheduler or optimizer reset. Task1 is the existing
verified checkpoint, not newly trained. Both heads continuously inherit their own
AdamW moments from step210 to2100. BERT is frozen/eval/no-gradient throughout.
No replay, Router, old loss, scalar shrink, distillation, LoRA or adapter.

Ten validation inputs duplicate train inputs and were excluded from the bank
before training with user approval. No replacement, repetition or train-order
change. Eight would have been in task10's old72 relations; the bank therefore
has10072 rather than10080 samples. Historical seven primary-test exclusions remain
unchanged. All clean controls and primary test inputs are hash-disjoint from train
and each other. This checks complete encoded inputs, not semantic equivalence.

Validation features were cached once; the existing task2 1200-row matrix was
reused byte-for-byte. The first10 clean current-relation validation examples are
read-only audit rows, not old safety constraints. Future validation rows are never
passed into earlier protection calls. Test goes only to the post-task read-only
evaluator after both arm training records are complete and hashed.

The native fast path preserves strictly positive margins; triggered projection
requires actual FP32 margin>=1e-6. These distinct paths exactly match the parent.
Only receiving-correct controls are protected. The reserve formula and100-sweep
cap never changed. First-safe stopping is not a claim of the global minimum-norm
solution, and finite-bank protection is not a theorem about unseen examples.

## Verification and reproduction

Preflight passed42 tests, including dynamic16/24/80-class checks and a historical
stored-state replay. During training, every task2 receiving/native/projected/
executed head exactly matched the parent trajectory. No formal restart, algorithm
change, additional arm, parameter search or test-driven decision was performed.
PyTorch emitted a warning about converting the training-loss tensor to a scalar
for logging after backward; it did not stop execution or alter optimization. The
frozen source was not changed to silence this warning during the run.

The independent checker reconstructs actual AdamW proposals from saved clipped
gradients and inherited states, then independently replays the Hildreth dual/
primal algebra. It checks every executed head, moment hash/counter, cross-task
continuity, endpoint, source identity and test barrier. Saved predictions are
used to recompute metrics; this is not a second complete BERT training run.
Independent verification **PASS**:3780 updates across the two arms,9973 projected
sweeps and42,902,160 endpoint margin values. Authoritative status and counts are
in `results/CHECKS.json`. Head/moment continuity and parent task2 exact identity
also passed. No original source or artifact was modified.

```text
python -B -m experiments.gfg_final_architecture_v1_ten_task.preflight
python -B -m experiments.gfg_final_architecture_v1_ten_task.run
python -B -m experiments.gfg_final_architecture_v1_ten_task.check
python -B -m experiments.gfg_final_architecture_v1_ten_task.package
```

Scripts refuse to overwrite the formal output root
`E:/gfg-evidence/final-architecture-v1-ten-task`. They do not automatically resume
or rerun failed training. Raw data, task1 BERT checkpoint, cached features, per-step
head/gradient/optimizer audit tensors remain local. Public records contain code,
protocol, generated results, predictions and compressed step ledgers with hashes.
The deleted historical Router ten-task final checkpoints are not dependencies.
