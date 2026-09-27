# Final Architecture v2 — fixed FewRel ten-task validation

Frozen BERT + existing shared linear head + actual AdamW candidate + exact
functional endpoint prediction + fixed per-relation task-entry floors +
unchanged Hildreth projection / FP32 reserve / actual FP32 recheck / commit.

Engineering base: `03a17843a8c7ad82b21eecf4b7b0360feb542c56`.
Historical Native-210 and V1-210 authority: `115534a09cd554fcd75fd46dde711fdb98988519`.
Production policy is imported unchanged from the previously validated
`gfg_relation_capability_floor_task8_v1` (task8 and fixed630 continuation).
Only this experiment directory is added. Main, original experiments and frozen
research tags/artifacts are not changed.

## Frozen comparison

Two independently continuous arms start from the SAME verified, already-trained
task1 BERT/head/AdamW checkpoint. Task1 is not retrained. Tasks2–10 train once:

- `v2_210`: 210 scheduled updates/task,1890 updates total.
- `v2_630`: 630 scheduled updates/task,5670 updates total, three repetitions of
  the SAME fixed batch order. It never starts from `v2_210` endpoints.

Seed2021, head LR5e-5, batch16, frozen/eval BERT,80-row biased linear head,
AdamW(.9,.999), weight_decay0, clipping1; no resets/scheduler/replay/old loss/
Router/distillation/feature expansion/adapter/LoRA. Initial optimizer counter210;
final counters2100 and5880. No outcome-driven training extension or rerun.

Every task fixes its own K[r] once, after expanding the seen-class readout and
before updating. K is not reset at epoch boundaries or ratcheted upward during
the task. Native candidates satisfying all counts commit unchanged; otherwise
the existing nearest-margin selection and Hildreth policy are used. Maximum100
total sweeps, first-floor-safe stopping. Final acceptance is C[r]>=K[r], not
per-sample immutability. See [PROTOCOL.md](PROTOCOL.md).

## Task endpoint results

All entries are relation-macro test percentages. Historical arms were not rerun.

| Task | Native old | Native new | V1 old | V1 new | V2-210 old | V2-210 new | V2-630 old | V2-630 new |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|2|55.98|90.00|92.32|85.00|91.07|85.89|87.77|90.80|
|3|23.01|97.68|83.88|94.82|83.21|95.18|82.54|97.77|
|4|21.06|95.26|85.71|84.37|85.26|84.37|84.82|91.78|
|5|14.74|96.16|84.08|81.34|83.28|84.29|83.97|91.07|
|6|19.18|91.01|81.30|76.30|80.78|78.44|82.50|86.16|
|7|16.89|90.80|78.53|77.14|77.92|78.66|80.78|84.73|
|8|14.32|89.11|76.44|65.98|75.86|69.11|78.33|79.64|
|9|17.49|96.61|74.22|78.92|73.76|80.00|76.96|86.96|
|10|11.81|92.59|73.09|65.41|72.92|70.33|75.40|79.00|

| Method | Task10 old | Task10 new | Task10 all | Mean old | Mean new | Mean all | Skips |
|---|---:|---:|---:|---:|---:|---:|---:|
|Native-210|11.81|92.59|19.89|21.61|93.25|35.55|0|
|V1-210|73.09|65.41|72.32|81.06|78.81|80.68|0|
|V2-210|72.92|70.33|72.66|80.45|80.70|80.53|0|
|V2-630|75.40|79.00|75.76|81.45|87.55|82.85|0|

Means are unweighted across the nine task2–10 endpoints, not weighted by update
count. Task10 old comprises72 relations; new comprises8; all comprises80.

At task10, V2-630 versus V1-210 improves old by2.31pp, new by13.59pp and all
by3.44pp. V2-630 versus V2-210 improves old by2.48pp, new by8.67pp and all
by3.10pp. This is a larger-budget comparison, not an equal-compute claim.
At equal210-step budget, V2 increases mean new by1.89pp, but mean old declines
by0.61pp and mean all by0.15pp. V2-210 does NOT uniformly dominate V1.

## Controller behavior and cost

| Quantity | V2-210 | V2-630 |
|---|---:|---:|
|Scheduled / executed updates|1890 /1890|5670 /5670|
|In-task relation-floor violations|0|0|
|Skipped updates|0|0|
|Correct-to-wrong events|3005|2695|
|Wrong-to-correct events|3016|2704|
|Largest sweep count|14|17|
|Training-loop wall|410.00s|1225.57s|
|Projection-call wall (included above)|191.87s|570.34s|

Events can revisit the same sample; they are not unique-example counts. The
recorded replacement relation sets describe changes in control correctness
membership, not a direct measurement of latent neural support identity.
Per-task constraint distributions, geometry, floors, recovery selections and
readout traces are retained in machine records/full local logs.

Training-loop wall includes instrumentation but excludes evaluation, audit
serialization, setup and independent checking. V2-630 training took20.43min;
V2-210 took6.83min. Historical Native/V1 recorded190.59s/424.15s; those are
historical measurements, not a fresh randomized timing benchmark.

## Task boundary and control–test gap

Bank sizes use the existing clean controls, reaching10072 at task10 (not10080).
One expansion-only flip occurred: V2-630, entering task3, relation5, dataset
ID4009 (bank index789). That relation's count changed136->135 before training;
the fixed task3 floor was135. V2-210 had zero expansion flips. These are separate
from in-task floor violations and have NOT been hidden by the new entry floor.

Task10 illustrates the finite-bank guarantee's generalization limit:

| Arm | Old control correct, entry -> end | Old test macro, expanded entry -> end |
|---|---|---|
|V2-210|8474 ->8476 /10072|74.46% ->72.92%|
|V2-630|8991 ->8995 /10072|78.07% ->75.40%|

All individual relation floors held; test retention is not a hard guarantee.
Full per-task/per-relation control–test differences are in RESULTS.json.
The gap demonstrates imperfect generalization of the bank constraint; it does
not by itself prove bank coverage is the unique causal explanation.

Test never enters loss, floors, constraint selection, epoch choice or tuning.
After both arms finish each task, one read-only test pass produces both current
readout and next-task expanded-readout diagnostics on already-seen examples.
Thus expanded-entry loss is not conflated with within-task learning loss.

## Integrity and documented engineering recoveries

Independent audit **PASS**: all7560 native candidates,35358 Hildreth sweeps,
302400 relation-step floor checks, continuous head/moment states, and actual
test predictions at every task2–10 endpoint were independently reconstructed.
The checker reconstructs every AdamW candidate from saved clipped gradients and
moments, independently replays selection/Hildreth/FP32 endpoints, verifies
continuity/floors, and recomputes actual BERT test predictions. Both arms' common
first210 steps are byte-identical. Source, data, cache and artifact identities
are retained; the finite100-sweep solver is not claimed to certify infeasibility
or globally minimum-distance solutions of the relation-count problem.

Two implementation issues are disclosed and preserved:

1. After task2, a task3 bookkeeping assertion compared old-only-batch margins
   with full-monitor-batch margins. FP32 differences were at most4.53e-6, with
   ZERO correctness differences. The check now uses the production batch shape.
   Completed task2 was hash-preserved; training resumed from each arm's own
   task2 head/moments. No completed training update was repeated. Original
   FAILURE.json, preflight, source snapshots and HARNESS_RECOVERY.json remain.
2. The initial independent checker reused same-device optimizer state between
   its two verification arms. This was corrected by deep-copying the state
   before loading each verification arm. The first210 verification steps were
   repeated, but NO training was repeated or changed. CHECKER_REVISION.json and
   the prior checker source are preserved.

Only the bookkeeping/checker changed; the production floor, selection,
projection, reserve, LR, optimizer and training budgets did not change.

## Scope and reproducibility

This fixed single-seed run supports greater new-learning plasticity under the
relation-level policy and improved long-run outcomes with a larger fixed budget.
V2 changes both per-sample locking and within-task ratcheting; their effects
are not isolated separately. The two continuous budget arms have different
later task-entry states. This is final validation on a previously studied
benchmark/seed, not a new blind benchmark or universal architecture claim.

Full local authority: `E:/gfg-evidence/final-architecture-v2-ten-task`.
The public results package contains protocols, compact per-step ledgers,
evaluation predictions, configurations, integrity reports and hashes.
Large AUDIT.pt / STEPS.jsonl files remain local, hash-addressed by each epoch's
TRAIN_RESULT.json. Independent tensor replay requires those artifacts and the
existing frozen FewRel/model/checkpoint dependencies; Git alone is not the full
tensor archive. No raw dataset, model weights or repeated BERT checkpoints
are committed. No new Zenodo version is created.

```powershell
python -m experiments.gfg_final_architecture_v2_ten_task.tests
python -m experiments.gfg_final_architecture_v2_ten_task.run
python -m experiments.gfg_final_architecture_v2_ten_task.check
```

These are once-only generation commands against an empty output root; the
writers refuse to overwrite existing evidence. `resume.py` records the one
specific historical harness recovery, not an automatic result-driven retry.
`package.ps1` packages only after CHECKS PASS, preserving original file bytes.
