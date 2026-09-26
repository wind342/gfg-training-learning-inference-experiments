# Router V4: full ten-task FewRel validation

Base: `7ed4fb4c6c0c61f4436ad944c1a88af1650b840f`.
New independent harness and generated records only. Router V4 and every earlier
experiment, result and frozen paper tag are unchanged.

## Outcome

All four fixed-seed runs completed once. Record/identity checks **PASS**.
The predeclared long-horizon success screen **did not pass**:

- Router final old/new/all accuracy: 64.0656% /
  96.6065% / 67.3197%.
- Relative to fixed replay: +0.0903
  percentage points old, +0.8036
  new, +0.1616 all.
- Old exposures: 27,614 versus
  30,240, or 91.3161% of fixed
  (8.6839% fewer), not the hoped-for 50–65% budget.
- At exactly matched exposures, random replay is
  0.5531
  points higher in final all accuracy. Router has only a
  0.1488-point advantage in
  mean task2–10 endpoint all accuracy, below the frozen 1-point screen.
- Router is substantially better than no replay, but this run does **not**
  establish an advantage over matched random replay. Its near-fixed retention
  does not establish economical long-horizon routing.
- Capacity saturation, long unserved-risk spans, weak-group failures and the
  computation overhead are retained. No strategy change, sweep or training rerun
  was made. Seeds2022/2023 were not launched because the frozen trigger failed.

These are single-seed descriptive comparisons, not statistical significance or
claims about all continual-learning settings.

## Fixed setup and admission

See [PROTOCOL.md](PROTOCOL.md) for the pre-training frozen protocol, success
screen and isolation decision. Seed2021; fixed80-relation order; ten tasks of
eight relations; 420 train examples/relation; batch16; LR5e-6; one epoch/task;
210 updates/task. Each arm continues the same retained original task1 checkpoint
for1,890 updates (tasks2–10). No task oracle at readout: classes expand to80.
The same original BERT, AdamW state, train batches and update/dropout seeds are used.

Unchanged mechanism: V4 state machine, threshold0.9, q10/rearm, memory10/group,
cap8, two replay examples/group, actual-sample-count loss weighting, sequential
microbatch backward, predictor and emergency gate. The harness imports these
implementations; no old source is patched on disk.

Fixed replay means16 old examples/update from eight rotating old groups, not all
historical examples per update. Random receives only
`step/task/old_batch_size`: it matches Router at **every update**, not merely
in total. It uses independent PCG64 sampling over old relations and the same
training-only memory, without state/q10/risk/prediction information. It uses
native alpha1. Because Router also has an emergency gate, this is a whole-V4
versus random-feedback comparison, not a pure routing-only ablation.

### Authorized isolation correction

Original data.npz is unchanged. Before training, exact encoded-input auditing found
control50124 duplicated training49887; control50130 is its first eligible unused
same-relation replacement. Exactly seven overlapping test indices are excluded
from the primary metric:7561,12566,18193,29275,29338,47545,55297.
Primary evaluation has11,193 examples; all11,200 original test predictions and
secondary metrics remain in every RESULT. Control remains10 examples/relation.
Memory eligibility continues to exclude the original full-test content.

No training example, order, label, original file or earlier result was changed.
This verifies exact encoded-input separation, not semantic near-duplicate removal.
The user approved this isolation correction before execution.

Comparisons below use **macro relation accuracy**. Micro accuracy, original-test
sensitivity, every relation and every cohort are also in CHECKS/RESULT.

## Final capabilities, exposure and cost

| Arm | Old accuracy | New accuracy | All accuracy | Old-example exposures | Continuation wall (s) | Update time (s) | Peak allocated / reserved (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| New-only native | 0.0998% | 97.8571% | 9.8755% | 0 | 931.146 | 571.565 | 4.930 / 5.129 |
| Full fixed replay | 63.9753% | 95.8029% | 67.1581% | 30240 | 1474.232 | 1113.330 | 5.360 / 5.568 |
| Router V4 | 64.0656% | 96.6065% | 67.3197% | 27614 | 4636.017 | 4276.702 | 6.183 / 6.438 |
| Matched-exposure random | 64.6801% | 96.6065% | 67.8728% | 27614 | 1418.402 | 1060.482 | 5.362 / 5.523 |

Router wall/fixed: 3.145x; Router update-time/fixed:
3.841x. Fewer old examples here did not
translate to lower wall time. Wall includes shared task1 evaluation and every
endpoint evaluation, but excludes checkpoint loading/saving and preflight/checking.
Update time includes each arm's in-step work, including Router observations,
prediction, gate and batch checks. These are serial single-run measurements,
not repeated isolated timing benchmarks; lightweight monitoring ran alongside.

| Arm | Load (s) | Evaluation (s) | Final save/hash (s) | Total run wall (s) | Mean endpoint old, tasks2–10 | Mean endpoint all, tasks2–10 |
|---|---:|---:|---:|---:|---:|---:|
| New-only native | 1.352 | 357.326 | 2.207 | 934.706 | 3.9268% | 23.3237% |
| Full fixed replay | 1.357 | 358.705 | 2.218 | 1477.807 | 76.9777% | 79.9304% |
| Router V4 | 1.746 | 356.803 | 2.244 | 4640.008 | 77.1896% | 80.1423% |
| Matched-exposure random | 1.287 | 355.717 | 2.270 | 1421.959 | 76.9234% | 79.9935% |

## Every task endpoint

Task1 is the shared prefix: 97.1429% for all arms;
it was not retrained four times.

| Task | Arm | Old | New | All |
|---|---|---:|---:|---:|
| 2 | New-only native | 5.6250% | 92.5893% | 49.1071% |
| 2 | Full fixed replay | 87.8571% | 90.0000% | 88.9286% |
| 2 | Router V4 | 86.6071% | 90.5357% | 88.5714% |
| 2 | Matched-exposure random | 85.8929% | 90.4464% | 88.1696% |
| 3 | New-only native | 5.8929% | 98.0357% | 36.6071% |
| 3 | Full fixed replay | 82.4994% | 97.3214% | 87.4400% |
| 3 | Router V4 | 82.0969% | 97.0536% | 87.0825% |
| 3 | Matched-exposure random | 82.0529% | 97.1429% | 87.0829% |
| 4 | New-only native | 6.0545% | 98.8374% | 29.2502% |
| 4 | Full fixed replay | 80.3747% | 97.4981% | 84.6555% |
| 4 | Router V4 | 81.4821% | 96.2481% | 85.1736% |
| 4 | Matched-exposure random | 81.6611% | 96.6052% | 85.3971% |
| 5 | New-only native | 5.4018% | 97.7679% | 23.8750% |
| 5 | Full fixed replay | 80.7295% | 96.5179% | 83.8872% |
| 5 | Router V4 | 80.6839% | 96.3393% | 83.8150% |
| 5 | Matched-exposure random | 80.0379% | 96.6964% | 83.3696% |
| 6 | New-only native | 6.1786% | 92.5634% | 20.5760% |
| 6 | Full fixed replay | 78.9960% | 93.7215% | 81.4503% |
| 6 | Router V4 | 80.1389% | 93.5443% | 82.3731% |
| 6 | Matched-exposure random | 79.3532% | 93.9920% | 81.7930% |
| 7 | New-only native | 2.4554% | 92.1429% | 15.2679% |
| 7 | Full fixed replay | 75.7892% | 91.5179% | 78.0361% |
| 7 | Router V4 | 75.6587% | 91.4286% | 77.9115% |
| 7 | Matched-exposure random | 74.9703% | 91.8750% | 77.3852% |
| 8 | New-only native | 1.6582% | 89.6429% | 12.6563% |
| 8 | Full fixed replay | 71.2962% | 90.6250% | 73.7123% |
| 8 | Router V4 | 72.7779% | 90.3571% | 74.9753% |
| 8 | Matched-exposure random | 72.4147% | 91.0714% | 74.7468% |
| 9 | New-only native | 1.9754% | 98.4821% | 12.6984% |
| 9 | Full fixed replay | 71.2819% | 96.6939% | 74.1055% |
| 9 | Router V4 | 71.1952% | 96.9624% | 74.0582% |
| 9 | Matched-exposure random | 71.2473% | 97.1416% | 74.1244% |
| 10 | New-only native | 0.0998% | 97.8571% | 9.8755% |
| 10 | Full fixed replay | 63.9753% | 95.8029% | 67.1581% |
| 10 | Router V4 | 64.0656% | 96.6065% | 67.3197% |
| 10 | Matched-exposure random | 64.6801% | 96.6065% | 67.8728% |

## Historical cohort retention at final task10

| Cohort learned at task | New-only | Fixed | Router | Matched random |
|---|---:|---:|---:|---:|
| 1 | 0.0000% | 72.6786% | 72.1429% | 70.5357% |
| 2 | 0.0000% | 53.7365% | 54.7219% | 53.5586% |
| 3 | 0.0000% | 79.2761% | 80.7040% | 79.1836% |
| 4 | 0.0000% | 66.8448% | 69.9756% | 68.0094% |
| 5 | 0.0000% | 61.6071% | 60.0893% | 58.6607% |
| 6 | 0.0000% | 78.4265% | 75.6599% | 77.8041% |
| 7 | 0.0000% | 43.6607% | 43.1250% | 49.4643% |
| 8 | 0.0000% | 44.1964% | 46.1607% | 50.1786% |
| 9 | 0.8980% | 75.3514% | 74.0114% | 74.7264% |
| 10 | 97.8571% | 95.8029% | 96.6065% | 96.6065% |

All intermediate cohort endpoints are retained in `results/CHECKS.json`
and each arm's `records/<arm>/RESULT.json`.

## Feedback, risk and capacity

Router feedback-active fraction: 99.8413%.
Modes: FORMATION3, BALANCE13,
RECOVERY1874; release events:1.
The router entered recovery almost continuously; this is recorded, not reset.

| Task | Router old exposures | Fixed old exposures | Ratio | Feedback-active | Cap8 saturated steps | Risk>8 steps | Risk-unserved group-steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1482 | 3360 | 44.1071% | 98.5714% | 0 | 0 | 0 |
| 3 | 2692 | 3360 | 80.1190% | 100.0000% | 21 | 0 | 0 |
| 4 | 3280 | 3360 | 97.6190% | 100.0000% | 170 | 165 | 995 |
| 5 | 3360 | 3360 | 100.0000% | 100.0000% | 210 | 210 | 2388 |
| 6 | 3360 | 3360 | 100.0000% | 100.0000% | 210 | 210 | 2920 |
| 7 | 3360 | 3360 | 100.0000% | 100.0000% | 210 | 210 | 4444 |
| 8 | 3360 | 3360 | 100.0000% | 100.0000% | 210 | 210 | 5489 |
| 9 | 3360 | 3360 | 100.0000% | 100.0000% | 210 | 210 | 5677 |
| 10 | 3360 | 3360 | 100.0000% | 100.0000% | 210 | 210 | 7697 |

From task5 onward the dose is16 old examples at **every** step: exposure has
become fixed-like, although relation selection remains state-routed. This is not
a claim that every old relation is replayed each step. The maximum active-risk
set reaches49 groups.
Overall,1425/1890 updates have more risks than
the cap;1451/1890 saturate the coalition.
Risk-unserved group-steps total29610; they count each
unselected currently marked old risk separately, not unique examples.

Every eligible old relation receives at least some feedback in every task.
That does not rule out long spans crossing a task boundary:

| Group | Replay occurrences | Eligible updates | Max no-feedback span | Risk-unserved steps | Max consecutive risk-unserved span | Final Router accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 39 | 310 | 1050 | 219 | 651 | 201 | 54.2857% |
| 19 | 218 | 1470 | 272 | 565 | 191 | 89.2857% |
| 42 | 202 | 840 | 189 | 613 | 189 | 62.8571% |
| 4 | 702 | 1890 | 208 | 846 | 181 | 92.8571% |
| 55 | 108 | 630 | 142 | 342 | 125 | 55.7143% |
| 12 | 216 | 1680 | 149 | 406 | 120 | 89.2857% |
| 31 | 516 | 1260 | 134 | 835 | 110 | 20.0000% |
| 15 | 780 | 1680 | 106 | 964 | 104 | 38.5714% |

The unchanged priority rule can spend all high-priority slots on DEGRADED
groups while other at-risk PROTECTED groups wait. Long waits are not uniformly
equivalent to failure (for example group19 still ends near89%); the records
show allocation and outcomes, not an isolated causal estimate of every deficit.

### PARTIAL groups9/14

Both receive520 replay occurrences across tasks3–10; neither again suffers
an entire210-step task with zero feedback. Their longest no-feedback span is13
updates. They remain PARTIAL at every task3–10 endpoint. Nevertheless:

| Group | Accuracy after own task2 | Final Router | Final fixed | Final random |
|---|---:|---:|---:|---:|
| 9 | 57.1429% | 0.7143% | 2.1429% | 5.0000% |
| 14 | 82.1429% | 57.1429% | 56.4286% | 55.7143% |

Thus fixing zero-feedback omission does not guarantee retention with low-priority
maintenance. Group9's control capability also declines to0 at task10 while it
remains PARTIAL; the unchanged rule does not promote never-protected PARTIAL
groups to DEGRADED on such decline. No policy change was made in response.

### Old-state distribution and dose

| Endpoint | Old PARTIAL | Old PROTECTED | Old DEGRADED |
|---|---:|---:|---:|
| 2 | 0 | 5 | 3 |
| 3 | 2 | 9 | 5 |
| 4 | 2 | 9 | 13 |
| 5 | 3 | 10 | 19 |
| 6 | 3 | 18 | 19 |
| 7 | 3 | 23 | 22 |
| 8 | 4 | 21 | 31 |
| 9 | 7 | 26 | 31 |
| 10 | 7 | 17 | 48 |

Final all80 states: {"DEGRADED":48,"PROTECTED":24,"PARTIAL":7,"FORMING":1}.
States belong to control observations; they are not invented for native baselines.

| Actual old batch size | Old-loss weight | Router updates | Random updates | Fixed updates |
|---|---:|---:|---:|---:|
| 0 | 0.0000% | 3 | 3 | 0 |
| 4 | 20.0000% | 17 | 17 | 0 |
| 6 | 27.2727% | 69 | 69 | 0 |
| 8 | 33.3333% | 105 | 105 | 0 |
| 10 | 38.4615% | 50 | 50 | 0 |
| 12 | 42.8571% | 77 | 77 | 0 |
| 14 | 46.6667% | 118 | 118 | 0 |
| 16 | 50.0000% | 1451 | 1451 | 1890 |

Weights are `n_old/(16+n_old)`, with sequential microbatch backward. Every
relation's replay counts, risk counts and longest gaps, per task and overall,
are in CHECKS. Random matches dose/timing/weight exactly without receiving the
Router's selected relations.

## Predictor and emergency gate

86 full predictor calls;41119/41120
selected-endpoint control-boundary signs agree with actual native evaluation
(99.9976%). Of107 actual
correct/incorrect transitions,107 agree
(100.0000%).

These figures concern control targets on invoked full measurements at the chosen
alpha, not every training/test example, unmeasured update or every candidate alpha.
The one sign error and full before/actual/prediction confusion are retained.
Prediction accuracy alone does not ensure long-term retention.

- Gate counts: {"NATIVE_SAFE":71,"NOT_MEASURED":1804,"EMERGENCY_SHRINK":14,"NO_SAFE_ALPHA":1}.
- Applied alpha counts: {"1.0":1876,"0.5":5,"0.25":7,"0.125":2}.
- Actual shrink:14. One NO_SAFE_ALPHA event is preserved;
  the inherited gate behavior was not replaced or tuned.
- Native, fixed and random have no full predictor or emergency shrink.

## Checks, artifacts and reproduction

- Preflight:131 test entries,130 passed,1 explicit skip. The old unstable
  recreated-tensor-view fixture is replaced only in the test runner by its already
  established stable-view equivalent; no production guard or old test is edited.
  A final separate rerun of the11 new unit tests also passed.
- All four arms completed once; four arm return codes and final record checker
  are0. `results/EXECUTION.json` and `results/CHECKS.json`:PASS.
- Checker validates source/protocol/data/checkpoint identities; current batch
  equality; memory/sample roles; all1,890-step Router decisions and state
  continuity; actual-count dose; random reproducibility and per-step matching;
  gate recomputation; every endpoint prediction-derived metric; final hashes.
- Separate final-checkpoint native replay:PASS. All11,200 original-test predictions
  per arm (44,800 total) match the saved ledgers with zero differences; primary
  and original per-relation metrics also match. See
  `results/FINAL_CHECKPOINT_REPLAY.json`. This is final-endpoint re-evaluation,
  not independent re-execution of all training/intermediate updates.
- Public `records/` contains lossless generated CONFIG/RESULT files,
  compressed update logs, exposure-only schedule and execution logs.
  `results/PUBLIC_RECORDS_MANIFEST.json` verifies both compressed bytes and
  uncompressed log hashes. No FewRel example text/token arrays, pretrained model
  or model/optimizer checkpoints are uploaded.
- A directory-local `.gitattributes` disables newline conversion only for these
  generated records/reports and the final-replay source copy. This preserves their
  byte-level hashes in Git; existing repository attributes and evidence are unchanged.
  The staged Git blobs were checked against all19 public-record manifest entries,
  all generated report bytes and all908 preflight-frozen source identities:PASS.
- Full local evidence is `E:/gfg-evidence/router-v4-ten-task-validation`.
  Only four final checkpoints are retained, not task-by-task model copies.

Pinned local runtime:
`E:/gfg-python/governor-v4-bench/Scripts/python.exe`.
Requires the existing original data.npz, local pinned BERT and retained task1
checkpoint; old deleted task2–10 checkpoints are not dependencies.

```text
python -B -m experiments.gfg_router_v4_ten_task_validation.preflight
python -B -m experiments.gfg_router_v4_ten_task_validation.run_all
```

These are **create-only original-run commands**, not commands to rerun over
completed evidence. They refuse existing output directories/files. Read-only
`check.verify` can inspect existing records without writing; the check CLI's
export is also create-only. Raw replay needs the same local data/checkpoints;
the public package alone is not a self-contained model/data distribution.
The separate final-replay source is `FINAL_CHECKPOINT_REPLAY_CODE.txt`;
execute it through Python/runpy only in an appropriately prepared environment,
without overwriting completed audit outputs.

### Pre-training development notes

An unmatched parenthesis prevented initial import of a new test. It was corrected
before preflight and before any GPU training; all11 new tests then passed.
No formal arm was rerun. A later console-summary parser encountered truncated
tool output; the underlying complete execution logs and experiments were
unaffected. That display error did not alter any source, policy or evidence.

### Frozen repeat decision

```json
{
  "criteria": {
    "completed": true,
    "final_old_close": true,
    "final_new_close": true,
    "mean_old_close": true,
    "exposure_at_most_65pct": false,
    "final_random_gap": false,
    "mean_random_gap": false,
    "beats_native_old": true,
    "beats_native_all": true
  },
  "eligible_if_integrity_pass": false,
  "seeds": [
    2022,
    2023
  ],
  "statistical_significance_claim": false
}
```

Integrity PASS is not a scientific-success verdict. The failed exposure and
matched-random superiority criteria stop the work at the primary seed.
