# Frozen ten-task FewRel validation of Router V4

Scientific/engineering base: 7ed4fb4c6c0c61f4436ad944c1a88af1650b840f.
Only this new directory may change. Original protocols, policy, state machine,
threshold .9, q10/rearm, cap8, two examples/group, memory10/group, sample-count
dose, predictor, emergency gate, AdamW and LR5e-6 remain unchanged.

## Primary experiment (freeze before any training)

Seed2021; existing fixed80-relation order and data split; eight relations/task;
420 training examples/relation; batch16; one epoch/task; 210 updates/task.
All four arms inherit the same original task1 model/optimizer/sentinel checkpoint
bd41bcbd84f280d4de7d721eeee1d7a7bad1ccbb04ceb7d84d0403a6bdbcb676.
Tasks2--10 add1,890 updates/arm. Task1 is shared, not independently retrained.
Current batch ordering and update/dropout seeds follow the existing implementation.
Full class readout expands to80; no task identity oracle. Test is observation only.

Arms (one execution each): router, native, replay, random. Execute router first
to derive an exposure-only schedule. Fixed replay means16 old examples/update,
eight rotating old relations/two rotating samples each, not all historical data
per update. Random replay reads ONLY step/task/old_batch_size from the schedule;
it matches Router's exposure at EVERY update, hence total/task budgets, feedback
timing and old-loss weight. It samples n_old/2 distinct old relations uniformly,
then two distinct memory samples per relation uniformly, using NumPy PCG64 with
seed2021+9000000, separate from all training RNG streams. No state/q10/risk,
predictor, emergency gate, alpha or Router-selected identity is provided to it.
Its AdamW alpha is1. The comparison is the complete V4 mechanism versus matched
random feedback, not a pure routing-only causal ablation when emergency gates fire.

Use sequential microbatch backward and actual sample-count weighting for all arms.
No sweep, no policy adjustments, no result-selected restart. Runtime cap6 hours
per arm (resource ceiling, not a performance target); preserve exceptions, partial
logs and failure results. Save ONLY each arm's final model/optimizer checkpoint,
not checkpoints at every task. Report wall/load/evaluation/update time separately.
Peak allocated AND reserved CUDA memory must be reported.

## Data isolation (approved before training)

Keep original data.npz and its SHA256 unchanged. Replace exactly control index
50124 with50130 (relation71, task9). It is the first eligible unused validation
sample of that relation with no exact encoded-input collision with any training,
test or other control sample. Keep10 controls/relation. No training sample/order
changes. Primary evaluation excludes test indices7561,12566,18193,29275,29338,
47545,55297, identified by encoded-input hashes regardless of label. Retain
original full-test predictions and metrics as secondary results. Clean test:
11,193 examples across80 relations. Do not assume approximate semantic deduplication.
Audit content separation of all training, chosen controls and primary tests.
Memory eligibility continues to exclude ORIGINAL full-test content (plus selected
controls), preserving the existing memory rule, even for excluded test examples.
Freeze controls, clean tests, memory identities, and current batch orders before
running; all arms share them. Old evidence remains untouched.

## Metrics and interpretation

For each endpoint: micro old/new/all accuracy, macro relation old/new/all accuracy,
each historical task cohort and per-relation accuracy, both primary and original
test sets. Use macro relation values for predeclared comparisons to avoid small
denominator differences after deduplication. Report exposures/ratio to fixed,
active fraction, per-relation exposures, old-weight histogram, FORMING/PARTIAL/
PROTECTED/DEGRADED distribution (Router only, no fictitious baseline states),
mode/release counts, active-risk histogram, risk>8 steps, cap-saturated steps,
risk-unserved group-steps and per-relation longest unserved streak, observed
feedback starvation, full predictor calls, selected endpoint sign agreement,
agreement on actual boundary transitions, alpha/gate/shrink counts and resource costs.
Report mean endpoint old retention across tasks2--10 and final old/new/all.
High aggregate sign agreement without boundary transitions is not enough.

The50--65% exposure band is an observation target, NEVER a controller constraint.
Single-seed continuation trigger, fixed now: all arms completed and checks PASS;
Router final old/new and mean task2--10 old are each within3 percentage points
of fixed; total exposure ratio<=.65; Router final all exceeds random by>=2 points
and mean task2--10 all exceeds random by>=1 point; Router final old/all exceed
native. These are operational screening margins, not statistical significance.
If ALL hold, repeat all four arms with fixed seeds2022 and2023, same data split and
relation order, but each seed's OWN newly trained shared task1 prefix. No claiming
independent multi-seed validation using a seed2021 checkpoint. If trigger fails,
stop after primary seed, retain negative evidence and do not tune/retry.

## Integrity

Unit tests before GPU; require historical V4 CHECKS PASS and source/base integrity.
Record immutable source/protocol/data/checkpoint identities in preflight/config.
Create-only run directories and result files. Each task stores real prediction
ledgers in RESULT, each update stores replay identity and native observations.
Independent checker recomputes metrics from prediction ledgers, verifies hashes,
batch/memory roles, exposure-only random schedule/reproducibility, Router choices,
state continuity, dose and gates. This is record verification, not an independent
native rerun. Store hashes for final model/optimizer checkpoints locally; publish
compact checked reports, never model weights or the underlying FewRel dataset.
