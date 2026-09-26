# FewRel: short mixed-feedback direction experiment

This is a fixed one-seed,60-update ablation from the same task-1 checkpoint.
It changes the training feedback mixture, not the predictor, v4 policy, model,
optimizer, initial optimizer memory or new-task sample sequence. See PROTOCOL.md.
Only this new directory is added; original evidence/results remain unchanged.

## Results

All arms started at97.1429% old-test accuracy and0% new-test accuracy.
Each arm used960 identical new-task training occurrences. Replay additionally
used960 old-task occurrences drawn from80 training examples, never control/test
examples. Test evaluation covers1120 old and1120 new examples with all16 classes
active and no task identity oracle. Test results never select updates.

| After60 updates | Old accuracy | New accuracy | All16 relations | Wall |
|---|---:|---:|---:|---:|
| New-only + original v4 governor |73.4821%|84.8214%|79.1518%|121.082s|
| Equal old/new replay + native AdamW |93.3929%|79.0179%|86.2054%|85.873s|
| Equal old/new replay + original v4 governor |93.1250%|79.1071%|86.1161%|124.296s|

At the same new-data exposure and update count, native mixed replay retains
19.9107 percentage points more old accuracy than new-only governor, while new
accuracy is5.8036 points lower. Combined balanced test accuracy is7.0536 points
higher. This is a real retention/plasticity trade-off, not preservation through
zero learning: new-task accuracy rises from0% to79.0179%.

Adding the current governor to matched mixed replay changes old accuracy by
-0.2679 points and new accuracy by+0.0893 points (three and one test examples),
and increases wall from85.873s to124.296s. There is no clear incremental benefit
from this governor in the short run. It selected alpha2 once, at offset37;
all other59 updates were native. New-only governor also selected alpha2 once.

Selected endpoint predictions: new-only640/640, replay-governor480/480. Actual
correct/incorrect transitions in those audited endpoints were only4 and1,
respectively, all correctly predicted. These small numbers are not a new broad
predictor-accuracy claim. Full calls numbered4 and3, respectively.

## Interpretation and limits

Letting old capabilities participate in forming subsequent gradients/updates
substantially reduces early forgetting here. Existing scalar selection applied
only to new-task updates did not deliver that outcome. This supports testing
feedback-directed update construction rather than only amplitude selection.
It does not prove that a direction change is mathematically necessary, identify
a uniquely optimal mixture, or establish restored capability after severe decay.
The fork precedes task2: this tests prevention/early retention, not the RL-E06
recovery intervention. No direction cosine or causal-support decomposition was
measured. Ordinary replay is the tested intervention; its gains must not be
attributed to a new GFG direction-selection algorithm.

Replay has additional old-data exposure and gradients. Wall includes these,
controller measurements, selected endpoint audits and fixed evaluations; model
loading is excluded. No new-only native arm was run, so the table does not
establish a speed advantage over new-only AdamW. No learning-rate tuning,
additional seeds, longer runs, or threshold stopping were performed.

## Integrity and resource record

The new-only path reproduced all60 original alpha/full-call decisions; maximum
selected endpoint difference was7.04e-6 (FP32 execution tolerance), not bitwise
identity. All three arms had exactly identical initial test predictions. Checker
recomputed test scores, batch identities, memory exclusions, source/checkpoint
hashes and selected endpoint agreement. Status: PASS. It is a log/data check,
not an independent second full training run; no final checkpoints were saved.

The inherited FewRel probe/test content duplicate remains disclosed as in the
parent experiment. A post-hoc sensitivity excludes test index7561; conclusions
are unchanged. No split or controller probe was changed to improve results.

The first replay-governor attempt failed before its first committed update:
the boundary workspace planner rejected the available driver-level memory budget.
The failure remains intact. A separate resource-only retry calls
`torch.cuda.empty_cache()` before full measurement, releasing unused allocator
reservations without altering live tensors or scientific settings. Successful
retry timing includes this operation; the failed attempt's time is separately
recorded in results/RESOURCE_FAILURE.json, not silently folded into a clean run.
Peak allocations: new-only6,167,232,512 bytes, replay-native5,753,887,232 bytes,
replay-governor6,629,484,544 bytes. No large new checkpoint or model copy on disk.

Raw records: E:/gfg-evidence/fewrel-replay-direction-smoke-v1/.
Compact results and raw record hashes: results/CHECKS.json.
Use the existing governor-v4-bench environment. All writers are create-only:

```text
python -m experiments.gfg_fewrel_replay_direction_smoke_v1.run new_governor
python -m experiments.gfg_fewrel_replay_direction_smoke_v1.run replay_native
python -m experiments.gfg_fewrel_replay_direction_smoke_v1.run replay_governor
# Only if the documented first-update resource failure occurs:
python -m experiments.gfg_fewrel_replay_direction_smoke_v1.retry_resource
python -m experiments.gfg_fewrel_replay_direction_smoke_v1.check
```
