# Head-only Hildreth numerical-roundoff repair

One new arm, `140_controls_hildreth_roundoff`, based on
`405702d51b0fff8f281e59c65b8fb2fbd273ce7c`. Historical experiments are unchanged
and reused, not rerun. The only algorithmic change is a deterministic internal
FP32 roundoff reserve in Hildreth's constraint RHS. Actual FP32 acceptance remains
at least `1e-6`; maximum sweeps remains 100. See [PROTOCOL.md](PROTOCOL.md).

## Result

| Method | Old test macro | New test macro | All test macro | Skips /210 |
|---|---:|---:|---:|---:|
| 140/current two-pass (historical A) |94.8214%|73.6588%|84.2401%|112|
| 140/Hildreth (historical, no reserve) |95.5357%|61.0599%|78.2978%|149|
| 140/Hildreth + numerical reserve |92.3214%|84.9994%|88.6604%|0|

The revised trajectory executed all 210 updates: 3 unchanged native updates and
207 safe, nonzero projected updates. No receiving-correct old protection sample
was executed across the wrong side of its boundary. Compared with the previous
Hildreth trajectory, new test accuracy increased by 23.9395 percentage points and
all-test accuracy by 10.3626 points; old test accuracy decreased by 3.2143 points.
Compared with A, old decreased by 2.5000 points, new increased by 11.3406 points,
and all increased by 4.4203 points.

This supports a numerical acceptance bottleneck in the former implementation:
adding an interior reserve made safe nonzero learning possible throughout this
fixed run. It does not show free retention or guaranteed safety on unseen data.
The finite validation bank was protected, while held-out old accuracy still fell.
Nor are these trajectories matched counterfactual recoveries of individual old
skipped steps: later receiving states and native updates differ.

## Execution diagnostics

| Diagnostic | Value |
|---|---:|
| Final old protection correctness |1094 /1120|
| Final task2 train16 accuracy |85.2381%|
| Native-safe steps |3|
| Hildreth-triggered steps |207|
| Safe nonzero projected steps |207|
| Failed/skipped steps |0|
| Executed old protected correct-to-wrong |0|
| Active constraints, triggered min/median/max |1 /39 /58|
| Sweeps, triggered min/median/max |1 /1 /4|
| Total sweeps |220|
| Mean projected/native norm ratio, triggered |0.808813|
| Mean cosine(native, projected), triggered |0.804115|
| Projection-call time, including checks/bookkeeping |4.3864 s|
| Training-loop wall |26.7557 s|

Previous Hildreth projection-call/training-loop times were 17.4025 /39.3388 s.
These are single-run local measurements, not controlled timing replicates;
loading, post-training full train/test evaluation and independent auditing are
outside training-loop wall. The total 220 sweeps replaces 14,987 in the parent.

## Numerical rule and its limitations

Compute a fixed per-update per-logit reserve using FP64 arithmetic:
`u=2^-24`, `k=1539`, `gamma=k*u/(1-k*u)`, and
`E=gamma*abs([h,1])@max(abs(receiving_head),abs(native_head)).T`.
Each active pair uses `eta=E[target]+E[competitor]` and aims at `1e-6+eta`.
Neither acceptance nor the native-safe fast path is loosened. The first-safe
stopping rule does not claim a globally minimum-distance projection.

The actual active-pair reserve ranged from 0.00172464 to 0.00197883. This is a
conservative scale estimate, considerably larger than observed FP32 error in the
three historical numerical cases (maximum absolute pair error about 3.1e-6 to
3.4e-6). It is not a universal GPU error certificate or a fitted optimal reserve.
It changes the interior target and can affect the trajectory. No alternative
reserve, multiplier, learning rate or seed was tried after viewing results.

## Fixed settings and evidence integrity

FewRel task2, seed2021, LR5e-5, batch16, 210 updates, original task1 checkpoint
and inherited AdamW moments, identical current-task batch order, frozen/eval BERT,
only the existing linear head trained, same 1120 old validation controls and
80 new audit controls. No replay, Router, scalar shrink, adapter, extra predictor
or model update. AdamW moments advance once per scheduled update.

Preflight: **36 unit tests PASS**; the first/middle/last failed historical calls
(steps 5,120,209) passed the fixed numerical audit and actual endpoint checks.
See [DEVELOPMENT_NOTE.md](DEVELOPMENT_NOTE.md) for a caught-and-fixed preflight
scalar-precision bug before numerical auditing or formal training.

Exactly one formal trajectory was run. Train/test evaluation followed the hashed
training-completion barrier; no test-driven tuning or rerun. Independent checking:
**210 steps, 220 sweeps, 1,008,000 endpoint margin values PASS**. It independently
reconstructs the numerical budget, active-set dual/primal iterates and their
hashes, compares every executed head state, and recomputes reported metrics from
saved predictions. This is not an independent rerun of all BERT inference.

The preflight manifest preserves previous artifacts, all experiment Python source
hashes, the new protocol, data and checkpoint identities. No previous experiment,
main branch, frozen tag or original evidence artifact is edited. Public records
contain generated results, predictions and step ledgers, not original datasets,
cached BERT features or model weights.

## Entry points

From this repository root, using the existing experiment environment:

```text
python -B -m experiments.gfg_head_only_hildreth_roundoff_smoke_v1.preflight
python -B -m experiments.gfg_head_only_hildreth_roundoff_smoke_v1.run
python -B -m experiments.gfg_head_only_hildreth_roundoff_smoke_v1.check
python -B -m experiments.gfg_head_only_hildreth_roundoff_smoke_v1.package
```

The scripts refuse to overwrite their existing local output root
`E:/gfg-evidence/head-only-hildreth-roundoff-smoke-v1`. Full saved head states are
local in `140_controls_hildreth_roundoff/AUDIT.pt`; the unchanged cache is reused
from `E:/gfg-evidence/head-only-coverage-cut-smoke-v1/CACHE.pt`. The public
`results/PUBLIC_RECORDS_MANIFEST.json` hashes every packaged machine record.
