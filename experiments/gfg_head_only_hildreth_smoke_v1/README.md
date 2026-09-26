# Head-only Hildreth projection smoke

One independent arm, `140_controls_hildreth`, based on
`9e0dc71c7692fa16ebeb81dbf73197aad93add4b`. Historical `A_140_current` is reused
as the comparison, never rerun. See [PROTOCOL.md](PROTOCOL.md) for the frozen
algorithm, stopping criterion, numerical precision and evidence contract.

## Result: the specified execution rule did not reduce skips

| Method | Old test macro | New test macro | Skips /210 |
|---|---:|---:|---:|
| A: 140/current two-pass (historical) |94.8214%|73.6588%|112|
| 140/Hildreth |95.5357%|61.0599%|149|

Hildreth all-test macro: **78.2978%** (historical A:84.2401%). Final task2
train16 accuracy: **63.8988%**. Final protected correctness: **1088/1120**.

| Requested execution diagnostic | Value |
|---|---:|
| Native-safe updates |3|
| Hildreth-triggered updates |207|
| Accepted safe, nonzero projected updates |58|
| Failed / skipped updates |149|
| Executed old protected correct-to-wrong |0|
| Active constraints on triggered steps, min/median/max |1 /21 /26|
| Sweeps for accepted projections, min/median/max |1 /1 /4|
| Sweeps for every failed projection |100|
| Total sweeps |14,987|
| Mean attempted projected/native norm ratio |0.914029|
| Mean attempted cosine |0.923985|
| Mean executed/native norm ratio on triggered steps |0.261029|
| Mean executed cosine, nonzero updates only |0.926784|
| Projection-call wall, including checks/bookkeeping |17.4025 s|
| Training-loop wall |39.3388 s|

### Where this run failed

This fixed rollout did NOT convert more scheduled steps into safe execution than
A, and did NOT restore new learning. It is not a positive accuracy result.
There is, however, a specific numerical distinction from the old cutter:

- For all 149 failures, the final FP64 affine minimum was within approximately
  1.2e-15 of the specified 1e-6 boundary. Its observed range was
  9.999999988075103e-7 to1.0000000003618226e-6. Small negative residuals remain
  in some constraints; no claim of exact real-arithmetic feasibility is made.
- After casting the head to FP32 and using the actual native affine forward,
  the failed endpoint minimum ranged from -4.76837158203125e-7 to
  9.5367431640625e-7, below the fixed 1e-6 acceptance boundary.
- In 133 of those failures, all eligible controls were still strictly correct
  (positive gaps), but did not meet 1e-6. The other16 had a zero or negative gap.
  Historical A accepted positive gaps, whereas triggered Hildreth calls in this
  protocol explicitly required at least1e-6. These acceptance rules differ.

The record therefore exposes a finite-precision boundary-acceptance limitation
of this implementation/protocol. It does NOT establish that no safe nonzero
update exists, that Hildreth is unsuitable in principle, or that the protection
bank/predictor must be changed. Replaying the 133 candidates under a different
acceptance rule would produce another trajectory; these counts cannot predict
that trajectory's final accuracy. No such rerun or rule change was performed.
The 58 accepted projected updates were all nonzero, but they are not matched
counterfactual recoveries of particular skipped steps along historical A.

Integrity **PASS**:34 preflight tests;210 independently replayed steps;14,987
independently reconstructed sweeps, including primal/dual hashes;1,008,000 endpoint
margin values; all actual head states and test/train metrics checked. Initial
head/moments and first native candidate match A. All old source/evidence identities
are unchanged. One formal run, no process failures/restarts, no parameter search.

## Fixed experiment

FewRel task2; seed 2021; LR 5e-5; batch 16; 210 scheduled updates; frozen/eval
BERT; existing linear head; original task1 checkpoint and AdamW moments. Same
current batches and the exact same cached protection features as A: all 140
validation examples per old relation, totaling 1,120 old controls. No replay,
Router, scalar shrink, new model, extra seed, parameter search, or result-driven
rerun. Only a new experiment directory is added; no original evidence is edited.

The native-safe path still uses receiving correctness and commits safe native
updates byte-for-byte. Triggered calls use Hildreth dual coordinate updates with
persistent nonnegative coefficients, permitting coefficient decreases. New pairs
are added after whole-bank checks. Maximum 100 sweeps; boundary exactly 1e-6.
Moments advance once per scheduled update even when parameter execution is skipped.

FP64 sparse two-row arithmetic avoids materializing full parameter-space
constraint vectors. The final actual FP32 head must pass all protected pair
constraints. No hidden epsilon inflation, rounding repair, relaxation, fallback,
extra sweeps or external solver. Finite-precision endpoint failures remain failures.
The iterate is stopped when first safe: this does NOT establish that the global
minimum-distance projection has been reached. Sweep-cap failure does NOT establish
infeasibility. Replacing the old cutter also changes the iteration budget and
projected endpoint criterion; this is not an isolated comparison of dual updates
against an otherwise identical 100-pass sequential implementation.

## Records and reproduction

The existing local FewRel/BERT environment and historical evidence are required.
Run from the repository root, only with an unused output directory:

```powershell
python -B -m experiments.gfg_head_only_hildreth_smoke_v1.preflight
python -B -m experiments.gfg_head_only_hildreth_smoke_v1.run
python -B -m experiments.gfg_head_only_hildreth_smoke_v1.check
python -B -m experiments.gfg_head_only_hildreth_smoke_v1.package
```

Outputs use exclusive creation, with no automatic retries. The sole training
result and its hash are fixed before any test inference. Final training accuracy
and test accuracy are read-only evaluations after training, not selection criteria.

`results/RESULTS.json` contains the A comparison and all requested aggregate
statistics. `records/STEPS.jsonl.gz` records every step, active pair IDs, final
dual values, sweep counts, full endpoint margins, primal/dual hashes per sweep,
and attempted versus executed geometry. `results/CHECKS.json` reports independent
reconstruction of every sweep without calling the production solver.

Original data, cached features and model/head snapshots are not uploaded. Local
snapshots reside at `E:/gfg-evidence/head-only-hildreth-smoke-v1`; the unchanged
feature cache is referenced from `E:/gfg-evidence/head-only-coverage-cut-smoke-v1`.
Public file identities are recorded in `PUBLIC_RECORDS_MANIFEST.json`.

Geometry means refer to triggered steps. Skips have zero actual executed norm
and undefined actual cosine; undefined cosines are excluded from their mean.
Attempted projection geometry is not misreported as executed geometry. Projection
wall includes endpoint checks and audit bookkeeping within the projection call;
training wall includes the training loop/logging but excludes setup, final tensor
serialization, evaluation and independent replay. These are measured timings,
not a controlled timing benchmark against the historical run.
