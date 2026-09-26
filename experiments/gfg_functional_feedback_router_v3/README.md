# Router V3: sample-count-normalized feedback dose

Base: `cf6c0f30d0e35173771ab2fc52fa2474b1d940ee` (Router V2).
Only training microbatch weights change. Each actual example has coefficient
`1 / (n_current + n_old)`, implemented as sequential backward calls on
`n_current / total * current_mean` and `n_old / total * old_mean`.
This does not imply equal gradient magnitudes across examples.

`dose.py` preserves seeds, training dropout, clipping and optimizer semantics.
`run.py` injects this accumulator and logs the effective coefficients without
editing inherited files. Router states, coalition rules, memory, predictor,
emergency gate, thresholds and q10 are unchanged. Closed-loop model states and
therefore actual later coalitions can change; this is not a forced replay of
the V2 coalition sequence.

## Fixed 60-step result

One completed run, no failed attempts, reruns, sweep or three-task experiment.
All methods start from the same task1 model/optimizer checkpoint at 97.1429%
old accuracy and 0% new accuracy and use the same 960 current-training examples.
Other methods below are historical results, not newly rerun baselines.

| Method | Old accuracy | New accuracy | All accuracy | Old exposures | Wall |
|---|---:|---:|---:|---:|---:|
| New-only native AdamW | 74.1071% | 84.5536% | 79.3304% | 0 | 70.803 s |
| Fixed replay native | 93.3929% | 79.0179% | 86.2054% | 960 | 85.873 s |
| Router V2 | 96.0714% | 69.6429% | 82.8571% | 282 | 118.433 s |
| Router V3 | 92.0536% | 80.9821% | 86.5179% | 318 | 130.782 s |

Against V2, new accuracy increases by 11.3393 percentage points, old accuracy
decreases by 4.0179 points, and all accuracy increases by 3.6607 points.
Thus this change restores more new formation, but does NOT preserve V2's old
retention advantage. It shifts the retention/formation trade-off.

Against fixed replay, V3 uses 66.875% fewer old-example occurrences, with new
accuracy 1.9643 points higher and old accuracy 1.3393 points lower. All accuracy
is 0.3125 points higher in this one fixed fork; this is not evidence of a
reliable general superiority. Wall time is longer, not shorter. Capability
outcomes are reported separately from exposure and runtime savings.

| Actual old batch size | Effective old-loss weight | V3 steps | V2 steps |
|---|---:|---:|---:|
| 0 | 0 | 3 | 3 |
| 4 | 4/20 = 20% | 17 | 30 |
| 6 | 6/22 = 27.2727% | 35 | 27 |
| 8 | 8/24 = 33.3333% | 5 | 0 |

The weight column applies to V3. In V2 every nonempty old batch had weight
50%, so its effective-weight histogram is 0:3, 50%:57. Fixed replay uses
16 old examples and 50% weight on all 60 steps; new-only uses zero on all steps.
V3's cumulative old/current coefficients are 14.612121/45.387879, versus
28.5/31.5 in V2. These are objective coefficients, not gradient-mass estimates.

Modes remain 44 RECOVERY, 13 BALANCE and 3 FORMATION, with one release and two
activations; feedback is active on 57/60 steps. Actual old exposure rises from
282 to 318 because the changed learning trajectory changes later selections,
not because coalition rules were modified.

## Gate, tests and evidence boundaries

V3 makes four full measurements: three NATIVE_SAFE and one EMERGENCY_SHRINK.
Alpha is 1 on 59 steps and 0.5 on one; 56 steps are NOT_MEASURED by the full
gate. Selected endpoint agreement is 640/640, including five actual boundary
transitions, all predicted. This is a selected-endpoint audit, not a claim
about unmeasured updates or general prediction accuracy. Peak allocation is
6,167,233,024 bytes. Wall includes runtime monitoring/audit and evaluations,
excluding initial loading, and is a single-run measurement.

Ten new dose tests passed before training. Final combined regression testing
passed all 87 tests. Sequential gradients match concatenated mean gradients
for 16+0, 16+4, 16+6 and 16+16 examples; tests also cover optimizer moments,
exact V2 equal-batch/native regressions, and a tiny real BERT. Dropout is disabled
only for the mathematical concatenation tests; training dropout is unchanged.

Checker PASS covers recorded source/checkpoint/data identities, inherited V1
and V2 source hashes, routing/cursors, sample roles, dose coefficients, gate
decisions, and accuracy recomputed from saved predictions. It consumes recorded
micro histories and does not independently rerun native inference. Historical
references retain their original result hashes and initial checkpoint identity.
The inherited control/test duplicate at test index 7561 remains disclosed;
excluding it gives new accuracy 80.9809%, without changing the conclusion.
Training memory excludes encoded control/test duplicates.

Compact results, record hashes and regression counts are in `results/CHECKS.json`
and `results/TESTS.json`. Raw create-only artifacts remain at
`E:/gfg-evidence/functional-feedback-router-v3/`; no model, checkpoint or dataset
is committed. Existing source files, evidence, frozen tag and main are unchanged.

```text
python -m experiments.gfg_functional_feedback_router_v3.tests
python -m experiments.gfg_functional_feedback_router_v3.run
python -m experiments.gfg_functional_feedback_router_v3.check
```

The runner/checker use the pinned local V2/V1 environment and reject overwriting
completed output. This is a scoped experiment, not an independent package.
