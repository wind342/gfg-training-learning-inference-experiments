# Cadence experiment: results

## Practical answer

Refreshing every five steps substantially reduces controller overhead and has a
positive signal for **first observed 99% accuracy per run**. It does not dominate
native AdamW at every threshold or every starting state. Refreshing every two
steps is less attractive in this experiment. No old frozen verdict is revised.

The twelve historical starts include **five already at >=99%** and **six already
at >=95%**. They contribute zero first-passage cost equally in every arm, not new
training successes. All twelve remain in the predeclared summaries. At 99%,
native newly reaches the target in 5/7 initially-below-threshold starts; each
controller reaches it in 6/7; fixed .25 reaches it in 4/7.

## First observed thresholds

Means below include all twelve runs. An unreached target is charged the full
200-step budget and its measured cost: these are **restricted first-passage
costs**, NOT average completion times among successful runs. Accuracy is observed
only every ten steps. Already-qualified states cost zero. Core seconds include
native training, selection and amplitude application.

| Arm | 95% reached | 95% mean steps | 95% core s | 99% reached | 99% restricted steps | 99% restricted core s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native AdamW | 12/12 | 17.50 | 0.4295 | 10/12 | 60.83 | 1.6507 |
| Fixed .25 | 12/12 | 13.33 | 0.3830 | 9/12 | 90.00 | 2.6484 |
| Controller every 1 | 12/12 | 13.33 | 1.0752 | 11/12 | 41.67 | 3.3241 |
| Controller every 2 | 12/12 | 20.00 | 1.1426 | 11/12 | 45.83 | 2.6164 |
| Controller every 5 | 12/12 | 15.00 | 0.5875 | 11/12 | 36.67 | 1.4164 |

For first observed 99%, every-five core cost is 57.39% lower than every-step
control and 14.19% lower than native in this measured suite. Including batch/RNG
setup and receiving-state snapshots, corresponding costs are 1.6797, 3.6176 and
2.0797 seconds: every-five is 19.23% below native on that broader cost measure.
For the seven initially-below-99% starts, multiply the all-run means by 12/7;
the percentages do not change. Do not silently drop the unreached start(s).

These are not unanimous time wins. Against native, every-five has lower/equal/
higher restricted core time in 3/5/4 runs (the five ties started qualified).
Including setup gives 4/5/3. Against every-step control, core time is lower in
all seven informative runs. Every-five has fewer/equal/more restricted steps
than native in 5/6/1 runs. At 95%, fixed .25 is the fastest observed arm and
every-five is still slower than native.

## Pooled accuracy and continued training

Pooling all targets at the same step answers a DIFFERENT question from each
run's first crossing. Its 99% crossings are native=110, fixed=130, every-one=50,
every-two=120, every-five=130. Thus the every-step controller's earlier pooled
99% crossing is not preserved by lowering frequency. These curves and each
run's full list of subsequent threshold drops are retained in RESULTS.json.

| Arm | Runs ever dropping below 99% after first crossing | Runs >=99% at final evaluation | Final pooled accuracy |
| --- | ---: | ---: | ---: |
| Native | 7 | 9 | 99.4497% |
| Fixed .25 | 4 | 8 | 97.5629% |
| Every 1 | 5 | 8 | 99.3711% |
| Every 2 | 7 | 9 | 99.0959% |
| Every 5 | 7 | 9 | 99.1745% |

First-hit efficiency is not sustained-accuracy efficiency. Neither the final
accuracy nor the full-horizon average is used to replace the requested first-hit
comparison. This experiment uses the same labeled targets for control and
evaluation, not a new independent generalization test.

## Measured full-horizon costs (all twelve runs, 200 steps each)

| Arm | Selection calls | Selection seconds | Core seconds | Operational seconds | Wall seconds including evidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 0 | 0 | 64.100 | 80.790 | 126.275 |
| Fixed .25 | 0 | 0 | 70.917 | 87.928 | 128.246 |
| Every 1 | 2400 | 121.230 | 191.885 | 208.948 | 308.792 |
| Every 2 | 1200 | 61.510 | 136.880 | 154.061 | 225.805 |
| Every 5 | 480 | 24.126 | 92.153 | 109.139 | 163.560 |

This table documents overhead; it is not the primary time-to-target comparison.
Operational = core + batch/RNG preparation + receiving snapshot. Evaluation and
hashing/persistence are excluded from operational and included in wall. Runtime
construction, synthetic warmup and independent checking are not training costs.
Timings are single executions on RTX 4060 Ti with rotated arm order, not precise
repeatability bounds or an energy measurement.

## Reproducibility and protected sources

The raw replay bundle has 16,200 files, 673,097,320 bytes (641.92 MiB).
ARTIFACT_MANIFEST.json identifies every file. Frozen code/protocol identities
are in FREEZE_SEAL.json. The independent checker is required to replay all
12,000 steps, recompute all 4,080 coordinate sets and match all 7,200 original
control states exactly; its machine-readable result is INDEPENDENT_CHECK.json.

**Verification completed: PASS.** All 60 branches, 12,000 native steps, 4,080
coordinate recomputations and 7,200 original-control state comparisons passed.
The separately implemented post-execution summary audit also passed all 16,200
file hashes, 12,000 chain records and 1,260 target-boundary evaluations. See
INDEPENDENT_CHECK.json and SUMMARY_AUDIT.json. No failed branch was excluded.

Only the new experiment directory is added. Original source implementations,
artifacts, manifests, frozen tag and main branch are unchanged. The new ledger
is compact and replayable, not a claim to store every intermediate raw tensor
or to rebuild the entire original GFG.
