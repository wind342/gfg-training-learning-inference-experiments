# Functional update governor v1

Project refactor with two fixed 70M smoke runs. The frozen v2 directory and all
older experiment results are unchanged. No new training-data policy, RL rebalancer,
learned trigger, Flash JVP or multi-GPU certification is included.

## Architecture

- `instrument.py`: expensive response measurement only. Reuses v2 canonical
  ownership, stagewise jets, target routing and workspace planning. Produces the
  baseline target minima and candidate endpoint minima. Exact, scalarized and
  streaming backends all accept a variable number of candidates. Four coordinate
  terms remain four terms; they are NOT the candidate dimension.
- `governor.py`: pure selection from measured minima. Separates new formation,
  damage to already-correct targets, and frontier progress. Returns all scores.
- `scheduler.py`: rolling past loss/gradient-norm median + MAD trigger and maximum
  interval. Also exposes optional release interpolation, explicitly an open-loop
  policy rather than a forecast of future updates.
- `runtime.py`: thin optimizer execution. Measurement steps form the exact native
  displacement, restore receiving parameters, measure/select and commit scaled
  displacement. Adam moments follow the native proposal. Ordinary alpha=1 steps
  call `optimizer.step()` directly: no candidate copying, swap, delta formation or
  JVP. The persistent two buffers remain allocated for later measurements.
- `tests.py`, `smoke.py`: ordinary tests and the two requested runs/comparison.

The default backend is exact. Scalarized/streaming remain alternative arithmetic
backends, not universally bit-identical substitutes. Baseline logits use v2's
existing adapter-native readout path; no new model forward is run by the sentinel.

### Defaults agreed after review

| Setting | Value |
|---|---|
| Candidate alpha | .125, .25, .5, 1, 2 |
| Preservation penalty | **1**, configurable (not 2) |
| Release | **0**: current measured step only, next step native |
| Maximum interval | 20 steps; first step always measured |
| Sentinel history / minimum history | 20 / 5 observations |
| MAD multiplier | 3 |
| MAD floors | max(MAD, abs(median)*.001, 1e-12) |
| Primary score | formation_gain - penalty * preservation_damage |

Ties use gain, less damage, unformed median improvement, formed median margin,
then proximity to 1 (remaining tie smaller alpha). For an entirely formed bank,
consider only alpha <= 1 and prefer 1 when it has the minimum predicted damage.
No claim that penalty 1 newly targets capability: predicted-correct-count already
equals a receiving-state constant plus formation gain minus damage. Explicit
partitioning and the frontier tie breaker are the substantive changes.

Sentinel signals are diagnostics, not capability measurements or safety guarantees.
Current observations are excluded from their own rolling threshold. An anomaly
can retrigger on consecutive steps; there is currently no cooldown/hysteresis.

With optional `release_steps=4`, a measured alpha 2 yields
2, 1.75, 1.5, 1.25, 1; this path still constructs a proposal for non-native steps.
It was unit-tested as a schedule but was NOT used in the two 70M runs.

### Failure behavior and distributed scope

Nonfinite loss/norm aborts before optimizer execution. Nonfinite boundary output
aborts without committing candidate parameters. If the native proposal has already
advanced Adam moments, the runtime is marked failed and cannot continue; restore
both model and optimizer in a fresh runtime. There is no implicit rollback or
silently substituted alpha. Single-step diagnostic stage-event lists are cleared.

The low-level boundary routing is retained, but the new runtime explicitly rejects
world sizes > 1: cross-rank sentinel consensus is not validated. Actual checks used
one RTX 4060 Ti, not NCCL multi-GPU. No claim of validated sharded JVP performance.

## Tests

16 tests passed, including the three backends' five-alpha endpoints against
independently evaluated NumPy expressions, variable alpha counts, alpha-0 baseline,
synthetic amplify/shrink/formed/unformed selection, release/trigger behavior,
nonfinite failures, no JVP on ordinary steps, exact parameters and Adam state across
scaled then native updates, true tied parameter identity, and a real tiny CUDA
nanoGPT adapter run. A PyTorch JIT deprecation warning appeared in the dependency;
there was no test failure, OOM or run retry.

```powershell
E:/gfg-python/torch-cu130/Scripts/python.exe -m experiments.gfg_functional_update_governor_v1.tests
```

## Actual 70M smoke results

Seed 201, 71,405,312 parameters, unchanged 212 control/317 evaluation queries,
batch128, AdamW (.9,.98), weight decay0, clip1, FP32, constant LR. Two warmup
updates on a throwaway model, exact reinitialization, stop at first >=99%, cap200.
The evaluation set is disjoint from the control bank but not from training data.
Only the requested SAFE/NEAR runs were performed, without a hyperparameter sweep.

Wall time includes every update and evaluation, excludes initialization/warmup.
Old references are previous runs on this hardware, not contemporaneously rerun
timings. Treat timing differences as indicative project measurements.

| Case | Method | Steps to99 | Wall to99 |
|---|---|---:|---:|
| SAFE, LR 3.7688918072220455e-6 | New governor | **119** | **14.96 s** |
| SAFE | Historical v2 dynamic | 143 | 22.67 s |
| SAFE | Historical same-LR native | 121 | 7.65 s |
| SAFE | Historical tuned native (LR doubled) | 106 | 6.78 s |
| NEAR, LR 6.030226891555273e-5 | New governor | **191** | **19.95 s** |
| NEAR | Historical v2 dynamic | 158 | 24.99 s |
| NEAR | Historical same-LR native | 174 | 11.00 s |
| NEAR | Historical tuned native (LR /8) | 106 | 6.82 s |

Both new runs reached 99.0536%, without OOM or the 300-second loop cap. Peak
allocated CUDA memory was 1.7625 GiB in each run. These are two smoke cases, not
confirmation of broad optimizer superiority. SAFE improved over old v2 in steps;
NEAR worsened in steps. Both were faster than historical v2 but slower than native.

### Was measurement actually sparser?

| Metric | SAFE | NEAR |
|---|---:|---:|
| Actual instrument calls | 29 | 32 |
| Fraction of new run | 24.37% | 16.75% |
| Ordinary native-fast steps | 90 | 159 |
| Cadence5 calls over the SAME new horizon | 24 | 39 |
| Change against that equal-horizon count | **20.83% more** | **17.95% fewer** |
| Historical v2 calls until its OWN 99% | 29 | 32 |

Total calls to each method's own threshold did not decrease in either case.
The speed difference therefore cannot simply be attributed to fewer total JVPs.
Fast native steps and changed trajectory/timing contribute; there was no ablation
isolating their separate contributions.

### Actions actually chosen at measurement steps

| Alpha | SAFE decisions | NEAR decisions |
|---|---:|---:|
| .125 | 3 | 6 |
| .25 | 3 | 6 |
| .5 | 1 | 4 |
| 1 | 5 | 12 |
| 2 | 17 | 4 |

Full executed-step alpha histograms: SAFE {.125:3, .25:3, .5:1, 1:95, 2:17};
NEAR {.125:6, .25:6, .5:4, 1:171, 2:4}. No alpha persists after its measured step.
Every decision's predicted formation gain, damage, utility and trigger reasons
are in `results/safe-RESULT.json` and `results/near-RESULT.json`.

SAFE had 26 gradient-norm anomaly reasons. NEAR had 27 gradient-norm, 3 loss and
4 interval reasons (multiple reasons can share one call). Consecutive anomaly
triggers are an observed performance issue. The next isolated engineering target
is event de-duplication/rearming during a sustained excursion, without silently
claiming that no anomaly means no capability risk. No threshold was retuned after
seeing these two outcomes.

## Use and files

```python
engine = GovernedTraining(model, NanoGPTAdapter(model))
info = engine.update(optimizer, step, batch_inputs, batch_targets,
                     probe_inputs, probe_targets, rng_seed, learning_rate)
```

`step` starts at1 and increases consecutively. Pass an already constructed model
and optimizer; the engine does not choose a dataset, seed or base LR. Optional
governor, instrument, scheduler and release configuration are separate objects.
Each return exposes alpha, measurement flag/reasons, loss/norm, elapsed steps since
measurement and execution path; measurement steps also return complete scores.

Raw per-step logits are local only under
`E:/gfg-evidence/functional-update-governor-v1/{safe,near}/attempt-1/`.
`results/` contains compact run logs and historical comparisons. Accuracy and
first99 were recomputed from raw logits; raw hashes are in the run logs.
No original files or frozen identities were changed. Work is on its own branch;
no remote push was performed.
