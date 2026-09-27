# Banking77 task2: one first-sweep exact check per outer

New stopping-policy experiment, based on `90ca7497789b29041dd19dc3005dd67b4ce3904e` and accepted Cached Hildreth Stage1 arithmetic. No historical experiment changed. Both fixed task2 budgets pass the original smoke gates and independent replay. No test or tasks3–11 accessed.

After the first complete Hildreth/Dykstra sweep in each outer, materialize and check the true FP32 relation floors. Accept immediately if safe; otherwise retain solver state and continue to the original PASS/cap. Reuse the checked candidate if termination occurs on sweep 1. The 1e-8 PASS criterion, rho, reserve, selection, Jacobian, predictor and AdamW remain unchanged. FIRST_SWEEP_ACTUAL_SAFE is a separate status, not a claim of QP convergence.

The measured benefit is lower algorithm time. New-task train accuracy is slightly lower; these observations do not establish learning equivalence or held-out performance. This is a new trajectory, not byte-identical engineering acceleration.

| Budget | Policy | Sweeps | Outers | Exact endpoint calls | Control median (s) | Algorithm median (s) | Algorithm range (s) |
|---|---|---:|---:|---:|---:|---:|---|
| 210 | baseline | 48,686 | 307 | 518 | 8.9619 | 9.8756 | 8.5193–10.4177 |
| 210 | firstcheck | 17,058 | 272 | 636 | 7.4249 | 8.3397 | 7.2019–8.3743 |
| 630 | baseline | 64,285 | 618 | 1249 | 16.2808 | 18.9592 | 17.8546–19.8668 |
| 630 | firstcheck | 627 | 618 | 1250 | 13.1935 | 15.7449 | 15.0410–16.1295 |

Three predeclared paired repetitions, rotated order; every repeated trajectory and final model/optimizer state matched its own validation run. All three pairs favor firstcheck on total/control wall in both budgets, but the 210 first-pair margin is small. No statistical-significance claim. Algorithm time uses the inherited entry + native update + control scope; excludes per-update observer/logging, independent replay and final train metrics. In-controller hashes, checks and allocations remain included.

| Budget | Control reduction | Algorithm reduction | Sweep reduction | Changed executed heads vs baseline |
|---|---:|---:|---:|---:|
| 210 | 17.15% | 15.55% | 64.96% | 207/210 (first at step 4) |
| 630 | 18.96% | 16.95% | 99.02% | 616/630 (first at step 15) |

| Budget | Policy | Old train macro (%) | New train macro (%) | All train macro (%) | Current CE entry → final |
|---|---|---:|---:|---:|---|
| 210 | baseline | 51.4938 | 58.8442 | 55.1690 | 2.686831 → 2.139460 |
| 210 | firstcheck | 52.0038 | 58.3944 | 55.1991 | 2.686831 → 2.142943 |
| 630 | baseline | 66.8580 | 76.5460 | 71.7020 | 3.042068 → 1.483108 |
| 630 | firstcheck | 66.9411 | 76.2038 | 71.5724 | 3.042068 → 1.488439 |

Metrics cover arrived training examples excluding the control bank, over 14 seen classes. They are not test scores. The two budgets use their corresponding frozen task1 checkpoint; compare policies within a budget.

| Budget | Unsafe native | First checks | First-safe accepts | Failed first checks | Ordinary safe accepts | Unsafe outer continuations | INNER_LIMIT | Rollback | Floor violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 210 | 208 | 272 | 119 | 153 | 89 | 64 | 0 | 0 | 0 |
| 630 | 618 | 618 | 617 | 1 | 1 | 0 | 0 | 0 | 0 |

In 210 steps, 153 failed first checks require continuing the original solver and doing its terminal check. Total endpoint calls therefore rise 518 → 636 even though outers fall 307 → 272. In 630 steps, 617/618 unsafe native updates accept after one sweep, and the remaining solve takes 10 sweeps; total endpoint calls rise only 1249 → 1250. The actual new trajectory differs from the prior fixed-state diagnostic, as expected.

| Budget | Policy | Norm-ratio median | Cosine median | Correct→wrong | Wrong→correct | Jacobian pairs |
|---|---|---:|---:|---:|---:|---:|
| 210 | baseline | 0.884034 | 0.882970 | 5 | 5 | 9002 |
| 210 | firstcheck | 0.888175 | 0.886621 | 5 | 5 | 7247 |
| 630 | baseline | 0.909822 | 0.909885 | 0 | 0 | 11698 |
| 630 | firstcheck | 0.910060 | 0.909463 | 0 | 0 | 10355 |

All policies/budgets have successful repair 100%, old CE backward examples/FLOPs 0, predictor false-safe 0 and all fixed smoke gates PASS. Correctness-transition counts refer to old controls across actual committed updates.

| Budget | Policy | Solver core | Parameter CPU conversion | Reconstruction + candidate diagnostics | Materialization | Endpoint | Predictor | Jacobian/Gram/guard |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 210 | baseline | 2.8636 | 0.7093 | 0.6664 | 0.9378 | 0.3560 | 1.0272 | 0.5143 |
| 210 | firstcheck | 0.8569 | 0.6743 | 0.8138 | 1.2001 | 0.4334 | 1.0124 | 0.4142 |
| 630 | baseline | 2.8580 | 2.0308 | 1.0602 | 1.7132 | 0.7144 | 2.9914 | 0.8218 |
| 630 | firstcheck | 0.0519 | 1.9301 | 1.0681 | 1.5127 | 0.6283 | 2.8270 | 0.8127 |

Profile values are seconds, medians of three runs; medians are not additive and these regions do not cover every control operation. Baseline includes candidate diagnostics inside reconstruction; firstcheck separates them. The first-check inclusive timer overlaps reconstruction/materialization/endpoint and must not be added again. At 630, solver core falls to 0.052s, while predictor takes 2.827s and parameter conversion + materialization takes 3.443s. These measured regions now matter more than remaining sweeps. No further optimization was mixed into this experiment.

| Budget | Policy | Endpoint FLOPs (G) | Cutter FLOPs (G) | Predictor FLOPs (G) | Old CE backward |
|---|---|---:|---:|---:|---:|
| 210 | baseline | 57.3715 | 13.4164 | 236.8261 | 0 |
| 210 | firstcheck | 70.4407 | 12.1847 | 236.8261 | 0 |
| 630 | baseline | 138.3340 | 16.6882 | 710.4784 | 0 |
| 630 | firstcheck | 138.4448 | 14.8843 | 710.4784 | 0 |

FLOPs are the inherited dense lower bounds and analytic estimates, including known removed duplicate Gram GEMVs; they are not hardware instruction counts. Lower wall time does not imply lower total estimated FLOPs, especially with additional endpoint calls in 210.

Independent audit replays both baseline and firstcheck from task1: every native AdamW proposal, moment tensor/counter, predictor, selection, independent Jacobian/solver, first-check candidate and failed-check continuation, outer endpoint, commit, and final model/optimizer state. Baseline additionally matches sealed Stage1 records byte for byte. New-policy audit checks match its new trajectory; they do not claim equality with baseline. The numerical suite verifies 40 solver cases, failed-callback continuity, first-complete-sweep identity and independent reconstruction/endpoint. Sources, frozen BERT, cache and input hashes are verified before/after.

`CHECKS.json` retains its at-generation `independent_audit_status: PENDING` because evidence is create-only. Final audit outcomes are in separate `AUDIT.json` and combined `SUMMARY.json`; both are PASS.

Code and compact results are in this directory. Full create-only evidence, checkpoints and compressed trajectories: `E:/gfg-evidence/banking77-first-sweep-check-task2-v1`. `ARTIFACT_MANIFEST.json` records every external artifact size/hash. Historical evidence remains unchanged.

Reproduction requires the frozen local inputs listed in `results/FREEZE.json`. Use a fresh, nonexistent evidence root through `GFG_FIRST_SWEEP_ROOT`, then invoke these modules with `python -B -m experiments.gfg_banking77_first_sweep_check_task2_v1.<module>` in order: `preflight`, `numerics`, `run baseline`, `compare`, `run firstcheck`, `audit firstcheck`, `audit baseline`, `benchmark`. The `report` command is the initial create-only publication step and refuses to overwrite the already committed compact results. Inspect the new evidence root for reproduction output.

Stopped after task2. No full experiment, tolerance/rho sweep, alternate interval, scalar shrink, solver replacement, test evaluation or automatic promotion.
