# Banking77 task2: defer nonessential evidence outside training timers

Implementation and verification complete. User requested immediate completion during repeated benchmarking; the remaining repetitions were stopped. This report uses the FIRST complete paired block, all four arms and both budgets. It is one timing observation per arm, not a three-repeat median. The original frozen protocol and partial repeated-run evidence remain intact; see SCOPE_ADJUSTMENT.json.

Unchanged: real FP32 floor checks, both stopping policies, selection, solver, reserve, Jacobian, rho rule, native AdamW and moments. Moved after the controller timer: SHA256, detailed records/list formatting, layer/cumulative diagnostic norms and safe-candidate repair diagnostics. Unsafe repair calculations that determine rho remain timed. Minimal trace retention/dual-copy cost remains timed. No buffer/numerical optimization added.

Both clean policies independently replayed for 210/630 steps. Every per-step record (including diagnostic scalars), executed/native head, optimizer moment, endpoint, selection and final checkpoint matches its own sealed policy. All smoke gates PASS, floor violations/rollback/old CE backward 0. No test or later tasks accessed.

| Budget | Policy | Original evidence placement: control / algorithm (s) | Deferred evidence: control / algorithm (s) | Deferred evidence work (s, outside) |
|---|---|---|---|---:|
| 210 | baseline | 7.4808 / 8.2671 | 7.1902 / 8.0954 | 1.4438 |
| 210 | firstcheck | 7.4396 / 8.3634 | 5.9550 / 6.9328 | 1.4540 |
| 630 | baseline | 16.2244 / 18.9112 | 12.8023 / 15.5270 | 2.9900 |
| 630 | firstcheck | 13.7220 / 16.3873 | 10.6876 / 13.3929 | 3.5670 |

Evidence was moved, not deleted. The algorithm timer excludes deferred observation, audit and final evaluation; it is not experiment-process elapsed time. The loop below includes both training and all per-step evidence. Do not interpret the placement change as eliminating this work or claim a stable speedup from one sample.

| Budget | Policy | Recorded full loop (s) | Deferred full loop (s) |
|---|---|---:|---:|
| 210 | baseline | 11.1213 | 13.8002 |
| 210 | firstcheck | 12.5436 | 12.6806 |
| 630 | baseline | 31.2903 | 31.0303 |
| 630 | firstcheck | 28.7584 | 29.6093 |

Additive ledger for the SAME completed run, clean firstcheck, seconds. Nested child time is removed from its parent; all synchronization is included. The parent-exclusive control line contains counting, commit, dispatch, small trace retention and timer overhead. Manual geometry subregion counters are not added again.

| Budget | Part | Seconds |
|---|---|---:|
| 210 | entry_wall | 0.002681 |
| 210 | common_wall | 0.975089 |
| 210 | control: control_total | 0.277047 |
| 210 | control: predictor | 1.084260 |
| 210 | control: endpoint | 0.447694 |
| 210 | control: parameter_geometry | 0.706695 |
| 210 | control: selection | 0.050912 |
| 210 | control: jacobian_gram_guard | 0.436792 |
| 210 | control: hildreth_ball | 0.876047 |
| 210 | control: first_check_bookkeeping | 0.056272 |
| 210 | control: reconstruction | 0.713655 |
| 210 | control: materialization | 1.282689 |
| 210 | control: unsafe_repair_rule | 0.022967 |
| 210 | **TOTAL algorithm** | **6.932799** |
| 630 | entry_wall | 0.003825 |
| 630 | common_wall | 2.701422 |
| 630 | control: control_total | 0.615552 |
| 630 | control: predictor | 3.034828 |
| 630 | control: endpoint | 0.790423 |
| 630 | control: parameter_geometry | 2.249322 |
| 630 | control: selection | 0.109104 |
| 630 | control: jacobian_gram_guard | 0.839289 |
| 630 | control: hildreth_ball | 0.053644 |
| 630 | control: first_check_bookkeeping | 0.127756 |
| 630 | control: reconstruction | 0.991922 |
| 630 | control: materialization | 1.875791 |
| 630 | **TOTAL algorithm** | **13.392878** |

Full source, protocol and compact checks are here; external raw checkpoints/logs remain at `E:/gfg-evidence/banking77-clean-timing-task2-v1`, hashed in ARTIFACT_MANIFEST.json. All old artifacts are unchanged. CHECKS at-generation audit fields remain PENDING because files are immutable; separate AUDIT.json files contain final PASS.

The original benchmark/report scripts preserve the initially frozen extended plan. That plan was curtailed by the user; benchmark/SUMMARY.json and the planned three-repeat report were not produced. FINAL_VALIDATION.json is the authoritative final summary for this request. No additional runs are required.
