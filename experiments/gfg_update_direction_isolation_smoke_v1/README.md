# Update-direction isolation smoke (task2)

Base: `f35b595c4e4582bd3a52e938b29e2cf253446d25`.
Independent new experiment; no prior experiment, Router V4 implementation,
historical result, main or frozen tag changed. One seed, no tuning or reruns.
See [PROTOCOL.md](PROTOCOL.md) for the pre-training freeze.

## Outcome and interpretation

All three arms completed; preflight15/15 tests, record checker and final native
checkpoint replay PASS. **The specified controller did not preserve old abilities
over the epoch or demonstrate a meaningful endpoint advantage over scalar shrink.**
However, its main preemptive direction-isolation question was not cleanly isolated:
all11 projection triggers (15 group-occurrences) were already unsafe under the
inherited q10 rule in the receiving state, before the proposed update. There were
zero newly group-threatened events relative to a receiving state that satisfied
the full safety rule. This is a post-run diagnostic, not a change of protocol.

Local directional evidence is more specific:
- There were15 gradient constraints and14 actual negative-dot cuts, over11 updates.
- Projected/native norm ratio ranged0.9985854369–0.9999826790.
- Native and full-projected proposals had identical predicted new-control correct
  counts at all11 projection sites. This is prediction evidence, not native
  execution of the unchosen full proposals.
- Four old probe-update predicted signs changed from wrong to correct after
  projection, with zero reverse changes at alpha1.
- Nevertheless, no full projected proposal satisfied **all group safety rules**;
  all11 sites were labeled NO_SAFE_PROJECTED_ALPHA and executed at0.125.
- Thus a local conflict can be removed while a pre-existing margin deficit remains.
  The result does not show that changing direction can never protect old ability.

## Fixed experiment and final capability

Seed2021, original shared task1 checkpoint, eight old/eight new relations,
batch16, LR5e-6, original AdamW betas/clipping/dropout and current batches.
Task2 only,210 updates/arm, no old-example replay loss. Scalar/projection have
the same21 detection offsets:1,11,...,201; all other updates are native alpha1.
Test is used only at the final endpoint.

| Arm | Old macro | New macro | All macro | Training wall (s) | Total wall (s) | Peak allocated / reserved (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| Native AdamW | 5.6250% | 92.5893% | 49.1071% | 60.827 | 76.299 | 4.929 / 5.135 |
| Scalar | 6.5179% | 92.3214% | 49.4196% | 317.359 | 332.659 | 5.744 / 6.016 |
| Single-pass projection | 6.6964% | 92.4107% | 49.5536% | 459.463 | 475.066 | 5.744 / 6.016 |

Projection minus scalar: 0.1786
percentage points old, 0.0893
new, 0.1339 all. These small
single-seed differences do not establish a general method advantage. Old capability
remains severely degraded in all arms; new capability forms to about92%.

| Relation | Native | Scalar | Projection |
|---|---:|---:|---:|
| 0 | 0.0000% | 0.0000% | 0.0000% |
| 1 | 6.4286% | 6.4286% | 6.4286% |
| 2 | 2.8571% | 2.8571% | 2.8571% |
| 3 | 0.0000% | 0.0000% | 0.0000% |
| 4 | 30.0000% | 34.2857% | 35.7143% |
| 5 | 4.2857% | 5.0000% | 5.0000% |
| 6 | 0.0000% | 0.0000% | 0.0000% |
| 7 | 1.4286% | 3.5714% | 3.5714% |
| 8 | 100.0000% | 100.0000% | 100.0000% |
| 9 | 67.1429% | 65.0000% | 65.7143% |
| 10 | 100.0000% | 100.0000% | 100.0000% |
| 11 | 99.2857% | 99.2857% | 99.2857% |
| 12 | 100.0000% | 100.0000% | 100.0000% |
| 13 | 97.1429% | 97.1429% | 97.1429% |
| 14 | 84.2857% | 84.2857% | 84.2857% |
| 15 | 92.8571% | 92.8571% | 92.8571% |

Primary metrics use relation-macro accuracy on the authorized deduplicated test
set (2,239 task2 examples). All2,240 original-test predictions and secondary
metrics are retained. Original source data is unchanged; control50124->50130
and the seven previously authorized full80-relation test exclusions are reused.
Only the applicable exclusion affects task2.

## Safety and predictor results

| Arm | Detection sites | Predictor calls | Native-threat sites | Projection sites | Full projected safe | Shrink sites | No-safe sites | Chosen endpoint boundary agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native AdamW | 0 | 0 | 0 | 0 | 0 | 0 | 0 | Not measured |
| Scalar | 21 | 21 | 10 | 0 | 0 | 10 | 10 | 3360/3360 |
| Single-pass projection | 21 | 32 | 11 | 11 | 0 | 11 | 11 | 3360/3360 |

Scalar has6 actual boundary transitions, all predicted correctly; projection
has4, all correct. These are160 controls across21 **executed measured updates**,
not an accuracy claim about every unmeasured step or unexecuted alpha1 proposal.
No action is reselected using the post-commit audit.

The control probes are **optimization-control data**. Their labels and margin
gradients inform projection. No replay loss or old training memory is used, but
this is not an old-data-free method, and these controls are not an independent
generalization test.

## Why zero full-safe projections does not settle the preemptive question

At offset11, old group2 was still PROTECTED but already in margin erosion:

| Quantity | Value |
|---|---:|
| Required q10 (existing anchor ×0.5) |0.7245355964|
| Receiving q10, before this update |0.6604279637|
| Native alpha1 predicted q10 |0.6098731499|
| Projected alpha1 predicted q10 |0.6520962277|
| Receiving worst margin |0.0447926521|
| Native alpha1 predicted worst margin |-0.0034663113|
| Projected alpha1 predicted worst margin |0.0450185586|
| Actual executed projected ×0.125 worst margin |0.0447943211|

The projection removed the selected negative local derivative and prevented that
probe's **predicted** alpha1 crossing, preserving2 predicted new-probe gains.
But even the receiving q10 was below the required floor. Making the local
derivative approximately zero does not repair a historical deficit of0.0641.
The frozen gate therefore still chose0.125; the executed proposal gained0 new
correct controls at that site. The full projected alpha1 was not executed, so
its safety is not claimed as a native causal outcome.

Every triggered group-occurrence in this run was already unsafe by the q10
condition. The post-run decomposition is in `results/POSTCHECK.json`; this
limitation was not used to change the rule, detection schedule or rerun an arm.

## All projection sites

"New gain" is predicted correct controls minus receiving correct controls.
Counts refer to the current80 new controls, not test. Negative/zero native gain
does not define a meaningful retention ratio; the machine record uses null.

| Offset | Threatened old groups | Actual cuts | Norm ratio | Cosine | Removed norm fraction | New gain native/projected/executed | Executed alpha |
|---|---|---:|---:|---:|---:|---|---:|
| 11 | 2 | 1/1 | 0.999783237 | 0.999783237 | 0.020820178 | 2/2/0 | 0.125 |
| 21 | 2 | 1/1 | 0.999976519 | 0.999976519 | 0.006852834 | 3/3/0 | 0.125 |
| 31 | 2,3 | 2/2 | 0.999479674 | 0.999480299 | 0.032235575 | 1/1/0 | 0.125 |
| 41 | 2,5 | 1/2 | 0.999480542 | 0.999480542 | 0.032228037 | 0/0/0 | 0.125 |
| 51 | 2 | 1/1 | 0.999810193 | 0.999810193 | 0.019482758 | 1/1/0 | 0.125 |
| 71 | 1,7 | 2/2 | 0.998585437 | 0.998525897 | 0.054277399 | 0/0/0 | 0.125 |
| 81 | 1,7 | 2/2 | 0.999339692 | 0.999311062 | 0.037113367 | 0/0/0 | 0.125 |
| 91 | 6 | 1/1 | 0.999888301 | 0.999888301 | 0.014946102 | 0/0/0 | 0.125 |
| 101 | 6 | 1/1 | 0.999980424 | 0.999980424 | 0.006257125 | 0/0/0 | 0.125 |
| 111 | 4 | 1/1 | 0.999780362 | 0.999780362 | 0.020957741 | 0/0/0 | 0.125 |
| 121 | 4 | 1/1 | 0.999982679 | 0.999982679 | 0.005885721 | -1/-1/-1 | 0.125 |

Only2/15 selected gradient probes were current-correct probes predicted to cross
under the native proposal. The current-worst-probe heuristic was retained exactly,
not replaced with a hindsight selection. Its gradient is a local margin direction,
not a guaranteed gradient of group capability or q10. Later sequential cuts may
reintroduce earlier conflicts; all receiving protected old groups are rechecked
after the single pass.

Native/projected/executed margins for every old and new control, per-group q10,
capability and new-group formation counts are retained in raw records and
POSTCHECK. No large historical gradient library is retained.

## Unguarded intervals and protection-set attrition

The shared task1 checkpoint has groups0/4 DEGRADED, six PROTECTED. Each measurement
refreshes actual receiving state using the original sentinel, so a previously
protected group that has already degraded can leave the eligible set. Previously
degraded groups can rearm under the same original rules. No new recovery policy
was added.

| Arm | Old correct->wrong across inter-detection intervals | Old correct->wrong in measured commits | Detection sites with zero protected old groups |
|---|---:|---:|---:|
| Scalar | 74 | 4 | 9 |
| Single-pass projection | 77 | 1 | 8 |

These are observed probe-transition occurrences, not unique probes; an interval
contains nine ordinary updates and may hide additional loss/recovery inside it.
The initial8->16 class-scope change is labeled separately, not falsely assigned
to an intervening optimizer update. The final nine ordinary updates after offset201
are not covered by another control audit; their combined effect appears in the
final test endpoint, not in these interval counts. Native has no control audits;
missing native interval statistics are not evidence of no damage.

All old relations are evaluated at the final test regardless of controller state.
Sparse checks plus a shrinking protection set left much of the epoch unprotected.
Thus this experiment is not evidence that all possible directional protection
would fail, nor a test of continuous protection on every update.

## Execution implementation and resource accounting

The native AdamW proposal is formed in each arm's own receiving state. After the
trajectories diverge, proposals across arms are not assumed numerically identical.
Moments advance once on the current-task gradient and are not altered by scaling
or projection. Original current gradients are cleared before margin differentiation;
control gradients are obtained with autograd.grad, never passed into AdamW.

The new projected commit explicitly applies receiving parameters+alpha*new_delta.
It does not call the old alpha1 shortcut, which would copy a stale native candidate.
The no-threat path uses the native candidate unchanged. Tests include flattened
toy equivalence, nonconflict identity, deterministic one-pass order, unchanged
moments and real small-BERT re-prediction after replacing delta. Tensorwise dot
products use float64 accumulation; eps1e-12 is fixed, not tuned.

| Timing component (seconds) | Native | Scalar | Projection |
|---|---:|---:|---:|
| training_gradient | 58.013 | 58.366 | 58.549 |
| proposal | 0.000 | 1.943 | 2.003 |
| first_prediction | 0.000 | 226.231 | 226.915 |
| projection_gradient | 0.000 | 0.000 | 1.331 |
| projection_pass | 0.000 | 0.000 | 1.135 |
| second_prediction | 0.000 | 0.000 | 118.981 |
| optimizer_commit | 2.784 | 2.664 | 2.628 |
| endpoint_audit | 0.000 | 19.811 | 19.840 |

Inclusive update time also contains recorded identities, CPU geometry/hash
calculations and instrumentation bookkeeping, so the component times do not
sum to inclusive wall. Total wall additionally includes loading, final test
evaluation and checkpoint saving; it excludes preflight, checker and postcheck.
Single serial measurements on the same GPU; no separate timing sweep.
The extra margin-gradient computation itself cost only1.331s;
re-prediction cost118.981s and dominated the added work.

## Integrity, artifacts and reproduction

- Preflight15/15 PASS; tests frozen before training. Three arms completed once.
  A final rerun of the15 tests also passed; an inherited torch.jit.script
  deprecation warning was emitted by the dependency, without a test failure.
- `results/CHECKS.json`:PASS. Identity, batch/RNG sequence, zero replay, fixed
  detection sites, state continuity, q10/capability decisions, all-protected safety
  checks, one-pass scalar algebra certificates, new-direction identities and
  endpoint metrics were recomputed from records.
- `results/POSTCHECK.json`:PASS. Reloading the three final checkpoints reproduced
  all6,720 original-test predictions exactly. This is final native inference,
  not an independent rerun of training or all projection gradients.
- `results/EXECUTION.json` records all process return codes.
- `records/` contains generated config/result ledgers, lossless gzip step logs
  and execution logs; manifest checks compressed and uncompressed byte hashes.
- Model/optimizer checkpoints and original FewRel data/token arrays remain local.
  Only three final checkpoints were created, not210 per-step model copies.
- Local evidence: `E:/gfg-evidence/update-direction-isolation-smoke-v1`.
- The post-run diagnostic/replay code is retained as `POSTCHECK_CODE.txt`;
  it is not part of the training decision algorithm.

Pinned runtime: `E:/gfg-python/governor-v4-bench/Scripts/python.exe`.
Original-run commands (create-only; do NOT run over completed directories):

```text
python -B -m experiments.gfg_update_direction_isolation_smoke_v1.preflight
python -B -m experiments.gfg_update_direction_isolation_smoke_v1.run_all
```

Requires the existing task1 checkpoint, data.npz and pinned local BERT. No existing
result or checkpoint was overwritten. Public records alone do not contain the
model/data needed for full native reproduction. This is a mechanistic smoke,
not a final continual-learning method or proof that replay/capacity changes are
unnecessary. No follow-up policy changes or additional training runs were performed.
