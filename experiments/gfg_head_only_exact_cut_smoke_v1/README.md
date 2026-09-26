# Head-only exact-cut smoke: FewRel task2

Base: `3f7a592a69947d944027c4761f99f7544119ed4a`.
Branch: `codex/gfg-head-only-exact-cut-smoke-v1`.
[Protocol](PROTOCOL.md), [machine results](results/RESULTS.json),
[independent checks](results/CHECKS.json).

## Result

**In this fixed single-seed smoke, exact-boundary directional cuts improved
held-out old retention substantially, while new-task accuracy changed little.**
The original78 correct old controls were preserved on every executed update.
No update was skipped, so the gain was not produced by stopping training.

| Method | Old test macro | New test macro | All test macro |
|---|---:|---:|---:|
| Head-only LR5e-5, frozen historical baseline | 55.9821% | 89.9994% | 72.9908% |
| Head-only exact cut, LR5e-5 | 76.6071% | 89.1958% | 82.9015% |
| Difference, percentage points | +20.6250 | -0.8036 | +9.9107 |

| Other endpoint | Historical baseline | Exact cut |
|---|---:|---:|
| Task2 train16 accuracy,3,360 examples | 89.9405% | 89.4643% |
| Old controls correct /80 | 42 | 78 |
| New controls correct /80 | 71 | 72 |

Primary test has2,239 examples with the same authorized isolation exclusions;
all2,240 original-test predictions and secondary metrics are retained. Test
scores are relation-macro. The baseline was referenced, not rerun. Only this
one new arm ran, once; test inference occurred only after training completed.

The held-out improvement is descriptive evidence from one fixed seed, not a
statistical significance claim or complete preservation of old capability.
Task1 old test was97.1429%; the cut endpoint remains below that. Protected
controls are optimization-control data, not held-out generalization evidence.

## Execution and cuts

Same task1 checkpoint, seed2021, LR5e-5, batch16,210 updates, original current
batch order and inherited head AdamW moments. BERT was byte-identical, frozen
and always eval. Current batches had fresh no_grad BERT forwards; only the160
fixed control features were cached. Only head weight and bias trained.

An old control is eligible if its receiving margin>0. A candidate margin<=0
violates. Every seen competitor is considered, including a newly dominant one.
For each violating target/competitor pair, analytic head gradient includes both
weight rows and biases. Negative directional margin change is projected out in
fixed control/competitor order, then all160 controls are exactly rechecked.
This is not a globally minimum-distance feasible projection. No scalar shrink,
alpha search, replay, Router, q10 policy, LoRA, adapter or BERT update was used.

| Event | Count |
|---|---:|
| Updates | 210 |
| Native-safe updates, unchanged native commit | 36 |
| Updates triggering projection | 174 |
| Safe after pass1 | 174 |
| Updates needing pass2 | 0 |
| CUT_FAILED_SKIP | 0 |
| Native old-control violating occurrences | 1,246 |
| Analytic pair certificates | 1,358 |
| Executed old correct-to-wrong occurrences | 0 |

Occurrences may repeat across updates;1,246 is not a count of unique controls.
All originally correct old controls remained correct. The initial two incorrect
old controls were not protected as correct examples and were not restored by
this rule. Pass2/skip semantics passed unit tests but were not reached in this
formal trajectory. AdamW moments advanced exactly once per step and were not
changed by the cut. The predefined skip policy would retain that advancement.

## How much did the direction change?

The following are means across the174 projected/accepted updates:

| Quantity | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| norm(cut)/norm(native) | 0.921469 | 0.851685 | 0.991375 |
| cosine(native,cut) | 0.886963 | 0.767430 | 0.991375 |
| norm(native-cut)/norm(native) | 0.448264 | 0.131052 | 0.647291 |

Across all210 executed updates the mean norm ratio is0.934931. Retaining92.15%
of the norm on projected steps does **not** mean changing only7.85% of the update
vector: the relative displacement norm is44.83%, with cosine0.887. Both amplitude
and directional measures are reported rather than equating norm with information.

Per-step new-control gains (correct-after minus receiving correct) are recorded
for native, cut and executed endpoints. On this cut trajectory their sums are:

| Update subset | Native candidate | Cut candidate | Executed |
|---|---:|---:|---:|
| All210 steps | 96 | 72 | 72 |
|174 triggered steps | 49 | 25 | 25 |

Unexecuted native candidates are exact cached-affine counterfactuals at the CUT
arm's receiving states, not the historical native training trajectory. Their
gain sum96 is not an endpoint score or96 distinct correct controls. Executed
gains telescope from0 to72 new controls; no candidate gain is credited to a skip.

## Timing and memory

- Exact-cut training wall:25.173231s.
- Historical LR5e-5 baseline training wall:approximately21.83s.
- Projection pair construction/arithmetic/certificates:2.990450s total.
- Post-cut exact rechecks:0.078010s total.
- Entire exact-check/cut filter:3.356519s total, including the above components.
- Training process wall including setup/artifact saving, excluding separate
  train/test evaluator and checker:30.259299s.
- Peak PyTorch allocated698,882,048bytes; reserved786,432,000bytes.

Filter and projection components overlap by containment; do not add them again.
Hashing, snapshots and logging are included in training wall but outside the
filter timer. The historical timing comparison is descriptive: one run each,
different audit payloads, not a controlled repeated timing benchmark.

## Per-relation test accuracy

| Relation | Baseline | Exact cut |
|---|---:|---:|
|0|1.4286%|37.8571%|
|1|85.0000%|90.0000%|
|2|64.2857%|82.1429%|
|3|0.0000%|47.8571%|
|4|80.7143%|92.1429%|
|5|79.2857%|89.2857%|
|6|90.7143%|92.8571%|
|7|46.4286%|80.7143%|
|8|100.0000%|100.0000%|
|9|61.4286%|55.0000%|
|10|99.2806%|99.2806%|
|11|95.7143%|95.0000%|
|12|100.0000%|100.0000%|
|13|97.1429%|93.5714%|
|14|79.2857%|81.4286%|
|15|87.1429%|89.2857%|

## Checks and reproduction

- 25 preflight tests PASS, including gradient/bias, switched competitor, zero
  ties, immutable inputs/RNG, two-pass failure and skip-moment semantics.
- One training run and final read-only evaluation completed without failure.
- Independent checker reconstructed1,358 pair operations, pass decisions,
  native/cut/executed head identities and128,640 affine endpoint values exactly.
- All210 batch IDs/LRs/seeds checked; moments inherited and final step420.
- BERT eval/no-gradient/version invariants and full state hashes unchanged.
- No protected old-control crossing on any actual committed step.
- Dataset isolation and original checkpoint/baseline artifact identities PASS.
- Train/test metrics recomputed from saved predictions. Test timestamp follows
  completed training. No result-dependent rule changes or reruns.
- CHECKS.json PASS. Algebra/affine replay is not an independent training rerun.

Local evidence: `E:/gfg-evidence/head-only-exact-cut-smoke-v1/`.
The local AUDIT.pt is313,650,140bytes and contains receiving/native/cut head
snapshots, cached control features and optimizer states, not a full BERT copy.
Original texts/token arrays, learned weights and control features are not public.
Public raw generated ledgers are losslessly gzipped with both byte hashes.
PUBLIC_RECORDS_MANIFEST.json records copied artifact identities.

Original create-only commands; do not rerun over completed evidence:

```text
E:/gfg-python/governor-v4-bench/Scripts/python.exe -B -m experiments.gfg_head_only_exact_cut_smoke_v1.preflight
E:/gfg-python/governor-v4-bench/Scripts/python.exe -B -m experiments.gfg_head_only_exact_cut_smoke_v1.run_all
```

All changes are in this independent directory. Old experiments/results, main
and the frozen paper tag remain unchanged. No additional arm or training was run.
