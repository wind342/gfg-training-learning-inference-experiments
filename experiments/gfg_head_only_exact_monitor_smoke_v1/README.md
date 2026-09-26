# Head-only exact-monitor smoke (FewRel task2)

Base: `e84014c2417e631025e9a8dc1b9a69d23997856c`.
Branch: `codex/gfg-head-only-exact-monitor-smoke-v1`.
Protocol: [PROTOCOL.md](PROTOCOL.md). Machine summary:
[results/RESULTS.json](results/RESULTS.json). Verification:
[results/CHECKS.json](results/CHECKS.json).

## Answer

**Every-step exact control monitoring became inexpensive, but the frozen
representation/head training did not form strong new task2 capability under
this fixed learning rate and budget.** This is not a successful protection
controller: no protection or replay was used.

Both new arms completed exactly once, 210 updates, seed2021, LR5e-6, batch16,
shared task1 checkpoint, inherited head AdamW moments, original batch order.
No tuning, sweeps, new full-FT run, changes to Router or previous evidence.

| Arm | Old macro test | New macro test | All macro test |
|---|---:|---:|---:|
| Full-FT native, frozen historical reference | 5.6250% | 92.5893% | 49.1071% |
| Head only | 97.1429% | 7.5000% | 52.3214% |
| Head only + every-step exact monitor | 97.1429% | 7.5000% | 52.3214% |

The equal-old/new macro average must not obscure the poor new-task formation.
Historical reference is read directly from the previous experiment's committed
native RESULT.json, whose hash is included in RESULTS.json. It was not rerun.
Primary test has2,239 examples; original2,240-example test predictions and
secondary metrics are also retained. Metrics above are relation-macro, not micro.

| Relation | Head only | Head only + monitor |
|---|---:|---:|
| 0 | 87.1429% | 87.1429% |
| 1 | 100.0000% | 100.0000% |
| 2 | 97.8571% | 97.8571% |
| 3 | 98.5714% | 98.5714% |
| 4 | 97.1429% | 97.1429% |
| 5 | 96.4286% | 96.4286% |
| 6 | 100.0000% | 100.0000% |
| 7 | 100.0000% | 100.0000% |
| 8 | 24.2857% | 24.2857% |
| 9 | 0.7143% | 0.7143% |
| 10 | 0.0000% | 0.0000% |
| 11 | 0.0000% | 0.0000% |
| 12 | 20.7143% | 20.7143% |
| 13 | 3.5714% | 3.5714% |
| 14 | 10.0000% | 10.0000% |
| 15 | 0.7143% | 0.7143% |

## Scope expansion versus training

| Point | Old test macro | Old control capability |
|---|---:|---:|
| Task1 checkpoint, seen8 | 97.1429% | 78/80 =97.5% |
| No update, seen16 | 97.1429% | 78/80 =97.5% |
| After210 head-only updates, seen16 | 97.1429% | 78/80 =97.5% |

The class expansion did not lower old accuracy here. Nor did this head-only
training lower aggregate old accuracy. This is a measured absence of decline
at these readouts, not a proof that all old margins or representations are the
same across configurations. BERT itself was byte-identical throughout the run.

Control group0 started at8/10, below the .9 threshold, and was already unsafe
at all210 steps. Other old groups started at10/10. There were:

- 0 newly-unsafe old-group steps /0 group-occurrences;
- 0 newly-unsafe receiving-PROTECTED group steps;
- 0 old control correct-to-wrong crossings;
- 210 already-unsafe group-occurrences, all group0;
- 9 total control boundary transitions, including new groups.

All old groups are recorded even when DEGRADED. Per-step q10, capability,
receiving anchor, PROTECTED identities and endpoint state are in raw ledgers.
The same receiving anchor is used on both sides of each safety comparison.
Inherited full refresh can follow gradual margin erosion; absence of a new
threshold crossing does not establish absence of cumulative margin changes.
The counts describe this uncontrolled trajectory, not a future filtered one.

## Actual cost

All times are seconds, one serial run per arm on RTX4060Ti8GB.

| Component, sum over210 updates | Head only | Head only + monitor |
|---|---:|---:|
| Fresh current-batch frozen BERT forward | 19.327154 | 19.321741 |
| Head forward and CE loss | 0.105440 | 0.100299 |
| Head backward and norm clipping | 0.286183 | 0.280597 |
| Native head AdamW candidate | 0.098792 | 0.101903 |
| Candidate commit | 0.019048 | 0.018786 |
| Cached baseline/endpoint logits and margins | 0 | 0.115087 |
| Full capability/q10/state statistics | 0 | 0.487646 |
| Compute update total, excluding research audit | 19.836617 | 20.426059 |
| Training wall, including audit and record I/O | 21.241490 | 21.951632 |
| All-in arm wall, including setup/evaluation/artifact save | 52.136021 | 52.827596 |

Per update monitor cost:

- actual cached affine endpoints/margins: **0.548034ms**;
- capability/q10/state statistics: **2.322122ms**;
- total monitor: **2.870157ms**;
- compute update ratio: **1.029715**, approximately2.97% added;
- inclusive update ratio: **1.031502**;
- full training-wall ratio: **1.033432**, approximately3.34% added.

Control caching took1.058767s /1.033826s respectively, included in setup/all-in
wall. H has shape160x1536, FP32,983,040bytes (0.9375MiB). It is created once per
arm; current training examples are never pre-cached. Comparing to the historical
~10.8s full-BERT predictor calls illustrates changed computational scope, not
an equivalent full-finetuning predictor acceleration.

Timing uses CUDA synchronization. Hash/RNG/version checks and native candidate
identity audit are excluded from monitor kernel/statistics and included in
training wall. Per-arm audit totals and JSON I/O are explicitly stored in
RESULTS.json/RESULT.json; no safety statistics were dropped to obtain the timing.
Two single serial measurements are not a precise statistical performance study.

| CUDA peak bytes, training (also process peak here) | Head only | Head only + monitor |
|---|---:|---:|
| Allocated | 697,405,952 | 697,408,000 |
| Reserved | 782,237,696 | 782,237,696 |

Approximately0.650GiB allocated and0.729GiB reserved. These are PyTorch allocator
measurements, not total graphics-driver/desktop usage from nvidia-smi.

## Exactness, identity and isolation checks

- Preflight15/15 tests PASS, including actual checkpoint head/moment migration.
- BERT requires_grad=False, no gradient, always eval after model.train().
- BERT parameter versions checked every step and full state SHA-256 unchanged
  from the source task1 checkpoint at the final endpoint.
- Only head weights/bias train. Both arms use native AdamW on the same head
  candidate path. Its moments advance exactly once per current gradient.
- Monitor touches neither real head nor optimizer moments, consumes no RNG,
  always commits the original alpha1 candidate, and never reads held-out test.
- All210 head endpoints and per-step optimizer-state hashes match across arms.
  Final head tensors, optimizer states, and test predictions match exactly.
- Cached H is unchanged; no control BERT forward within the training loop.
- 33,600 candidate-control margins independently recalculated from cached H
  and actual saved head endpoints agree exactly with the recorded values.
- 4,480 original-test predictions reproduced exactly by loading each final
  head on the frozen original BERT and doing native inference.
- Full inherited data isolation PASS. Original data unchanged; previously
  authorized control50124->50130 and seven test exclusions retained.
- All runs and the separate checker returned0. See EXECUTION.json.

The checker reconstructs affine endpoints and native final inference, and
checks records/hashes. It is NOT an independent full training rerun. Exact
control evaluation does not imply exact prediction of unseen generalization.

During development, eight initial tiny unit tests failed because their training
loss mistakenly called the no_grad monitor helper. Test training was changed to
the differentiable head forward before preflight/source freeze. The final15
tests passed; neither formal training arm failed, restarted or changed policy.

## Scientific boundary

The computational question is supported: exact every-step evaluation of all160
controls is cheap when the large representation is fixed. Under LR5e-6 and210
updates, new test macro is only7.5%, so this run does not provide a strong-learning
head-only candidate for a protection experiment. This does not show that
head-only learning is impossible with another representation/LR/budget.

Nor can the full-FT/head-only accuracy difference be labeled an isolated backbone
causal contribution: trainable scope, BERT dropout and the resulting trajectories
differ. No additive three-part causal decomposition is claimed.

## Files and reproduction

Local evidence: `E:/gfg-evidence/head-only-exact-monitor-smoke-v1/`.
Each arm hasCONFIG.json,RESULT.json,rawSTEPS.jsonl andAUDIT.pt. AUDIT.pt contains
211 small head snapshots, cached control features and initial/final head optimizer
state, anchored to the immutable task1 checkpoint. Two files total~204MiB; no
duplicate full BERT checkpoint. They remain local, not in public Git.

Public records contain generated metrics, example indices, control margins,
hashes and lossless gzip step logs. No original texts, token arrays, learned
weights or cached feature tensors are included. PUBLIC_RECORDS_MANIFEST.json
records compressed and uncompressed identities. package_records.txt is a
post-run export utility, never imported by frozen training/check code.

Pinned Python: `E:/gfg-python/governor-v4-bench/Scripts/python.exe`.
Original create-only commands, **do not run over completed evidence**:

```text
python -B -m experiments.gfg_head_only_exact_monitor_smoke_v1.preflight
python -B -m experiments.gfg_head_only_exact_monitor_smoke_v1.run_all
```

Source model/data are required for full native reproduction. Old experiments,
results, Router code, main and frozen tag remain unchanged. No subsequent
intervention experiment, LR search or extra training was performed.
