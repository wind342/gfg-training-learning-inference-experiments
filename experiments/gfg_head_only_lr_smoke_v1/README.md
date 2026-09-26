# Head-only LR sanity check: FewRel task2

Base: `c6490aa26f83d046e0101dc2661ab8dae750cb6a`.
Branch: `codex/gfg-head-only-lr-smoke-v1`.
[Frozen protocol](PROTOCOL.md), [machine results](results/RESULTS.json),
[independent checks](results/CHECKS.json).

## Conclusion

**Case A is supported in this fixed task: the previous7.5% new-test score
primarily reflected insufficient head optimization at LR5e-6, not an inability
of the frozen BERT representation to support task2 learning.** Changing only LR,
with the same210 updates, raised16-class training accuracy to95.45–96.07% and
new-test macro to92.50–92.95%. No backbone update, replay or protection was used.
This does not locate the ultimate representation ceiling or establish retention.
Old capability declines are reported without exclusion. No LR is selected from
test for a subsequent experiment.

| LR | Train16-class | Train new-only (diagnostic) | Old control /80 | New control /80 | Old test macro | New test macro |
|---|---:|---:|---:|---:|---:|---:|
| 5e-6, existing reference | 7.4405% | 40.1190% | 78 | 9 | 97.1429% | 7.5000% |
| 5e-5 | 89.9405% | 90.1488% | 42 | 71 | 55.9821% | 89.9994% |
| 5e-4 | 95.4464% | 95.4464% | 9 | 75 | 15.6250% | 92.4994% |
| 1e-3 | 96.0714% | 96.0714% | 8 | 75 | 12.5893% | 92.9458% |

All-test macro, in the same LR order:52.3214%,72.9908%,54.0622%,52.7675%.
Training uses all3,360 task2 training examples. Test primary uses the same2,239
authorized examples; original2,240-example predictions and secondary metrics are
retained. New-only restricts competitors to eight new classes; it is a diagnostic,
not the formal16-class score. Test numbers are relation-macro averages.

## Fixed execution and delayed evaluation

Only three new arms ran, once each:5e-5,5e-4,1e-3. Seed2021, original task1
checkpoint, batch16,210 updates, inherited head AdamW moments and step, original
betas/weight decay/norm clip and current batch order. LR was set AFTER loading
optimizer state and checked on every update. Nothing else was tuned.

Every current batch received a fresh no_grad frozen-BERT forward. BERT remained
eval, with no gradients or parameter changes. Only head weight/bias updated.
Every step used the inherited exact160-control monitor; its labels/statistics
did not change the update. Alpha was always1. No replay, projection, scalar
shrink, protection, scheduler, LoRA, adapters or new predictor.

Trainers contain no test inference. After all three TRAIN_RESULT.json files
were COMPLETED, their hashes were sealed in TRAINING_COMPLETE.json. Only then
did evaluate.py load heads and execute the single read-only evaluation phase.
That phase evaluated training examples and then test, with shared frozen BERT
features across heads. The5e-6 head was loaded from its previous local artifact,
not retrained; its native test predictions exactly reproduced the frozen result.
No training occurred after evaluation and no additional LR/seed was run.

## Requested losses

These are current-batch, pre-update cross-entropy values. The batch at each
listed step is identical across arms, but different steps have different batches;
a late high value alone does not establish divergence.

| LR | Step1 | Step50 | Step100 | Step150 | Step210 |
|---|---:|---:|---:|---:|---:|
| 5e-6 | 3.089324 | 3.041420 | 2.721660 | 2.858146 | 2.476681 |
| 5e-5 | 3.089324 | 1.595111 | 0.879488 | 0.622081 | 0.576258 |
| 5e-4 | 3.089324 | 0.457713 | 0.098457 | 0.062424 | 0.366792 |
| 1e-3 | 3.089324 | 0.480602 | 0.080934 | 0.058783 | 0.401537 |

## Controls and actual old-boundary changes

Each cell below is correct controls /10 for that relation at the final endpoint.

| Relation | 5e-6 | 5e-5 | 5e-4 | 1e-3 |
|---|---:|---:|---:|---:|
| 0 | 8 | 0 | 0 | 0 |
| 1 | 10 | 10 | 5 | 4 |
| 2 | 10 | 4 | 0 | 0 |
| 3 | 10 | 0 | 0 | 0 |
| 4 | 10 | 8 | 4 | 4 |
| 5 | 10 | 5 | 0 | 0 |
| 6 | 10 | 9 | 0 | 0 |
| 7 | 10 | 6 | 0 | 0 |
| 8 | 4 | 10 | 10 | 10 |
| 9 | 0 | 6 | 10 | 10 |
| 10 | 0 | 10 | 10 | 10 |
| 11 | 0 | 10 | 10 | 10 |
| 12 | 3 | 9 | 10 | 10 |
| 13 | 1 | 10 | 8 | 8 |
| 14 | 1 | 6 | 7 | 7 |
| 15 | 0 | 10 | 10 | 10 |

| LR | Old correct-to-wrong occurrences | Steps with an old safe-to-unsafe group | Group transition occurrences |
|---|---:|---:|---:|
| 5e-6 | 0 | 0 | 0 |
| 5e-5 | 36 | 5 | 5 |
| 5e-4 | 82 | 6 | 7 |
| 1e-3 | 104 | 5 | 7 |

Occurrences can repeat for the same probe/group; they are not unique counts.
All old groups, including already-unsafe ones, remain in the records. Group
safety uses the inherited receiving-state anchor, fixed across each baseline/
candidate comparison. Anchor refresh/rearm semantics are unchanged.

Every relation's test accuracy, per-step margins/capability/q10, already-unsafe
and newly-unsafe identities are included in EVALUATION.json, RESULTS.json and
lossless step logs. No result is removed because it loses old capability.

## Verification and artifacts

- 18 preflight tests PASS, including three-LR override, exact moment inheritance,
  native AdamW candidate equivalence and monitor noninterference at all LRs.
- All three trainers, unified evaluation and checker exited0; no formal restarts.
- Original data/checkpoint/reference artifact hashes unchanged.
- All210 batch IDs, LR values, seeds and alpha1 commits checked per arm.
- Initial head moments match the original reference; final AdamW step420.
- BERT state hashes unchanged; eval/no-gradient/version invariants checked.
- **100,800 new candidate-control endpoints** independently recomputed from
  cached features and saved head states; exact agreement.
- Train/test metrics recalculated from saved predictions; delayed-test barrier
  timestamps and result hashes validated. Original5e-6 test reproduced exactly.
- CHECKS.json PASS. This is record/affine replay, not independent retraining.

PyTorch emitted a non-fatal warning for converting the logged differentiable
loss tensor to a Python scalar. This is after backward and only for the ledger;
no runtime failure or policy change occurred. Frozen source was not altered
after preflight to suppress the warning. Execution logs are retained locally.

Local evidence: `E:/gfg-evidence/head-only-lr-smoke-v1/`. Each new arm retains
CONFIG.json, TRAIN_RESULT.json, rawSTEPS.jsonl and AUDIT.pt (small head snapshots,
control features and optimizer states). No full BERT checkpoint was duplicated.
Weights, feature tensors and original texts/token arrays remain local.

Public generated records are hashed in PUBLIC_RECORDS_MANIFEST.json; gzip logs
preserve raw bytes. package_records.txt only packages artifacts after completion.
Old experiments, results, Router, main and frozen paper tag remain unchanged.

Original create-only commands, not for rerunning over completed evidence:

```text
E:/gfg-python/governor-v4-bench/Scripts/python.exe -B -m experiments.gfg_head_only_lr_smoke_v1.preflight
E:/gfg-python/governor-v4-bench/Scripts/python.exe -B -m experiments.gfg_head_only_lr_smoke_v1.run_all
```
