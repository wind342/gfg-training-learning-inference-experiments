# Task1: current architecture without old-task conflicts

One fixed paired smoke on RTX 4060 Ti 8 GB; seed 2021, 210 updates,
batch16, head LR 5e-5. Frozen local pretrained BERT, fresh head and fresh
AdamW moments. No already-task1-trained checkpoint was loaded. Production
relation-floor projection is imported unchanged; with zero old relations
there are no constraints. Eighty new-task monitors are observational only.

## Measured result

| Arm | Training compute wall | Median update | Train accuracy | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native head AdamW | 19.717 s | 93.703 ms | 83.244% | 80.625% | 82.857% |
| Current architecture | 20.096 s | 95.663 ms | 83.244% | 80.625% | 82.857% |

Additional measured wall: 0.379 s / 1.921%. Both arms produced byte-identical
head parameters AND AdamW moments at all 210 updates. Zero projections,
zero skips, and every proposed native update was committed unchanged.
Independent test recomputation: 928 / 1120 correct in both arms.

| Step | Full task1 validation accuracy |
| --- | ---: |
| 30 | 19.286% |
| 60 | 41.250% |
| 90 | 58.304% |
| 120 | 69.643% |
| 150 | 76.339% |
| 180 | 79.554% |
| 210 | 80.625% |

Neither arm reached 90% or 95% validation on the prespecified observation grid.
This run establishes low controller overhead with no historical constraints;
it does not establish fewer updates than AdamW, and does not measure historical
task1 full-BERT fine-tuning. The earlier long-run experiments froze a BERT
already trained on task1. Here the pretrained BERT and fresh entity-marker
embeddings were frozen from the beginning, so that task1 learning is not leaked
through a previously trained backbone.

## Timing scope and checks

Per-batch arm order alternated. Each arm independently performed its BERT
forward (19.320 s native, 19.301 s controller in total). Head backward/update
took 0.395 / 0.426 s. The controller prediction/commit/recheck segment took
0.369 s. This is synchronized training compute, excluding per-step archival
hashing, validation evaluation, model load, warmup and final test evaluation.
Full validation feature caching cost 6.485 s separately. Peak allocated CUDA
memory across the paired training run: 707,249,152 bytes (~0.659 GiB).
One timing run only; a 1.9% difference is not a statistical performance claim.

CHECKS PASS: 210 paired parameter/moment identities, 210 exact FP32 endpoint
checks, frozen BERT hash, unchanged production source hashes, original batch
order, clean input split identities, and 1120 independently reconstructed test
predictions. The PyTorch tensor-to-scalar warning in logging was non-fatal;
it did not change training or results. No failed training or outcome-driven rerun.

Only this new experiment directory was added. Main, frozen tags, prior
experiments and their artifacts remain unchanged. Local full artifacts:
`E:/gfg-evidence/task1-conflict-free-timing-v1`; hashes are in
`results/ARTIFACT_MANIFEST.json`. No BERT weights or raw dataset uploaded.

## Reproduction

The existing local dataset and pretrained model paths are imported from the
earlier FewRel setup. With those dependencies and the existing Python runtime:

```powershell
python -m experiments.gfg_task1_conflict_free_timing_v1.run
python -m experiments.gfg_task1_conflict_free_timing_v1.check
```

The runner refuses to overwrite an existing run. The checker reconstructs
test features from local pretrained BERT without retraining. Output writers
also refuse overwrite. See PROTOCOL.md for the frozen comparison scope.
