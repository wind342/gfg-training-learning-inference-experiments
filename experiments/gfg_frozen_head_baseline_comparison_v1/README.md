# Same-representation FewRel baseline comparison

Independent branch extension of `0a62264bcbad0a6dae8bd4e755aae96aec205a07`.
Scientific protocol: [PROTOCOL.md](PROTOCOL.md). No previous results or frozen
release files are edited. Final v2 is imported from its existing production
implementation, not rewritten for this comparison.

## What is compared

- Native head training, 210 updates/task.
- Equal-example feature replay (ER), 210 / 630 updates/task.
- A-GEM-AdamW and zero-margin GEM-AdamW shared-head adaptations, 210 / 630.
- Final v2 relation-floor functional endpoint control, 210 / 630.
- Accumulated ridge without random projection.
- RanPAC **classifier-head adaptation**, fixed Gaussian RP + ReLU, M=10000.

These are controlled adaptations to the same task1-trained frozen BERT features,
not the original papers' reported vision benchmark results. In particular GEM's
original SGD and task-dependent test masks are not used, and RanPAC's original
first-session visual-model PETL is not reproduced.

The same labeled stream is made available. Statistical methods retain all past
training contributions in G/Q, while other methods retain the learned head,
moments and the released old validation bank. This is NOT equal historical
memory storage. Controls are algorithmic training information, not held-out
generalization evidence. Only test metrics assess held-out generalization.

## Reproduction

Run from the repository root in the existing FewRel runtime:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'E:\gfg-python\governor-v4-bench\Scripts\python.exe' -m experiments.gfg_frozen_head_baseline_comparison_v1.prepare
& 'E:\gfg-python\governor-v4-bench\Scripts\python.exe' -m experiments.gfg_frozen_head_baseline_comparison_v1.tests
& 'E:\gfg-python\governor-v4-bench\Scripts\python.exe' -m experiments.gfg_frozen_head_baseline_comparison_v1.run
& 'E:\gfg-python\governor-v4-bench\Scripts\python.exe' -m experiments.gfg_frozen_head_baseline_comparison_v1.check --replay
```

Output authority: `E:/gfg-evidence/frozen-head-baseline-comparison-v1`.
Commands are create-only and refuse to overwrite an existing run. Relocate or
archive an existing authority before a deliberate new reproduction; never
overwrite this run. No training process reads TEST_FEATURES.pt; only the
post-task evaluator and independent checker do so.

Required pre-existing authorities:

- `E:/gfg-evidence/governor-fewrel-v1/data.npz`, SHA256
  `29450295331dcd56040b9ec60ab2998303988c0e27500e353b7025b977734a8a`.
- `E:/gfg-evidence/governor-fewrel-v1/3tasks-governor/attempt-1/task1.pt`,
  SHA256 `bd41bcbd84f280d4de7d721eeee1d7a7bad1ccbb04ceb7d84d0403a6bdbcb676`.
- Existing verified Final v1 control cache and manifest.
- Existing relation-floor task8 audit for the inherited numerical regression
  tests; the old Final v2 records for historical equality comparison.
- Local BERT model assets used by the existing FewRel preparation code.

The input dataset, full task1 backbone, and cache authorities are not included
in the public compact results. Their original acquisition/preparation code
remains in the parent experiments. Therefore public records alone support
inspection of the comparison but are not claimed to contain every large input
needed for native reexecution.

## Compact but reproducible storage

Keep one shared FP32 feature cache, one evaluator-only feature cache, a small
head/moments checkpoint after each task, sampled IDs, compressed step records,
predictions, and input/source/checkpoint hashes. For statistics retain per-task
classifier weights plus one final G/Q state and one RP matrix. Do not retain
thousands of full head/gradient/candidate tensors or ten copies of the Gram.
The directory-local Git attributes disable newline conversion for frozen
`results/` bytes so that published records retain the local authority hashes.

Task2 of every gradient arm is rerun deterministically to compare all step head,
moment and count hashes. An independent checker recomputes sampling, data
arrival, metric aggregation, every saved classifier's predictions and final
control counts. This is not an assertion that all later updates were
independently reexecuted or that a separate implementation reproduced every
algorithmic operation. See CHECKS for the exact achieved verification scope.

Algorithm wall includes mandatory memory backward, projection and solve costs.
Read-only observers, audit/hash writes and common test are separate. Shared
feature-extraction and statistical task1 initialization are separately recorded.
Recorded CUDA peaks are **shared-process high-water marks**, with other arm
states resident; they are not isolated per-method VRAM requirements. Old read
counts are logical row evaluations, not bytes or equal FLOPs. V2 uses no old
CE backward/replay, but it does use labeled old controls for boundary constraints.

## Baseline source review

- [GEM implementation](https://github.com/facebookresearch/GradientEpisodicMemory/blob/34c6b8e9a0607db7567301c48b727430d20bee7e/model/gem.py):
  past-task gradients, QP, gradient replacement, optimizer step. Our homogeneous
  zero-margin QP is explicitly distinguished from the code's positive dual
  lower-bound option. GEM source is CC-BY-NC-4.0; no source was copied.
- [A-GEM implementation](https://github.com/facebookresearch/agem/blob/45421499483b28935491251e9e821c55e8b3c089/model/model.py):
  averaged memory reference gradient and single-halfspace projection. No source
  was copied; projection equations are implemented and unit-tested locally.
- [RanPAC implementation](https://github.com/McDonnell-Research-Lab/RanPAC/blob/cf4b301d18b0c27db030f4371b72b768005ae58a/RanPAC.py):
  `replace_fc`, `optimise_ridge_parameter`, `setup_RP`. No-RP uses cumulative
  ridge (M<=0), not the alternative NCM path. FP64 statistics/solving and
  algebraically equivalent primal/dual ridge CV are explicit numerical choices.
  Its [CosineLinear.forward RP path](https://github.com/McDonnell-Research-Lab/RanPAC/blob/cf4b301d18b0c27db030f4371b72b768005ae58a/inc_net.py)
  uses an unnormalized linear readout; despite the class name, this path does
  not apply the non-RP/NCM branch's cosine normalization.

No license change is made to this research-evidence repository.
