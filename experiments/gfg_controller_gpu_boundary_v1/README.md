# GPU boundary and independent runtime entry

Base: `992ab35645b60d60f5eb4ab4590b7faaa79ff2df`.
Only this directory is added. The previously certified runtime and all frozen
research evidence remain untouched.

## Implemented

- `GPUBlockwiseController` uses the certified two-buffer native proposal/commit
  machinery and blockwise h/h'/h'' propagation.
- Logits, hidden arrays, readout weights and actual readout updates stay on CUDA.
  FP64 target/competitor differences and dot products are tiled; the same explicit
  FP32 coordinate rounding and FP64 endpoint association are retained.
- Every competitor participates. No target-bank reduction, top-k shortcut,
  missing curvature term, new alpha candidate, or cadence change.
- Four correct counts, four median margins and a finite-data flag form one
  9-double / **72-byte boundary transfer per decision**. Loss/evaluation scalar
  synchronization still exists; 72 bytes is not total training I/O.
- `entry.py` is an independent callable/CLI loop. It does not import benchmark
  optimizers, read old private ledgers, or hash/copy the full model per step.
- Target/competitor tile sizes are explicit arguments, default `256 x 128`.
  The default was fixed before timing, not chosen retrospectively.

## Verification

All 30 synthetic shape/dtype/tile cases were byte-exact in coordinates and
decisions, including large vocabulary, odd/even medians and tied-zero behavior.

Across 12 old historical states and three learning rates:

- 36 audited GPU trajectories: 1,730 updates and 353 boundary decisions;
- 36 actual standalone-entry trajectories: another 1,730 updates;
- all historical post-state/logit comparisons in the audited trajectories and
  final optimizer identities match the old reference;
- every checked coordinate, minimum, score and alpha was exact; maximum coordinate
  difference was zero, with no tolerance fallback;
- standalone alpha/accuracy trajectories, final logits and optimizer identities
  match the old reference; their final hashes are taken OUTSIDE the entry/timer;
- 30 of the 72 trajectories were zero-step initial-success cases, not new training;
- independent offline audit: PASS.

A further guarded 49-update standalone execution prohibited model snapshot/hash
helpers and any explicit Tensor.cpu transfer larger than nine elements. It passed:
ten boundary transfers, 720 bytes total. This does not claim the absence of scalar
device synchronization or measure every CUDA-runtime transfer.

These finite tests establish observed byte equality on this hardware/environment,
not a universal CPU/GPU bit-equivalence theorem across all inputs and devices.

## Matched time-to-capability check

One preselected historical start, LR .003, unchanged target99 rule: all paths
reached the same result at **49 updates**. One full warmup excluded per path,
then three fresh-state repetitions; values below are medians. Timing includes
initial/every-step evaluation, excludes model/optimizer/controller construction
and the external final hash. Both CPU and GPU entries use the same timing scope.

| Boundary backend / tile | Wall to target (s) | Peak CUDA allocated (bytes) |
|---|---:|---:|
| CPU reference / 32x8 | 0.8457835 | 101,108,736 |
| GPU / 32x8 | 1.1098437 | 101,108,736 |
| GPU / 64x64 | 0.8836272 | 101,108,736 |
| GPU / 256x128 | 0.7789562 | 101,108,736 |

The preset 256x128 GPU path reduced median wall time by approximately **7.9%**
against the already-refactored CPU-boundary controller in this single small case.
Smaller GPU tiles were slower. This is not a result against native AdamW or fixed
shrinkage, and not a large-model speed claim. The three CPU times were
0.9300025/0.8457835/0.8110780 s; GPU256 times were
0.7772687/0.7789562/0.8075802 s. Longer/larger measurements remain necessary.

## Usage

From repository root, in the existing CUDA environment:

```text
python -m experiments.gfg_controller_gpu_boundary_v1.entry --help
python -m experiments.gfg_controller_gpu_boundary_v1.entry --factory your_local_module:make_run --backend gpu --target-tile 256 --competitor-tile 128 --out path/to/new_result.json
```

`make_run()` returns keyword arguments for `entry.train`: model, optimizer,
batches, seeds, learning_rates, probe_inputs, probe_targets; optional separate
evaluation_inputs/evaluation_targets. Caller supplies deterministic backend
settings, actual weights/data and a scientifically frozen schedule/bank.
No data or weights are downloaded and existing result files are never overwritten.

The API defaults to evaluate_every=1. Changing evaluation cadence changes observed
first-crossing resolution; it was not changed in this validation. Probe and
evaluation banks are the same 212 instances in this old witness, not a new
held-out-target generalization claim.

Reproduction/audit with existing evidence dependencies:

```text
python -m experiments.gfg_controller_gpu_boundary_v1.validate
python -m experiments.gfg_controller_gpu_boundary_v1.audit
python -m experiments.gfg_controller_gpu_boundary_v1.transfer_guard
```

Runners refuse existing seals/output paths. Raw arrays/trajectories are in
`E:/gfg-evidence/controller-gpu-boundary-v1`, outside the source repository.
`AUDIT.json` lists their hashes. The source contains compact reports, not weights.

## Remaining limits

Still two full model-size control buffers, still bounded 3-D tile temporaries,
still full pre-logits on GPU. Difference storage scales as O(B*C*H), not zero.
No vocabulary GEMM reassociation, distributed sharding, mixed precision, CUDA
graphs or arbitrary-model adapter is certified. The GPU uses FP64 boundary math;
hardware-dependent speed is measured, not assumed.

70M/160M training has NOT started. This delivers a usable independent runtime
for the next scale-validation stage, not proof that those scales fit or win.
