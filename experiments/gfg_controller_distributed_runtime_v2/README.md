# Functional controller distributed runtime v2

Engineering base: `ec891393178e52ab5f0e5e2571e0f3880dfa7e13`.
Scientific baseline: `d17f4510adf46ebae4fc391382ca6be1134eec03`.
Independent single-GPU oracle: `e49cb97db7467fbda8fa9d06c8c4a2a945707d43`.
The paper tag and all earlier experiment sources/results are unchanged.

## Scope and interpretation

The controller still scores **actual native optimizer displacements**, not a
loss-risk proxy or a learned gate. The proposal, optimizer moment recurrence,
complete quadratic coordinates, explicit FP32 coordinate rounding, all
competitors, positive-gap criterion and alpha/count/median tie breaking remain
fixed. Reference cadence is 5, with alphas `{.25,.5,.75,1}`. Native model
parameters are FP32; boundary coordinate arithmetic accumulates in FP64.

Default `backend='exact'` implements the original subtract-then-dot arithmetic
with bounded tiling and exact target-row deduplication. `scalarized` and
`streaming` are explicit **candidates**; passing finite witnesses is not a proof
of universal floating-point equivalence. Neither has silently become the default.

This is a **distributed research prototype**, not a multi-GPU production
certification. One-dimensional FSDP ownership doubles as the vocabulary-row
partition for boundary MIN. Arbitrary two-dimensional DP/TP meshes and an
independent Megatron transformer tensor-parallel execution engine are not
implemented. Replica placements are rejected rather than double counted.

## Module boundaries

- `scientific.py`: endpoint expression and selector only.
- `identity.py`: canonical logical identity, true tied aliases and shape checks.
- `ownership.py`: local candidate/delta, packed stage gather, explicit stage life.
- `clipping.py`: global DTensor-aware clipping with ownership checks.
- `adapters.py`: generic stage contract and the only model-specific implementation,
  `NanoGPTAdapter`; native names do not leak into the generic runtime.
- `boundary.py`: workspace planner, exact/deduplicated, scalarized and streamed
  boundary calculations, owner routing and global per-target MIN.
- `controller.py`: standalone update/train API. It imports none of the old
  ledgers, witness loaders, evidence artifacts or full-model hash routines.
- `distributed_train.py`: caller-supplied factory / torchrun entry.
- Other files: external gates, fixed protocols, scale jobs, diagnostics and audits;
  they are not required by the runtime API.

`GenericDecoderAdapter` is a contract for embedding, ordered stage functions,
readout and directional propagation. Only nanoGPT has a tested model-specific
FSDP wrapping adapter. No Llama compatibility is claimed.

## Identity, ownership and collectives

Registry construction groups **Parameter object identity before wrapping**, then
assigns stable canonical IDs and aliases. Resolution after wrapping checks the
alias identity and global shape. Tied embedding/head have one optimizer state,
candidate, delta and commit. Distinct Parameter objects sharing storage are
explicitly unsupported.

Candidate and delta are two persistent local FP32 buffers: `8 * P_local` bytes,
approximately `8P/R` under one-dimensional row sharding (including uneven rows).
The swap scratch is bounded at 65,536 elements. Model parameters, gradients,
optimizer states and stage/activation workspaces are **additional**, not included
in this controller-buffer figure. No five full-model control buffers are used.

Each directional stage explicitly unshards its owning unit, gathers one packed
delta, computes the jet through a stateless META template, drops full delta
storage and reshards. The tied root unit's whole materialized parameter footprint
is counted, even when a substage reads fewer parameters. There is no per-parameter
`full_tensor()` in production execution. Audit helpers intentionally gather full
identities outside production timings.

For `R>1`, delta gathering uses **one data collective per stage**. Packing derives
ceil-row partitions from canonical global shapes and pads ranks to the largest
stage payload. At `R=1`, gathering is a local identity operation with zero
communication; it must not be described as measured multi-GPU communication.

Boundary routing performs two SUM collectives (target logits and target payload),
then MIN over the four per-target minima and a validity flag. Count and median
are computed **after** the global competitor MIN. No full-vocabulary gather is
performed. The rank group must have identical ordered probes. A small bank hash
checks this on bank changes; this is not a full-model CPU hash.

Exact target payload is `4N + 8UH` bytes, where `U` is the number of unique targets;
without dedup it would be `4N + 8NH`. Scalarization's payload is `36N` bytes. The
minima payload is `32N` bytes plus one int32 validity flag. These are logical
collective buffer sizes, **not measured network traffic**. FSDP forward/backward
parameter gathers and gradient reductions are extra. Wire traffic depends on
collective algorithm/topology and is not measured on this one-GPU machine.

For the current 212-probe, 23-unique-target bank, target row traffic is reduced
from 212 rows to 23 rows; it is not a task-independent promise of that ratio.

## Workspace and optimized candidates

The planner estimates temporary tiles using hidden width, local
vocabulary/probe sizes and the lesser of 64 MiB and 10% of free CUDA memory after
a 16 MiB reserve. It shrinks tiles, never deletes competitors. If even the minimum
tile cannot fit, it raises a resource error. This is an allocation estimate, not
a CUDA allocator hard limit. Measured budget compliance below is restricted to
the recorded shapes; larger probe/vocabulary sizes need their own measurement.

The streamed candidate bounds hidden chunks at 128. It avoids a full-hidden
`B_target * B_competitor * H` intermediate but still writes **chunk intermediates
to HBM**. It is not a fused SRAM-only FlashAttention-style kernel. Triton is not
installed; no untested Triton implementation or `torch.compile` speedup is claimed.

## Reproduction

External raw output must be explicit; `evidence.py`'s historical default is
attempt-1 and must not be reused for new runs. Original output directories are
append-only. Reports ship under `verification/`; raw NPZs remain outside Git.

```powershell
$env:GFG_V2_OUT='E:/gfg-evidence/controller-distributed-runtime-v2/attempt-2'
E:/gfg-python/torch-cu130/Scripts/python.exe -m experiments.gfg_controller_distributed_runtime_v2.verify
```

Portable Linux / NCCL runtime (caller supplies the factory; no audit data required):

```bash
torchrun --standalone --nproc-per-node=2 \
  -m experiments.gfg_controller_distributed_runtime_v2.distributed_train \
  --factory your_training_module:factory
```

The factory supplies model, pre-wrap adapter/registry, optimizer factory, rank-local
slices of one global batch sequence, seeds, learning rates, identical control
bank/targets and optional callback. Unequal local batch sizes need explicit loss
weighting; this prototype does not silently correct such a caller error.

Portable real oracle gate, with historical witness material available locally:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m experiments.gfg_controller_distributed_runtime_v2.distributed_gate --lr .003
```

Repeat at `.001` and `.01`. This records raw gradients, norms, proposals, actual
deltas, moments, jets, minima, selected alpha, parameters and logits against an
independent full-batch oracle. Any bit difference stays a mismatch pending review;
there is no tolerance selected after seeing results.

`launcher_compat.py` supplies an optional Windows FileStore rendezvous workaround
without patching installed Torch. Ordinary Windows TCP rendezvous failures are
preserved alongside its results.

## Claim boundaries

The 70M/160M checks retain vocabulary 24 and sequence length 3. They test scaling
the controller/model state on modular addition, **not natural-language pretraining**.
All 529 input pairs can appear in training. The 317 evaluation queries are disjoint
from controller probes, not from the training distribution. Evaluation is included
in measured time-to-capability. Two throwaway warmup updates and initialization
are excluded and the same initial model/optimizer is rebuilt before timing.

SAFE/NEAR/SEVERE are calibration-seed, budget-defined groups, not universal
instability classifications. Formal seeds are independent. Tuned-native AdamW's
rate is selected on calibration only. Missing regimes remain missing. Cadence
and smaller probe banks are **different control policies**, not exact backend
substitutions.

Unresolved for 1B/7B deployment: actual multi-GPU collectives/gradient reduction
rounding and timing; large vocabulary and long-sequence memory/compute; additional
adapters; mixed precision; arbitrary meshes; fully fused boundary kernels; and
bounded long-run telemetry (current per-stage diagnostic event lists accumulate).
Single-GPU short-witness equality does not remove these limits.

## Measured results

Hardware: Windows 11, Python 3.12.10, PyTorch 2.13.0+cu130, one RTX 4060 Ti
(approximately 8 GiB). Gloo and FSDP2 are available; NCCL and Triton are not.
Maximum actual CUDA world size is **1**; real CPU Gloo world size is **2**.

The 12 original historical cases (6 updates each) passed byte equality against
e49 for proposals, actual deltas, optimizer moments, norms, clipped gradients,
jets, minima, scores, alpha, parameters and logits. These comprise three rates
for each of exact/single, scalarized/single, streamed/single and exact/FSDP1.
The corrected portable torchrun/FSDP1 oracle gate also passed all three rates.
Canonical tied identity was checked before and after wrapping and after every
update. Independent NumPy recomputation of the clipping coefficient and scaled
gradients was exact in all recorded cases. Real CUDA multi-rank clipping remains
**unverified** despite the world1 and CPU collective checks.

All 12 shape/seed synthetic cases had identical minima, scores and alpha for all
four tested modes. Exact dedup and streaming also matched every coordinate.
Scalarization produced 1,016 tiny differences in the **masked target-versus-itself
entries only**, and zero recorded competitor-coordinate mismatches. This finite
observation does not certify its changed arithmetic on arbitrary inputs.

Maximum measured temporary increments across both seeds, **N=33, V=73**:

| Hidden width | Exact dedup | Scalarized candidate | Streaming candidate |
|---|---:|---:|---:|
| 4096 | 21.94 MiB | 7.44 MiB | 15.05 MiB |
| 8192 | 24.65 MiB | 15.00 MiB | 18.61 MiB |

All measured cases were below their frozen 64 MiB budget. These figures are not
whole-model memory or measurements at a natural-language vocabulary size.

### Capability comparison

56 isolated warmed runs completed: 20 calibration jobs and 36 independent-seed
jobs. No scale job hit OOM or the 300-second loop cap. Each formal row below uses
seeds 201/202/203 and the same strict 99% criterion; no training extends beyond
200 updates. Failure is assigned 201 **only for restricted-step aggregation**.
Wall time includes per-update evaluation, excludes model initialization/warmup.

| Scale / calibrated regime | Method | Reach99 | Mean restricted steps | Mean time to99, successful runs |
|---|---|---:|---:|---:|
| 70M SAFE | Native | 3/3 | 129.00 | 8.14 s |
| 70M SAFE | Fixed .25 | 0/3 | 201.00 | Not reached |
| 70M SAFE | Dynamic | 3/3 | 151.33 | 23.97 s |
| 70M SAFE | Tuned native | 3/3 | 112.00 | 7.12 s |
| 70M NEAR | Native | 3/3 | 185.00 | 11.71 s |
| 70M NEAR | Fixed .25 | 3/3 | 125.00 | 15.33 s |
| 70M NEAR | Dynamic | 3/3 | 171.67 | 27.30 s |
| 70M NEAR | Tuned native | 3/3 | 112.00 | 7.17 s |
| 160M SAFE | Native | 3/3 | 92.33 | 11.51 s |
| 160M SAFE | Fixed .25 | 2/3 | 193.00 | 48.91 s (2 successes only) |
| 160M SAFE | Dynamic | 3/3 | 100.67 | 31.52 s |
| 160M SAFE | Tuned native | 3/3 | 82.67 | 10.15 s |

70M SAFE LR is 3.7688918072220455e-6; NEAR LR is 6.030226891555273e-5.
160M SAFE LR is 3.125e-6. Tuned-native rates are 7.537783614444091e-6 and
6.25e-6 respectively, chosen by calibration steps before the formal seeds.

The 70M NEAR label was established by the calibration seed, **not** by failures
in every formal seed: all three independent native runs also reached99 within
200. Dynamic beat same-LR native in steps in all three NEAR cases, but lost to
fixed .25 in steps and time in all three. At SAFE rates dynamic beat excessive
fixed shrinking, but not native/tuned-native. Neither scale demonstrated a
time-to-capability advantage over native/tuned-native in the measured regimes.

70M SEVERE and 160M NEAR/SEVERE were **NOT_ESTABLISHED**. Since the frozen NEAR
dynamic-benefit trigger failed, Phase3 is **NOT_RUN**. No cadence/probe Pareto
frontier or production preset is recommended. K=5/full bank remains the tested
reference, not an empirically selected optimum.

### Costs and bottlenecks

Measured single-rank persistent bytes:

| Component | 71,405,312 parameters | 163,633,152 parameters |
|---|---:|---:|
| Parameters | 285,621,248 | 654,532,608 |
| Gradients | 285,621,248 | 654,532,608 |
| Optimizer state | 571,242,796 | 1,309,065,540 |
| Candidate | 285,621,248 | 654,532,608 |
| Delta | 285,621,248 | 654,532,608 |

Candidate+delta are **544.78 MiB / 1,248.42 MiB**. Maximum formal dynamic peak
allocation was **1.7625 / 3.8706 GiB**, with reservation **1.9668 / 3.9609 GiB**.
These are single-GPU measurements, not measurements of a 1/R multi-GPU saving.
The configured largest transformer block contains 22.69 / 48.01 MiB of FP32
parameters respectively (computed from layer shapes). This is not a measured
multi-rank peak: receiving parameters, packed/gathered/unpacked delta buffers and
directional activations coexist. Actual tiny-FSDP stage events are in
`verification/BLOCKWISE_LIFETIME_TRACE.json`.

Across formal runs, median dynamic decision updates cost about .281 s (70M) and
.495 s (160M); ordinary dynamic updates cost .096-.099 s and .213 s. Evaluation
is additional to these per-update figures but included in time-to-capability.

A separate synchronizing diagnostic at seed701/update6 measured approximately:

| Component | 70M | 160M |
|---|---:|---:|
| Base forward + backward | 27.77 ms | 49.80 ms |
| Proposal including optimizer | 61.53 ms | 147.05 ms |
| Optimizer alone, included above | 8.45 ms | 18.82 ms |
| Directional jet + materialization | 182.96 ms | 268.35 ms |
| Boundary + selector | 7.98 ms | 7.85 ms |
| Evaluation | 21.31 ms | 41.21 ms |

These are instrumented samples, not a statistically stable fused-kernel
benchmark. They locate substantial overhead in bounded parameter swapping and
directional propagation, not just boundary evaluation. Isolated directional
workspace was **NOT_MEASURED**; aggregate GPU peaks and stage lifetime traces are
available. Real multi-GPU wire bytes, FSDP native traffic and synchronization
time were **NOT_MEASURED**. Controller network traffic on world1 is zero.

## Audits, preserved failures and final status

`verification/FINAL_AUDIT.json` validates source/old-file/tag identities and
recomputes outcomes from raw arrays. It is an **integrity verdict**, not a claim
that every candidate or performance hypothesis succeeded. `verify.py` replays
the audit read-only; raw NPZs are under the explicitly identified external output
directory, with hashes in the report. Additional report sources were added after
the main pre-outcome seal and are distinguished from sealed scientific sources.

Failures preserved without rewriting original results:

1. Attempt1 process exit in the first FSDP1 case; exact native cause unresolved.
   Original source snapshot and receipts remain in external attempt-1. Attempt2
   removed unnecessary singleton collectives and reran all gates.
2. Default Windows TCP/libuv rendezvous failed, including a configuration retry.
   FileStore with disabled shared TCP rendezvous subsequently worked.
3. The added portable oracle harness initially passed LR as cached alpha. Its
   three mismatch receipts and original source remain preserved. A separately
   sealed two-line harness correction reran all three rates to BYTE_EXACT.
   Production runtime and the 56 scale jobs were unaffected.
4. Near-regime dynamic superiority to fixed shrinking was not supported; absent
   regimes and the skipped Phase3 remain explicit negative/absent findings.

Final labels: scientific reference **PASS**; canonical identity **PASS**;
global clipping / exact distributed boundary **EMULATED_ONLY** (world1 CUDA plus
real CPU2 collectives); stage-packed FSDP **WORLD_SIZE1_ONLY**; scalarized and
streaming **CANDIDATE**; real multi-GPU **NOT_TESTED**; 70M advantage transport
**MIXED**; 160M near-boundary advantage **NOT_ESTABLISHED**; sparse control
**NOT_RUN**. Engineering status: **DISTRIBUTED_RESEARCH_PROTOTYPE**.
