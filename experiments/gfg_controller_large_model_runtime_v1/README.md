# Architecture gate before 70M / 160M validation

Scientific baseline: `d17f4510adf46ebae4fc391382ca6be1134eec03`.
Frozen paper tag remains at `36dab5ce347dbbdac157ef23205f556606d18294`.
Only this new directory is added. No old protocols, results or evidence changed.

## What changed

1. Native fused AdamW proposes into separate candidate storage, then boundary
   adjudication chooses alpha, then parameters are committed. Native optimizer
   moments are unchanged and never rescaled. Actual rounded delta is preserved.
2. Two full parameter buffers replace five. Direction storage becomes the scaled
   endpoint only after its derivative/decision lifetime ends.
3. Exact target/competitor tiling retains every competitor and the original
   float64 calculation + float32 coordinate rounding. No top-k pruning or GEMM
   reassociation changes the scientific arithmetic.
4. `BlockwiseController` propagates h/h'/h'' through embeddings, decoder blocks
   and final normalization. It includes input curvature and mixed terms.
5. RUNTIME omits full coordinate retention and CPU parameter hashing. AUDIT and
   the external validation harness keep those checks. Alpha candidates, cadence,
   targets, all three second-order terms and evaluation criterion are unchanged.

## Results

The global-JVP/two-buffer bridge passed 72 trajectories and 3,460 controller
updates. Its 353 blockwise derivative comparisons were byte-identical.
The separately sealed integrated blockwise backend then passed another 72
trajectories and 3,460 updates, with another 353 exact derivative comparisons.
Each bridge independently executed 1,730 additional old-controller oracle steps.

These are repeated validations of the SAME 12 historical states at three learning
rates in two execution modes, not 144 independent new training histories.
Five states already met the stopping threshold: 60 of the 144 trajectories have
zero updates. All trajectories are retained, including 200-step failures.

Across both bridges: native candidate, h/h'/h'', boundary coordinates, all
competitor minima, scores, selected alpha, updated parameters, logits, capability
crossings and final optimizer identities matched the prior artifacts. No
tolerance-based replacement of byte identity was used. Independent audits PASS.

## Memory and timing scope

Persistent control allocation: **20P -> 8P bytes in FP32, 60% less in this category**.
This is not a 60% reduction in total GPU memory.

An isolated first-refresh diagnostic on the existing 100,352-parameter model
(three fresh-state repetitions per mode) measured median CUDA peaks:

| Path | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|
| Native optimizer | 99,492,352 | 102,760,448 |
| Old controller | 101,499,392 | 109,051,904 |
| New global-JVP controller | 100,295,168 | 109,051,904 |
| New blockwise controller | 100,295,168 | 109,051,904 |

The new blockwise peak saves 1,204,224 bytes versus the old controller here.
Single-refresh timing is diagnostic only, not a warmed benchmark or a
time-to-capability comparison. No net speed advantage over AdamW/fixed shrinkage
is claimed by this architecture gate.

FP32 parameter+gradient+Adam moments+new control buffers have a lower-bound
24P-byte footprint: 1.68 GB at 70M and 3.84 GB at 160M (decimal GB). Activations,
directional workspaces, CUDA runtime and allocator reserve are additional.
These are formulas, **not measured large-model peaks or feasibility guarantees**.

## Limits and status

- PHASE1_RUNTIME_EQUIVALENCE: PASS, including integrated blockwise execution.
- PHASE2_SCALE_TRANSPORT: NOT_STARTED; no 70M/160M training in this gate.
- PHASE3_SPARSE_CONTROL: NOT_STARTED; no cadence/probe-bank changes.
- Largest validated model in this gate: 100,352 parameters, existing witnesses.
- Engineering status: research prototype with a verified runtime refactor.
- No new OOM or numerical mismatch occurred during these checks.
- Single GPU, eager FP32 execution only. No FSDP, tensor parallel, optimizer
  offload, CUDA graphs or arbitrary-decoder compatibility is claimed.
- Full pre-logits and head weights are still materialized on CPU. Only the
  target x competitor x hidden differences and coordinate retention are bounded.
- The specialized adapter is for the existing nanoGPT decoder; the proposal and
  tiled-boundary primitives do not assume 212 targets or a fixed model size.
- Buffer savings do not establish cross-scale scientific benefit. Large-model
  feasibility, dynamic-vs-fixed results and time-to-capability remain untested.

## Files and reproduction

`runtime.py`: proposal storage, tiled boundary core and reference-global backend.
`blockwise_runtime.py`: blockwise backend for subsequent scale tests.
`phase1.py`, `blockwise_validation.py`: sealed native replay bridges.
`audit.py`, `final_audit.py`: offline reconstruction/identity checks.
`memory.py`: separately sealed first-refresh allocation diagnostic.

Run from repository root with the existing CUDA/dependency environment:

```text
python -m experiments.gfg_controller_large_model_runtime_v1.phase1
python -m experiments.gfg_controller_large_model_runtime_v1.audit
python -m experiments.gfg_controller_large_model_runtime_v1.blockwise_validation
python -m experiments.gfg_controller_large_model_runtime_v1.memory
python -m experiments.gfg_controller_large_model_runtime_v1.final_audit
```

Runners intentionally refuse to overwrite existing seals/results. Reproduction
requires separate output destinations and retained prior seals, not deletion of
evidence. The published source is a companion to the existing dependency/data
archives, not a standalone installer.

Raw hashes/logits/coordinates/hidden arrays remain under:
`E:/gfg-evidence/controller-large-model-runtime-phase1-v1` and
`E:/gfg-evidence/controller-large-model-runtime-blockwise-v1`.
Both `AUDIT.json` files contain the full raw artifact manifests. Git contains
code, protocols, seals and compact audit/results only, not raw data/checkpoints.
