# Cross-system frozen-inference projection protocol v1

Status: **FROZEN BEFORE FORMAL EXECUTION**.

## Scientific question

Does the relation established in nanoGPT—frozen inference calls and combines
query-conditioned distributed support formed during training without
persistently updating the learned state—transport independently to both new
cross-system training–learning systems?

The two arms are the already executed ResNet-18/CIFAR-100/SGD-momentum system
from TL-G01 and the DDPM-style U-Net/CIFAR-10/AdamW system from TL-G02. A joint
positive verdict requires every frozen test to pass in every retained seed of
both systems. No seed, query or component may be removed after its response is
read.

## Source authority and version identity

No new training is performed. Each arm loads the exact final checkpoint and
optimizer state produced by its source training experiment. The checkpoint
file is verified against the source manifest. The loaded model state is also
verified against the source run summary where that state hash is present.

The pre-learning component versions are reconstructed from the frozen source
seed and initialization recipe. Their complete state hash must equal the
`pre_state_sha256` recorded before the first actual training action in the
source experiment. A mismatch is an integrity failure, not an approximate
rollback.

## Frozen inference

For ResNet, a query is an identified CIFAR-100 image and inference is its
frozen forward pass. For diffusion, a query is an identified noisy image and
diffusion timestep; the complete generative inference additionally performs a
deterministic 100-step DDIM-style projection from fixed noise.

Diffusion sample state changes during denoising. It is a transient state of the
current inference occurrence, not a persistent learned state. The model and
optimizer states must remain byte-identical before and after both one-step
functional projections and complete sampling.

## Component calls and support combination

Each arm retains the four support components registered by its source
training–learning experiment. Native forward hooks establish that all four
components are actually called and have nonzero outputs. All sixteen gate
coalitions are then executed on the fixed query set. Target margins are the
native class boundary for ResNet and the frozen residual-candidate boundary for
diffusion.

Exact four-player Shapley effects provide a query-level support profile. The
experiment tests:

1. at least two queries have support profiles separated by the frozen L1
   threshold;
2. at least one single gate changes the complete functional output;
3. at least one component pair has a non-additive interaction above the
   system-specific frozen threshold.

## Training-version rollback

At the trained checkpoint, each registered component is replaced in turn by
that component's exact pre-learning version. All other trained parameters and
buffers remain fixed. ResNet records the change in complete logits. Diffusion
records both the one-step epsilon output and the complete deterministic sample.

After every intervention, the trained version is restored. The complete model
state hash and complete baseline output must be recovered exactly. Accuracy or
boundary correctness need not decrease for every rollback; all positive,
negative and zero effects are retained.

## Frozen gates

For every seed, the arm passes only if:

1. source checkpoint and pre-learning version identities are exact;
2. repeated inference is exact and model/optimizer state is unchanged;
3. every component is called with nonzero output;
4. gating changes the functional output;
5. support profiles are query-conditioned;
6. at least one pair interaction is non-additive;
7. at least one exact pre-learning component rollback changes the trained
   inference result;
8. every restoration is exact.

The verdicts are `CROSS_SYSTEM_FROZEN_PROJECTION_SUPPORTED`,
`CROSS_SYSTEM_FROZEN_PROJECTION_PARTIALLY_SUPPORTED`,
`CROSS_SYSTEM_FROZEN_PROJECTION_NOT_SUPPORTED` and `INTEGRITY_FAILURE`.

The strongest positive result establishes transport to the two executed
systems. It does not assert an unconditional law for every possible model or
inference procedure.
