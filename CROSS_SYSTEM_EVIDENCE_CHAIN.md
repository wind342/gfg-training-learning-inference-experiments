# Cross-system training-learning and inference evidence chain

TL-G01, TL-G02 and INF-G01 form the original cumulative falsification chain.
TL-G03 and INF-G02 add a separately frozen routed-expert chain. They do
not replace the sealed nanoGPT experiments. They ask whether the same proposed
relations survive when architecture, modality, objective and optimizer are
changed, and whether the resulting learned states are subsequently used by
frozen inference.

## TL-G01 — discriminative vision with SGD momentum

TL-G01 changes the system from nanoGPT/text/Adam to
ResNet-18/CIFAR-100/SGD momentum. Three formal seeds retest receiving-state
conditioning, finite-amplitude nonlinear response, distributed support,
target-specific boundaries and held-out transport in the three primary
coordinate families. The frozen aggregate verdict is
`CROSS_SYSTEM_GENERALIZATION_SUPPORTED`.

## TL-G02 — generative diffusion with AdamW

TL-G02 changes the task again, to time-conditioned diffusion residual
prediction with a compact U-Net on CIFAR-10 under AdamW. Target identity is an
image--timestep--noise occurrence and the readout boundary is defined from
residual error rather than class correctness. Three formal seeds retest the
same mechanism relations and produce the frozen aggregate verdict
`CROSS_SYSTEM_GENERALIZATION_SUPPORTED`.

## INF-G01 — frozen projection in both learned systems

INF-G01 loads the exact six trained checkpoints established by TL-G01 and
TL-G02. With persistent model and optimizer state frozen, it tests exact repeat
inference, causal component recruitment, query-conditioned support profiles,
non-additive component combination, pre-learning rollback dependence and exact
restoration. All three ResNet seeds and all three diffusion seeds pass all nine
frozen criteria, producing the verdict
`CROSS_SYSTEM_FROZEN_PROJECTION_SUPPORTED`.

## TL-G03 and INF-G02 — routed expert organization

TL-G03 changes the organization again, to a deterministic top-2 Vision
Mixture-of-Experts classifier on CIFAR-100 under AdamW. Three formal seeds
retest receiving parameter state and Adam memory, finite-amplitude response,
distributed expert support, update-induced support and routing reallocation,
target boundaries and run-isolated coordinate transport. All seven frozen
training--learning tests pass, producing the verdict
`MOE_TRAINING_LEARNING_GENERALIZATION_SUPPORTED`.

INF-G02 then tests the exact trained MoE versions under frozen inference. All
three seeds preserve model and optimizer state exactly, route exactly two
experts per token, recruit all four experts across queries, change output under
expert gating and expert/router rollback, exhibit non-additive paired-expert
interactions and restore the trained result exactly. The verdict is
`MOE_FROZEN_INFERENCE_PROJECTION_SUPPORTED`, and the joint frozen verdict is
`MOE_UNIFIED_DYNAMICS_SUPPORTED`.

The implementation evaluates all expert branches and gives non-selected
experts exactly zero routing weight. The experiment therefore tests the exact
deterministic top-2 routed function and its expert organization, not sparse-
kernel throughput.

The INF-G01 package does not duplicate the six checkpoints. Its
`CHECKPOINT_AUTHORITIES.json` resolves each checkpoint to the TL-G01 or TL-G02
bundle by byte length and SHA-256 identity. The top-level archive verifier
checks those cross-bundle authorities before accepting INF-G01.

## Public adjudication boundary

The archive permits independent hash validation and result recomputation from
the complete formal machine records. It also includes the exact trained
checkpoints required to rerun the frozen-inference interventions. The standard
CIFAR-100 and CIFAR-10 datasets are not redistributed; a fresh training run
requires obtaining those third-party datasets and recreating the documented
PyTorch environment.
