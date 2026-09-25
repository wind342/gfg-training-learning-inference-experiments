# Runtime v2: pre-outcome contract

Base ec891393178e52ab5f0e5e2571e0f3880dfa7e13; science d17f451;
single-GPU oracle e49cb97. Only this directory may change. Old files and paper
tag 36dab5ce347dbbdac157ef23205f556606d18294 are read-only.

Seal all source, this protocol, environment and old tracked identities before
formal outcomes. Software repairs preserve failed sources/receipts, make a new
seal and use a new attempt destination. No outcome-dependent tolerance changes.

## Scientific plane

FP32 native fused AdamW, betas .9/.98, weight_decay0, global L2 clipping1;
actual rounded candidate minus receiving; native moments; alpha only changes
displacement; alphas .25/.5/.75/1; complete second order coordinates; strict gap>0;
same count/median/closeness-to1 selector; cadence5 and full control bank reference.
All competitors retained. Scalarization and hidden-streaming are CANDIDATES,
not automatically exact. Streaming PyTorch still writes CHUNK intermediates to
HBM; it is not a fused FlashAttention-class CUDA kernel.

## Architecture scope

Canonical identity established before wrapping; true tied Parameter aliases share
one update and one moment state. Distinct Parameter views sharing storage rejected.
Only one-dimensional FSDP Shard(0) is supported. Boundary vocabulary partition
uses those same owners and identical ordered probes. No claim of arbitrary 2-D
DP/TP composition. Replica norm double-counting is prevented by rejecting unsupported
placements. Standard DTensor clip_grad_norm is checked against independent norms.
Native training batches in a real multi-rank launcher must partition one fixed
global batch; no duplicated batch / double reduction equivalence claims.

Packed stage deltas: stable canonical ordering, deterministic uneven shard lengths,
padding, one data all-gather per stage, explicit unpack workspace. Root tied
embedding group is counted in full, even if a substage uses fewer parameters.
META functional stage templates avoid entering FSDP hooks during nested JVP.
No full model materialization for controller execution. Audit may materialize
state identities and is excluded from runtime memory/performance claims.

## Gates and numerical policy

First historical witness, 6 updates x LRs .001/.003/.01. Compare exact, scalarized,
streaming against original e49: proposals, delta, optimizer moments, gradients,
norm, h/h'/h'', minima, scores, alpha, poststate, logits. Exact means byte equality,
not a posterior tolerance. Float numerical diagnostics report max_abs and counts;
no tolerances authorize a default-backend upgrade. Candidate alpha/trajectory
agreement on these bounded witnesses is NOT universal certification.

Real FSDP2 world1 runs all 3 rates through the same nanoGPT controller, including
tied readout and next steps. If >=2 CUDA devices AND NCCL exist, launch world2
through torchrun; otherwise preserve REAL_MULTIGPU_NOT_TESTED.
CPU Gloo world2 tests real stage packing, target routing, MIN and disjoint-gradient
norm composition; no CUDA-memory or CUDA-timing claims from CPU tests.

Synthetic boundary shapes: H64/512/1024/2048/4096/8192, N33,V73, seeds111/222,
with random and near-zero output margins. Additional uneven-tail V7,N5 case.
Compare all coordinates/minima/scores/alpha against e49. Workspace budget64MiB,
free memory fraction .1, reserve16MiB; conservative estimate includes target
payload, row conversions and scratch. Record estimated and measured increments.
Do not tune tiles from observed timing. Any measured budget miss is a failure.

## Scale regime discovery

Reuse v1 71,405,312 / 163,633,152 parameter configurations, vocabulary24, sequence3,
FP32 batch128. Same SHA-fixed 212 control /317 evaluation queries; training includes
all529. Controller-disjoint evaluation is NOT held-out-example generalization.
Seed101 calibration, native and fixed.25; L0 is v1 minimum LR; grid
[.125,.25,.5,1,2]*L0. Constant rate, 200-update cap, threshold99 unchanged.
Each isolated job has 300-second training-loop cap, checked after complete updates;
RESOURCE_LIMITED is censored, never treated as completed200 or theory failure.
Warmup2 updates on a throwaway same-shape model/arm, delete it, rebuild exact initial
model/optimizer before timing. Full evaluation each actual update is timed.

SAFE=lowest grid rate native reaches99. NEAR=lowest higher rate native fails and
fixed reaches99. SEVERE=next higher qualifying rate. These are operational
budget-defined groups, not proofs of dynamical instability. Missing groups stay
NOT_ESTABLISHED. Run each identified group independently; do NOT require all3.

Formal seeds201/202/203: native, fixed.25, dynamic exact at each available group's
rate, plus tuned-native at calibration's fastest step-to99 native rate (lower rate
tie-break). Baseline tuning uses calibration only, same grid; fixed.25 is preselected
from v1, not retrospectively best. If no native reaches99, no safe/near group is
invented. Best native baseline is included to distinguish fixed effective LR from
state-conditioned improvement. Record steps/tokens/wall-to99, reach, restricted
steps (fail=201), alpha distribution, complete output margins and memory.

Do not run v2 scale dynamic if its exact single-GPU gate fails. Native/fixed
calibration can still run independently. No transport claim from smoke or initialization.

## Conditional Phase3, a new policy configuration experiment

Trigger only if a NEAR formal group dynamic has more reaches than fixed, or equal
reaches and lower paired mean restricted steps over all3 seeds. This is a benefit
gate, not automatically a time-to-capability advantage.
Pick the smaller qualifying scale. Seeds301/302, full evaluation always. Reference
K5/fullN; K10/20/50/fullN; K5 at N/2,N/4,N/8 using subset seeds11/29. Fractions floor,
minimum1. These policies may change decisions and are not execution-equivalent.
Preserve the full non-dominated single-variable set without arbitrary weighted
score. Select best cadence and two best subsets by reach, restricted steps, then
wall (lexicographic) to construct at most two cadence/subset combinations. Freeze these
in a separate selection receipt BEFORE their confirmation. Confirmation seeds401/402/403
include reference and selected candidates; only confirmed non-worse reach and better
wall permit a ProductionPreset. Otherwise NO_RECOMMENDED_PRESET.

## Audit and failures

Reopen raw arrays, verify hashes and numerical identities, regenerate outcomes.
Reports cannot self-certify. Raw artifacts outside repo; no checkpoints uploaded.
Every failed gate/backend/compile/OOM retained. Runtime unavailable != scientific
refutation. Freeze vs hardware limitation vs no observed benefit are distinct.
