# Distributed research prototype and bounded scale feasibility

Base e49cb97db7467fbda8fa9d06c8c4a2a945707d43, only this new directory.
Frozen paper commit 36dab5ce347dbbdac157ef23205f556606d18294 unchanged.
One Windows GPU, NCCL unavailable: EMULATED ownership is NOT real FSDP.

Before tests seal all sources, this protocol, old tracked hashes and hardware.
Pure-software repairs preserve failed receipts/sources/seals and use new destinations.
All competitors, four alphas, cadence5 and complete quadratic coordinates unchanged.

## Gates

Emulated logical ranks split each parameter's flat coordinates contiguously.
Each rank owns separate optimizer moments and two local control buffers; proxy
parameters reference its receiving slice of the SINGLE full backward model.
This is O(P/R) per virtual owner but O(P) aggregate on the one physical GPU.
Do not advertise single-device savings or measured network traffic.

Native fused AdamW advances each local moment recurrence. Candidate/before
exchange uses bounded 65536-element swap tiles to preserve bytes and DTensor
ownership, never inverse floating-point reconstruction of a native proposal.
Alpha scales only realized displacement. Block materialization gathers receiving
and actual direction for ONE named stage; release before next stage. No whole
controller model clone. Full backward model still exists in emulation.

Local readout logits are computed per vocabulary shard, target row/update/score
payloads are made available, then local exact margins reduce with MIN. Recompute
global median only AFTER reducing all per-target minima. Record target payload
separately from 4*N*8-byte MIN input; estimates are not actual wire bytes.
Different output GEMM shapes may change FP32 logits. Such differences are a valid
FAIL/PARTIAL outcome, not a reason to relabel numerical tolerance as byte identity.

First six actual updates of first historical assignment, LRs .001/.003/.01,
world sizes1/2/3: compare native candidate/delta/moments, block h/h'/h'', minima,
scores, alpha, post-state and logits against e49cb97. All 6 steps run for this
bounded gate, independent of early target crossing. Store raw gate tensors and
hash receipts. If byte gate fails, report difference and do not claim certified
distributed runtime or launch a long distributed trajectory. Single-GPU scale
smoke using the already-certified e49cb97 remains an independent allowed check.

Real FSDP2 ownership code uses DTensor.to_local and block unshard/reshard.
Probe native FSDP2 world-size1 in an isolated process on current backend; preserve
traceback on unsupported CUDA/Gloo. Multi-GPU FSDP remains NOT_TESTED. A separate
two-process CPU Gloo MIN test checks actual collectives only, not CUDA/FSDP.

## Scale checks, independent single GPU e49cb97

Same local nanoGPT family: (layers12,width704,heads11),
(layers13,width1024,heads16), vocab24, sequence3, biasFalse, dropout0, FP32.
These are about71M/164M controlled systems, not pretrained language models.
No downloads. Never change precision/cadence/bank to make an OOM disappear.

Task is modular addition over all529 inputs (a,b,separator23), final target(a+b)%23.
SHA256 ordering fixes212 control queries and317 distinct evaluation queries.
Training samples all529 queries: evaluation is controller-disjoint but NOT
training-disjoint. Do not call it held-out-example generalization. Batch size128,
deterministic seed-specific sampling. Token targets at first two positions -1.

Each scale: isolated six-update native/fixed.25/dynamic smoke from seed101,
LR .001*sqrt(64/width), retain capability outputs, timing, allocation and OOM.
This checks engineering feasibility, not scientific advantage.

If smoke succeeds, bounded regime calibration: seed101, native and fixed.25,
80-update cap, base LRs [.0001,.0003,.001,.003]*sqrt(64/width), constant LR,
stop when evaluation accuracy>=.99; optional 180s hard job cap records RESOURCE_LIMITED.
Safe=lowest LR where native reaches99; near-boundary=lowest higher LR where
native fails but fixed succeeds; severe=next higher LR with native failure and
fixed success. Missing regimes remain NOT_ESTABLISHED, never forced labels.
Only if all three exist, compare native/fixed.25/dynamic on independent seeds
201,202 at those frozen rates with identical80-update cap. No calibration on
dynamic performance. Report time-to-capability, reach and failures separately.
If regimes do not exist in the bounded calibration, report no transport verdict.

Independent audit reopens raw gate tensors/logits, checks MIN and selection,
source and old-file hashes, scale banks, capability crossing and raw receipts.
PASS of record integrity must not be confused with PASS of distributed equivalence.
