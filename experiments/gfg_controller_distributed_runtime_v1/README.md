# Distributed exact-controller prototype v1

Engineering baseline: `e49cb97db7467fbda8fa9d06c8c4a2a945707d43`.
Science baseline remains `d17f4510adf46ebae4fc391382ca6be1134eec03`.
Only this new directory was added. Original evidence, main, and the annotated
paper tag pointing to `36dab5ce347dbbdac157ef23205f556606d18294` were not changed.

**Overall: PARTIAL distributed-runtime implementation/validation, not production
multi-GPU certification.** The bounded EMULATED equivalence gate passed; real
multi-GPU FSDP/TP execution remains NOT_TESTED. Single-GPU scale feasibility passed.
Dynamic-versus-fixed capability advantage at the larger scales is NOT_ASSESSED.

## What actually ran

- Windows, one RTX 4060 Ti (8 GiB), 32 GiB system RAM, PyTorch 2.13.0+cu130.
  Gloo available; NCCL unavailable. See the pre-run hardware/source seal.
- Logical shard counts 1, 2, 3, three learning rates, six updates each:
  **9 cases, 54 updates and 18 decisions**, each compared with the e49 oracle.
- Candidates, rounded deltas, native moments, h/h'/h'', all-competitor minima,
  scores, selected alpha, post-parameters and logits: **BYTE_EXACT in all cases**.
  This is a bounded gate, not a complete long-trajectory or all-model guarantee.
- Real FSDP2, world size 1, small linear network: local proposal restoration and
  block unshard/delta-materialize/reshard lifecycle passed. This is not a full
  FSDP nanoGPT controller test, a tied-embedding test, or multi-GPU certification.
- Real two-process **CPU** Gloo MIN reduction: exact on both ranks. This is not
  a CUDA/NCCL communication benchmark or integrated tensor-parallel model run.
- Two scale points, 22 isolated runs, **1,316 updates**, all COMPLETED, no OOM.

## Memory ownership and communication

Candidate and delta are per-owner shards. The tiny witness has 100,352 parameters.
Their combined persistent bytes per logical rank are:

| Logical ranks | Candidate + delta bytes per rank |
|---|---|
| 1 | 802,816 |
| 2 | 401,408; 401,408 |
| 3 | 267,576; 267,576; 267,664 |

Thus the *logical-owner* buffers scale approximately as 1/R. All virtual owners
share one full backward model on the same GPU, so aggregate storage does NOT
shrink with emulated world size. Gradients and optimizer moments follow the
same local coordinate ownership; measured byte receipts are in
`MEMORY_DISTRIBUTED.json`. Physical tiny-gate CUDA peaks include the oracle and
audit tensors and must not be called production per-rank memory.

The largest materialized receiving+delta block in the tiny witness is 394,240
bytes. Stage contexts release it before the next stage. Real FSDP ownership
supports unshard/reshard contexts, but only the isolated world-size-1 lifecycle
probe was exercised. The backward model is still full in the emulation.

Vocabulary boundaries are logically sharded and then reduced with MIN; all
competitors are retained, and the global median is taken AFTER per-target MIN.
For 212 probes and width 64:

- MIN input: **6,784 bytes per rank per decision** (4 x 212 x 8).
- Target rows, update rows and scores: **109,392 bytes of logical payload**;
  replicated recipient costs are additional and implementation-dependent.
- Host decision summary: 72 bytes.
- Emulation's actual network traffic: zero. No measured CUDA wire-byte claim.

There is no full-vocabulary gather in the runtime boundary. The external audit
option reconstructs shard logits on CPU only to save and verify evidence.
Target rows are not yet deduplicated. Boundary tile allocation is not separately
profiled: for the tiny full-vocabulary case the two FP64 difference arrays alone
occupy 5,210,112 logical bytes; this is not a measured allocator peak.

## 70M / 160M feasibility

These are newly initialized controlled nanoGPT-family systems, not pretrained
language models: vocabulary 24, sequence length 3, FP32, batch 128, dropout zero.
The predeclared SHA-ordered control bank has 212 queries and the evaluation bank
317 disjoint queries. Training samples all 529 possible inputs. Therefore the
evaluation is controller-disjoint, NOT held-out-example generalization.

The scale tests use the certified e49 single-GPU controller, not the new emulated
or real FSDP runtime. There is no extrapolation to long-context language training.

| Item | 70M point | 160M point |
|---|---:|---:|
| Actual parameters | 71,405,312 | 163,633,152 |
| Native/fixed/dynamic six-update smoke | all PASS | all PASS |
| Native peak allocated | 1.163 GiB | 2.548 GiB |
| Dynamic peak allocated | 1.808 GiB | 3.962 GiB |
| Dynamic extra peak vs native | +55.44% | +55.47% |
| Candidate + delta buffers | 571,242,496 bytes | 1,309,065,216 bytes |
| Native six-update wall | 0.5333 s | 0.8323 s |
| Fixed .25 six-update wall | 0.6403 s | 1.0813 s |
| Dynamic six-update wall | 1.4283 s | 1.8271 s |
| Dynamic extra wall vs native | +167.84% | +119.52% |
| Dynamic extra wall vs fixed | +123.06% | +68.97% |

These are single cold six-update feasibility runs, including two decision steps
and per-step evaluation, excluding model construction and initial evaluation.
They are NOT repeated steady-state benchmarks, nor time-to-capability results.
No scientific advantage can be inferred from these overhead numbers alone.

## Regime finding and capability outcome

The frozen native/fixed-only calibration used four width-normalized rates,
80 updates, seed 101, fixed alpha .25, and a 99% threshold. Neither scale reached
99% in any calibration run. SAFE / NEAR_BOUNDARY / SEVERE were therefore
**NOT_ESTABLISHED**, and independent-seed formal dynamic comparisons were not run.
The cap was not increased after observing outcomes and the threshold was not lowered.

| Scale | Actual learning rate | Native accuracy at 80 | Fixed .25 accuracy at 80 |
|---|---:|---:|---:|
| 70M | 0.0000301511344578 | 4.73% | 26.81% |
| 70M | 0.0000904534033733 | 5.05% | 6.31% |
| 70M | 0.000301511344578 | 5.05% | 5.05% |
| 70M | 0.000904534033733 | 5.05% | 5.05% |
| 160M | 0.000025 | 19.56% | 97.48% |
| 160M | 0.000075 | 3.79% | 32.18% |
| 160M | 0.00025 | 5.05% | 5.36% |
| 160M | 0.00075 | 5.05% | 5.05% |

The 97.48% fixed-shrinkage result is preserved, but is not 99% and is not a dynamic
controller result. Dynamic-versus-fixed wins/losses and time-to-99 are **not
adjudicated**. Missing calibration regimes are not evidence disproving the old
near-boundary result or proving its transport to larger scales.

## Implementation and verification

- `storage.py`: local proposals, emulated ownership, real FSDP2 ownership binding.
- `adapters.py`: configurable stage interface and nanoGPT binding. No untested
  Llama/GPT adapter-support claim.
- `boundary.py`: local exact boundaries, global MIN primitive, emulated readout.
- `distributed_entry.py`: caller-provided models/data, EMULATED standalone entry;
  no per-step CPU model hashing. **Not an end-to-end torchrun/FSDP launcher.**
- `checks.py`, `backend_probe.py`, `scale.py`, `run_scale.py`: isolated evidence
  harnesses; these are distinct from the standalone runtime.
- `audit.py`: sealed before outcomes; rebuilds proposals/moments/state identities,
  directional comparisons, all-competitor minima, selector, logits/accuracy,
  bank membership and regime selection from raw artifacts.
- `verify_existing.py`: post-run read-only wrapper of the frozen audit; requires
  recomputed JSON to equal the archived receipts, never overwrites them.

Raw artifacts were written outside the repository to:
`E:/gfg-evidence/controller-distributed-runtime-v1`.
They are hash-indexed by `FINAL_AUDIT.json`; weights/raw arrays were not committed.
Full local recheck (requires those raw artifacts and existing frozen dependencies):

```powershell
E:\gfg-python\torch-cu130\Scripts\python.exe -m experiments.gfg_controller_distributed_runtime_v1.verify_existing
```

The original `checks`/`run_scale` commands are one-shot protocol runs with exclusive
output creation, not commands to overwrite archived evidence. The GitHub summary
alone does not replace a raw-artifact recheck.

**Audit integrity: PASS. Bounded EMULATED equivalence: PASS. Overall integrated
distributed controller: PARTIAL. Multi-GPU CUDA FSDP/TP: NOT_TESTED. Scale smoke:
PASS. Large-scale time-to-capability advantage: NOT_ASSESSED.**
