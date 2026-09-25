# Shape-independent batched boundary-coordinate pilot

## Outcome

Exactness **PASS**. Whole-training acceleration **not established** by this
small pilot: the first history was slower and the second faster, with noisy
paired repetitions. The isolated aggregation kernel is faster, but this does
not establish a reliable end-to-end win. This is an implementation experiment,
not a change to the theory, predictor or frozen experimental verdicts.

Each row is the median of five runs per implementation on RTX 4060 Ti 8GB.
Stop at the first >=99% target accuracy; both implementations take the same
number of updates. No training after first attainment.

| Historical entry suffix | Updates | Original compute | Batched compute | Compute time reduction | Original checked wall | Batched checked wall |
|---|---:|---:|---:|---:|---:|---:|
| 481b86f81d58d496a687 | 50 | 0.9953 s | 1.0552 s | -6.01% | 1.1543 s | 1.2180 s |
| 786d0a3628f6f791399f | 55 | 1.1441 s | 1.0449 s | +8.67% | 1.3234 s | 1.2059 s |

The raw nine-repetition isolated-kernel timings and all five paired complete
trajectory timings are in `verification_v2/RESULTS.json`. These two histories
are not evidence of general performance across tasks or model scales.

## What changed

- `kernel.py` accepts logits, target IDs, linear-readout weights and actual
  deltas, and hidden directional derivatives. Batch size, number of classes
  and hidden width come from inputs, not the modular-addition experiment.
- Competition includes every non-target output class; none are discarded.
- `nanogpt_adapter.py` supplies these quantities from the existing nanoGPT
  model. Other architectures need their own correct adapter; their efficacy
  has not been tested here. A linear-readout boundary is the explicit contract.
- Per-target matrix-vector operations are batched in bounded chunks. Precision
  conversions, quadratic terms, endpoint addition order and selection rule
  remain unchanged. The frozen selector is imported unchanged for selection.
- CPU/GPU copies and internal hashes were deliberately not removed: this pilot
  isolates batching and does not count removed verification as acceleration.

## Verification

Five shape tests cover target counts 1/7/53/212/257, class counts 2/5/11/24/37,
and widths 3/17/32/64/19, with tied scores and three chunk sizes. These are
interface/equivalence checks, not additional scientific model experiments.

Twenty timed trajectories (1,050 updates) plus four warm-up trajectories
reproduce the two previous frozen search ledgers exactly. The runner compares
coordinate hashes, scores and selected amplitudes, all post-update parameter
hashes and evaluation logits, and final parameter/optimizer state hashes.
`audit.py` independently checks persisted per-step identities/accuracy/amplitudes,
final identities, source hashes, summary arithmetic and unchanged frozen tag.

Two preliminary test-harness failures are disclosed in `ATTEMPT_01.md` and
`ATTEMPT_02.md`; their seals remain. Neither kernel nor adapter was changed
after the initial seal. Only `verification_v2` is the completed result.

## Reproduce

From this worktree root, using the existing CUDA environment:

```powershell
& E:/gfg-python/torch-cu130/Scripts/python.exe -m experiments.gfg_controller_batched_coordinates_v1.run --self-test
& E:/gfg-python/torch-cu130/Scripts/python.exe -m experiments.gfg_controller_batched_coordinates_v1.run --receipt-dir E:/gfg-evidence/controller-batched-new-run
```

The new receipt destination must be unused. Source history and runtime
dependencies are read from the previous benchmark's existing local paths;
there is no installation, remote write, source-data rewrite or tag movement.
The NumPy-only generic core does not depend on these local experiment paths.

The next potential bottlenecks are copies and derivative acquisition, but this
pilot does not profile their separate shares or claim a speedup from changing
them. No further controller changes were made.
