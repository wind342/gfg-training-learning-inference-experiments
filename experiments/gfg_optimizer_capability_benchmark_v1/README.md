# Capability-first optimizer comparison

Completed results: [Chinese summary](SUMMARY_ZH.md), [results](RESULTS.md),
[per-history outcomes and costs](PER_HISTORY_AND_COSTS.md).

This is a separate experimental extension; it does not alter any frozen paper
experiment, tag, bundle, or prior verdict. The model has 100,352 parameters.

Five arms: cosine-scheduled AdamW, official Sophia-G, official classic Distributed
Shampoo with Adam grafting, official Armijo SLS, and the existing target-boundary
controller refreshed every five updates. Four fixed LR configurations each,
leave-one-history-out selection, exact first-95%/99% crossings, and three native
timing replays of each selected trajectory. Stop at first 99%, maximum 200.

Read **PROTOCOL_FREEZE.md** before interpreting results. In particular, historical
weights are shared but all optimizer statistics are freshly initialized, and
weight decay is zero across methods. This differs from the original Adam-history
continuation; SOURCE_BRIDGE_CHECK.json separately confirms that the unmodified
original controller still reproduces its old trajectory. Old results are not
silently substituted into this comparison.

## Dependencies and reproduction

Use the existing CUDA Python environment (measured: Python/PyTorch identities in
FREEZE_SEAL.json). No dependency upgrade is needed. The local official checkouts
are ignored under runtime/, not copied into the public source tree:

| Directory | Official repository | Pinned commit |
|---|---|---|
| runtime/Sophia | https://github.com/Liuhong99/Sophia | a7e157229b71d58cf995d32854f1be15c265b350 |
| runtime/sls | https://github.com/IssamLaradji/sls | 19346532fc45956e025461bfad516796736a4ec0 |
| runtime/optimizers | https://github.com/facebookresearch/optimizers | d24a1491797e5bf2c2985d8789f3a28293d87a6f |

Sophia has an MIT license file, Meta optimizers a BSD license file. SLS declares
MIT in setup.py but has no root LICENSE file in this checkout. Therefore this
extension references and locally imports that official checkout; it does not
redistribute its source. All imported Python files are hash-pinned in the seal.
SLS's wall-clock seed call is addressed by a per-occurrence deterministic seed
during execution; optimizer logic remains the official implementation.

From the repository root:

```powershell
E:/gfg-python/torch-cu130/Scripts/python.exe -m experiments.gfg_optimizer_capability_benchmark_v1.benchmark search
E:/gfg-python/torch-cu130/Scripts/python.exe -m experiments.gfg_optimizer_capability_benchmark_v1.timing_cached
E:/gfg-python/torch-cu130/Scripts/python.exe -m experiments.gfg_optimizer_capability_benchmark_v1.audit
E:/gfg-python/torch-cu130/Scripts/python.exe -m experiments.gfg_optimizer_capability_benchmark_v1.cost_accounting
```

The one-time freeze has already happened. Do not overwrite sealed output.
Raw execution records: E:/gfg-evidence/optimizer-capability-benchmark-v1.
Source artifact paths are inherited from the unchanged online-controller common
module and remain read-only. Completed trajectories can be skipped on resume;
partial failed trajectories are preserved and require explicit diagnosis.

`timing_cached.py` is an I/O-only staging wrapper added after search began. It
calls the unchanged frozen LOO selection and training functions, caching only
crossing fields to avoid decoding each full trajectory eleven times. Full/compact
crossings must match for every loaded trajectory, and its own source hash is
sealed before timing. Running `benchmark timing` is the equivalent uncached path.
The final independent audit reads complete original trajectories. The separate
post-execution `cost_accounting.py` reports tuning cost, per-history results and
evidence-inclusive wall time; neither helper changes optimizer or selector logic.

## Interpretation

- The same labeled targets control and evaluate the controller. This is NOT a
  new held-out-target generalization claim.
- LOO tuning excludes the evaluated history's outcomes, but this still uses the
  same historically studied twelve runs. Four LR configurations are not exhaustive.
- Cold optimizer statistics and zero weight decay make this a specific optimizer
  switch comparison, not a measurement of unchanged historical receiving dynamics.
- Shampoo uses Adam grafting until step 10, then refreshes matrix roots every 10
  steps. Success before step 10 cannot be attributed to matrix preconditioning.
- SLS uses its native unclipped Armijo gradient; others use norm clipping 1.
- Timings include all method-specific computation. The controller path also
  performs an internal nonmutation hash check, so this is the measured audited
  implementation, not a best-case optimized controller kernel.
- Already-at-target histories are separate, not counted as successful learning.
- First hit is not a statement of sustained accuracy after the stopping point.
- Compact replay ledgers are NOT a claim to store every raw intermediate GFG
  tensor. Hashes, source identities, decisions, coordinates and final states
  support native re-execution and boundary auditing.
