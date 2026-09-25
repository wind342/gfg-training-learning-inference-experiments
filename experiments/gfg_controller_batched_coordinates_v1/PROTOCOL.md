# Bounded implementation-equivalence and speed experiment

Base: 77333c8aa890aa1b406ff62086fb555d16eba16b. All old files remain unchanged.
Only target-wise coordinate aggregation is batched. Actual-update construction,
copies, model forward/JVPs, four candidate amplitudes, second-order terms,
every-five-update cadence, optimizer, evaluation and internal hash checks remain
unchanged. No scheduling change, reduced target set, new training task or tuning.

Generic core: any nonempty target batch, >=2 output classes, positive hidden
width; a linear readout with first/second hidden directional derivatives along
the realized update. This interface is not a claim that arbitrary architectures
or non-classification tasks have already been validated. nanoGPT is an adapter.

Before native runs, test multiple dimensions, tied logits, chunk sizes and
invalid inputs against a simple independent loop. Require exact arrays, scores
and selected amplitudes, not tolerance-based replacement of scientific results.

Use the first two lexically ordered initially-below-99% histories from the
previous benchmark's fixed selection table (reference reaches 99% at 50 / 55 steps).
Reuse their selected controller configurations, historical initial weights,
source RNG identities and actual batches. Fresh AdamW as in that benchmark.
Stop at first 99%, cap 60 updates. No retuning. One full warm-up for each
implementation then five paired repetitions, alternating order. Compare every
step to the frozen search ledger (coordinates, decisions, parameter hashes,
evaluation) and final optimizer/parameter identity. Any mismatch fails.

Time synchronized GPU updates including ALL old internal integrity checks and
optimizer construction. Separately report complete loop wall time including
evaluation and per-step comparison/hash checks, excluding final report writing
and source loading. Same checks for both arms. Report median per history,
paired ratios, and raw repetitions; avoid inferring large-model generalization.
Microbenchmark aggregation separately (input copying/derivative acquisition
excluded identically); this is NOT full training speed. Source files and protocol
are hash-sealed before native outcomes. No old evidence outputs are written.
