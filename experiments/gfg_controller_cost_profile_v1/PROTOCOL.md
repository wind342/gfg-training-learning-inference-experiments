# Small cost-attribution pilot

Base: ada27cd1dbd1f30cab6769cf93c967d2bdfcdf5c. Add files only in this directory.
Reuse the two previous 50/55-update histories and their already selected learning
rates. Use the batched controller, unchanged target set, actual updates, second
order terms, alpha rule, five-step cadence and complete verification.

Compare an instrumented copy of the update/coordinate execution to the unchanged
batched reference. Require exact coordinates, decision scores/alphas, every
post-update parameter identity and evaluation logits, and final optimizer state
against the original frozen benchmark ledger. Stop on first >=99%; cap 60.

Measure ordinary update, hidden first directional derivative, nested second
directional derivative, target readout forward, coordinate aggregation, alpha
selection, transfers/snapshots, CPU delta/application, internal hash checking,
external evaluation and external hash/ledger comparison separately. Synchronize
CUDA before and after each non-overlapping segment. CPU timings include Python
overhead inside each segment; transfers include packing/casting where stated.
These are wall-time segments, not pure GPU-kernel times. No double-counted nested
timers. Added synchronization can perturb runtime, so also measure unchanged
non-segmented reference runs with the same outside checks.

One full warm-up for each implementation, then three paired repetitions per
history with alternating order. Record all raw stage times and total update /
checked-wall times. Report synchronization/instrumentation overhead rather than
claiming exact uninstrumented attribution. Exclude source loading and report IO.
No power measurement, tuning, optimizer redesign or performance optimization.
All old sources, ledgers and frozen identities remain unchanged.
