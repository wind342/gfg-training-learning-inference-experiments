# Functional controller cadence v1

New prospective timing/threshold experiment, based on commit
706795db4a838fd1487d6ce278dec93b57b5f2e2. No old experiment or verdict is changed.

## Fixed comparison

Reuse all twelve entries, their starting states, 212 labeled control/evaluation
targets, and 200 ordered batches from the online-controller RUN_ASSIGNMENT.json.
Retain that experiment's protocol hash as the RNG schedule input, so native,
fixed and every-step controls must reproduce their old trajectories exactly.
This is continuation from historical states, not training from initialization.
The labeled control targets are also evaluated: no new held-out accuracy claim.

Five arms: native alpha=1; fixed alpha=.25; controller every 1, 2, or 5 steps.
For interval K select on steps 1, 1+K, 1+2K, ... using the UNCHANGED online
selector (four candidates .25/.5/.75/1, quadratic_complete coordinate arithmetic
and the existing lexicographic selection rule). Between these steps reuse ONLY
alpha. Always form a fresh native gradient, Adam moments, step and parameter
proposal using that arm's current state. Never reuse an old delta or scale
moments. Alpha=1 preserves native endpoint bytes. No candidate endpoint forward.
Record/seal selection before installing the scaled parameters or evaluating them.
Same pinned nanoGPT, CUDA float32, optimizer and deterministic settings as the
source experiment. All arms execute the full 200-step horizon; no exclusions.
Rotate five-arm execution order by run index. One measured execution per arm;
timing is machine-specific, not a precise population estimate.

## Outcomes decided before execution

Evaluate at 0,10,...,200, strict target-minus-largest-competitor margin >0.
Report per-run first OBSERVED 95% and 99% accuracy, corresponding cumulative
measured training+selection+application seconds, and observed subsequent drops
below each threshold. Already at threshold means step/time zero. Unreached
thresholds are retained as null/censored, not discarded. Also report restricted
time/step cost through the horizon (unreached charged full horizon), success
counts, paired run comparisons, final accuracy and descriptive pooled curves.
Pooled threshold crossings must be explicitly distinguished from every run
reaching the threshold. Sustained crossing means all later GRID observations
remain above threshold, not proof of stability between evaluations.

Training timing includes the native runtime and proposed-state snapshot;
application includes scaling/copy and synchronization; receiving snapshots and
batch setup are separately timed. Core time = training+selection+application.
Operational time also includes receiving snapshots and batch setup. Evaluation,
hashing and evidence I/O are separate; total wall time is also reported. CUDA
synchronization brackets timed compute. No energy claim without power sampling.

No new binary scientific verdict or retuned decision rule is introduced. The
practical question is whether interval 2 or 5 preserves threshold-step benefits
while reducing measured cost relative to the every-step controller AND native
AdamW. Failure or tradeoffs are reported. Original AUC NOT_SUPPORTED stands.

## Integrity and space use

Keep original code/artifacts/tag untouched. This new compact execution ledger
stores every pre/proposal/post full-state commitment, gradient and delta hashes,
batch/seed identity, selection seal, coordinate file, and evaluation logits.
It is a compact replay ledger, not a claim to retain every raw intermediate
tensor or a replacement for the original full GFG. Immutable initial sources
plus complete ordered actions permit full independent native replay. Save final
raw states. Chain step records by content SHA-256; seal code/protocol before
formal execution. Independent checking must recompute selector choices, replay
all 12000 native steps and all 4080 coordinate calculations, compare full state
hashes and evaluation bytes, and compare three controls to original outcomes.
No source data is read within the selector boundary. Synthetic warmup/self-tests
use no formal checkpoint. Interrupted/failed attempts are preserved, not erased.
