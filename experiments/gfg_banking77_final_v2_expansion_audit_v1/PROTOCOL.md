# Banking77 nonlinear Final V2: expansion-only boundary audit

Base 534130d8995fbf9f6cec176afb915f1a56845702; separate branch/directory.
Continue each verified task2 checkpoint (210 and 630 updates/task) through
tasks3-11. Never reset to task1 at a boundary. Reuse original task/label order,
seed2021, batch16, LR5e-5, inherited AdamW moments, frozen BERT features,
768-256-GELU-77 head, actual-update JVP predictor and the unchanged nonlinear
Jacobian-Hildreth cutter with corrected INNER_LIMIT actual-safe acceptance.
No replay, old CE backward, calibration, class/logit gate, scaling, bias
correction, protection change, tolerance/rho/EPS change, or test read.

Original task2 source is imported unchanged, verified against its frozen hashes
and BASE Git blobs. A new continuation harness generalizes ONLY task indexing
and control-bank selection. 7 classes/task; all arrived old algorithmic controls
only (10/class). Original K is computed by the production Bank AFTER expansion,
before the first new-task update; observer values never construct/override K.
Check that production K equals observed post-expansion counts, for audit only.

At each of 18 boundaries (T2->T3 through T10->T11 for each arm), use separate
scratch buffers to compute fixed256 full77 logits on the same old controls.
Take two slices from that one output for before/after class competition. Repeat
the full forward independently to check deterministic bytes. Preserve head,
shadow head, gradients, moments, counters, hyperparameters, module modes and
Python/NumPy/Torch/CUDA RNG identity. No backward/update/controller call inside
the observer. Save complete compact logits, IDs, labels and per-control strict
competition/tie attribution. Use margin>0 for controller correctness and record
argmax correctness separately. Pure expansion cannot produce wrong->correct;
any such event is an integrity failure, not an experimental benefit.

Record old/new seen, bank size, per-relation pre/post counts, losses, zero-count
relations, newly-zero relations, K-zero counts, strict flips, ties, and new-class
logit distribution. No predefined claim that future-head rows are stationary:
boundary logits show competition, not proof of what earlier parameter changes
created those logits. Shared hidden evolution is a possible history mechanism,
not a conclusion assumed from this observer.

Even catastrophic expansion does not stop or modify subsequent training. Only
execution/integrity/resource failures stop, with preserved records. Predictor
accuracy and controller diagnostics are recorded, not used to tune settings or
abort merely for a scientific negative result. Scheduled updates advance native
AdamW moments once even when controller rolls back; production per-task floor
must hold for executed updates. Task11 is trained as requested after its audit.

Preflight: parent authority PASS, input and BERT identity hashes, model-source
identity; strict-flip/tie/no-flip synthetic observer tests; actual task2 restart
counter/head identity; observer on/off first-step differential for both arms
from disposable task2 clones. Restore deterministic setup before formal run.

Compact per-step records retain proposal/executed/moment/endpoint identities,
prediction comparisons and actual cutter decisions; per-task endpoint checkpoint
and fixed K allow replay. Boundary observer records are independently recomputed
from each predecessor checkpoint with a separately written endpoint/readout.
Continuation is additionally reexecuted without observer calls, comparing every
scheduled update/head/moment/decision. No test evaluation in this task. Python
read allowlist guards evidence/model inputs; OS/native-library internals are
outside its scope. Old results stay untouched. No automatic tuning or retries.

Local create-only authority E:/gfg-evidence/banking77-final-v2-expansion-audit-v1.
Disk guard at task boundaries; only small head checkpoints, no BERT/cache copies.
Report per-arm nine-boundary tables and cumulative EVENTS (not unique controls
across boundaries), including maximum boundary/relation loss and relation-to-zero
events. Zero events means absence on these two tested trajectories, not universal
impossibility, nor proof that the predictor itself prevented prior failures.
