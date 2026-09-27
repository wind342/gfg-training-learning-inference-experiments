# Task1 conflict-free timing smoke

Base: f24d37cd0d23c4d32c68bebefd3345f662016788.
One seed (2021), one task, one epoch (210 batches of 16), LR 5e-5.
No policy modifications, sweeps, retries based on results, or extra epochs.

Start from local pretrained BERT, freshly initialized existing 80-class linear
head, and fresh AdamW moments. Never load the already task1-trained checkpoint:
that would leak task1 learning into this measurement. BERT including newly
initialized entity marker embeddings is frozen/eval throughout. This differs
from the historical task1 bootstrap which fine-tuned BERT; do not compare their
accuracies as if they had identical training.

Two arms share identical initial parameters and current-task batches:
1. Native fused AdamW on the linear head.
2. Current actual-candidate + unchanged relation-floor projection + FP32 commit
   and exact recheck, with zero old relations and an empty protection bank.
   The existing 80 current validation monitors are observational only, as in
   the continuation harness. They cannot trigger protection.

Use original gradient clipping (1), AdamW betas (.9,.999), zero weight decay,
TF32 disabled, 8 seen classes. No replay, LR changes, BERT backward, or protection
of newly correct current-task examples. "Conflict-free" means no old-task
constraints, not absence of competition between the eight current classes.

Alternate arm execution order each batch to reduce order/warmup timing bias.
Warm up on training inputs and a throwaway head/optimizer before either arm.
Each arm independently runs BERT on every training batch; do not amortize its
cost using the other arm's output. Synchronized stage timings include input
transfer + BERT, head backward/AdamW, and controller commit/recheck. Per-step
hashing and full validation evaluation are outside timing, equally for both.
Record all 210 times, median and total; no timing significance claim.

Observe full clean task1 validation at steps 0,30,60,90,120,150,180,210.
First observed 90/95% is only on this grid, not exact first-passage time.
Read test only after both arms finish. No test-driven selection or tuning.

Require byte-identical head and optimizer states between arms at every step,
native candidate unchanged by control, zero projection and skip, frozen BERT
identity, and exact input train/validation/test disjointness. Preserve final
states, step ledger, hashes, final test predictions, and local input identities.
Only this new experiment directory may change.
