# Matched fused AdamW + cosine, with/without capability-boundary control

Base e5a7dc724d5ac7d99e62dc36264c8fc644edc083. Old files/evidence untouched.
First bridge the identity-step shortcut against the two previous 50/55-update
ledgers under their OLD optimizer and fixed LR: require exact decisions,
coordinates, parameters, evaluation and final optimizer state. Then perform a
NEW comparison; fused arithmetic and matched cosine change trajectories and must
not be described as the same old scientific result.

Both methods: same 12 historical nanoGPT weights, actual 200 ordered batches,
fresh optimizer state, all 212 control/evaluation targets, FP32 deterministic
CUDA, TF32 off, norm-1 gradient clipping, AdamW betas(.9,.98), eps1e-8,
weight_decay=0, foreach=False, fused=True. Decay factor at update t:
0.1 + 0.9*0.5*(1+cos(pi*(t-1)/199)). These are continuation runs, not new models.

Controller: same quadratic_complete prediction, all competitors, four alpha
candidates (.25,.5,.75,1), refresh every5, unchanged moments after amplitude
application. All derivative and hash checks retained. Cached alpha=1 steps skip
only pre-state capture and delta construction. Baseline directly applies native
AdamW. Both arms use the same packed post-state audit code and overhead; allocate
the same reusable buffers in both arms to keep the external checker identical.

Equal search budgets: LR .0003, .001, .003, .01, 12 histories, 2 arms =96 runs.
Stop at first >=99%, cap200; record first95 every step. Initially-at-threshold
cases have zero updates and are excluded from threshold-specific aggregate.
Failure costs200 updates and retains full measured cap runtime, not dropped.
Use capability thresholds, not loss-AUC, as primary outcomes.

Primary matched LR selection: for EACH held-out history, pool restricted first99
costs across BOTH methods on the OTHER11 histories; rank the FOUR common LR
configurations by mean first99 cost, then mean first95 cost, then index. Select
the SAME LR for BOTH arms on the held-out history. No within-history selection.
Preserve all four same-LR paired search outcomes so the shared choice is visible.
This symmetric common-LR rule differs from the older method-specific selection;
do not attribute all differences versus old tables solely to implementation.

After all selection is frozen, replay both selected arms three times per history,
alternating arm order. Require exact replay of logits, all coordinate hashes,
scores/alphas, post-state hashes and final optimizer identity. Time GPU-synchronized
updates plus construction separately from full checked wall (including evaluation
and hashes). Source loading, preloading replay artifacts, warmup and final artifact
writes excluded identically. Report tuning cost separately. No old/new timing splice.
Source/protocol hash seal precedes outcomes. Native outputs use a new E: evidence
directory; compact results and audit remain in this new experiment directory.
