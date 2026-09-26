# Router V3: sample-count-normalized feedback dose only

Basecf6c0f30d0e35173771ab2fc52fa2474b1d940ee. Preserve all prior code/results.
Only the training microbatch mean coefficients change: n_i/sum(n_i), not1/2.
Current16+old0/4/6/16 => old weight0,1/5,3/11,1/2 respectively.
Sequential backwards, one live microbatch graph; no extra model/checkpoint.

Unchanged: V2 router state/coalition/memory/cursors, q10/rearm thresholds,
observer timing, model/dropout, seed-per-microbatch, optimizer/initial moments,
LR5e-6, clipping1, predictor, emergency gate and cache-release workaround.
Closed-loop observations may change later coalitions; do not force the V2
sample sequence or282 old exposures. This is a dose-policy ablation, not
an open-loop fixed-coalition causal comparison.

Before running: test sequential weighted gradient against concatenated sample
mean gradient with dropout disabled, including sizes16+0,16+4,16+6,16+16,
equal-batch regression to V2, real tiny BERT and AdamW state continuity.
Training dropout remains unchanged; tests permit ordinary floating-point error.

Run ONLY the same task1 checkpoint -> first60 task2 updates, seed2021,
960 identical new occurrences, control/test schedule0/20/40/60.900s limit,
create-only artifacts. No3-task run, sweep, post-result tuning or extra arms.
Compare hash-checked historical new-only native/fixed native replay/V2.
All fixed choices precede the run; preserve failure if one occurs.

Record old/new/all accuracy, actual feedback batch histogram/exposures,
effective old-loss weight histogram, modes/releases, alpha/gate, wall/peak,
selected endpoint audit and source/record hashes. Capability retention and
new formation take priority over sample or runtime savings. If this merely
shifts the trade-off back toward new capability, say so.

Raw output: E:/gfg-evidence/functional-feedback-router-v3. Commit+push a new
codex branch after checking, without changing main or frozen tags.
