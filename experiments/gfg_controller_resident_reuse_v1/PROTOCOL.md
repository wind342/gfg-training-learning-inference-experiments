# Combined equivalent implementation pilot

Base: 689140d90fe7508392e9fcd65b7c9da44352c72a. Only add this experiment.
Reference is the previous batched controller. Combined implementation:
1. retain h as auxiliary output of the nested JVP yielding h' and h'';
2. resident reusable before/delta/proposal/scaled/audit GPU slabs, grouped by
   device/dtype, batched separate copy/subtract/multiply/add operations;
3. packed CPU snapshots for hashing, reusing a post-audit snapshot only when
   alpha=1 and no subsequent parameter mutation occurs;
4. existing shape-independent quadratic boundary core.

All targets, all competitors, both derivative orders, floating storage rounding,
four alphas and their selection rule, cadence 5, native AdamW moments and update
are unchanged. AdamW remains foreach=False/fused=False in BOTH arms. No loss of
checks, decreased precision, approximate derivative, target subsampling or
schedule change. Exact equality is mandatory; a mismatch is a failure to report,
not permission to loosen tolerance. New generic buffer/derivative code is not a
claim of scientific efficacy on an untested architecture.

Use precisely the same two histories and learning rates as the batched pilot
(first attainment of 99% at updates 50 and 55). Stop at first attainment, cap60.
No tuning. CPU/GPU buffer tests across dimensions, exact analytic derivative
tests and historical native replay precede timing. One full warm-up per arm,
then five paired repetitions in alternating order. Ten timed trajectories per
history, twenty in total. This is not a rerun of the AdamW optimizer comparison.

Synchronize only around complete updates, not individual inner stages. Include
optimizer, adapter and resident-buffer setup in compute and full checked-wall
time. Both arms compare coordinates, scores, selected alpha, every post-update
parameter identity, evaluation logits, and final parameter/optimizer hash to the
old frozen benchmark ledger. Include checks in full wall time. Source loading,
warm-up and final report IO excluded identically. Report all repetitions,
median time per history, memory, and paired ratios. Hash-seal code/protocol
before outcomes, preserve prior artifacts and tagged commit identities.
