# V2: repair anchor overfill only

Base41a9724e1d0f32a7ffbdf07fd360779823c26cac. Original V1 code/results unchanged.
V2 changes exactly stable-anchor allocation: at most ONE rotating stable group,
never fill remaining slots. Risk priority and per-tier/per-exemplar rotation,
state/rearm rules, gate, observations, optimizer and losses remain V1.
One risk group => two risk examples plus at most two anchor examples, not16.
Multiple genuine risk groups may still legitimately require a full16-example batch.
Group0's existing degraded state must NOT be cleared artificially.

Risk tiers remain degraded before at-risk. Reserve one of at most8 group slots
for an anchor when available. Lower risk tiers may wait under sustained high
degraded demand, exactly as V1. Each selected group supplies two examples.
Current16 and variable old batch use equal MEAN weighting, not equal per-example
weighting. Thus each selected old group receives a larger share when the coalition
is smaller. This is an intended consequence, not a reduction of old-loss weight.

Tests precede GPU execution: one risk among8 and80 old groups, two inherited risks,
anchor rotation, persistent real debt, group/example bounds and no memory read in
FORMATION. Runtime checks each coalition before backward. At10 updates, check
actual anchor bound, single-risk bound and data-role/log integrity before continuing.
Stop immediately on violations; do not finish a known-invalid allocation run.
Do NOT require no feedback or RELEASE while real risk persists.

Then complete at most60 identical task2 updates from the same task1 checkpoint,
seed2021/LR5e-6/960 new occurrences, old memory80 examples. Test schedule0/20/40/60;
no test-driven routing. Same900s limit. No three-task or ten-task experiment in this
fix. Native, fixed replay and V1 results are historical identity-checked references.
Judge old/new capability first; exposure savings alone are not success. In
particular, reduced maintenance may allow stable old groups to degrade: report it
without resetting state, modifying memory or tuning after results.

Raw output is create-only under E:/gfg-evidence/functional-feedback-router-v2.
Keep all failures. Wrapper reuses V1 training/runtime and records inherited source
hashes plus V2 sources. No new large checkpoint. Commit separately on a new branch.
