# Functional Feedback Router V1: fixed engineering evaluation

Base6fc8f88e9dcc0320ba0bae7599f5f87c4350c9c7. New directories only.
No changes to original code, data, results, model or frozen identities.

Generic router uses existing FORMING/PROTECTED/DEGRADED and hysteresis levels.
Old DEGRADED => RECOVERY; otherwise old PROTECTED level1 => BALANCE;
otherwise FORMATION/no feedback. Transition to FORMATION records RELEASE.
No TTL/cooldown. Checkpoint history is inherited, including old groups0 and4
already DEGRADED; there is no forced initial current-only phase.

Feedback: up to8 groups,2 examples each. DEGRADED precedes level1 risk. Reserve
one slot for a stable PROTECTED level0 anchor if one exists, filling spare slots
with additional anchors. Each tier rotates deterministically; each memory group
cycles through its ten exemplars. If fewer than8 eligible groups exist, use a
smaller feedback batch (never duplicate groups merely to fill16). Separate tier
rotation prevents starvation within a tier; strict priorities do not guarantee
service to a lower risk tier while higher-priority demand fills all risk slots.
Anchors retain a reserved slot. These are fixed V1 choices, not optimality claims.

Memory is append-only, ten TRAIN exemplars per completed old relation; no current
or future relation is eligible. Reuse the prior smoke's exact80-example memory.
Exclude encoded duplicates of all currently visible control/test items. Test
content is used only for duplicate exclusion/integrity, never scores or routing.
Memory identity is fixed; future-scope collision fails closed rather than quietly
replacing memory. All gradient sample IDs are checked against split/role pools.

Timing: router freezes selection from already existing history. Current pre-update
observations may trigger a full gate, but never reselect feedback. After commit,
selected endpoint audit/full or same-group post-micro observations update next-step
history. Newly escalated post events schedule the next full check. Full measurements
of receiving state are not mislabeled as committed state. No audit-driven redo.

Gradient: current loss alone, or(.5 current mean+.5 feedback mean), two sequential
backwards. Original AdamW LR5e-6, clipping1, betas(.9,.999), decay0; original BERT,
data, new batch order and seeds. No additional geometric optimizer. Gate evaluates
four alphas(.125,.25,.5,1) on the actually formed mixed AdamW proposal. Native-safe
=>alpha1. Otherwise lexicographic minimum protected violation count/worst/total,
then nearest1. Protected means current receiving-state group accuracy>=90%,
including currently capable groups with unresolved historical debt. No alpha2.
If all violate, commit the least violating allowed candidate with NO_SAFE_ALPHA;
this is not a safety guarantee. Gate does not optimize formation or recovery.

Observer: existing complete-group round robin, max full interval50, erosion.5,
rearm.75. Current-only path does not read memory exemplars. All kernel/source
identities, decisions, sampled memory cursors and actual endpoints are logged.

## Fixed60-step comparison

Common task1 checkpoint SHA:
bd41bcbd84f280d4de7d721eeee1d7a7bad1ccbb04ceb7d84d0403a6bdbcb676.
Task2 first60 updates, seed2021,960 identical new samples; evaluation offsets0/20/40/60.
A new-only native AdamW is newly run. B always-replay native uses the existing exact
60-step reference, with checked source/checkpoint/batch/memory identities. C router
uses the same fork. No test-based stopping.900s cap per newly executed arm.
Replay samples and time are measured; capability quality precedes exposure saving.

## Conditional3-task extension

Proceed only if60-step checks pass, router BOTH activates and releases, and final
new-test accuracy>=70% (fixed engineering gate; test never enters training control).
If unmet, stop at60 steps without tuning to force a release. This gate does not
assert effectiveness; report old/new trade-offs even if it passes.

For a fair matched extension reuse the common completed task1 prefix, then run
the full210 original updates of task2 and210 of task3 from that same checkpoint
for A native/B always-replay/C router. Do not extend the60-step trajectories or
reuse a differently trained task1 native checkpoint. This is a three-task sequence
with a shared frozen task1 prefix, not three independently trained prefixes or
router application during task1. Report prefix separately from measured420-update
continuation time. No ten-task run. Each continuation has3600s cap.
At task2 completion append ten exemplars for each newly old relation, preserve old
memory/cursors, expand control scope explicitly and fail on old-memory collisions.
Always-replay rotates up to8 old groups per update,2 examples/group, always active.
Metrics: task current/old/all, forgetting/BWT using common task1 scores; feedback
fraction/exposures, per-group counts, mode/release, gate/full counts, wall/peak.
No per-exposure scalar leaderboard; report resource diagnostics beside accuracy.

Preserve failures, including resource issues. Cache release before full measurement
is inherited from the previous documented workaround and included in wall. Raw
data/checkpoints stay external; compact reports/hashes may be committed. No sweep.
