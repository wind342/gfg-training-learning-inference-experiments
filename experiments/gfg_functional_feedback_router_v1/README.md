# Functional Feedback Router V1

`policy.py`: bounded deterministic routing and emergency-only alpha gate.
`runtime.py`: pre-step feedback selection, sequential gradient accumulation,
actual AdamW proposal, sparse boundary measurement, commit and post observations.
`tests.py`:24 tests, including real tiny-BERT CUDA execution.
FewRel integration/results are in `../gfg_governor_fewrel_feedback_router_v1/`.

The router acts BEFORE gradient formation. It does not predict or project a
parameter direction. Current data and selected old TRAIN memory form either the
current mean loss or an equal mixture of current/old means. AdamW retains its
moments and produces the update. Probe/test examples never enter backward.

Modes: old DEGRADED => RECOVERY; otherwise old PROTECTED level1 => BALANCE;
otherwise FORMATION with no feedback. Returning to FORMATION emits RELEASE,
without TTL or cooldown. Existing q10/hysteresis definitions remain unchanged.
State is copied before routing; current endpoint observations affect only future
feedback. Fresh receiving-state measurements may affect the current emergency
gate, without changing that step's feedback coalition.

At most8 distinct old groups contribute2 memory examples each. Degraded precedes
at-risk groups. One stable-anchor slot is reserved when available; remaining
vacancies are filled with stable anchors. Each tier and exemplar cursor rotates
deterministically. Fewer eligible groups means fewer than16 examples. Priority
does not promise service to lower risk tiers when the higher tier fills all risk
slots. Memory is fixed, append-only and only available after a task completes.

Gate candidates are(.125,.25,.5,1), never2. Native safe =>1. Otherwise minimize
current protected violation count, worst deficit, total deficit, then distance
to1. All-violating candidate sets emit NO_SAFE_ALPHA and commit the least-violating
allowed choice: this is explicitly NOT a guarantee of preserved capability.
The gate does not optimize formation/recovery gains or q10.

Full selected endpoint measurements update next-step history; pre-update full
measurements are never mislabeled as post-update states. Pre/post micro forwards
are recorded as different event phases. Post escalations request the next full
check. Endpoint auditing does not reselect the committed action. Existing
identity checks and fail-closed ownership paths are reused.

```text
python -m experiments.gfg_functional_feedback_router_v1.tests
```

No original v1-v4 code or research evidence is changed. This component has been
tested in a world1 FewRel short fork, not validated across arbitrary tasks.
