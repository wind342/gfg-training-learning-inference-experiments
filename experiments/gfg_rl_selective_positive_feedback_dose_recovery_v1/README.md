# RL-E06: selective positive-feedback dose, duration and recovery

This falsification-first experiment tests whether the concentration and
duration of exact positive feedback are causal coordinates of functional
support allocation in a shared learned policy, and whether redistributing
feedback reverses the resulting trade-off.

Four Boolean skills are first trained to exact mastery in one attention-free
GRU. Byte-identical parameter--AdamW states then enter balanced, mild, high,
exclusive and frozen feedback branches. Every positive consequence is the
exact terminal task result; no proxy reward, negative reward or entropy bonus
is used. The four updated branches receive the same number of episodes and
updates. Functional support is measured with all 16 component coalitions and
exact Shapley attribution of identity-aligned correct-action margins.

Two recovery branches fork from the exact exclusive-feedback state at update
800. One restores balanced feedback; the other trains only the three
previously unreinforced skills. This separates general recovery from the
specific trade-off created by redirecting feedback.

The frozen protocol and decision gates are in `PROTOCOL_FREEZE.md` and
`MODEL_CONTRACT.json`; `CONTRACT_FREEZE.json` seals the formal implementation.

## Outcome

All preregistered gates passed in all 12 retained formal seeds. Increasing
feedback concentration showed a strictly positive association with reinforced
skill support share and a strictly negative association with the other skills'
mean margin in 12/12 seeds. Exclusive feedback increased the reinforced skill's
support share by 9.67 percentage points relative to balanced feedback and
reduced the other skills' mean accuracy by 39.06 percentage points. Balanced
recovery restored 38.54 percentage points while retaining the reinforced skill
at 100% accuracy. The independent checker fully re-executed all 12 formal seeds
and passed.

Run and independently check with:

```bash
python -m experiments.gfg_rl_selective_positive_feedback_dose_recovery_v1.runner --artifact-root <new-formal-output-directory> --device cpu --mode formal
python -m experiments.gfg_rl_selective_positive_feedback_dose_recovery_v1.independent_checker --formal-root <formal-output-directory> --audit-root <new-audit-output-directory> --device cpu
```

The conclusion is bounded to the executed shared finite GRU policy. It does not
assert that all reinforcement learning inevitably weakens unrelated abilities,
or that functional support is a conserved physical quantity.
