# RL-E05: selective positive feedback and functional-support concentration

This falsification-first experiment tests whether prolonged, correct, selective
positive feedback can concentrate shared distributed functional support around
the reinforced capability and crowd other already formed capabilities.  The
reward authority is the exact task rule; no proxy reward, negative feedback or
entropy bonus is used.

Four capabilities are first formed to exact mastery in one attention-free GRU.
Byte-identical parameter--AdamW states then enter selective, balanced and frozen
branches.  Functional support is measured by all 16 component-coalition gates
and exact Shapley attribution of identity-aligned target margins.  All 15
non-empty component-version rollback subsets are executed at the selective
endpoint.

The frozen protocol and thresholds are in `PROTOCOL_FREEZE.md` and
`MODEL_CONTRACT.json`.  Formal execution is sealed by `CONTRACT_FREEZE.json`.

## Outcome

The formal run retained all 12 seeds and the independent check passed.  Ten of
eleven preregistered gates passed; the composite status is `NOT_SUPPORTED` because
the frozen 0.03 temporal-precedence gate passed in only 2/12 seeds.  The narrower
concentration/crowding mechanism passed its behavioural, support, control and
component-version intervention tests.  See `RESULTS.md` and
`SCIENTIFIC_ASSESSMENT.md`; the post-hoc timing analysis is explicitly labelled
`DIAGNOSTIC_ONLY`.

Run and independently check with:

```bash
python -m experiments.gfg_rl_selective_positive_feedback_support_concentration_v1.runner --artifact-root <new-output-directory> --device cpu
python -m experiments.gfg_rl_selective_positive_feedback_support_concentration_v1.independent_checker --artifact-root <output-directory> --device cpu
```

The claim is deliberately bounded to the executed shared finite policy.  This
experiment does not assert that every reinforcement-learning system must
collapse.
