# Functional controller cadence experiment

This independent extension asks whether selecting the functional amplitude less
often reduces measured cost to 95%/99% target accuracy. It does not modify the
prior online controller's frozen NOT_SUPPORTED AUC verdict.

Five arms share twelve immutable historical starting states and ordered batches:
native AdamW, fixed amplitude .25, and the existing controller refreshed every
1, 2, or 5 steps. Only the scalar amplitude is retained between selections;
each step still forms its own native gradient, moments and parameter update.

Read `PROTOCOL_FREEZE.md` for the pre-execution rules and `FREEZE_SEAL.json` for
code, protocol, environment and source identities. The source assignment and
unchanged implementation dependencies are in the sibling
`gfg_nanogpt_functional_online_controller_v1` directory.

From the repository root, using the same pinned CUDA/PyTorch environment:

```powershell
python -m experiments.gfg_nanogpt_functional_controller_cadence_v1.experiment run
python -m experiments.gfg_nanogpt_functional_controller_cadence_v1.experiment check
python -m experiments.gfg_nanogpt_functional_controller_cadence_v1.analysis
python -m experiments.gfg_nanogpt_functional_controller_cadence_v1.audit_summary
```

`freeze` was performed once, before the formal outcomes. Do not overwrite the
seal or existing output. An incomplete branch is deliberately not auto-resumed.

Local raw output: `E:/gfg-evidence/nanogpt-functional-controller-cadence-v1`.
Records are a compact hash-linked replay ledger, not a full raw-tensor GFG.
They contain durable per-step selections, coordinate arrays at refresh steps,
every state/gradient/update commitment, evaluation logits and final raw states.
The checker natively re-executes every branch and recalculates every refreshed
coordinate; it independently scores amplitude choices and requires exact matches
to the previous experiment for native/fixed/every-step controller trajectories.

The 212 labeled targets serve both control and evaluation. This is not a new
held-out generalization test or a from-initialization training comparison.
Threshold crossings and drops are observed only on a 10-step grid. Distinguish
pooled accuracy from all twelve individual runs reaching a threshold. Report
unreached runs rather than dropping them from the average. Single-run machine
timings are measurements, not universal speedup or electricity estimates.

Results: [RESULTS.md](RESULTS.md) and [RESULTS.json](RESULTS.json).
The separate `audit_summary.py` was added after execution for read-only checking
of saved artifacts, boundary correctness and summaries; it did not participate
in action selection or the frozen experiment. Its result is `SUMMARY_AUDIT.json`.
