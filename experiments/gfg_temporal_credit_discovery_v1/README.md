# GFG temporal-credit discovery v1

This experiment extends RL-E01 from *predeclared* consequence binding and
temporal credit to *discovered* long-range credit.  A 64-step terminal-only
environment supplies no ground-truth credit relation to the method.  The base
GFG retrieves formation-path candidates, matched replay adjudicates causal
effects and the resulting derived credit relations form policy updates.

The experiment is intentionally separated into development and frozen formal
execution.  Formal thresholds and source hashes are not created until the
development runs have completed and the implementation has passed replay and
leakage checks.

## Formal execution

```console
python -m experiments.gfg_temporal_credit_discovery_v1.runner \
  --mode formal \
  --artifact-root /external/artifact/path/temporal-credit-discovery-v1-formal

python -m experiments.gfg_temporal_credit_discovery_v1.independent_checker \
  --aggregate /external/artifact/path/temporal-credit-discovery-v1-formal/formal/AGGREGATE_RESULT.json \
  --output /external/artifact/path/temporal-credit-discovery-v1-formal/formal/INDEPENDENT_CHECK.json
```

The formal experiment passed all nine frozen gates and the independent check.
See `RESULTS.md` and `FORMAL_RESULT_SUMMARY.json` for the bounded scientific
claim, complete comparator table, replay cost and limitations.
