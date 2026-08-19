# GFG reinforcement-learning feedback closure v1

This package executes a delayed two-decision reinforcement-learning system and
tests whether correct action--consequence binding and temporal credit are
causally necessary for policy re-formation.  It uses a small GRU policy and
AdamW; no attention layer is present.

The frozen conditions are:

- `A`: true consequence binding and true temporal credit;
- `B`: the same within-stage consequence multiset is permuted across episodes;
- `C`: physical consequence binding is retained, but the two action-stage
  credit targets are exchanged.

The formal run uses 12 paired seeds.  Raw ledgers, checkpoints, Core v3
snapshots and GFGs are kept on the E drive.  Only compact manifests and results
are committed.

## Reproduction

With the repository root on `PYTHONPATH` and the Core source directory added:

```text
python -m experiments.gfg_rl_feedback_closure_v1.runner \
  --mode formal \
  --artifact-root E:\gfg-evidence\rl-feedback-closure-v1\formal-reproduction \
  --require-cuda

python -m experiments.gfg_rl_feedback_closure_v1.evidence \
  --aggregate E:\gfg-evidence\rl-feedback-closure-v1\formal-reproduction\formal\AGGREGATE_RESULT.json \
  --output-root E:\gfg-evidence\rl-feedback-closure-v1\formal-reproduction-evidence

python -m experiments.gfg_rl_feedback_closure_v1.independent_checker \
  --aggregate E:\gfg-evidence\rl-feedback-closure-v1\formal-reproduction\formal\AGGREGATE_RESULT.json \
  --evidence-manifest E:\gfg-evidence\rl-feedback-closure-v1\formal-reproduction-evidence\EVIDENCE_MANIFEST.json \
  --output E:\gfg-evidence\rl-feedback-closure-v1\formal-reproduction\INDEPENDENT_CHECK.json
```

Artifact directories must be new.  The runner refuses to overwrite an existing
formal execution.

