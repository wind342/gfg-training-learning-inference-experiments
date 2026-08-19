# Public evidence recomputation

The generated INF-E01 payload is distributed in the publication archive as
`inf_e01_frozen_inference_gfg_evidence_v1.zip`. After extracting the bundle,
run from the matching Git release:

```bash
python -m experiments.gfg_nanogpt_training_learning_inference_projection_v1.PUBLIC_EVIDENCE_CHECKER path/to/inf_e01_frozen_inference_projection_v1
```

The checker is read-only by default. It validates all 13 per-run GFG manifest,
validation, SQLite and content-addressed tensor hashes. It then recomputes the
52 checkpoint-phase logit-level component interactions and query-conditioned
support-profile distances and compares them with the frozen audit at a
`1e-12` numerical tolerance.

This verifies the archived derived GFG result. Fresh regeneration of the
rollback and gated-inference executions requires the original historical
parameter checkpoints and is a separate, more expensive operation.
