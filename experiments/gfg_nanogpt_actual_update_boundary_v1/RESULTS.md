# Actual-update target-boundary prediction

## Verdict

`ACTUAL_UPDATE_BOUNDARY_PREDICTION_REPRODUCED`

This experiment predicts only whether each concrete target is correct after one already formed actual training update. It does not predict response curves or CSRG support states, and it applies no difficult-sample partition or exemption.

## Primary confirmation result

The frozen `quadratic_complete` algorithm predicted 4,652 of 5,088 confirmation targets correctly:

- final-boundary accuracy: **91.4308%**;
- balanced accuracy: **92.1661%**;
- four-way transition macro recall: **91.4877%**.

Per-transition recall was:

- maintain correct: 87.1805%;
- correct to wrong: 98.9121%;
- maintain wrong: 95.9041%;
- wrong to correct: 83.9542%.

## All-run audit

Across all twelve runs, the same algorithm predicted 14,069 of 15,264 targets correctly:

- final-boundary accuracy: **92.1711%**;
- four-way transition macro recall: **91.2952%**.

All 15,264 targets were evaluated under one rule. Total errors were 1,195; none were removed or relabelled as a separate evaluation class.

## Interpretation

The algorithm uses the full pre-update parameter state, the actual formed parameter update, the concrete evaluation input, and every target-versus-competitor boundary. It computes the hidden representation's first and second directional response and the joint rotation of output boundaries, then derives final correctness from the signs of all predicted endpoint gaps.

The result supports direct prediction of the immediate functional consequence of an actual training update. It does not claim long-horizon prediction or prediction of an update before its gradient and optimizer action have formed. This run is a raw-state re-execution of an already established algorithm, not a new prospective confirmation.

## Public recomputation

The complete machine records are distributed in the publication archive as
`tl_p01_actual_update_boundary_evidence_v1.zip`. After extracting the bundle,
run from the matching Git release:

```bash
python -m experiments.gfg_nanogpt_actual_update_boundary_v1.INDEPENDENT_CHECKER path/to/tl_p01_actual_update_boundary_v1
```

The checker validates the frozen file hashes and independently recomputes the
all-run and four-run confirmation metrics from every target record. Absolute
filesystem strings retained in `SOURCE_MANIFEST.json` are native provenance
labels and are not dereferenced by this public recomputation.
