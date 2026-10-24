# Hard-spiking Adagrad TLI/TLIR falsification stress test

Status: formal four-seed execution, all 12 per-seed frozen gates passed; two complete native executions have identical per-seed raw files; independent checker passed both.

IMPORTANT: This is an original research stress test, not a native GFG Core v3 validation or a direct repeat of GFG instruments.

## Contents

- `snn_tlir_adversarial.py`: native trainer and frozen formal measurements.
- `snn_tlir_frozen_protocol.json`: pre-formal freeze, including source and checker hashes.
- `snn_tlir_independent_checker.py`: independent numerical audit using saved raw parameter states, coalitions and gradients.
- `formal/`: all four seed original native raw-state evidence files, per-seed measurements, gates, and machine aggregate.
- `replay/`: a second full independent native training execution with its own summary and identical original scientific output files.
- `SNN_TLIR_replay_comparison.json`: checksums comparing original and replay, distinguishing runtime_seconds.
- `SNN_TLIR_postcapability_exploratory.json`: explicitly post-hoc support reallocation analysis among samples retaining correct classification.
- `SNN_TLIR_local_derivative_exploratory.json`: explicitly post-hoc comparison between surrogate JVP and physical finite differences.
- `snn_derivative_boundary_exploratory.py`: code used for this post-hoc diagnostic; assumes `/mnt/data/SNN_TLIR_formal` as supplied in the original research execution, edit input path to run elsewhere.
- `TLIR_脉冲网络独立反证实验报告.md`: Chinese scientific report and boundaries.

## Dependencies

Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0 were used for the actual execution. The digits dataset comes with sklearn; no external download required.

## Reproduce in a clean folder

Put code, checker and protocol into the same directory. Then run:

```shell
python snn_tlir_adversarial.py --formal --protocol snn_tlir_frozen_protocol.json --output output_fresh
python snn_tlir_independent_checker.py --root output_fresh --protocol snn_tlir_frozen_protocol.json
```

The script rejects writing to an already existing formal output folder. The checker verifies the SHA-256 of the frozen source and all native evidence payloads.

This re-execution takes its own wall-clock time, so a byte-identical complete `FORMAL_SUMMARY.json` is not expected because it carries elapsed runtime. Every per-seed native `.pt`, `.json` measurements and frozen gates was byte-identical in the recorded repeat.

`torch.load(weights_only=False)` is used solely to read this trusted native research package. Do not load untrusted pickle files.
