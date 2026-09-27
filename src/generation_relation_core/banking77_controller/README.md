# Final optimized Banking77 controller

Fixed production entry point: `Controller.step`. This integrates the accepted
optimizations; it does not introduce a new actuator or experimental variant:

- A: reuse post-ball `G @ t` in each Hildreth sweep.
- B: first outer uses the known zero native offset; skip zero-direction apply.
- C: reuse the checked endpoint at commit, including native-safe and rollback.
- Check the real FP32 floor after the first full sweep of each outer; accept
  immediately if safe, otherwise continue existing solver state to PASS/cap.
- Detailed hashes/records/diagnostic norms are disabled by default. Optional
  evidence is generated after training timers end. Only requested evidence
  retains intermediate geometry or copies the first-sweep dual.

Frozen MLP/AdamW, actual-update predictor, analytic Jacobian, selection, reserve,
rho, EPS, solver tolerances and buffer256 arithmetic remain as previously verified.
Actual floor checks and unsafe-candidate calculations needed for rho stay enabled.
The rejected unique-row directional arithmetic (D) is not included.

Set `OPENBLAS_NUM_THREADS=2` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing
NumPy/Torch, and call `setup()` once to use the verified deterministic settings.
Dependencies are NumPy, SciPy and CUDA PyTorch from the existing local environment.

```python
from generation_relation_core.banking77_controller import Controller

# head and shadow have the existing task-entry parameters.
# opt owns shadow parameters and keeps its moments across tasks.
controller = Controller(head, old_control_features, old_control_labels,
                        old_control_ids, seen=14)
result = controller.step(head, shadow, opt, current_x, current_y)
# Optional only when detailed diagnostics are needed:
# result = controller.step(..., capture_evidence=True)
```

Use a fresh Controller at each task entry; keep head/shadow/optimizer. Current
batches contain only the current seven intents and exclude reserved control rows.
The task1 training/checkpoint, data/cache generation and task scheduler remain
the caller's existing responsibilities. This change does not start later tasks.

The small integration check is `python -B tools/verify_banking77_final_controller.py`.
It checks only the first 20 task2 updates from each existing 210/630 task1 checkpoint
against the sealed firstcheck trajectory, including native-safe, first-sweep-safe
and solver continuation paths where present. It checks native/executed parameters,
moments, decisions, selections, sweeps and independent actual endpoints. There is
no performance benchmark, hyperparameter search or full training run.

Historical implementations and artifacts remain unchanged. The local validation
requires the already cached training features and sealed checkpoints; no raw data,
checkpoint or test data is distributed in this module.
