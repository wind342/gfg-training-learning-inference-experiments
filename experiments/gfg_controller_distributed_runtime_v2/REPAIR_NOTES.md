# Preserved software/backend attempt history

Attempt1 sources were sealed before outcomes. All12 synthetic shapes and all9
single-GPU historical cases completed. The process then exited with code1 during
the first FSDP1 case, immediately after the explicit CUDA/Gloo packed all-gather
warning, without a Python exception receipt. This is NOT recorded as a pass.
The exact native failure cause is not established.

Original seal, outputs, process-failure receipt and complete source snapshot are
preserved in external attempt-1. Attempt2 eliminates explicit singleton group
collectives: for world_size1, gather/reduction is an identity copy/no-op. This is
mathematically exact and requires no CUDA/Gloo communication. Real world2 still
uses actual collectives. The audit now checks zero singleton data collectives
versus one per packed stage when world_size>1. No scientific formula, tolerance,
learning rate, threshold, seed or candidate decision rule was changed. Gates now
run in isolated subprocesses with captured stdout/stderr/exit codes, so a native
backend crash cannot erase the surrounding suite's failure receipts.
