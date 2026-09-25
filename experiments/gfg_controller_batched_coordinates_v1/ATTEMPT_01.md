# Preserved initial harness failure

The original RUN_SEAL.json is retained unchanged. First-history batched and
reference warm-up trajectories both passed exact checks against the frozen
50-update trajectory. The following microbenchmark assertion then failed before
paired timings were collected. Its independent loop omitted the frozen
selector's float64 conversion of the float32 readout and update arrays, whereas
the actual reference selector and new generic core both performed it.

Corrected only this test-reference conversion and added separate receipt output
directories. No change to kernel.py or nanogpt_adapter.py. The corrected native
run starts again from the original historical state and seals its sources in
verification/RUN_SEAL.json. This is not a measured speed result or a changed
scientific verdict. Do not treat the initial seal as the completed-run seal.
