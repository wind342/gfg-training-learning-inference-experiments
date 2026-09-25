# Preserved eligibility-filter harness failure

verification/RUN_SEAL.json records the second attempt. First-history warm-ups
and all ten timed trajectories passed exact reference comparisons. The next
lexical entry was already >=99% at initialization, so the explicit nontrivial-
history assertion stopped the run. This exposed a missing eligibility filter,
not a controller failure. No final results were written by that attempt.

Corrected selection to the first two lexically sorted histories BELOW 99%
initially, matching the intended 50/55-step pair already named in the protocol.
No choices based on new speed outcomes. Added per-history receipt persistence.
The completed attempt uses verification_v2; both old seals remain unchanged.
No core, adapter, optimization rule, reference ledger or frozen files changed.
